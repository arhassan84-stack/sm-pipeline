# FCS Single-Molecule Inference — Algorithm, Current State
**Generated:** 2026-04-16
**Source files:** measurement.py, pipeline.py, stage3_v2.py, model_registry.py

> This document describes the algorithm **as it is actually implemented today**.
> Where the implementation differs from `algorithm_sm_inference.md` (the reference design doc),
> the difference is noted explicitly.

---

## Overview

```
Input: .fcs file  +  RAW photon-arrival files (optional)
          ↓
Stage −1:  Model Registry — load noise & diffusion models from JSON catalogues
          ↓
Stage  0:  Preprocessing — FCS/RAW ingestion, multi-dt channel derivation, validation
          ↓
Stage  1:  Sliding-window noise estimation → n_abs (kHz)
          ↓
Stage  2:  Background subtraction → i_min_clean, i050_clean, i100_clean
          ↓
Stage  3:  Peak-finding + D-guided window expansion → events
          ↓
Stage  4:  Conformal uncertainty intervals added to events
          ↓
Stage  5:  Per-event quality filter
          ↓
Stage  6:  Output — per-event CSV + annotated plots
```

---

## Stage −1 — Model Registry

**Files:** `model_registry.py`, `noise_models.json`, `diffusion_models.json`

Two JSON catalogues define all available models. A `ModelRegistry` object is instantiated
once at startup and passed through all subsequent stages.

### Noise models (noise_models.json)
Each entry specifies: `label`, `window_ms`, `mape`, `requires_raw`, `dt_channels`,
`feat_dim`, `requires_acf_features`, `model_file` (and optional scaler files).

Current models:
| Label | window_ms | MAPE | requires_raw | Features |
|---|---|---|---|---|
| multidt_noise | 4096 | 2.2% | true | 916 (910 ACF + 6 raw) |
| noise_w2048   | 2048 | 3.8% | true | 6 raw stats |
| noise_w1024   | 1024 | 6.0% | true | 6 raw stats |
| noise_w512    |  512 | 10.5% | true | 6 raw stats |
| noise_w256    |  256 | 14.4% | true | 6 raw stats |

> **In progress:** `noise_w512_v2` (job 8483290 running) — same architecture as `noise_w512`
> but trained with traces in kHz (/ 1000 at load) to fix a unit mismatch that caused the
> old models to predict near-zero n_abs values on real FCS data.

### Diffusion models (diffusion_models.json)
| Label | window_ms | MAPE | requires_raw | dt channels |
|---|---|---|---|---|
| multidt_B_large | 4096 | 9.5% | true  | 0.1, 0.5, 1.0 ms |
| multidt_B       | 4096 | 9.9% | true  | 0.1, 0.5, 1.0 ms |
| wavenet_wide_aug | 4096 | 12.3% | false | 1.0 ms only |

### Registry API
- `registry.noise_models.best_for(trace_length_ms, has_raw)` → lowest-MAPE admissible model
- `registry.diffusion_models.best_for(trace_length_ms, has_raw)` → best admissible D model
- `registry.dt_min` → finest dt across all loaded models (currently 0.1 ms)
- Each `ModelWrapper` exposes `predict(traces, features)` → n_abs (kHz) or D (µm²/s)

---

## Stage 0 — Preprocessing

**File:** `measurement.py` → `load_measurement(fcs_path, data_dir, registry, raw_opts)`
**Output:** fully-populated `Measurement` object

### 0.1 FCS Ingestion
Parse `.fcs` file via `fcs_io.parse_fcs_file`. Extract:
- `name` (filename stem), `n_channels`, `n_submeasurements`, `total_duration_ms`
- Per-sub-measurement intensity traces (1ms bins, kHz) via `fcs.intensity_trace(channel)`

### 0.2 Admissibility Check
Computed minimum required duration:
```
min_duration_ms = max(
    min(m.window_ms for m in noise_models),     # currently 256 ms
    min(m.window_ms for m in diffusion_models)  # currently 4096 ms
) = 4096 ms
```
Raises `MeasurementError` if `total_duration_ms < 4096 ms` or `n_submeasurements < 1`.

### 0.3 RAW File Discovery and Binning
Scan `data_dir` for files matching `{name}_*_R{r}_*_Ch*.raw`.
Parse `R{r}` → sub-measurement index; `Ch{channel_id}` → channel label (e.g. `S1`, `S2`).
`has_raw = True` if any matching files are found.

For each RAW file: call `raw_io.read_arrivals` → photon timestamps → `raw_io.bin_arrivals`
at `registry.dt_min` (0.1 ms). Result: `i_min_full` at 0.1 ms resolution (kHz).

RAW options default: `bits=16, n_header=48, clock_rate_hz=15_000_000`.

### 0.4 Concatenation
All sub-measurement RAW segments are concatenated in R-order to form `i_min_full`.
If `has_raw=False`, only `i100_full` (from FCS) is populated.

### 0.5 Multi-dt Derivation and Validation
Derived by block-summing `i_min_full` (sum preserves kHz count-rate units):
```
B050 = round(0.5 / dt_min) = 5    (bins at 0.1ms per 0.5ms bin)
B100 = round(1.0 / dt_min) = 10   (bins at 0.1ms per 1.0ms bin)

i050_full = i_min_full.reshape(-1, B050).sum(axis=1)   # kHz
i100_full = i_min_full.reshape(-1, B100).sum(axis=1)   # kHz
```

**Validation against FCS 1ms trace** (per sub-measurement):
- Primary criterion: total photon count match — `|sum(i100_derived) − sum(i100_fcs)| / sum(i100_fcs) ≤ 5%`
- Pearson r is computed and logged as a diagnostic but does NOT gate pass/fail
  (rationale: SM-FCS traces are mostly background, so r ≈ 0 even for correctly binned data)
- If > 50% of evaluated sub-measurements fail: downgrade to `has_raw=False`, strip i_min and i050

> **Difference from design doc:** design doc specifies r ≥ 0.99 and MARD ≤ 2% as hard criteria.
> Implementation uses count-match ≤ 5% only (r logged for diagnostics).

### 0.6 Output — Measurement Object
```
Measurement
├── name, id, path_fcs
├── has_raw (bool)           # True if RAW loaded & validated
├── suspect_raw (bool)       # True if >50% sub-measurements failed validation
├── n_channels, total_duration_ms, dt_min_ms
├── channel_ids              # e.g. ['S1', 'S2']
├── channels[ch_id]:
│     ├── i_min_full         # (10N,) kHz — if has_raw
│     ├── i050_full          # (2N,)  kHz — if has_raw
│     ├── i100_full          # (N,)   kHz — always
│     └── validation: {r, counts_match, failed_submeasurements}
└── source_files[ch_id]      # [fcs_path, raw_paths ...]
```

---

## Stage 1 — Sliding-Window Noise Estimation

**File:** `pipeline.py` → `_stage1_noise(meas, ch_data, registry)`
**Output:** `n_abs` (kHz scalar), `noise_label` (str)

Select noise model: `registry.noise_models.best_for(trace_length_ms, has_raw)`.
If no admissible model (e.g. `has_raw=False`, no noise model without RAW): return `None`.

**Window parameters (read from model metadata):**
```
W        = noise_model.window_ms        # e.g. 512 ms
stride   = W // 8                       # e.g. 64 ms
R_n0     = round(1.0 / dt_n0)          # bins per 1ms in finest channel
                                         # (e.g. 10 for dt=0.1ms)
```

**Per-window inference:**
For each window `j` with `t0 = j × stride`, `t1 = t0 + W`:
1. Extract `w100 = i100[t0:t1]` (W bins)
2. Extract `w050 = i050[t0*2 : t1*2]` (2W bins) if available
3. Extract `w_min = i_min[t0*R_n0 : t1*R_n0]` (W×R_n0 bins) if available
4. Compute 6 raw stats: `[mean, std]` × 3 channels — always required by the model
5. Build `noise_traces = {dt: window_array}` for each dt in `noise_model.dt_channels`
6. Call `noise_model.predict(traces=noise_traces, features=feats)` → `n_hat` (kHz)
7. Append to `n_abs_local`

**Global estimate:**
```
n_abs = percentile(n_abs_local, 20)
```
20th percentile: windows containing molecular signal overestimate background;
low-percentile windows correspond to background-only periods.

---

## Stage 2 — Background Subtraction

**File:** `pipeline.py` → `_stage2_subtract(ch_data, n_abs, meas)`
**Output:** `(i_min_clean, i050_clean, i100_clean)` — float64, kHz, clipped ≥ 0

If `n_abs` is available (Stage 1 succeeded):
```
bg = n_abs
```

If `n_abs` is None (`has_raw=False`):
```
bg = percentile(i100_full, 5)    # 5th-percentile fallback
meas.background_estimated = True
```

**Safety cap** (prevents over-subtraction):
```
clip_cap = percentile(i100_full, 10)
if bg > clip_cap:
    bg = clip_cap    # cap so at most 10% of bins are zeroed after subtraction
```
Warns if cap is applied or if > 5% of bins clip to zero after subtraction.

**Subtraction:**
```
i_min_clean = maximum(0, i_min_full − bg)   # if has_raw
i050_clean  = maximum(0, i050_full  − bg)   # if has_raw
i100_clean  = maximum(0, i100_full  − bg)   # always
```

---

## Stage 3 — Event Detection (v2: Peak-Finding + D-Guided Expansion)

**File:** `stage3_v2.py` → `run_stage3_v2(bundle, d_model)`
**Input:** `TraceBundle` + diffusion `ModelWrapper`
**Output:** `(events: list[dict], z_score: ndarray)`

> **Difference from design doc:** the design doc describes a matched-filter bank (Stage 3.1–3.5)
> that operated on `i100_clean`. The current implementation (v2) replaces this entirely with
> the peak-finding + D-guided expansion algorithm described below.

### TraceBundle
```python
@dataclass
class TraceBundle:
    i1ms:    ndarray      # (N,)   raw intensity, 1ms bins, kHz
    baseline: object      # scalar or (N,) background from noise model, kHz
    i_dt050: ndarray|None # (2N,)  background-subtracted 0.5ms intensity, kHz
    i_dt010: ndarray|None # (10N,) background-subtracted 0.1ms intensity, kHz
```
A scalar `baseline` is broadcast to shape `(N,)`.
`i_dt050` and `i_dt010` come from Stage 2 (already background-subtracted).

### Visualisation z-score (computed unconditionally)
```
z_score = (i1ms − baseline) / sqrt(max(baseline, 1e-9))
```
Poisson z-score: σ = √(background count rate). Returned alongside events for plotting.

---

### Step 1 — Peak Finding + Claiming
**Function:** `_step1(bundle)` → `(cat1, cat2, claimed)`

#### 1a. Peak Detection
```python
peak_idx, _ = find_peaks(
    i1ms,
    height   = bundle.baseline,   # each peak must exceed its local baseline
    distance = 50,                 # minimum inter-peak spacing: 50 ms
)
```
Uses raw `i1ms` (not background-subtracted), with `baseline` as the height threshold.

#### 1b–c. Half-Width and Classification
For each detected peak at `tp`:

**w12 computation** (`_compute_w12`):
```
I_cut     = TRNST_LVL × signal[tp]     # TRNST_LVL = 0.90
t_left_w  = last index < tp where signal < I_cut    (or 0 if none)
t_right_w = first index > tp where signal < I_cut   (or N-1 if none)
w12       = max((t_right_w − t_left_w) // 2,  1)
```

**Classification:**
- `2 × w12 ≥ 200 ms` → **Cat 1** (broad / slow dwellers)
- `2 × w12 < 200 ms` → **Cat 2** (narrow / fast transients)

Both lists are sorted by `h_peak = signal[tp]` descending before claiming.

#### 1d. Cat 1 Claiming Loop (descending h_peak)
For each Cat 1 event (highest peak first):
```
if event already excluded: skip
claim [tp − w12, tp + w12]     (capped to [0, N-1])
exclude all other peaks (Cat 1 + Cat 2) whose tp falls inside the claimed window
```
Cat 1 peaks whose `tp` was excluded by a higher-amplitude Cat 1 are removed.

#### 1e. Cat 2 Claiming Loop (descending h_peak)
```
half = W_START2_MS // 2 = 64 ms
if event excluded by Cat 1 loop: skip
claim [tp − 64, tp + 64]
exclude other Cat 2 peaks whose tp falls inside
```
Cat 2 peaks cannot exclude Cat 1 events (Cat 1 loop already ran).

#### 1f. Output
```
cat1:    surviving Cat 1 seeds, sorted by h_peak desc  (have t_start, t_end)
cat2:    surviving Cat 2 seeds, sorted by h_peak desc
claimed: bool (N,) — all seed bins marked True
```

**Key parameters:**
```
PEAK_DISTANCE_MS  = 50    # scipy find_peaks distance
TRNST_LVL         = 0.90  # threshold fraction of peak for w12
CAT1_MIN_WIDTH_MS = 200   # 2*w12 >= this → Cat 1
W_START2_MS       = 128   # Cat 2 seed window total width (ms)
```

---

### Step 2 — D-Guided Window Expansion
**Function:** `_step2(bundle, seed_events, claimed, d_model, category)` → `list[dict]`
Called separately for Cat 1 and Cat 2 seeds. Modifies `claimed` in-place.

For each seed event:

#### 2a. Initial D Prediction
Call `_predict_D(bundle, tp, d_model)`:
1. Build a W_D = 4096ms window centred on `tp` (always fixed, unconditional):
   ```
   inf_s = max(0, tp − W_D // 2)
   inf_e = min(N−1, tp + W_D // 2)
   ```
2. For each dt channel in `d_model.dt_channels`:
   - `dt=1.0ms`: subtract `baseline[inf_s:inf_e+1]`, clip at 0; zero-pad to W_D bins
   - `dt=0.5ms`: use `bundle.i_dt050[inf_s*2 : (inf_e+1)*2]` directly; zero-pad to 2×W_D bins
   - `dt=0.1ms`: use `bundle.i_dt010[inf_s*10 : (inf_e+1)*10]` directly; zero-pad to 10×W_D bins
3. If `d_model.requires_acf_features`: compute ACF features per dt, concatenate
4. Call `d_model.predict(traces, features)` → D̂ (µm²/s)

#### 2b. Fast-Group Check
```
if D̂ > D_FAST_CUTOFF (= 1.0 µm²/s):
    fast_group = True
    keep seed window; skip expansion
```

#### 2c–e. Window Doubling (slow group only)
Look up `tau_max = _lookup_tau_max(D̂)` from expansion table:

| D̂ range (µm²/s) | tau_max (ms) |
|---|---|
| D̂ ≤ 0.02 | 2048 |
| D̂ ≤ 0.06 | 1024 |
| D̂ ≤ 0.20 |  512 |
| D̂ ≤ 0.50 |  256 |
| D̂ ≤ 1.00 |  128 |

Doubling loop:
```
w = current window width (= t_end − t_start)

while w < W_MAX_MS (= 4096 ms):
    w_new  = 2 × w
    ts_new = max(0, tp − w_new // 2)
    te_new = min(N−1, tp + w_new // 2)

    Stop 1: break if w_new ≥ W_MAX_MS
             OR any bin in [ts_new, t_start) is already claimed
             OR any bin in (t_end, te_new] is already claimed

    Stop 2: break if w_new ≥ tau_max

    Accept expansion:
        mark [ts_new, t_start) and (t_end, te_new] as claimed
        t_start, t_end = ts_new, te_new
        w = w_new

        Re-predict D̂ on the same W_D = 4096ms window centred on tp
        if D̂ > D_FAST_CUTOFF: fast_group = True; break
        update tau_max = _lookup_tau_max(D̂)
```

#### 2f. Output Dict per Event
```python
{
    't_peak'    : tp,
    't_left'    : t_start,
    't_right'   : t_end,
    'W_seg'     : (t_end − t_start + 1) // 2,
    'D_hat'     : D̂,                     # µm²/s
    'log10_D'   : log10(D̂),
    'category'  : 1 or 2,
    'fast_group': bool,
    'score'     : (i1ms[tp] − baseline[tp]) / sqrt(max(baseline[tp], 1e-9)),
    'pad_frac'  : max(0, (W_D − dur) / W_D) if dur < W_D else 0.0,
    'n_win'     : 1 if dur ≤ W_D else max(1, (dur−W_D) // (W_D//8) + 1),
    'k_win'     : 0,   # kept for pipeline compatibility; no template bank in v2
}
```

### Step 3 — Residual Pass (deferred)
Not yet implemented. Intended to detect molecular events in unclaimed trace bins using
a lower-sensitivity method. Will be added after evaluating Step 1–2 results.

---

## Stage 4 — Conformal Uncertainty Intervals

**File:** `pipeline.py` → `_stage4_conformal(events, d_model)`

> **Difference from design doc:** the design doc describes Stage 4 as performing D-prediction
> (window assembly, feature computation, model inference). In the current implementation,
> D prediction is performed inside `_predict_D` during Stage 3 (Step 2). Stage 4 only
> adds conformal uncertainty intervals and bookkeeping fields. A full `_stage4_predict`
> function exists in pipeline.py but is not called.

For each event carrying `D_hat` and `pad_frac`:
```
if d_model.conformal_intervals is not None:
    q      = conformal_intervals.lookup(pad_frac)  # pre-computed quantile by pad stratum
    D_low  = 10 ^ (log10_D − q)
    D_high = 10 ^ (log10_D + q)
else:
    D_low = D_high = None

if pad_frac > 0.75: flags.append('low_signal')
```

Adds to each event: `D_low`, `D_high`, `d_hat_all = [D_hat]`, `d_model_label`, `flags`.

**Conformal interval semantics:** `q_log10` is the 90th percentile of
`|log10(D_pred) − log10(D_true)|` computed on the held-out test set, stratified by
`pad_fraction`. Coverage is ≥ 90% empirically.

---

## Stage 5 — Quality Filter

**File:** `pipeline.py` → `_stage5_filter(events, meas)`
**Output:** `(passing: list, rejected: list)`

| Criterion | Threshold | Action |
|---|---|---|
| Duration `t_right − t_left` | < 64 ms | Hard discard → `too_short` |
| `D_hat` | outside (0, 14) µm²/s | Hard discard → `d_out_of_range` |
| `D_cv` (log10 std across windows, Case 3 only) | > 0.35 | Soft flag → `variable_D` |
| `n_frac` | not implemented yet | `noise_unavailable` flag always added |

`n_frac` (background fraction at peak): computation is stubbed — requires `i100_clean`
to be in scope. Currently the check is skipped and every event gets `noise_unavailable`.

> **Note:** `pad_frac` is not a filter criterion. Fast-diffuser events (small w12, large
> pad_frac) are valid detections; pad_frac is reported as a diagnostic.

---

## Stage 6 — Output

**File:** `pipeline.py` → `_stage6_output(result, z_score, i100_clean, out_dir, ...)`

### CSV (`{stem}_events.csv`)
One row per passing event. Columns:
```
t_peak_ms, t_left_ms, t_right_ms, duration_ms,
D_hat, log10_D, D_low, D_high,
pad_fraction, n_win, D_cv, k_win, score,
n_frac, model_used, noise_model, flags
```

### Plots (if `save_plots=True`)

1. **`{stem}_trace.png`** — `i100_clean` intensity trace (bright green line on black
   background); passing events shaded by log10(D̂) via RdYlBu_r colormap; rejected events
   in grey. Colormap range: log10(D) ∈ [−2, 1].

2. **`{stem}_d_scatter.png`** — scatter plot of t_peak (s) vs log10(D̂), points coloured
   by `pad_fraction` (viridis_r), conformal intervals as error bars.

3. **`{stem}_d_histogram.png`** — histogram of log10(D̂) for all passing events,
   bin width 0.1, range [−2.5, 1.5].

---

## Key Parameters Summary

| Parameter | Value | Stage | Purpose |
|---|---|---|---|
| W_D | 4096 ms | 3, 4 | D-model inference window (fixed) |
| TRNST_LVL | 0.90 | 3 Step 1 | Fraction of peak height for w12 |
| CAT1_MIN_WIDTH_MS | 200 ms | 3 Step 1 | Minimum 2×w12 for Cat 1 |
| W_START2_MS | 128 ms | 3 Step 1 | Cat 2 seed window width |
| PEAK_DISTANCE_MS | 50 ms | 3 Step 1 | Min inter-peak spacing |
| D_FAST_CUTOFF | 1.0 µm²/s | 3 Step 2 | Fast-group threshold |
| W_MAX_MS | 4096 ms | 3 Step 2 | Max window after expansion |
| D_CUTOFFS | [0.02, 0.06, 0.20, 0.50, 1.00] | 3 Step 2 | Expansion table |
| TRANSIT_TIMES | [2048, 1024, 512, 256, 128] ms | 3 Step 2 | Expansion table |
| Noise percentile | 20th | 1 | Background aggregation |
| Bg cap percentile | 10th | 2 | Prevents over-subtraction |
| QF_MIN_DURATION_MS | 64 ms | 5 | Hard discard threshold |
| QF_D_MAX | 14 µm²/s | 5 | Hard discard threshold |
| QF_D_CV_SOFT | 0.35 (log10 units) | 5 | Soft flag for variable D |

---

## Data Flow: Units and Array Dimensions

All intensity arrays are in **kHz** throughout the pipeline (count rate, not counts per bin).

| Array | Resolution | Length relative to N (total 1ms bins) |
|---|---|---|
| `i_min_full` / `i_dt010` | 0.1 ms | 10N |
| `i050_full` / `i_dt050` | 0.5 ms | 2N |
| `i100_full` / `i1ms` | 1.0 ms | N |
| `baseline` | 1.0 ms | N (or scalar broadcast) |
| `z_score` | 1.0 ms | N |

Background subtraction (`bg`) is a single scalar applied to all dt channels identically:
block-averaging preserves mean count rate, so the same n_abs (kHz) is valid at all resolutions.

---

## Known Issues / Pending Work

1. **n_abs underestimation** — old noise models trained with traces in Hz instead of kHz
   → raw-stat features 1000× out of distribution at inference. Fixed in `noise_w512_v2`
   (retraining in progress, job 8483290). All `noise_w*` and `multidt_noise` models
   need retraining with the v2 unit fix before n_abs values are trustworthy.

2. **n_frac check in Stage 5** — stubbed. Requires i100_clean at the peak bin;
   needs to be threaded through from Stage 3 output.

3. **Step 3 (residual pass)** — deferred. Will add detection in unclaimed trace bins
   after evaluating Step 1–2 results on real data.

4. **Stage 3 uses raw i1ms for peak finding** — peaks are found in the raw (not
   background-subtracted) trace, but the `height=baseline` requirement ensures only
   supra-background peaks are accepted. This means bright background fluctuations
   that are spatially uniform (spikes) could trigger false peaks; no spike-rejection
   filter is currently implemented.

5. **algorithm_sm_inference.md is out of date** — describes the old matched-filter Stage 3
   (stage3_version_1.py / stage3.py). Should be updated to reflect the v2 algorithm.

---

*End of document*
