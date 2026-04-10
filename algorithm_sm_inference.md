# FCS Single-Molecule Segmentation and Inference Algorithm

**Project:** FCS Diffusion Coefficient and Noise Inference
**Last updated:** 2026-04-10
**Status:** Draft — updated iteratively

---

## Overview

The pipeline takes a long photon-count time series composed of randomly stitched single-molecule FCS traces (each segment stationary with its own diffusion coefficient D) and:

1. Loads and registers inference models
2. Preprocesses raw FCS and photon-arrival-time (RAW) data
3. Estimates the global background noise level
4. Detects segment boundaries via intensity thresholding
5. Predicts D per segment using a sliding window
6. Applies change-point detection to resolve transitions
7. Outputs a per-segment table and annotated plots

```
Input: FCS file(s) + RAW photon-arrival files
          ↓
Stage -1:  Model registry — load noise and diffusion models
          ↓
Stage  0:  Preprocessing — FCS/RAW ingestion, channel derivation, validation
          ↓
Stage  1:  Sliding-window noise estimation → n_abs
          ↓
Stage  2:  Background subtraction → cleaned channels (i_min_clean, i050_clean, i100_clean)
          ↓
Stage  3:  Matched-filter event detection → candidate segments
          ↓
Stage  4:  Per-segment D-prediction → D̂ per event
          ↓
Stage  5:  Per-event quality filter and aggregation
          ↓
Stage  6:  Output — table + annotated plots
```

---

## Stage −1 — Model Registry

### −1.1 Model Catalogue Files

Two JSON catalogue files define all available models.

**`noise_models.json`**

```json
[
  {"label": "multidt_noise",  "window_ms": 4096, "mape": 2.2,
   "requires_raw": true,  "dt_channels": [0.1, 0.5, 1.0],
   "feat_dim": 916, "requires_acf_features": true,
   "model_file": "model_wavenet_multidt_noise.pt",
   "scaler_mean": "scaler_wavenet_multidt_noise_mean.npy",
   "scaler_scale": "scaler_wavenet_multidt_noise_scale.npy"},

  {"label": "noise_w2048",    "window_ms": 2048, "mape": 3.8,
   "requires_raw": true,  "dt_channels": [0.1, 0.5, 1.0],
   "feat_dim": 6,  "requires_acf_features": false,
   "model_file": "model_wavenet_noise_w2048.pt"},

  {"label": "noise_w1024",    "window_ms": 1024, "mape": 6.0,
   "requires_raw": true,  "dt_channels": [0.1, 0.5, 1.0],
   "feat_dim": 6,  "requires_acf_features": false,
   "model_file": "model_wavenet_noise_w1024.pt"},

  {"label": "noise_w512",     "window_ms": 512,  "mape": 10.5,
   "requires_raw": true,  "dt_channels": [0.1, 0.5, 1.0],
   "feat_dim": 6,  "requires_acf_features": false,
   "model_file": "model_wavenet_noise_w512.pt"},

  {"label": "noise_w256",     "window_ms": 256,  "mape": 14.4,
   "requires_raw": true,  "dt_channels": [0.1, 0.5, 1.0],
   "feat_dim": 6,  "requires_acf_features": false,
   "model_file": "model_wavenet_noise_w256.pt"}
]
```

**`diffusion_models.json`**

```json
[
  {"label": "multidt_B_large", "window_ms": 4096, "mape": 9.5,
   "requires_raw": true,  "dt_channels": [0.1, 0.5, 1.0],
   "feat_dim": 910, "requires_acf_features": true,
   "model_file": "model_wavenet_multidt_B_large.pt",
   "scaler_mean": "scaler_wavenet_multidt_B_large_mean.npy",
   "scaler_scale": "scaler_wavenet_multidt_B_large_scale.npy"},

  {"label": "wavenet_wide_aug", "window_ms": 4096, "mape": 12.3,
   "requires_raw": false, "dt_channels": [1.0],
   "feat_dim": "ACF features (1ms lags)", "requires_acf_features": true,
   "model_file": "model_pt_wavenet_wide_aug.pt",
   "note": "Best single-dt model; fallback when RAW files are unavailable"}
]
```

> **Note on the single-dt fallback model:** `wavenet_wide_aug` is a single-branch WaveNet
> trained on dt=1.0ms traces (N_BINS=4096, tMax=4096ms). It is the best model in its class
> (MAPE=12.3%, d<1=9.8%, d≥1=17.6%) and is used as the fallback diffusion model when RAW
> photon-arrival files are unavailable and only the FCS 1ms intensity trace exists. It is
> inferior to `multidt_B_large` (MAPE=9.5%) but fully functional on dt=1ms-only data.

---

### −1.2 Model Wrapper Class

Each model is wrapped in an object exposing a common interface:

```
ModelWrapper
├── metadata:       label, window_ms, mape, dt_channels, requires_raw,
│                   feat_dim, requires_acf_features
│
├── is_admissible(trace_length_ms, has_raw) → (bool, reason_string)
│     Checks:
│       - trace_length_ms >= window_ms
│       - if requires_raw and not has_raw → inadmissible
│       - model weights loaded without error
│
├── predict(i010, i050, i100, features=None) → float
│     Checks before inference:
│       - len(i010) == window_ms * 10   (for dt=0.1ms models)
│       - len(i100) == window_ms
│       - if requires_acf_features and features is None → raise ValueError
│       - all input arrays are finite and non-negative
│     Returns:
│       - noise models      → n_abs  (kHz)
│       - diffusion models  → D      (µm²/s)
│
└── __repr__() → one-line summary string
```

---

### −1.3 Registry Loader

A `ModelRegistry` object is instantiated once at program start:

```
ModelRegistry
├── noise_models:      dict[window_ms → NoiseModelWrapper]
│     best_for(trace_length_ms, has_raw)
│       → returns the admissible noise model with the lowest MAPE
│
├── diffusion_models:  dict[label → DiffusionModelWrapper]
│     best_for(trace_length_ms, has_raw)
│       → returns best admissible D model
│          (multi-dt preferred over single-dt when has_raw=True)
│
├── dt_min  (float, ms)
│     Computed at load time:
│       dt_min = min(dt for model in all_models for dt in model.dt_channels)
│     Currently: 0.1 ms (from multidt_B_large and all noise_w* models)
│     Automatically updates if finer-resolution models are added to the catalogue
│     (e.g., a future multidt_E model with dt=0.05ms would set dt_min=0.05ms)
│     This value is passed to Stage 0.2 to control RAW file binning resolution.
│
└── load_all()
      - loads all .pt weights to GPU (CPU fallback if no GPU)
      - verifies each model file exists and loads without error
      - logs a summary table of loaded vs. failed/unavailable models
      - computes and logs dt_min after loading
```

The registry is loaded once and passed through all subsequent pipeline stages. No model is
loaded on demand during inference.

---

## Stage 0 — Preprocessing

### 0.1 FCS File Ingestion

**Source:** FCS file from a Zeiss confocal microscope.
**Translation required:** Matlab FCS reader → Python (to be provided).

Steps:

1. Parse FCS file and extract:
   - `name`: filename stem (no extension) → used as the measurement identifier throughout
   - `id`: sequential integer assigned at load time
   - `n_channels`: number of detector channels (typically 1 or 2)
   - `n_submeasurements`: number of sub-measurement blocks (typically 2-second duration each)
   - `submeasurements[ch][k]`: intensity trace array for channel `ch`, block `k`
     - Each array: 2000 bins at dt=1ms (2 s × 1000 bins/s)
   - `dt_fcs = 1.0 ms` (fixed)

2. Store all metadata and traces in a `Measurement` object indexed by `(name, id)`.

---

### 0.2 Admissibility Check

> ⚠️ **TO BE COMPLETED** — The checks below cover the minimum requirements derived from
> the noise and diffusion model windows. Additional checks should be added here as each
> subsequent algorithm stage is finalised (e.g., minimum signal level, maximum gap fraction,
> channel count requirements). Revisit this section once Stages 1–7 are fully defined.

Performed immediately after FCS ingestion, before any RAW file loading, to avoid
unnecessary I/O on measurements that cannot be processed.

**Compute minimum required duration from registry:**
```
min_noise_window = min(m.window_ms for m in registry.noise_models.values())
min_d_window     = min(m.window_ms for m in registry.diffusion_models.values())
min_duration_ms  = max(min_noise_window, min_d_window)
                   # currently: max(256ms, 4096ms) = 4096ms
```

**Checks applied to each `Measurement`:**

| Check | Criterion | Action on failure |
|---|---|---|
| Minimum total duration | `total_duration_ms ≥ min_duration_ms` | omit measurement, log reason |
| At least one admissible D model | `registry.diffusion_models.best_for(has_raw)` exists | omit measurement, log reason |
| Sub-measurement count | `n_submeasurements ≥ 1` | omit measurement, log reason |
| *(further checks — TO BE ADDED)* | | |

Omitted measurements are recorded in a run-level rejection log:
```
rejection_log[name] = {id, reason, total_duration_ms, has_raw}
```
and excluded from all downstream stages. They are not silently dropped — the log is
written to `{run_id}_rejected_measurements.csv` at the end of the run.

---

### 0.3 RAW File Discovery and Photon-Arrival Binning

**Source:** RAW photon-arrival-time files (Sony detector, acquired via Zeiss system).
**Translation required:** Matlab RAW reader → Python (to be provided).

**Naming convention:**
```
{name}_{hash}_R{r}_P{p}_K{k}_Ch{channel_id}.raw
```

Fields:
- `{name}` — FCS filename stem, exactly matching the parent `.fcs` file (e.g. `lateralDorsal_nt_1`)
- `{hash}` — 32-character hexadecimal identifier string (e.g. `6e127ff34069a65c902167b49f354dcd`)
- `R{r}` — **sub-measurement index** (maps 1:1 to sub-measurements in the FCS file)
- `P{p}` — position index (meaning to be confirmed from Matlab script)
- `K{k}` — additional index (meaning to be confirmed from Matlab script)
- `Ch{channel_id}` — detector channel label (e.g. `S1`, `S2`, or other labels; open-ended)

**Example:** `lateralDorsal_nt_1_6e127ff34069a65c902167b49f354dcd_R2_P1_K1_ChS2.raw`
corresponds to sub-measurement R=2, channel S2 of `lateralDorsal_nt_1.fcs`.

Steps:

1. For each `Measurement` with stem `{name}`:
   - Scan the data directory for all files matching `{name}_*_R{r}_*_Ch*.raw`
   - Parse `R{r}` → sub-measurement index, `Ch{channel_id}` → channel label
   - Build a map: `raw_files[channel_id][r] = filepath`  (or `None` if missing)
   - Discover channel labels dynamically from the filenames found (do not hardcode)

2. **Completeness check:**
   - `has_raw = True`  if all R indices [1…R_max] have at least one channel present
   - `has_raw = False` if any sub-measurement index R has no RAW file for any channel
   - Attach `has_raw` flag to the `Measurement` → controls model selection in Stage −1.3
   - Log a warning listing any missing (r, channel_id) pairs

3. If `has_raw = True`:
   - For each RAW file: read photon arrival timestamps
   - **Timestamp unit:** open parameter `dt_raw_unit` (to be determined from the Matlab
     script and Sony RAW file format specification; e.g. seconds, milliseconds, or
     hardware clock ticks requiring a conversion factor)
   - Convert timestamps to seconds if not already, then bin into `registry.dt_min` bins
     using a histogram over `[0, t_end_submeasurement]`
   - Result: `i_min_sub[channel_id][r]` — array of `t_submeasurement_s / dt_min_s` bins
     (e.g. 20,000 bins for 2 s at dt_min=0.1ms; automatically adjusts if dt_min changes)

> The Matlab RAW reader will be translated to Python and adapted to output binned arrays
> at dt_min = 0.1 ms. Binning uses a standard histogram on the photon arrival timestamps.
> The exact binary format of the Sony RAW files, the timestamp unit, and the meanings of
> P{p} and K{k} will be determined from the Matlab script once provided.

---

### 0.4 Concatenation

For each measurement and each channel, concatenate all sub-measurement traces in order:

```
i010_full[ch] = concatenate( i010_sub[ch][0],
                              i010_sub[ch][1],
                              ...,
                              i010_sub[ch][K−1] )
```

If `has_raw = False`: skip `i010_full` — only `i100_full` will be constructed from FCS
intensity traces (Stage 0.4).

Each resulting trace is stored in the `Measurement` object with full provenance:

```
Measurement.channels[ch]:
  - i010_full             (if has_raw; length = K × 20000)
  - fcs_submeasurements   (always;    list of K arrays, each length 2000)
  - source_files          [list of raw paths + fcs path]
  - total_duration_ms
```

---

### 0.5 Multi-dt Channel Derivation and Validation

**If `has_raw = True`:**

Derive channels by block-averaging `i_min_full` using `registry.dt_min` as the base
(units: kHz throughout). Block sizes are computed as `round(dt_target / dt_min)`:

```
i050[k] = mean( i_min[ k*B050 : k*B050+B050 ] )   B050 = round(0.5ms / dt_min)
i100[k] = mean( i_min[ k*B100 : k*B100+B100 ] )   B100 = round(1.0ms / dt_min)
```

At current dt_min=0.1ms: B050=5, B100=10 (i.e. same as before).
If dt_min changes (e.g. 0.05ms), block sizes update automatically: B050=10, B100=20.

**Validation — compare derived `i100` against FCS intensity trace:**

For each sub-measurement `k` and channel `ch`:

```
i100_derived  =  i100[ k*2000 : (k+1)*2000 ]     (from RAW binning)
i100_fcs      =  fcs_submeasurements[ch][k]       (from FCS file, 1ms bins)
```

Three checks applied per sub-measurement:

| Check | Criterion | Action on failure |
|---|---|---|
| Pearson correlation | r ≥ 0.99 | warn + flag sub-measurement |
| Mean absolute relative difference | MARD ≤ 2% | warn + flag sub-measurement |
| Total photon count match | \|sum(derived) − sum(fcs)\| / sum(fcs) ≤ 1% | warn + flag sub-measurement |

- Flag any failing sub-measurement as `validation_failed` → excluded from downstream
- If > 20% of sub-measurements fail: flag entire measurement as `suspect_raw` and
  downgrade to `has_raw = False` for model selection

**If `has_raw = False`:**

Concatenate FCS sub-measurement traces directly:

```
i100_full[ch] = concatenate( fcs_submeasurements[ch][0..K−1] )
```

`i010` and `i050` are unavailable. Only models with `requires_raw = False` are admissible.

---

### 0.6 Output of Stage 0

```
Measurement
├── name                  (str)
├── id                    (int)
├── has_raw               (bool)
├── suspect_raw           (bool)
├── n_channels            (int)
├── total_duration_ms     (float)
├── channel_ids           (list[str], e.g. ['S1', 'S2'])
├── channels[channel_id]:
│     ├── i_min_full      (ndarray, if has_raw; binned at registry.dt_min)
│     ├── i050_full       (ndarray, if has_raw; block-averaged to dt=0.5ms)
│     ├── i100_full       (ndarray, always;     block-averaged to dt=1.0ms
│     │                    or directly from FCS if has_raw=False)
│     └── validation:
│           ├── r                       per sub-measurement
│           ├── mard                    per sub-measurement
│           ├── counts_match            per sub-measurement
│           └── failed_submeasurements  list of indices
└── source_files[channel_id]  [fcs_path, raw_paths ...]
```

This object is passed to Stage 1, which calls:
```
registry.noise_models.best_for(has_raw=measurement.has_raw)
```
to automatically select the most accurate admissible noise model.

---

## Stage 1 — Sliding-Window Noise Estimation

**Model selection:**
```
noise_model = registry.noise_models.best_for(has_raw=measurement.has_raw)
```
Selection is based solely on channel availability (`has_raw`), since the noise model is
always applied in a sliding window across the full trace — the total trace duration is not
a selection criterion. The registry returns the admissible noise model with the lowest MAPE.
If `has_raw=False`, only models with `requires_raw=False` are considered; currently no such
noise model exists, so this stage is skipped and `n_abs` is flagged as unavailable.

**Parameters** (read from selected model's metadata):
```
W_noise   = noise_model.window_ms          # e.g. 2048ms
stride_ms = W_noise // 8                   # e.g. 256ms
B_min     = round(W_noise / registry.dt_min)   # bins in i_min per window
B050      = round(W_noise / 0.5)               # bins in i050 per window
B100      = W_noise                            # bins in i100 per window
```

Steps:

1. Slide a window of `W_noise` ms across all available channels with stride `stride_ms`
2. For each window position `j = 0, 1, …, N_windows-1`:
   - `N_windows ≈ floor(total_duration_ms / stride_ms)`  — much smaller than the trace
     length; e.g. ~234 windows for a 60 s trace with stride 256 ms
   - Extract `i_min[window]` (B_min bins), `i050[window]` (B050 bins),
     `i100[window]` (B100 bins)
   - If `noise_model.requires_acf_features`: compute ACF features from `i_min[window]`
   - Compute 6 raw stats: `[mean, std]` per channel (always required)
   - Z-score normalize each channel
   - Run `noise_model.predict(...)` → scalar noise estimate for this window
   - Store as `n_abs_local[j]` (kHz)
3. `n_abs_local` is therefore a 1-D array of length `N_windows` — one noise estimate
   per stride position, not one per trace bin
4. **Global noise estimate:** `n_abs = percentile(n_abs_local, 20)`
   - 20th percentile used because windows containing strong molecular signal overestimate
     the background; low-percentile windows correspond to background-only periods
   - Collapses `n_abs_local` to a single scalar representing the measurement background

**Output:**
- `n_abs_local` (array, length N_windows, kHz) — time-resolved noise estimates
- `n_abs` (scalar, kHz) — global background estimate used for thresholding in Stage 2
- `None` for both if no admissible noise model exists for this measurement

---

## Stage 2 — Background Subtraction

**Purpose:** remove the estimated background component from all channels so that the
cleaned traces are admissible to the diffusion models, which were trained on noise-free
(`noise0`) simulations with zero background.

The background adds a constant mean offset to every channel. Subtracting it restores the
traces to the zero-background regime expected by the D models. Residual shot-noise
fluctuations from the background cannot be fully eliminated (they are Poisson-distributed
around the mean), but their relative magnitude is reduced once the mean is removed.

**If `n_abs` is available** (Stage 1 succeeded — `has_raw=True`):
```
i_min_clean = np.maximum(0,  i_min_full − n_abs)
i050_clean  = np.maximum(0,  i050_full  − n_abs)
i100_clean  = np.maximum(0,  i100_full  − n_abs)
```
`n_abs` is in kHz and is the same scalar subtracted from all channels, since block-averaging
preserves the mean photon rate across dt resolutions.

**Non-negativity is strictly enforced:** `np.maximum(0, ...)` is applied element-wise to
every bin of every channel. No bin is permitted to be negative after subtraction. This
ensures physical admissibility (photon counts cannot be negative) and compatibility with
the model's training distribution. Log the fraction of clipped bins per channel as a
diagnostic — a high fraction (> 5%) may indicate that `n_abs` was overestimated.

**If `n_abs` is unavailable** (`has_raw=False`, noise stage skipped):
Estimate background from the low-intensity tail of the dt=1ms trace:
```
n_abs_est   = np.percentile(i100_full, 5)     # robust low-end estimate
i100_clean  = np.maximum(0, i100_full − n_abs_est)
```
Flag the measurement as `background_estimated` in the output. `i_min` and `i050` are
unavailable in this case and are not processed.

**Output:**
- `i_min_clean`, `i050_clean`, `i100_clean` — background-subtracted channels (kHz)
- All downstream stages operate on the cleaned channels, not the raw channels
- `n_abs` (or `n_abs_est`) stored in the `Measurement` object for reporting

---

## Stage 3 — Matched-Filter Event Detection

**Purpose:** locate single-molecule passage events in `i100_clean` and demarcate a segment
around each event for input to the D model. Detection is performed by a bank of
matched filters derived from training-data burst templates, followed by adaptive
non-maximum suppression, signal-driven boundary expansion, and event merging.

**Model selection** (determines inference window `W_D`):
```
d_model = registry.diffusion_models.best_for(has_raw=measurement.has_raw)
W_D     = d_model.window_ms    # e.g. 4096ms — D model fixed input length
```

---

### 3.1 Burst Templates

Five template waveforms are pre-computed from the training data — one per D-bin — by
averaging the background-subtracted burst shapes aligned to their intensity peak.
Each template spans a half-width matched to the typical dwell time of its D-bin:

| D-bin (µm²/s) | Template half-width `W` (ms) |
|---|---|
| 0.01 – 0.05 | 1024 |
| 0.05 – 0.2  | 512  |
| 0.2  – 0.7  | 256  |
| 0.7  – 2.5  | 128  |
| 2.5  – 15   | 64   |

Each template `T_k` is unit-energy normalised before storage:
```
T̃_k = T_k / ‖T_k‖₂
```

Templates are loaded once at pipeline initialisation (see `stage3.py`) and cached for
the lifetime of the process.

---

### 3.2 Matched-Filter Bank

Each template is slid across `i100_clean` via cross-correlation, producing a per-template
score curve. The score at position `t` measures how well a window of `i100_clean` centred
at `t` matches the burst shape of D-bin `k`:

```
score_k(t) = dot( i100_clean[t−W_k : t+W_k+1],  T̃_k )
```

The combined response `R(t)` and winning template index `k*(t)` are then:
```
R(t)   = max_k  score_k(t)
k*(t)  = argmax_k  score_k(t)
```

`R(t)` is large wherever the local signal shape matches any template well. Positions
near the trace edges (within `W_k` of either boundary) are not scored by template `k`
and remain `NaN`; `R(t)` takes the maximum over the templates that do have valid scores
at each position.

---

### 3.3 Adaptive Greedy Non-Maximum Suppression (NMS)

**Detection threshold:**
```
R_thresh = percentile( R(t), 85 )
```
The 85th percentile of the full trace's combined response is used as a data-adaptive
threshold. This ensures the threshold rises with overall signal level and falls on
background-only traces.

**Candidate peaks:** all local maxima of `R(t)` that exceed `R_thresh` and are separated
by at least `d_min = 10 ms`. These are found with `scipy.signal.find_peaks`.

**Greedy NMS** (accepts peaks in descending score order):

```
sort candidates by R(t_peak) descending
for each candidate p (highest score first):
    if p is still active:
        accept p as a detected event
        radius = min( W_HALVES[k*(p)],  NMS_CAP )     # NMS_CAP = 128 ms
        suppress all remaining candidates q with |q − p| ≤ radius
```

The suppression radius is set to the winning template's half-width, capped at 128 ms.
This prevents a slow template winning spuriously from suppressing a wide neighbourhood
of fast events, while still removing closely spaced duplicates within a single burst.

**Parameters:**
```
NMS_CAP  = 128   # ms — maximum suppression radius (tunable)
d_min    = 10    # ms — minimum raw peak separation (tunable)
```

---

### 3.4 Signal-Adaptive Window Expansion

The winning template half-width `W_HALVES[k*(p)]` sets an initial window around each
accepted peak:
```
t_left  = t_peak − W_HALVES[k*(p)]
t_right = t_peak + W_HALVES[k*(p)]
```

This initial window may under-cover broad events (particularly slow diffusers). Each
boundary is therefore walked outward until the local signal returns to the noise floor:

```
background = percentile( i100_clean, 10 )
smooth     = gaussian_filter1d( i100_clean, σ=30 )
stop_level = background + 0.10 × max( smooth[t_peak] − background,  0 )

# expand left
while t_left  > t_peak − 3 × W_seg  and  smooth[t_left  − 1] > stop_level:
    t_left  -= 1

# expand right
while t_right < t_peak + 3 × W_seg  and  smooth[t_right + 1] > stop_level:
    t_right += 1
```

The stop level is 10% of the peak height above background — expansion halts when the
smoothed signal falls to this level on either side. A hard cap of ±3 × W_seg prevents
runaway expansion into neighbouring events or extended background regions.

**Parameters:**
```
EXPAND_THRESH_FRAC = 0.10   # stop when smooth < bg + 10%×peak  (tunable)
EXPAND_SIGMA       = 30     # ms — smoothing sigma for expansion decision (tunable)
EXPAND_MAX_MULT    = 3      # hard cap: expand at most ±3×W_seg from peak (tunable)
BG_PCTILE          = 10     # trace percentile used as background estimate (tunable)
```

**Coverage** (measured on 500 training traces per D-bin with the above defaults):

| D-bin | Median photon coverage | Traces <80% coverage |
|---|---|---|
| 0.01 – 0.05 | 96% | 11% |
| 0.05 – 0.2  | 99% | 6%  |
| 0.2  – 0.7  | ≥99% | 3%  |
| 0.7  – 15   | ≥99% | <2% |

The residual under-coverage in the slowest bin arises from events wider than the
±3 × 1024 ms = ±3072 ms cap; these are molecule passages spanning the majority of
the 4096 ms trace.

---

### 3.5 Event Merging

Adjacent events whose demarcated windows are closer to each other than to the outer
edges of their own windows are merged into a single segment. Formally, events A and B
(A to the left of B) are merged when:

```
gap = t_left_B − t_right_A  <  min( W_seg_A,  W_seg_B )
```

A negative gap (overlapping windows) always satisfies this condition. When events are
merged, the result inherits:
- `t_left`  = min(t_left_A,  t_left_B)
- `t_right` = max(t_right_A, t_right_B)
- `t_peak`, `k_win`, `score` from whichever event had the higher filter score

The procedure iterates over all adjacent pairs repeatedly until no further merges occur.

**Rationale:** clustered detections within a single extended burst — e.g., a slow molecule
whose filter response produces two local maxima separated by a transient dip — are
collapsed into one segment before being passed to the D model. This avoids presenting
the model with an artificially truncated view of the same event.

---

### 3.6 Padding and Windowing for D-Model Admissibility

Each identified segment of length `L = t_right − t_left + 1` bins must be made
compatible with the D model's fixed input length `W_D`. Three cases:

**Case 1 — L < W_D (short segment → zero-pad symmetrically):**
```
pad_total = W_D − L
pad_left  = pad_total // 2
pad_right = pad_total − pad_left
seg_padded = concatenate([ zeros(pad_left),
                            i100_clean[t_left : t_right+1],
                            zeros(pad_right) ])
pad_fraction = pad_total / W_D
```
Zero-padding represents post-subtraction background and is consistent with the model's
`noise0` training distribution. Segments with `pad_fraction > 0.75` yield less reliable
D estimates and are flagged.

**Case 2 — L = W_D (exact match):**
Use `i100_clean[t_left : t_right+1]` directly. `pad_fraction = 0`.

**Case 3 — L > W_D (long segment → sliding windows):**
Slide a window of length `W_D` with stride `W_D // 8`, producing `n_win` overlapping
windows. The D model is called once per window; predictions are aggregated via the
geometric mean in Stage 4:
```
D_segment = 10^( mean( log10( D_window[j] ) for j in 0…n_win−1 ) )
```

---

### 3.7 Output of Stage 3

For each detected event (sorted by `t_peak`), store a dict:

```
{
  't_peak_ms'   : t_peak,            # detected event centre (ms index)
  't_left_ms'   : t_left,            # segment left boundary after expansion
  't_right_ms'  : t_right,           # segment right boundary after expansion
  'duration_ms' : t_right − t_left,
  'W_seg'       : W_HALVES[k_win],   # template half-width (ms)
  'k_win'       : k_win,             # winning template D-bin index (0=slowest)
  'score'       : R(t_peak),         # combined filter response at peak
  'pad_fraction': pad_fraction,      # 0.0 for Cases 2 and 3
  'n_win'       : n_win,             # number of W_D windows (≥1)
  'd_model_label': d_model.label,
}
```

D-prediction (calling `d_model.predict(...)` on each window) is performed in
**Stage 4**, which receives this segment list along with the padded/windowed traces.

**Implementation reference:** `stage3.py` — `run_stage3(signal)` returns this list
together with the combined filter response `R(t)` for visualisation.

---

## Stage 4 — Per-Segment D-Prediction

**Purpose:** for each Stage 3 segment, assemble a model-admissible input tensor at
every required dt resolution, run the D model, and aggregate any multi-window
predictions into a single D estimate per event.

**Model:** `d_model` selected in Stage 3 and carried through.

---

### 4.1 Resolution Parameters

Derived once from the model catalogue and the registry:

```
W_D     = d_model.window_ms          # fixed input length at dt=1ms, e.g. 4096 ms
dt_min  = registry.dt_min            # finest available channel, e.g. 0.1 ms
stride  = W_D // 8                   # sliding-window stride (Case 3), e.g. 512 ms

# Per-window bin counts at each channel resolution:
B_min   = round(W_D  / dt_min)       # e.g. 40960 bins (dt=0.1ms channel)
B_050   = round(W_D  / 0.5  )        # e.g.  8192 bins (dt=0.5ms channel)
B_100   = W_D                        # e.g.  4096 bins (dt=1.0ms channel)

# dt→bin-count multipliers relative to dt=1ms:
R_min   = round(1.0  / dt_min)       # e.g. 10   (bins per 1ms bin in i_min)
R_050   = round(1.0  / 0.5  )        # = 2  (bins per 1ms bin in i_050)
```

---

### 4.2 Multi-dt Segment Retrieval

The Stage 3 segment boundaries `[t_left, t_right]` are in dt=1ms bins. The
corresponding regions in the finer channels are obtained by scaling the boundaries:

```
# dt=1ms (always available)
seg_100 = i100_clean[ t_left            :  t_right + 1          ]   # L bins

# dt=0.5ms (if has_raw and 0.5ms in d_model.dt_channels)
seg_050 = i050_clean[ t_left * R_050    : (t_right + 1) * R_050 ]   # L × R_050 bins

# dt=dt_min (if has_raw and dt_min in d_model.dt_channels)
seg_min = i_min_clean[ t_left * R_min   : (t_right + 1) * R_min ]   # L × R_min bins
```

The right-end index `(t_right + 1) * R_min` is exclusive, giving exactly `L × R_min`
bins — one multi-dt block per original dt=1ms bin. No interpolation is required;
the finer channels are already in register with `i100_clean` by construction
(Stage 0.4 block-averaging preserves alignment).

---

### 4.3 Padding to W_D

Three cases mirror Stage 3.6. Padding is applied identically to all channels,
scaled by the bin-count multiplier of each channel so that every channel spans
exactly one model input window:

**Case 1 — L < W_D  (segment shorter than model input — zero-pad):**

```
pad_total   = W_D − L
pad_l_100   = pad_total // 2
pad_r_100   = pad_total − pad_l_100

# Scale padding to each channel:
pad_l_050, pad_r_050 = pad_l_100 * R_050, pad_r_100 * R_050
pad_l_min,  pad_r_min  = pad_l_100 * R_min,  pad_r_100 * R_min

win_100 = concatenate([ zeros(pad_l_100), seg_100, zeros(pad_r_100) ])  # B_100 bins
win_050 = concatenate([ zeros(pad_l_050), seg_050, zeros(pad_r_050) ])  # B_050 bins
win_min = concatenate([ zeros(pad_l_min),  seg_min,  zeros(pad_r_min)  ])  # B_min bins
```

Zero padding represents the post-subtraction noise floor and is consistent with the
`noise0` training regime. Segments with `pad_fraction > 0.75` are flagged (see §4.6).

**Case 2 — L = W_D  (exact fit — no padding):**

```
win_100 = seg_100          # B_100 bins
win_050 = seg_050          # B_050 bins
win_min = seg_min          # B_min bins
```

**Case 3 — L > W_D  (segment longer than model input — sliding windows):**

```
stride_100 = W_D // 8
stride_050 = stride_100 * R_050
stride_min  = stride_100 * R_min

windows = []
start = 0
while start + W_D <= L:
    windows.append({
        'win_100': seg_100[ start            : start + B_100 ],
        'win_050': seg_050[ start * R_050    : start * R_050 + B_050 ],
        'win_min':  seg_min[  start * R_min    : start * R_min  + B_min  ],
    })
    start += stride_100
# n_win = len(windows);  guaranteed ≥ 1 by Stage 3 construction
```

---

### 4.4 Feature Computation

If `d_model.requires_acf_features`:

For each window, call the model's own feature method with the finest available channel
window:

```
features = d_model.compute_features( win_finest )   # → vector of length d_model.feat_dim
```

`win_finest` is `win_min` if the dt_min channel is present, else `win_050` or `win_100`.

`compute_features` is a method of the `ModelWrapper` object. It encapsulates the exact
lag set, normalisation, and feature ordering used during training — Stage 4 does not
implement or know the feature pipeline details. Any change to the feature computation
is a model-level change, not a pipeline-level change.

If `d_model.requires_acf_features = False`, the method is not called and `features = None`
is passed to §4.6.

---

### 4.5 Normalisation

Feature normalisation is handled internally by `d_model.predict()` using the scalers
loaded at model initialisation. Stage 4 passes raw features; the model wrapper applies
`(features − scaler_mean) / scaler_scale` before forwarding to the network.

The waveform channels (`win_100`, `win_050`, `win_min`) are passed as-is — the WaveNet
branches handle amplitude variation through their internal batch normalisation layers.

---

### 4.6 Model Inference

For each window assembled in §4.3 (one window for Cases 1–2, `n_win` windows for Case 3):

```
log10_d_hat = d_model.predict(
    win_min  = win_min   if 0.1ms in d_model.dt_channels else None,
    win_050  = win_050   if 0.5ms in d_model.dt_channels else None,
    win_100  = win_100,
    features = features,   # raw; normalisation applied internally by the wrapper
)
d_hat = 10 ** log10_d_hat
```

**Aggregation for Case 3 (multiple windows):**
```
D_segment = 10^( mean( log10(d_hat[j]) for j in 0…n_win−1 ) )
```
The geometric mean in log-space is used because D predictions are approximately
log-normally distributed around the true value.

Flag segments with `pad_fraction > 0.75` as `low_signal` in the output — their D
estimates are dominated by the zero-padded background region and are less reliable.

---

### 4.7 Conformal Uncertainty Interval

The prediction interval is a property of the model, pre-computed once during model
evaluation on the held-out test set and stored in the model catalogue. The algorithm
only performs a lookup and application at inference time.

**Catalogue field** (added to each D model entry in `diffusion_models.json`):
```json
"conformal_intervals": {
  "coverage": 0.90,
  "pad_strata": [
    {"pad_max": 0.25, "q_log10": ...},
    {"pad_max": 0.50, "q_log10": ...},
    {"pad_max": 0.75, "q_log10": ...},
    {"pad_max": 1.00, "q_log10": ...}
  ]
}
```

`q_log10` is the 90th percentile of `|log10(D_pred) − log10(D_true)|` computed on the
test set within each `pad_fraction` stratum. Wider intervals for more heavily padded
events reflect the reduced signal available to the model.

**At inference**, after obtaining `log10_D` from §4.6:
```
q = conformal_intervals.lookup( pad_fraction )   # selects the matching stratum

D_low  = 10 ^ ( log10_D − q )
D_high = 10 ^ ( log10_D + q )
```

The resulting interval `[D_low, D_high]` has ≥ 90% empirical coverage — at least 90%
of predictions on held-out data contain the true D within this range.

> **Note — calibration is the model's responsibility, not the algorithm's.**
> The `q_log10` values are populated during model evaluation (see Planned Extension P1
> calibration step) and are frozen before deployment. Updating a model requires
> recomputing its conformal quantiles on a fresh held-out set.

---

### 4.8 Output of Stage 4

For each Stage 3 event, append D prediction fields to the existing segment dict:

```
{
  # — inherited from Stage 3 —
  't_peak_ms', 't_left_ms', 't_right_ms', 'duration_ms',
  'W_seg', 'k_win', 'score', 'pad_fraction', 'n_win', 'd_model_label',

  # — added by Stage 4 —
  'D_hat'      : D_segment,      # µm²/s  (geometric mean over windows if n_win > 1)
  'log10_D'    : log10(D_segment),
  'D_low'      : D_low,          # µm²/s  — lower bound of 90% conformal interval
  'D_high'     : D_high,         # µm²/s  — upper bound of 90% conformal interval
  'd_hat_all'  : [d_hat_0, …],   # per-window predictions (length n_win); list of length 1
                                  # for Cases 1–2
  'flags'      : [],              # e.g. ['low_signal'] if pad_fraction > 0.75
}
```

This list of per-event dicts is passed to **Stage 5** for quality filtering.

---

## Stage 5 — Per-Event Quality Filter

**Input:** the list of per-event dicts produced by Stage 4, each carrying `D_hat`,
`log10_D`, `pad_fraction`, `n_win`, `duration_ms`, and any flags set during inference.

For each event, apply the following quality checks. Events failing any hard criterion
are discarded; events failing soft criteria are retained but flagged.

| Metric | Definition | Hard discard | Soft flag |
|---|---|---|---|
| `duration_ms` | `t_right − t_left` | < 64 ms | — |
| `D_hat` | model prediction | outside (0, 14) µm²/s | — |
| `D_cv` | spread of per-window log₁₀D predictions (Case 3 only) | — | > 0.35 → `variable_D` ⚠️ threshold TBD |
| `n_frac` | `n_abs / I_peak_kHz` at event peak (`None` if noise unavailable) | > 0.90 | > 0.50 → `high_background` |

> **Note — `pad_fraction` is not used as a filter criterion.** Fast diffuser events
> (W_seg=64ms padded to W_D=4096ms) have pad_fraction≈0.97 by construction and are
> perfectly valid detections. `pad_fraction` is reported in the output table as a
> diagnostic but does not gate event acceptance.

`D_cv` is only meaningful for Case 3 segments (n_win ≥ 2); it measures consistency of D
across the sliding windows of a long event. A high value suggests the event spans two
molecules with different D, or that the matched-filter peak is misplaced.
> ⚠️ **D_cv definition and threshold to be finalised next session.**

If `n_frac` is unavailable (`has_raw=False`), the `n_frac` checks are skipped and
`noise_unavailable` is added to every event's flags.

**Output:** filtered list of per-event dicts, each with a `flags` list. Passed to
Stage 6 for output.

---

## Stage 6 — Output

**Per-event results table** (one row per event surviving Stage 5):

```
| t_peak (ms) | t_left (ms) | t_right (ms) | duration (ms) | D (µm²/s) | D_low (µm²/s) | D_high (µm²/s) | pad_frac | n_win | n_frac | model_used | flags |
```

- `flags`: comma-separated quality warnings (e.g. `low_signal`, `variable_D`,
  `high_background`, `noise_unavailable`)
- `model_used`: label of the D model (e.g. `multidt_B_large`) and noise model
  (e.g. `noise_w2048`)

Saved as CSV named `{name}_{id}_events.csv`.

**Plots:**

1. `i100_clean` intensity trace with each detected event shaded by log10(D̂) on a
   diverging colormap; discarded events outlined in grey
2. D̂ scatter plot — one point per event at t_peak, y = log10(D̂), coloured by
   `pad_fraction` (blue=well-covered → yellow=heavily padded)
3. D̂ histogram across all passing events (log10-scale x-axis, bin width 0.1)

All plots saved as `{name}_{id}_{plot_type}.png`.

---

## Key Design Notes

### The 4096ms window vs. short transits
The D model requires a 4096ms window to form a reliable multi-lag ACF. Individual
molecule transits in free-solution FCS last 1–50ms — far shorter. Three regimes:

- **Long-dwelling molecules** (surface-tethered, confined, or slow diffusers): transits
  lasting seconds are directly compatible with the 4096ms window.
- **Burst accumulation:** many consecutive short bursts from the same population are
  averaged by the sliding window, effectively performing standard ensemble FCS.
- **Background-padded windows:** a 4096ms window around an isolated short burst contains
  signal + background padding; the ACF has a non-decaying component and the model returns
  a biased (slow) D estimate. Treat these as lower bounds; flag with `D_cv > threshold`.

### PSF calibration
All D predictions are in the simulation PSF coordinate system (ω_xy, ω_z). A calibration
factor `α = D_known / D_measured` from a standard sample (e.g., Alexa 488 in water at
25°C, D ≈ 435 µm²/s) must be applied to all outputs before reporting absolute values.

### Model selection logic
All model selection is handled by `registry.best_for(total_duration_ms, has_raw)` and
requires no hardcoded logic in the pipeline stages. The registry evaluates each model's
`is_admissible()` method and returns the one with the lowest MAPE. Adding or updating a
model in the catalogue automatically propagates to all stages.

For reference, the current precedence under typical conditions (`has_raw=True`):

| Stage | Preferred model | Fallback |
|---|---|---|
| Noise (Stage 1) | `noise_w2048` (MAPE=3.8%) | next best admissible by window |
| Diffusion (Stage 4) | `multidt_B_large` (MAPE=9.5%) | `wavenet_wide_aug` if `has_raw=False` |

---

## Planned Extensions

The following analyses are deferred to future pipeline stages. They are not part of the
current implementation but are core scientific goals.

### P1 — D Population Analysis
Given the per-event D̂ values from Stage 4, fit a Gaussian mixture model (GMM) in
log₁₀D-space to identify distinct molecular populations automatically. Outputs:
- number of populations K (selected by BIC)
- per-population mean log₁₀D, spread σ, and weight (fractional abundance)
- per-event posterior probability of belonging to each population

This answers the primary question: "how many diffusive species are present, and what
are their D values and relative abundances?"

### P2 — Temporal Consistency Check
For long acquisitions, test whether D̂ drifts systematically over time (photobleaching,
focus drift, or sample evolution). Applies a Mann-Kendall trend test or simple
linear regression on log₁₀D̂ vs. t_peak across all events. A significant trend
triggers a `temporal_drift` warning in the run-level QC report.

If no admissible model exists for a stage, that stage is skipped and the measurement is
flagged accordingly in the output.

---

## Model Performance Reference

### Noise Models

| Model | Trace length | dt channels | Features | Test MAPE | R² |
|---|---|---|---|---|---|
| multidt_noise | 4096 ms | 0.1, 0.5, 1.0 ms | 916 (910 ACF + 6 raw) | 2.2% | 0.9997 |
| noise_w2048   | 2048 ms | 0.1, 0.5, 1.0 ms | 6 raw stats only       | 3.8% | 0.9988 |
| noise_w1024   | 1024 ms | 0.1, 0.5, 1.0 ms | 6 raw stats only       | 6.0% | 0.9967 |
| noise_w512    |  512 ms | 0.1, 0.5, 1.0 ms | 6 raw stats only       |10.5% | 0.9933 |
| noise_w256    |  256 ms | 0.1, 0.5, 1.0 ms | 6 raw stats only       |14.4% | 0.9870 |

### Diffusion Models

| Model | Trace length | dt channels | Features | Test MAPE | d<1 MAPE | d≥1 MAPE |
|---|---|---|---|---|---|---|
| multidt_B_large | 4096 ms | 0.1, 0.5, 1.0 ms | 910 ACF | 9.5% | 7.8% | 12.8% |
| multidt_B       | 4096 ms | 0.1, 0.5, 1.0 ms | 910 ACF | 9.9% | 8.3% | 13.3% |
| wavenet_wide_aug | 4096 ms | 1.0 ms only     | ACF (1ms lags) | 12.3% | 9.8% | 17.6% |

---

*End of document — updated iteratively as algorithm evolves*
