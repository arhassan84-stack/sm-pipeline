# Burst Expansion Analysis — Notes

**Script:** `analyze_burst_expansion.py`
**Output figure:** `burst_expansion_analysis.png`
**Date:** 2026-04-10
**Data:** `cache_i_train.npy` + `cache_d_train.npy` — 50,000 noise-free simulated traces, dt=1ms, 4096 bins each

---

## What the Analysis Does

For every simulated trace in the training set:

1. Find the peak bin via a Gaussian-smoothed trace (σ=10ms)
2. Expand symmetrically from the peak and compute the cumulative photon sum:
   ```
   S(t) = sum( i[peak−t : peak+t+1] )   for t = 0, 1, 2, …, 512 ms
   ```
3. Normalize: `F(t) = S(t) / S(T_max) ∈ [0, 1]` — the cumulative photon fraction curve
4. Extract per-trace radii `t_90`, `t_95`, `t_99` (ms): the expansion radius at which F first exceeds 90%, 95%, 99%
5. Compute total photon count per trace: `N_ph = sum(i) × dt_s`  (dt_s = 0.001 s)

Results stratified into 5 D-bins (log-spaced in µm²/s):
`[0.01–0.05]`, `[0.05–0.2]`, `[0.2–0.7]`, `[0.7–2.5]`, `[2.5–15]`

---

## Figure Layout

**White background, black axes, 3 panels side by side (18×6 inches, 150 dpi)**

### Panel A — Cumulative photon fraction curves
- x-axis: expansion radius t (ms), range 0–512
- y-axis: F(t), range 0–1.05
- One curve per D-bin: median F(t) with shaded 10–90th percentile band
- Horizontal reference lines at 90%, 95%, 99% (gray dashed/dotted/dash-dot)
- Legend lower-right

### Panel B — Distribution of t₉₅ per D-bin
- x-axis: t₉₅ (ms), range 0–512, 80 bins
- y-axis: density (normalized histogram)
- Overlapping filled histograms per D-bin, α=0.55
- Dashed vertical line at median t₉₅ per D-bin

### Panel C — Distribution of total photons per D-bin
- x-axis: log₁₀(total photons), 80 log-spaced bins
- y-axis: density
- Same format as Panel B

---

## Colour Palette

| D-bin (µm²/s) | Hex colour |
|---|---|
| 0.01–0.05 | `#4477AA` (dark blue) |
| 0.05–0.2  | `#66CCEE` (light blue) |
| 0.2–0.7   | `#228833` (green) |
| 0.7–2.5   | `#CCBB44` (yellow) |
| 2.5–15    | `#EE6677` (red) |

---

## Font Sizes

```python
FS_SUPTITLE = 20
FS_TITLE    = 14
FS_LABEL    = 15
FS_YLABEL   = 14
FS_TICK     = 12
SPINE_LW    = 1.5
```

All axes: white facecolor, black spines (lw=1.5), black tick labels.
Figure: white facecolor.
`plt.rcParams` set globally: `text.color`, `axes.labelcolor`, `xtick.color`, `ytick.color` all black.

---

## Numerical Results

| D range (µm²/s) | N traces | t₉₀ median (ms) | t₉₅ median (ms) | t₉₉ median (ms) | N_ph median | N_ph 5th pct |
|---|---|---|---|---|---|---|
| 0.01–0.05 | 11,124 | 430 | 469 | 504 | 51,970 | 19,078 |
| 0.05–0.2  |  9,559 | 341 | 396 | 473 | 17,899 |  5,723 |
| 0.2–0.7   |  8,644 | 225 | 272 | 347 |  6,262 |  1,870 |
| 0.7–2.5   |  8,720 | 144 | 178 | 233 |  2,159 |    588 |
| 2.5–15    | 11,953 |  85 | 104 | 138 |    592 |    153 |

Key observations:
- **Fastest diffusers (D=2.5–15) are the limiting case** in all metrics
- t₉₅ for fastest bin: median=104ms, meaning 95% of photons are within ±104ms of peak
- Total photon count spans >2 orders of magnitude across D range (median 592 vs 51,970)
- 5th percentile photon count for fast diffusers: ~153 photons — this is the hard lower bound for any N_min criterion
- D-bins are well-separated in both t₉₅ and N_ph: the analysis can distinguish them cleanly

---

## Key Parameters

```python
DT_MS        = 1.0     # ms per bin
SIGMA_SMOOTH = 10      # bins; Gaussian smoothing σ for peak finding
MAX_RADIUS   = 512     # ms; maximum expansion radius computed
N_SAMPLE     = 50_000  # all training traces used
SEED         = 42
```

---

## Intended Use

This analysis motivates data-driven stopping criteria E (minimum duration) and F (minimum photon count) for the peak-expansion segmentation in Stage 3 of `algorithm_sm_inference.md`. The curves and statistics quantify how much of the molecular signal is captured as a function of expansion radius, stratified by D.

Pending discussion: what thresholds for T_min and N_min to derive from these curves.
