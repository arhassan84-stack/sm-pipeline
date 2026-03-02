"""
Compute nonlinear FCS ACF fit features from existing cached feature files.

Uses the ACF values already stored in the 283-feature caches (cols 10-38).
Fits the standard FCS diffusion model (2D Gaussian beam):

    G(τ) = G0 / (1 + τ/τD)

by minimising RMSE over a dense log-grid of τD values.
This is a 1-parameter fit — fully vectorised (no scipy loop per sample).

Output files (local):
  nlfit_tauD_train.npy   (n_train,)  best-fit τD for each training sample
  nlfit_rmse_train.npy   (n_train,)  RMSE of the fit
  nlfit_tauD_test.npy    (n_test,)
  nlfit_rmse_test.npy    (n_test,)

These are loaded by meta_learner_5fold.py as additional meta-features.

Note: G0 is already available in the feature cache (col 8).
"""
import numpy as np, time

N       = 4096
N_GRID  = 300           # τD grid resolution (log-spaced)
BATCH   = 10_000        # samples per batch (memory: ~300 MB per batch)

# ACF lags used in cache: must match build_cache_90pct.py exactly
LAGS = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
assert len(LAGS) == 29, f"Expected 29 unique lag values, got {len(LAGS)}"
LAGS_F = LAGS.astype(np.float64)   # (29,)

# τD grid: 1 to N/2 (log-spaced)
TAU_D_GRID = np.logspace(0, np.log10(N // 2), N_GRID)  # (N_GRID,)

COL_G0        = 8
COL_ACF_START = 10
N_ACF_LAGS    = 29

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_nlfit(X, label=""):
    """
    X: (n, 283) feature array
    Returns: tauD (n,), rmse (n,)
    """
    n = len(X)
    G0  = X[:, COL_G0].astype(np.float64)                        # (n,)
    acf = X[:, COL_ACF_START:COL_ACF_START+N_ACF_LAGS].astype(np.float64)  # (n, 29)

    # Normalize: y_i = G(τ_i) / G0, model predicts 1/(1+τ/τD)
    G0_safe = np.maximum(G0, 1e-10)
    y_norm  = acf / G0_safe[:, None]                              # (n, 29)
    y_norm  = np.clip(y_norm, 0.0, 2.0)                          # (n, 29)

    # G_pred[j, k] = 1 / (1 + LAGS[j] / TAU_D_GRID[k])
    # shape: (29, N_GRID)
    G_pred = 1.0 / (1.0 + LAGS_F[:, None] / TAU_D_GRID[None, :])

    best_tauD = np.zeros(n)
    best_rmse = np.zeros(n)

    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        y_b = y_norm[i:j, :, None]              # (bs, 29, 1)
        gp  = G_pred[None, :, :]                # (1,  29, N_GRID)
        rmse_b = np.sqrt(((y_b - gp)**2).mean(axis=1))  # (bs, N_GRID)
        idx = rmse_b.argmin(axis=1)             # (bs,)
        best_tauD[i:j] = TAU_D_GRID[idx]
        best_rmse[i:j] = rmse_b[np.arange(j-i), idx]
        if (i // BATCH) % 5 == 0:
            log(f"  {label} {j}/{n} ({100*j/n:.0f}%)  sample tauD range [{best_tauD[:j].min():.1f},{best_tauD[:j].max():.1f}]")

    return best_tauD, best_rmse


log("=== Nonlinear FCS ACF fit ===")
log(f"ACF lags ({len(LAGS)}): {LAGS[:5]}...{LAGS[-3:]}")
log(f"τD grid: {N_GRID} points from {TAU_D_GRID[0]:.1f} to {TAU_D_GRID[-1]:.0f}")

for split, x_file, tauD_out, rmse_out in [
    ('train', 'cache_X_train_90pct.npy', 'nlfit_tauD_train.npy', 'nlfit_rmse_train.npy'),
    ('test',  'cache_X_test_90pct.npy',  'nlfit_tauD_test.npy',  'nlfit_rmse_test.npy'),
]:
    log(f"\n--- {split} ---")
    X = np.load(x_file)
    log(f"  Loaded {x_file}  shape={X.shape}")

    # Filter to d<=10 to match model training
    # NOTE: we don't have d labels here, so we process all samples
    # The meta-learner will apply the same d<=10 filter when loading targets

    t0 = time.time()
    tauD, rmse = compute_nlfit(X, label=split)
    log(f"  Done in {time.time()-t0:.1f}s")
    log(f"  τD: min={tauD.min():.2f}  median={np.median(tauD):.2f}  max={tauD.max():.2f}")
    log(f"  RMSE: mean={rmse.mean():.4f}  std={rmse.std():.4f}")

    np.save(tauD_out, tauD.astype(np.float32))
    np.save(rmse_out, rmse.astype(np.float32))
    log(f"  Saved {tauD_out}  {rmse_out}")

log("\nDone.")
log("Output: nlfit_tauD_{train,test}.npy  nlfit_rmse_{train,test}.npy")
log("Use in meta_learner_5fold.py as additional physical meta-features.")
