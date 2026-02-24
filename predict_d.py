import numpy as np
import h5py
import time
from scipy.stats import skew, kurtosis
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

t_total = time.time()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def log(msg, t_ref=None):
    elapsed = f"  [{time.time()-t_ref:.1f}s]" if t_ref is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}{elapsed}", flush=True)

# ── Helper: extract features from a raw i matrix ─────────────────────────────
def extract_features(i, label=""):
    N = i.shape[1]  # 4096
    t = time.time()

    log(f"{label} Computing intensity statistics ({len(i)} traces)...")
    means     = i.mean(axis=1)
    variances = i.var(axis=1)
    q_mandel  = (variances - means) / means
    skewness  = skew(i, axis=1)
    kurt      = kurtosis(i, axis=1)

    log(f"{label} Computing factorial cumulants...", t)
    kappa1      = means
    kappa2      = variances - means
    kappa2_norm = kappa2 / kappa1**2

    log(f"{label} Computing ACF via FFT...", t)
    lags  = np.unique(np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
    fluct = i - means[:, np.newaxis]
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    F     = np.fft.rfft(fluct, n=n_fft, axis=1)
    corr  = np.fft.irfft(np.abs(F)**2, n=n_fft, axis=1)

    acf = np.zeros((len(i), len(lags)))
    for j, lag in enumerate(lags):
        acf[:, j] = corr[:, lag] / (N - lag) / means**2

    log(f"{label} Computing half-decay lag...", t)
    g0       = variances / means**2
    half_g0  = g0 / 2
    half_decay = np.zeros(len(i))
    for k in range(len(i)):
        crossed = np.where(acf[k] < half_g0[k])[0]
        half_decay[k] = lags[crossed[0]] if len(crossed) > 0 else lags[-1]

    log(f"{label} Computing scattering transform (J=8, Q=2)...", t)
    J, Q  = 8, 2
    n_sc  = 2 ** int(np.ceil(np.log2(N)))
    freqs = np.fft.rfftfreq(n_sc)

    # Build Morlet filterbank: J octaves x Q wavelets per octave
    psi_bank = []
    for j in range(J):
        for q in range(Q):
            xi    = 0.5 / (2 ** (j + q / Q))
            sigma = xi / (Q * 2 * np.sqrt(2 * np.log(2)))
            psi_bank.append(np.exp(-0.5 * ((freqs - xi) / sigma) ** 2))

    X_fft  = np.fft.rfft(i, n=n_sc, axis=1)   # (n_traces, n_sc//2+1)
    n_filt = len(psi_bank)                      # J*Q = 16

    # Order 1: S1_k = mean(|W_k * x|)
    U1, S1 = [], []
    for psi in psi_bank:
        u1 = np.abs(np.fft.irfft(X_fft * psi, n=n_sc, axis=1)[:, :N])
        S1.append(u1.mean(axis=1))
        U1.append(u1)
    S1 = np.column_stack(S1)   # (n_traces, n_filt)

    # Order 2: S2_{k1,k2} = mean(|W_k2 * |W_k1*x||) for k2 > k1
    S2 = []
    for k1 in range(n_filt):
        U1_fft = np.fft.rfft(U1[k1], n=n_sc, axis=1)
        for k2 in range(k1 + 1, n_filt):
            u2 = np.abs(np.fft.irfft(U1_fft * psi_bank[k2], n=n_sc, axis=1)[:, :N])
            S2.append(u2.mean(axis=1))
    S2 = np.column_stack(S2)   # (n_traces, n_filt*(n_filt-1)/2 = 120)

    log(f"{label} Feature extraction done. "
        f"ACF={len(lags)}  S1={S1.shape[1]}  S2={S2.shape[1]}", t)

    return np.column_stack([
        means, variances, q_mandel, skewness, kurt,
        kappa1, kappa2, kappa2_norm,
        g0, half_decay,
        acf,
        S1, S2,
    ]), lags


CACHE_TRAIN = 'cache_X_train.npy'
CACHE_DTRAIN = 'cache_d_train.npy'
CACHE_TEST  = 'cache_X_test.npy'
CACHE_DTEST = 'cache_d_test.npy'

import os
if all(os.path.exists(p) for p in [CACHE_TRAIN, CACHE_DTRAIN, CACHE_TEST, CACHE_DTEST]):
    log("Cache found — loading pre-computed features...")
    X_train  = np.load(CACHE_TRAIN)
    d_train  = np.load(CACHE_DTRAIN)
    X_test   = np.load(CACHE_TEST)
    d_test_raw = np.load(CACHE_DTEST)
    lags     = None   # not needed after feature extraction
    log(f"Train: {X_train.shape}  Test: {X_test.shape}")
else:
    log("No cache — computing features from scratch...")
    f = h5py.File('unrollIntensity.mat', 'r')
    uInt = f['uInt']

    n_all     = uInt[1, :]
    valid_idx = np.where(n_all == 0)[0]          # 100,000 rows

    rng        = np.random.default_rng(42)
    train_idx  = rng.choice(valid_idx, size=len(valid_idx) // 2, replace=False)   # 50%
    train_set  = set(train_idx.tolist())
    train_idx.sort()

    log(f"Train: {len(train_idx)} samples — reading from HDF5...")
    d_train    = uInt[0, train_idx]
    i_train    = uInt[2:, train_idx].T.astype(np.float64)
    log("Train data loaded — extracting features...")
    X_train, lags = extract_features(i_train, label="[Train]")
    log(f"Train features ready: {X_train.shape}")

    test_idx_all = np.array([idx for idx in valid_idx if idx not in train_set])
    rng2     = np.random.default_rng(99)
    test_idx = rng2.choice(test_idx_all, size=len(test_idx_all) // 10, replace=False)
    test_idx.sort()
    log(f"Test pool: {len(test_idx)} samples — reading from HDF5...")

    d_test_raw = uInt[0, test_idx]
    i_test_raw = uInt[2:, test_idx].T.astype(np.float64)
    log("Test data loaded — extracting features...")
    X_test, _ = extract_features(i_test_raw, label="[Test]")
    log(f"Test features ready: {X_test.shape}")

    log("Saving feature cache to disk...")
    np.save(CACHE_TRAIN,  X_train)
    np.save(CACHE_DTRAIN, d_train)
    np.save(CACHE_TEST,   X_test)
    np.save(CACHE_DTEST,  d_test_raw)
    log("Cache saved.")


# ── Exclude d > 10 from training ─────────────────────────────────────────────
train_mask  = d_train <= 10
X_train     = X_train[train_mask]
d_train     = d_train[train_mask]
log(f"Train samples with d<=10: {train_mask.sum()} / {len(train_mask)}")

# ── Log-transform target ──────────────────────────────────────────────────────
log_d_train = np.log(d_train)
log_d_test  = np.log(d_test_raw)

# ── Train model ───────────────────────────────────────────────────────────────
log("Training HistGradientBoostingRegressor on log(d)...")
model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42,
    verbose=1,
)
model.fit(X_train, log_d_train)
log("Training done — evaluating...")

# ── Evaluate (exponentiate predictions back to d space) ───────────────────────
d_pred_train = np.exp(model.predict(X_train))
d_pred_test  = np.exp(model.predict(X_test))

# Exclude d > 10 from test evaluation
test_mask    = d_test_raw <= 10
d_test_eval  = d_test_raw[test_mask]
d_pred_eval  = d_pred_test[test_mask]
log(f"Test samples with d<=10: {test_mask.sum()} / {len(d_test_raw)}")

def mape(true, pred):
    mask = true > 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

print(f"\n{'='*50}")
print(f"Train     R²={r2_score(d_train, d_pred_train):.4f}  MAE={mean_absolute_error(d_train, d_pred_train):.4f}  MAPE={mape(d_train, d_pred_train):.1f}%")
print(f"Test(all) R²={r2_score(d_test_raw, d_pred_test):.4f}  MAE={mean_absolute_error(d_test_raw, d_pred_test):.4f}  MAPE={mape(d_test_raw, d_pred_test):.1f}%")
print(f"Test(d≤10) R²={r2_score(d_test_eval, d_pred_eval):.4f}  MAE={mean_absolute_error(d_test_eval, d_pred_eval):.4f}  MAPE={mape(d_test_eval, d_pred_eval):.1f}%")
print(f"{'='*50}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Predicted vs actual (d <= 10)
ax = axes[0]
ax.scatter(d_test_eval, d_pred_eval, alpha=0.1, s=3, color='steelblue')
lims = [min(d_test_eval.min(), d_pred_eval.min()), max(d_test_eval.max(), d_pred_eval.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Predicted d')
ax.set_title(f'Test (d≤10) R²={r2_score(d_test_eval, d_pred_eval):.4f}')

# Residuals (d <= 10)
ax = axes[1]
ax.scatter(d_test_eval, d_pred_eval - d_test_eval, alpha=0.1, s=3, color='steelblue')
ax.axhline(0, color='r', lw=1, ls='--')
ax.set_xscale('log')
ax.set_xlabel('True d'); ax.set_ylabel('Residual (pred - true)')
ax.set_title('Residuals vs True d (d≤10)')


plt.tight_layout()
plt.savefig('model_results.png', dpi=150)
log(f"Saved model_results.png  — Total runtime: {time.time()-t_total:.1f}s")
print("Done.")
