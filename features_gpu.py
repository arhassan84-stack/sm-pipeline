"""
features_gpu.py — GPU-native FCS feature pipeline.

Faithfully ports build_cache_multidt.py to PyTorch so that all features can
be computed on GPU each training batch (on-the-fly training).

Public API
----------
make_constants(N, device)   →  dict of precomputed tensors (call once per N)
compute_features_gpu(I, C)  →  (B, F) float32 tensor on same device as I

Feature groups (matches build_cache_multidt.py order):
  extract_base_features:   10 stats + n_lags ACF + 16 S1 + 120 S2 + 32 PSD
  compute_transit_features: 5 threshold fractions
  compute_expanded_features:
      6 ACF fit params (exp + stretched-exp OLS)
      4 ACF decay lags (interpolated)
      n_short short ACF lags
     20 burst / crossing stats  (vectorised via scatter_add)
     15 segment variance features
      2 higher cumulants
     10 multitau Mandel Q
      4 nonstationarity
  compute_acf_ratio_features: 19 consecutive-lag log-ratios

Safe-length design
------------------
- ACF lags are clamped to corr size (n_fft−1) to prevent OOB indexing; lags
  beyond N//2 are zeroed out (unreliable for short windows).
- OLS ACF fits use only the valid (non-zeroed) lags.
- Segment and multitau features are zeroed when the window is too short
  (n_segs < 2 or n_b < 2).
- These guards are no-ops for full-length (N=40960) traces.

Variable-length usage (Option 2)
---------------------------------
Call make_constants(N_win, device) then override the lag arrays with those
from a reference (full-trace) constants dict to maintain a fixed feat_dim:

    C = make_constants(N_win, device)
    for key in ('lags', 'short_lags', 'n_lags', 'n_short'):
        C[key] = C_ref[key]
"""

import torch
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed constants  (call once per trace length N)
# ─────────────────────────────────────────────────────────────────────────────

def make_constants(N: int, device: torch.device) -> dict:
    """
    Precompute all length-dependent constants for N-bin FCS traces.

    Returns a dict whose tensors live on `device`.  Pass this dict to
    compute_features_gpu() on every batch.
    """
    # ── ACF log-spaced lags ───────────────────────────────────────────────
    lags_np = np.unique(
        np.round(np.logspace(0, np.log10(N // 2), 32)).astype(int))
    lags = torch.from_numpy(lags_np.astype(np.int64)).to(device)

    # Short ACF lags: {1..20} not already in lags_np
    short_np = np.setdiff1d(np.arange(1, 21), lags_np).astype(np.int64)
    short_lags = torch.from_numpy(short_np).to(device)

    # ── Gabor filter bank  (J=8, Q=2 → 16 filters) ────────────────────────
    J, Q = 8, 2
    n_sc = 2 ** math.ceil(math.log2(N))          # next power-of-2 ≥ N
    freqs = torch.fft.rfftfreq(n_sc, device=device)
    psi_list = []
    for j in range(J):
        for q in range(Q):
            xi    = 0.5 / 2 ** (j + q / Q)
            sigma = xi / (Q * 2 * math.sqrt(2 * math.log(2)))
            psi_list.append(torch.exp(-0.5 * ((freqs - xi) / sigma) ** 2))
    psi_bank = torch.stack(psi_list, dim=0)       # (16, n_sc//2+1)

    # S2 pair indices  (all k1 < k2 combinations of the 16 filters)
    n_filt = len(psi_list)                         # 16
    pairs  = [(k1, k2)
              for k1 in range(n_filt)
              for k2 in range(k1 + 1, n_filt)]
    k1_idx = torch.tensor([p[0] for p in pairs], device=device)  # (120,)
    k2_idx = torch.tensor([p[1] for p in pairs], device=device)  # (120,)

    # ── PSD log-bin edges ─────────────────────────────────────────────────
    n_fft2    = 2 ** math.ceil(math.log2(N))
    freqs2    = torch.fft.rfftfreq(n_fft2, device=device)
    psd_edges = torch.logspace(
        math.log10(freqs2[1].item()),
        math.log10(freqs2[-1].item()),
        33, device=device)                         # 33 edges → 32 bins

    return dict(
        N=N, n_sc=n_sc,
        n_fft=2 ** math.ceil(math.log2(2 * N)),  # for ACF circular cov
        n_fft2=n_fft2,
        lags=lags, short_lags=short_lags,
        psi_bank=psi_bank,
        k1_idx=k1_idx, k2_idx=k2_idx,
        freqs2=freqs2, psd_edges=psd_edges,
        n_lags=lags.shape[0], n_short=short_lags.shape[0],
        short_offset_in_expanded=10,              # 6 fit + 4 decay cols before acf_short
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main feature computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_features_gpu(
        I:        torch.Tensor,
        C:        dict,
        S2_CHUNK: int = 30) -> torch.Tensor:
    """
    Compute the full FCS feature vector for a batch of intensity traces.

    Parameters
    ----------
    I        : (B, N) float32 on GPU — raw intensity (counts/s)
    C        : dict returned by make_constants(N, device)
    S2_CHUNK : S2 pairs per IFFT call  (trade memory vs. kernel launches)

    Returns
    -------
    (B, F) float32 on same device as I
    """
    B, N   = I.shape
    device = I.device
    eps    = 1e-12

    lags       = C['lags']        # (n_lags,) int64
    short_lags = C['short_lags']  # (n_short,) int64
    psi_bank   = C['psi_bank']    # (16, H) float32
    k1_idx     = C['k1_idx']      # (120,) int64
    k2_idx     = C['k2_idx']      # (120,) int64
    n_sc       = C['n_sc']
    n_fft      = C['n_fft']
    n_fft2     = C['n_fft2']
    freqs2     = C['freqs2']
    psd_edges  = C['psd_edges']
    n_lags     = C['n_lags']
    n_short    = C['n_short']
    n_pairs    = k1_idx.shape[0]  # 120

    lag_f      = lags.float()     # (n_lags,) original values for physical meaning

    # ─────────────────────────────────────────────────────────────────────
    # 1.  Basic statistics
    # ─────────────────────────────────────────────────────────────────────
    means = I.mean(1)                               # (B,)
    var_  = I.var(1, unbiased=False)                # (B,)
    m1    = means.clamp(min=eps)
    fluct = I - means.unsqueeze(1)                  # (B, N) centred
    std_  = var_.sqrt().clamp(min=eps)

    skewness = (fluct ** 3).mean(1) / std_ ** 3
    kurtosis = (fluct ** 4).mean(1) / (var_ ** 2 + eps) - 3.0   # excess

    kappa1   = means
    kappa2   = var_ - means
    kappa2_n = kappa2 / m1 ** 2
    g0       = var_  / m1 ** 2

    # ─────────────────────────────────────────────────────────────────────
    # 2.  ACF via FFT  (safe for short windows: clamp + validity mask)
    # ─────────────────────────────────────────────────────────────────────
    F_   = torch.fft.rfft(fluct, n=n_fft, dim=1)
    corr = torch.fft.irfft(F_.abs() ** 2, n=n_fft, dim=1)    # (B, n_fft)

    # Clamp lag indices to corr length; zero ACF values beyond N//2
    safe_lags  = lags.clamp(max=n_fft - 1)                    # prevent OOB
    valid_lag  = lags <= (N // 2)                              # (n_lags,) bool
    norm_acf   = ((N - safe_lags.float()).clamp(min=1).unsqueeze(0)
                  * m1.pow(2).unsqueeze(1) + eps)
    acf_raw    = corr[:, safe_lags] / norm_acf                 # (B, n_lags)
    acf        = torch.where(valid_lag.unsqueeze(0),
                             acf_raw,
                             torch.zeros_like(acf_raw))        # (B, n_lags)

    # Short ACF lags — same treatment
    if n_short > 0:
        safe_sl    = short_lags.clamp(max=n_fft - 1)
        valid_sl   = short_lags <= (N // 2)
        norm_s     = ((N - safe_sl.float()).clamp(min=1).unsqueeze(0)
                      * m1.pow(2).unsqueeze(1) + eps)
        acf_raw_s  = corr[:, safe_sl] / norm_s
        acf_short  = torch.where(valid_sl.unsqueeze(0),
                                 acf_raw_s,
                                 torch.zeros_like(acf_raw_s))
    else:
        acf_short = torch.zeros(B, 0, device=device)

    # Half-decay lag (only among valid lags; set to last valid if none found)
    half_g0 = (g0 / 2).unsqueeze(1)
    # For invalid (zeroed) lags, use +inf so they never trigger "below"
    acf_for_hd = torch.where(valid_lag.unsqueeze(0), acf,
                              torch.full_like(acf, float('inf')))
    below_hd   = acf_for_hd < half_g0
    first_idx  = torch.where(
        below_hd.any(1),
        below_hd.long().argmax(1),
        torch.full((B,), n_lags - 1, device=device, dtype=torch.long))
    half_decay = lag_f[first_idx]                              # (B,)

    # ─────────────────────────────────────────────────────────────────────
    # 3.  Scattering  (S1 + S2)
    # ─────────────────────────────────────────────────────────────────────
    X_fft = torch.fft.rfft(I, n=n_sc, dim=1)                  # (B, H)

    # S1 — all 16 filters in one batched IFFT
    u1_all = torch.fft.irfft(
        X_fft.unsqueeze(1) * psi_bank.unsqueeze(0),
        n=n_sc, dim=2
    )[:, :, :N].abs()                                          # (B, 16, N)
    S1_feats = u1_all.mean(2)                                  # (B, 16)

    # S2 — 120 pairs, chunked to control memory
    U1_fft   = torch.fft.rfft(u1_all, n=n_sc, dim=2)          # (B, 16, H)
    del u1_all
    S2_feats = torch.empty(B, n_pairs, device=device)
    for cs in range(0, n_pairs, S2_CHUNK):
        ce   = min(cs + S2_CHUNK, n_pairs)
        k1c  = k1_idx[cs:ce];  k2c = k2_idx[cs:ce]
        prod = U1_fft[:, k1c, :] * psi_bank[k2c].unsqueeze(0) # (B, C, H)
        S2_feats[:, cs:ce] = (
            torch.fft.irfft(prod, n=n_sc, dim=2)[:, :, :N].abs().mean(2))
    del U1_fft

    # ─────────────────────────────────────────────────────────────────────
    # 4.  PSD log-bins
    # ─────────────────────────────────────────────────────────────────────
    psd      = torch.fft.rfft(fluct, n=n_fft2, dim=1).abs() ** 2 / N
    psd_bins = torch.zeros(B, 32, device=device)
    for b in range(32):
        mask = (freqs2 >= psd_edges[b]) & (freqs2 < psd_edges[b + 1])
        if mask.any():
            psd_bins[:, b] = psd[:, mask].mean(1)

    # ─────────────────────────────────────────────────────────────────────
    # 5.  Transit (threshold) features
    # ─────────────────────────────────────────────────────────────────────
    TRANSIT_THR = [0.80, 0.60, 0.40, 0.20, 0.10]
    i_max = I.max(1, keepdim=True).values.clamp(min=eps)       # (B, 1)
    transit = torch.stack(
        [(I >= thr * i_max).float().sum(1) / N for thr in TRANSIT_THR],
        dim=1)                                                 # (B, 5)

    # ── X_base  (10 + n_lags + 16 + 120 + 32 cols) ────────────────────────
    X_base = torch.cat([
        means[:, None], var_[:, None], ((var_ - means) / m1)[:, None],
        skewness[:, None], kurtosis[:, None],
        kappa1[:, None], kappa2[:, None], kappa2_n[:, None],
        g0[:, None], half_decay[:, None],
        acf, S1_feats, S2_feats, psd_bins,
    ], dim=1)
    X_212 = torch.cat([X_base, transit], dim=1)

    # ─────────────────────────────────────────────────────────────────────
    # 6.  Expanded features
    # ─────────────────────────────────────────────────────────────────────

    # (a) OLS exponential fit using only valid lags
    n_vl = int(valid_lag.sum().item())
    if n_vl >= 2:
        vl_f     = lag_f[valid_lag]                            # (n_vl,)
        acf_vl   = acf[:, valid_lag]                           # (B, n_vl)
        log_acf  = acf_vl.double().clamp(min=1e-10).log()        # float64 matches CPU
        vl_f_d   = vl_f.double()
        X_ols    = torch.stack([torch.ones(n_vl, device=device, dtype=torch.float64), vl_f_d], dim=1)
        XtX_inv  = torch.linalg.inv(X_ols.T @ X_ols)
        coeffs   = log_acf @ X_ols @ XtX_inv                  # (B, 2) float64
        A_exp      = coeffs[:, 0].exp().clamp(0, 1e6).float()
        inv_tau    = -coeffs[:, 1].float()
        tau_D_exp  = torch.where(
            inv_tau > 1e-10,
            1.0 / inv_tau.clamp(min=1e-10),
            torch.full_like(inv_tau, float(lag_f[-1].item())))
        tau_D_exp  = tau_D_exp.clamp(0.1, float(N))
        pred_exp   = A_exp[:, None] * torch.exp(
            -vl_f[None, :] / tau_D_exp[:, None])
        rmse_exp   = ((acf_vl - pred_exp) ** 2).mean(1).sqrt()
    else:
        A_exp     = torch.ones(B, device=device)
        tau_D_exp = torch.full((B,), float(N), device=device)
        rmse_exp  = torch.zeros(B, device=device)

    # (b) OLS stretched-exponential fit (log-log), valid lags only
    if n_vl >= 2:
        g0c    = g0.clamp(min=eps)
        ratio  = (acf_vl / g0c[:, None]).clamp(1e-10, 1.0 - 1e-10)
        y_ll   = torch.log(-torch.log(ratio.double()))           # float64 matches CPU
        ll_f_d = vl_f.clamp(min=1.0).double().log()
        X_ll   = torch.stack([ll_f_d, torch.ones(n_vl, device=device, dtype=torch.float64)], dim=1)
        XtX_ll = torch.linalg.inv(X_ll.T @ X_ll)
        c_ll_v = y_ll @ X_ll @ XtX_ll                         # (B, 2) float64
        beta   = c_ll_v[:, 0].float().clamp(0.05, 5.0)
        c_ll   = c_ll_v[:, 1].float()
        tau_D_s = torch.exp(
            (-c_ll / beta.clamp(min=0.05)).clamp(0.0, math.log(N)))
        pred_s  = g0c[:, None] * torch.exp(
            -(vl_f[None, :] / tau_D_s[:, None].clamp(min=0.1)) ** beta[:, None])
        rmse_s  = ((acf_vl - pred_s) ** 2).mean(1).sqrt()
    else:
        g0c   = g0.clamp(min=eps)
        beta  = torch.ones(B, device=device)
        tau_D_s = torch.full((B,), float(N), device=device)
        rmse_s  = torch.zeros(B, device=device)

    # (c) ACF decay lags (interpolated, using all lags with inf-fill for invalid)
    DECAY_FRACS = [0.90, 0.75, 0.25, 0.10]
    acf_for_decay = torch.where(valid_lag.unsqueeze(0), acf,
                                torch.full_like(acf, float('inf')))
    decay_lags_t  = torch.zeros(B, 4, device=device)
    for fi, frac in enumerate(DECAY_FRACS):
        thresh    = frac * g0c                                 # (B,)
        below_m   = acf_for_decay < thresh[:, None]            # (B, n_lags)
        has_below = below_m.any(1)
        first     = torch.where(
            has_below,
            below_m.long().argmax(1),
            torch.full((B,), n_lags - 1, device=device, dtype=torch.long))
        i0  = (first - 1).clamp(0);  i1 = first.clamp(max=n_lags - 1)
        t0v = lag_f[i0];             t1v = lag_f[i1]
        a0  = acf.gather(1, i0[:, None]).squeeze(1)
        a1  = acf.gather(1, i1[:, None]).squeeze(1)
        denom  = (a1 - a0).abs().clamp(min=1e-12)
        t_lerp = (t0v + (thresh - a0) / denom * (t1v - t0v)).clamp(t0v, t1v)
        decay_lags_t[:, fi] = torch.where(
            has_below, t_lerp,
            torch.full_like(t_lerp, float(lag_f[-1].item())))

    # (d) Burst / crossing features  (vectorised via scatter_add)
    EXP_THR     = [0.80, 0.60, 0.40, 0.20, 0.10]
    cross_parts = []
    for thr in EXP_THR:
        above = I >= thr * i_max                               # (B, N) bool
        pad0     = torch.zeros(B, 1, dtype=torch.bool, device=device)
        x_pad    = torch.cat([pad0, above, pad0], dim=1)
        diff_x   = x_pad.int().diff(dim=1)
        starts_m = (diff_x == 1)[:, :N]                        # (B, N)

        n_bursts_raw = starts_m.float().sum(1)
        n_bursts     = n_bursts_raw.clamp(min=1)

        run_id   = starts_m.long().cumsum(1)
        masked   = run_id * above.long()
        max_k    = max(int(run_id.max().item()), 1)
        counts_k = torch.zeros(B, max_k + 1, device=device)
        counts_k.scatter_add_(1, masked, above.float())
        burst_lens = counts_k[:, 1:]

        total_on  = above.float().sum(1)
        max_burst = burst_lens.max(1).values / N
        mean_len  = total_on / n_bursts
        sum_sq    = (burst_lens ** 2).sum(1)
        var_burst = (sum_sq / n_bursts - mean_len ** 2).clamp(min=0)
        std_burst = var_burst.sqrt() / N
        # Mean inter-burst interval: mean(diff(start_positions)) / N
        #   = (last_start - first_start) / (n_bursts_raw - 1) / N  when n_bursts > 1
        #   matches CPU: np.diff(starts).mean() / N
        has_multi  = n_bursts_raw > 1
        first_pos  = starts_m.float().argmax(1).float()
        last_pos   = (N - 1 - starts_m.float().flip(1).argmax(1)).float()
        mean_inter = torch.where(
            has_multi,
            (last_pos - first_pos) / (n_bursts_raw - 1).clamp(min=1) / N,
            torch.ones(B, device=device))

        cross_parts += [n_bursts_raw / N, max_burst, std_burst, mean_inter]

    cross_feats = torch.stack(cross_parts, dim=1)              # (B, 20)

    # (e) Segment features  — zeroed when window too short
    WIN_SIZES = [32, 64, 128, 512, 1024]
    seg_parts = []
    for w in WIN_SIZES:
        n_segs = N // w
        if n_segs < 2:
            seg_parts += [torch.zeros(B, device=device)] * 3
            continue
        segs  = I[:, :n_segs * w].reshape(B, n_segs, w)
        sv    = segs.var(2, unbiased=False)
        sm    = segs.mean(2)
        msv   = sv.mean(1)
        ssm   = sm.std(1, unbiased=False)
        cv_sv = torch.where(msv > 0,
                            sv.std(1, unbiased=False) / msv.clamp(min=eps),
                            torch.zeros_like(msv))
        seg_parts += [msv, ssm, cv_sv]
    seg_feats = torch.stack(seg_parts, dim=1)                  # (B, 15)

    # (f) Higher cumulants
    kappa3_n = (fluct ** 3).mean(1) / (m1 ** 3 + eps)
    kappa4_n = (fluct ** 4).mean(1) / (m1 ** 4 + eps)

    # (g) Multitau Mandel Q  — zeroed when window too short
    BIN_TIMES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    mt_parts  = []
    for T_b in BIN_TIMES:
        n_b = N // T_b
        if n_b < 2:
            mt_parts.append(torch.zeros(B, device=device))
            continue
        bnd  = I[:, :n_b * T_b].reshape(B, n_b, T_b).sum(2)
        mu_t = bnd.mean(1);  vt = bnd.var(1, unbiased=False)
        mt_parts.append(vt / (mu_t ** 2 + eps))
    multitau = torch.stack(mt_parts, dim=1)                    # (B, 10)

    # (h) Nonstationarity
    half  = max(N // 2, 1)
    h1    = I[:, :half];          h2 = I[:, half:2 * half]
    m1_h  = h1.mean(1);           m2 = h2.mean(1).clamp(min=eps)
    v1    = h1.var(1, unbiased=False)
    v2    = h2.var(1, unbiased=False).clamp(min=eps)
    seg_w = max(N // 16, 1)
    s16   = I[:, :16 * seg_w].reshape(B, 16, seg_w).mean(2)
    idx_c = torch.arange(16, dtype=torch.float32, device=device) - 7.5
    slope = (s16 * idx_c).sum(1) / (idx_c ** 2).sum()
    dh1   = h1 - m1_h[:, None]
    dh2   = h2 - h2.mean(1)[:, None]
    cc    = (dh1 * dh2).mean(1) / (v1 * v2).sqrt().clamp(min=eps)
    nonstat = torch.stack([m1_h / m2, v1 / v2, slope / m1, cc], dim=1)

    # ── Assemble X_283 ────────────────────────────────────────────────────
    X_expanded = torch.cat([
        A_exp[:, None], tau_D_exp[:, None], rmse_exp[:, None],
        tau_D_s[:, None], beta[:, None], rmse_s[:, None],
        decay_lags_t,
        acf_short,
        cross_feats,
        seg_feats,
        kappa3_n[:, None], kappa4_n[:, None],
        multitau,
        nonstat,
    ], dim=1)
    X_283 = torch.cat([X_212, X_expanded], dim=1)

    # ─────────────────────────────────────────────────────────────────────
    # 7.  ACF ratio features  (19 consecutive-lag log-ratios)
    # ─────────────────────────────────────────────────────────────────────
    n_X_212      = X_212.shape[1]
    short_offset = C['short_offset_in_expanded']

    lag_to_col: dict = {}
    for ji in range(n_lags):
        lag_to_col[int(lags[ji].item())] = 10 + ji
    for ji in range(n_short):
        lag_to_col[int(short_lags[ji].item())] = n_X_212 + short_offset + ji

    acf_ratio_parts = []
    for i in range(19):
        lo_col = lag_to_col.get(i + 1)
        hi_col = lag_to_col.get(i + 2)
        if lo_col is not None and hi_col is not None:
            g_lo = X_283[:, lo_col].clamp(min=eps)
            g_hi = X_283[:, hi_col].clamp(min=eps)
            acf_ratio_parts.append((g_hi / g_lo).log())
        else:
            acf_ratio_parts.append(torch.zeros(B, device=device))
    acf_ratios = torch.stack(acf_ratio_parts, dim=1)           # (B, 19)

    result = torch.cat([X_283, acf_ratios], dim=1).float()
    # Replace any NaN/inf with 0 — rare edge cases (all-zero traces, singular
    # ACF, etc.) should not corrupt training or validation loss.
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
