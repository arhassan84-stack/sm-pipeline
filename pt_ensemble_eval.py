"""
Ensemble averaging: combine seed=0 (original), seed=1, seed=2 predictions for each dt.

For each dt, averages log(D) predictions from up to 3 seeds and reports MAPE.
Also compares individual seed performance vs ensemble.

Output: results_ensemble.txt, pred_logd_ensemble_{label}_test.npy for each dt.
"""

import numpy as np
import os

def mape(true, pred):
    m = true > 0
    return np.mean(np.abs((pred[m] - true[m]) / true[m])) * 100

D_MAX = 10.0

# Configuration for each dt
CONFIGS = [
    dict(
        label    = 'wavenet_wide_aug',
        d_test_f = 'cache_d_test_90pct.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_wavenet_wide_aug_test.npy',
            'pred_logd_wavenet_wide_aug_s1_test.npy',
            'pred_logd_wavenet_wide_aug_s2_test.npy',
        ],
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt020',
        d_test_f = 'cache_d_test_dt020.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_pt_wavenet_wide_aug_dt020_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt020_s1_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt020_s2_test.npy',
        ],
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt025',
        d_test_f = 'cache_d_test_dt025.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_pt_wavenet_wide_aug_dt025_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt025_s1_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt025_s2_test.npy',
        ],
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt040',
        d_test_f = 'cache_d_test_dt040.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_pt_wavenet_wide_aug_dt040_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt040_s1_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt040_s2_test.npy',
        ],
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt050',
        d_test_f = 'cache_d_test_dt050.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_pt_wavenet_wide_aug_dt050_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt050_s1_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt050_s2_test.npy',
        ],
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt060',
        d_test_f = 'cache_d_test_dt060.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_pt_wavenet_wide_aug_dt060_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt060_s1_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt060_s2_test.npy',
        ],
    ),
    dict(
        label    = 'pt_wavenet_wide_aug_dt080',
        d_test_f = 'cache_d_test_dt080.npy',
        seeds    = [0, 1, 2],
        pred_files = [
            'pred_logd_pt_wavenet_wide_aug_dt080_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt080_s1_test.npy',
            'pred_logd_pt_wavenet_wide_aug_dt080_s2_test.npy',
        ],
    ),
]

summary_lines = []
print(f"\n{'='*70}")
print(f"{'Model':<38} {'S0':>7} {'S1':>7} {'S2':>7} {'Ens':>7}")
print(f"{'='*70}")

for cfg in CONFIGS:
    label = cfg['label']

    d_test_raw = np.load(cfg['d_test_f'])
    mask       = d_test_raw <= D_MAX
    d_test     = d_test_raw[mask].astype(np.float32)
    n          = len(d_test)
    m_slow     = d_test < 1.0
    m_fast     = d_test >= 1.0

    # Load available predictions
    available_preds = []
    available_seeds = []
    for seed, pf in zip(cfg['seeds'], cfg['pred_files']):
        if os.path.exists(pf):
            available_preds.append(np.load(pf).astype(np.float64))
            available_seeds.append(seed)
        else:
            print(f"  MISSING: {pf}")

    if len(available_preds) == 0:
        print(f"{label}: no predictions found, skipping")
        continue

    # Individual MAPEs
    individual_mapes = []
    for seed, logd_pred in zip(available_seeds, available_preds):
        m = mape(d_test, np.exp(logd_pred))
        individual_mapes.append((seed, m))

    # Ensemble: average in log space
    logd_ensemble = np.mean(available_preds, axis=0).astype(np.float32)
    d_ens = np.exp(logd_ensemble)

    mape_ens      = mape(d_test, d_ens)
    mape_ens_slow = mape(d_test[m_slow], d_ens[m_slow])
    mape_ens_fast = mape(d_test[m_fast], d_ens[m_fast])

    # Save ensemble predictions
    out_f = f'pred_logd_ensemble_{label}_test.npy'
    np.save(out_f, logd_ensemble)

    # Print table row
    seed_mapes = {s: m for s, m in individual_mapes}
    s0_str = f"{seed_mapes.get(0, float('nan')):.2f}%" if 0 in seed_mapes else "  N/A "
    s1_str = f"{seed_mapes.get(1, float('nan')):.2f}%" if 1 in seed_mapes else "  N/A "
    s2_str = f"{seed_mapes.get(2, float('nan')):.2f}%" if 2 in seed_mapes else "  N/A "
    print(f"{label:<38} {s0_str:>7} {s1_str:>7} {s2_str:>7} {mape_ens:.2f}%")

    res = (
        f"Task: ensemble_{label}\n"
        f"Seeds: {available_seeds}  n_test={n}\n"
        + "".join(f"  seed={s}: MAPE={m:.2f}%\n" for s, m in individual_mapes)
        + f"Ensemble MAPE={mape_ens:.2f}%\n"
        f"  d<1  ({m_slow.sum():5d}): MAPE={mape_ens_slow:.2f}%\n"
        f"  d>=1 ({m_fast.sum():5d}): MAPE={mape_ens_fast:.2f}%\n"
    )
    summary_lines.append(res)

print(f"{'='*70}\n")

with open('results_ensemble.txt', 'w') as f:
    f.write('\n'.join(summary_lines))
print("Saved results_ensemble.txt")
