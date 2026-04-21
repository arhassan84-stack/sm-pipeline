# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
verify_raw_fcs_alignment.py

Quick sanity-check for the two-stage truncation fix in _bin_raw_channels_paired.

For each channel:
  - Reads all R files via the new paired function
  - Compares length, photon count, and per-block Pearson r against FCS trace
  - Checks alignment of the 112 kHz peak that was previously at t≈30499ms

Expected outputs after the fix:
  - len(RAW) == len(FCS)  (no ±1 bin residual)
  - Pearson r ≈ 1.000 per block
  - Peak location offset = 0 ms
"""

import sys
from pathlib import Path
import numpy as np

PROJECT  = Path(__file__).parent
DATA_DIR = PROJECT / '01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF'

sys.path.insert(0, str(PROJECT))
from fcs_io  import parse_fcs_file
from raw_io  import read_arrivals, bin_arrivals

FCS_STEM  = 'nt_dorsal_1'
CHANNELS  = ['S1', 'S2']      # k_ch = 0, 1
DT_MIN_MS = 0.1
DT_RATIO  = round(1.0 / DT_MIN_MS)   # 10

def load_raw_paired(fcs, stem):
    """
    Replicate _bin_raw_channels_paired logic directly so we can inspect per-R.
    """
    bits          = 32
    n_header      = 32
    clock_rate_hz = 15_000_000

    # Discover R files
    raw_map = {ch: {} for ch in CHANNELS}
    raw_dir = DATA_DIR
    ch_tags = {'S1': 'ChS1', 'S2': 'ChS2'}
    for r in range(1, fcs.n_blocks + 1):
        for ch in CHANNELS:
            tag = ch_tags[ch]
            # Pattern: stem_*_R{r}_P*_K*_{tag}.raw
            cands = sorted(raw_dir.glob(f'{stem}_*_R{r}_P*_K*_{tag}.raw'))
            if cands:
                raw_map[ch][r] = cands[0]

    # Only keep R files present in ALL channels
    common_rs = sorted(set.intersection(*(set(raw_map[ch].keys()) for ch in CHANNELS)))
    print(f'  Common R files: {len(common_rs)} (R{common_rs[0]}…R{common_rs[-1]})')

    fcs_blocks = fcs.blocks

    def fcs_bins(r, k_ch):
        blk_idx = 2 * (r - 1) + k_ch
        if blk_idx < len(fcs_blocks) and fcs_blocks[blk_idx].count_rates:
            return fcs_blocks[blk_idx].count_rates[0].shape[0] * DT_RATIO
        return 0

    segs    = {ch: [] for ch in CHANNELS}
    mismatches = []

    for r in common_rs:
        ch_segs = {}
        for ch in CHANNELS:
            arr = read_arrivals(raw_map[ch][r], bits=bits, n_header=n_header,
                                clock_rate_hz=clock_rate_hz)
            ch_segs[ch] = bin_arrivals(arr, bin_ms=DT_MIN_MS).astype(np.float64)

        # Stage 1: per-channel FCS block ceiling
        trimmed = {}
        for k_ch, ch in enumerate(CHANNELS):
            lim = fcs_bins(r, k_ch)
            trimmed[ch] = ch_segs[ch][:lim] if lim > 0 else ch_segs[ch]

        # Stage 2: cross-channel min
        min_len = min(len(trimmed[ch]) for ch in CHANNELS)
        max_len = max(len(trimmed[ch]) for ch in CHANNELS)
        mismatches.append(max_len - min_len)

        for ch in CHANNELS:
            segs[ch].append(trimmed[ch][:min_len])

    print(f'  Per-R mismatch after stage-1: '
          f'max={max(mismatches)} bins, total={sum(mismatches)} bins')

    return {ch: np.concatenate(segs[ch]) for ch in CHANNELS}


def main():
    fcs_path = DATA_DIR / f'{FCS_STEM}.fcs'
    print(f'\nLoading FCS: {fcs_path.name}')
    fcs = parse_fcs_file(fcs_path)
    print(f'  Blocks: {fcs.n_blocks}   (expect 370 = 185×2 channels)')

    # ── Build RAW traces ──────────────────────────────────────────────────────
    print('\nBuilding paired RAW traces …')
    raw_traces = load_raw_paired(fcs, FCS_STEM)

    # ── Compare lengths ────────────────────────────────────────────────────────
    print('\n── Length comparison ──────────────────────────────────────────────')
    for k_ch, ch in enumerate(CHANNELS):
        fcs_i100 = fcs.intensity_trace(channel=k_ch)
        raw_i100 = raw_traces[ch].reshape(-1, DT_RATIO).mean(axis=1) * (1.0 / (DT_MIN_MS * 1e-3) / 1e3)
        # length in 1ms bins
        print(f'  {ch}:  len(RAW@1ms)={len(raw_i100):7d}  len(FCS@1ms)={len(fcs_i100):7d}  '
              f'diff={len(raw_i100)-len(fcs_i100):+d}')

    # ── Per-block Pearson r (at 1ms) ──────────────────────────────────────────
    print('\n── Per-block Pearson r (first 5 blocks per channel) ───────────────')
    bins_per_block = fcs.blocks[0].count_rates[0].shape[0] if (
        fcs.blocks and fcs.blocks[0].count_rates) else 2000

    khz_factor = 1.0 / (DT_MIN_MS * 1e-3) / 1e3

    for k_ch, ch in enumerate(CHANNELS):
        fcs_i100 = fcs.intensity_trace(channel=k_ch)
        raw_i100 = raw_traces[ch].reshape(-1, DT_RATIO).mean(axis=1) * khz_factor
        n_blk    = min(5, len(raw_i100) // bins_per_block)
        rs = []
        for k in range(n_blk):
            sl = slice(k * bins_per_block, (k + 1) * bins_per_block)
            ri = raw_i100[sl]; fi = fcs_i100[sl]
            if len(ri) == len(fi) and ri.std() > 0 and fi.std() > 0:
                rs.append(np.corrcoef(ri, fi)[0, 1])
        r_str = '  '.join(f'{r:.6f}' for r in rs)
        print(f'  {ch} blocks 0-{n_blk-1}: {r_str}')

    # ── Peak alignment check (S1 around t=30499ms) ───────────────────────────
    print('\n── Peak alignment check (S1, t≈30499ms) ───────────────────────────')
    ch = 'S1'
    fcs_i100 = fcs.intensity_trace(channel=0)
    raw_i100 = raw_traces[ch].reshape(-1, DT_RATIO).mean(axis=1) * khz_factor

    t0, t1 = 30480, 30520   # 40ms window around the known peak
    for label, trace in [('FCS', fcs_i100), ('RAW', raw_i100)]:
        seg  = trace[t0:min(t1, len(trace))]
        tpk  = int(np.argmax(seg)) + t0
        apk  = float(seg.max())
        print(f'  {label}: peak at t={tpk}ms  amplitude={apk:.1f} kHz')

    # Offset
    fcs_pk = int(np.argmax(fcs_i100[t0:t1])) + t0
    raw_pk = int(np.argmax(raw_i100[t0:min(t1, len(raw_i100))])) + t0
    print(f'  Offset (RAW - FCS): {raw_pk - fcs_pk} ms  (expect 0)')

    print('\nDone.')


if __name__ == '__main__':
    main()
