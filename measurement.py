"""
measurement.py — Stage 0: Preprocessing
════════════════════════════════════════
Implements Stage 0 of algorithm_sm_inference.md:
  0.1  FCS file ingestion → Measurement object
  0.2  Admissibility check (duration, model availability)
  0.3  RAW file discovery and photon-arrival binning
  0.4  Sub-measurement concatenation
  0.5  Multi-dt channel derivation and FCS↔RAW validation
  0.6  Output: fully-populated Measurement object

Quickstart
----------
    from measurement import load_measurement, MeasurementError

    meas = load_measurement(
        fcs_path  = 'sample.fcs',
        data_dir  = '.',
        registry  = registry,          # ModelRegistry from model_registry.py
        raw_opts  = dict(bits=32, n_header=24, clock_rate_hz=15_000_000),
    )
    # meas.channels['S1'].i100_full  → (N,) float64 kHz
    # meas.has_raw                   → True / False
"""

from __future__ import annotations

import logging
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fcs_io import parse_fcs_file, FCSFile
from raw_io  import read_arrivals, bin_arrivals, parse_raw_filename

log = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class MeasurementError(ValueError):
    """Raised when a measurement cannot be processed."""


# ── Per-channel data container ────────────────────────────────────────────────

@dataclass
class ChannelData:
    """
    All multi-dt trace arrays for one detector channel.
    Arrays are in kHz throughout.
    """
    channel_id: str

    # finest-resolution channel (dt = registry.dt_min, e.g. 0.1 ms)
    i_min_full:  Optional[np.ndarray] = None  # only if has_raw

    # intermediate channel (dt = 0.5 ms)
    i050_full:   Optional[np.ndarray] = None  # only if has_raw

    # 1 ms channel (always present)
    i100_full:   Optional[np.ndarray] = None

    # per-sub-measurement validation results (Stage 0.5)
    validation: dict = field(default_factory=lambda: {
        'r':                     [],   # Pearson r per sub-measurement
        'mard':                  [],   # mean abs relative diff per sub-measurement
        'counts_match':          [],   # bool per sub-measurement
        'failed_submeasurements': [],  # list of indices that failed any check
    })


# ── Main Measurement object ───────────────────────────────────────────────────

@dataclass
class Measurement:
    """
    One FCS acquisition, fully pre-processed through Stage 0.

    Attributes
    ----------
    name             : FCS filename stem (e.g. 'lateralDorsal_nt_1')
    id               : sequential integer assigned at load time
    path_fcs         : Path to the .fcs file
    has_raw          : whether RAW photon-arrival files were found and loaded
    suspect_raw      : True if >20% of sub-measurements failed FCS↔RAW validation
    background_estimated : True if n_abs was estimated from the 5th percentile
                           rather than from the noise model (has_raw=False case)
    n_channels       : number of detector channels
    total_duration_ms: total trace duration
    channel_ids      : ordered list of channel label strings
    channels         : dict[channel_id → ChannelData]
    source_files     : dict[channel_id → list of file paths (fcs + raw)]
    n_submeasurements: number of FCS sub-measurement blocks
    dt_min_ms        : finest time resolution (from registry)
    n_abs            : global background estimate (kHz); None if unavailable
    """
    name:                  str
    id:                    int
    path_fcs:              Path
    has_raw:               bool                = False
    suspect_raw:           bool                = False
    background_estimated:  bool                = False
    n_channels:            int                 = 1
    total_duration_ms:     float               = 0.0
    channel_ids:           List[str]           = field(default_factory=list)
    channels:              Dict[str, ChannelData] = field(default_factory=dict)
    source_files:          Dict[str, List[str]]   = field(default_factory=dict)
    n_submeasurements:     int                 = 0
    dt_min_ms:             float               = 0.1
    n_abs:                 Optional[float]     = None
    _fcs_file:             Optional[FCSFile]   = field(default=None, repr=False)

    def __repr__(self):
        return (f'Measurement({self.name!r}, id={self.id}, '
                f'has_raw={self.has_raw}, '
                f'dur={self.total_duration_ms:.0f}ms, '
                f'ch={self.channel_ids})')


# ── Stage 0 loader ────────────────────────────────────────────────────────────

_meas_counter = 0   # module-level sequential ID


def load_measurement(
    fcs_path:  str | Path,
    data_dir:  str | Path,
    registry,                      # ModelRegistry (imported from model_registry)
    raw_opts:  dict | None = None,
    meas_id:   int  | None = None,
) -> Measurement:
    """
    Run Stage 0 on a single FCS file: ingest → discover RAW → bin → validate.

    Parameters
    ----------
    fcs_path  : path to the .fcs file
    data_dir  : directory to search for matching .raw files
    registry  : ModelRegistry (provides dt_min, model admissibility checks)
    raw_opts  : keyword args forwarded to read_arrivals:
                  bits, n_header, clock_rate_hz
                Defaults: bits=32, n_header=24, clock_rate_hz=15_000_000
    meas_id   : explicit integer ID; auto-increments if None

    Returns
    -------
    Measurement  (fully populated)

    Raises
    ------
    MeasurementError  if the measurement fails admissibility (Stage 0.2)
    """
    global _meas_counter

    fcs_path = Path(fcs_path)
    data_dir = Path(data_dir)
    raw_opts = raw_opts or {}

    if meas_id is None:
        _meas_counter += 1
        meas_id = _meas_counter

    # ── 0.1 FCS ingestion ─────────────────────────────────────────────────────
    log.info(f'[Stage 0.1] Parsing {fcs_path.name}')
    fcs = parse_fcs_file(fcs_path)

    meas = Measurement(
        name              = fcs.name,
        id                = meas_id,
        path_fcs          = fcs_path,
        n_submeasurements = fcs.n_blocks,
        total_duration_ms = fcs.total_duration_ms,
        n_channels        = fcs.n_channels,
        dt_min_ms         = registry.dt_min,
        _fcs_file         = fcs,
    )

    if fcs.n_blocks == 0:
        raise MeasurementError(
            f'{fcs.name}: no sub-measurement blocks found in FCS file'
        )

    # ── 0.2 Admissibility check ───────────────────────────────────────────────
    log.info(f'[Stage 0.2] Admissibility check')
    _check_admissibility(meas, registry)

    # ── 0.3 RAW file discovery and binning ────────────────────────────────────
    log.info(f'[Stage 0.3] Discovering RAW files in {data_dir}')
    raw_map = _discover_raw_files(fcs.name, data_dir, fcs.n_blocks)

    has_raw = bool(raw_map)
    meas.has_raw = has_raw

    # Determine channel IDs
    if has_raw:
        channel_ids = sorted(raw_map.keys())
    else:
        # fall back to 'Ch0', 'Ch1', … based on FCS n_channels
        channel_ids = [f'Ch{k}' for k in range(fcs.n_channels or 1)]

    meas.channel_ids = channel_ids
    meas.n_channels  = len(channel_ids)

    # ── 0.3 Paired binning (all channels together, min-length truncation) ──────
    if has_raw:
        log.info(f'[Stage 0.3] Binning RAW files (paired, channels: {channel_ids})')
        paired = _bin_raw_channels_paired(raw_map, fcs.n_blocks, registry, raw_opts, fcs)
    else:
        paired = {}

    # ── 0.4  Concatenation + 0.5 Multi-dt derivation ─────────────────────────
    for ch_id in channel_ids:
        ch_data = ChannelData(channel_id=ch_id)
        src     = [str(fcs_path)]

        if has_raw:
            ch_data, src_raw   = paired[ch_id]
            ch_data.channel_id = ch_id
            src += src_raw

            # FCS 1ms trace for validation
            fcs_i100 = fcs.intensity_trace(channel=channel_ids.index(ch_id))

            # 0.5 derive i050 and i100 from i_min, and validate against FCS
            ch_data, n_failed, n_evaluated = _derive_and_validate(
                ch_data, fcs_i100, fcs, registry
            )

            log.info(
                f'  Validation: {n_failed}/{n_evaluated} evaluated sub-measurements '
                f'failed (total FCS blocks: {fcs.n_blocks})'
            )
            if n_evaluated > 0 and n_failed / n_evaluated > 0.50:
                meas.suspect_raw = True
                meas.has_raw     = False   # downgrade
                log.warning(
                    f'{fcs.name}/{ch_id}: {n_failed}/{n_evaluated} evaluated '
                    f'sub-measurements failed FCS↔RAW validation — downgrading to has_raw=False'
                )
        else:
            # No RAW — only i100 from FCS
            fcs_i100 = fcs.intensity_trace(channel=channel_ids.index(ch_id))
            ch_data.i100_full = fcs_i100.astype(np.float64)

        meas.channels[ch_id]    = ch_data
        meas.source_files[ch_id] = src

    # If all channels were downgraded, ensure flag is consistent
    if not meas.has_raw:
        _strip_raw_channels(meas)

    return meas


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_admissibility(meas: Measurement, registry) -> None:
    """
    Stage 0.2 — raise MeasurementError if any hard criterion fails.
    """
    # Minimum duration
    try:
        min_noise  = min(m.window_ms for m in registry.noise_models.values())
    except (AttributeError, ValueError):
        min_noise  = 0
    try:
        min_d      = min(m.window_ms for m in registry.diffusion_models.values())
    except (AttributeError, ValueError):
        min_d      = 4096
    min_dur_ms = max(min_noise, min_d)

    if meas.total_duration_ms < min_dur_ms:
        raise MeasurementError(
            f'{meas.name}: total duration {meas.total_duration_ms:.0f} ms < '
            f'required {min_dur_ms} ms'
        )

    if meas.n_submeasurements < 1:
        raise MeasurementError(
            f'{meas.name}: no sub-measurements found'
        )


def _discover_raw_files(
    name: str,
    data_dir: Path,
    n_blocks: int,
) -> dict:
    """
    Scan data_dir for .raw files belonging to this measurement.

    Returns
    -------
    raw_map : dict[channel_id → dict[r_index → Path]]
    or {} if no matching files found.
    """
    pattern = f'{name}_*_R*_*_Ch*.raw'
    found   = list(data_dir.glob(pattern))

    if not found:
        log.info(f'  No RAW files matching {pattern} in {data_dir}')
        return {}

    raw_map: dict[str, dict[int, Path]] = {}
    for fp in found:
        info = parse_raw_filename(fp.name)
        if info is None:
            log.warning(f'  Could not parse filename: {fp.name}')
            continue
        ch  = info['channel_id']
        r   = info['r']
        raw_map.setdefault(ch, {})[r] = fp

    # Check completeness — collect missing (r, channel) pairs and log a summary.
    # n_blocks is the total FCS block count (all channels interleaved), so divide
    # by the number of channels to get the expected number of R-files per channel.
    n_channels_found = max(len(raw_map), 1)
    n_reps           = n_blocks // n_channels_found
    missing = []
    for ch, r_map in raw_map.items():
        for r in range(1, n_reps + 1):
            if r not in r_map:
                missing.append((ch, r))
                log.debug(f'  Missing RAW file for channel={ch}, R={r}')
    if missing:
        log.warning(f'  {len(missing)} RAW file(s) missing out of '
                    f'{n_reps * len(raw_map)} expected '
                    f'(channels: {sorted(raw_map)})')

    # Drop sub-measurements missing from every channel (e.g. interrupted last block).
    # Only abort entirely if NO sub-measurement has any RAW data.
    all_rs = set().union(*(set(rm.keys()) for rm in raw_map.values()))
    n_skipped = 0
    for r in range(1, n_reps + 1):
        if r not in all_rs:
            n_skipped += 1
            log.debug(f'  R={r} has no RAW file for any channel — skipping')
            for ch in raw_map:
                raw_map[ch].pop(r, None)
    if n_skipped:
        log.warning(f'  {n_skipped} sub-measurement(s) skipped (no RAW for any channel)')

    if not any(raw_map.values()):
        log.warning('  No sub-measurements have any RAW data — has_raw=False')
        return {}

    return raw_map


def _bin_raw_channels_paired(
    raw_map:  dict,   # ch_id → {r_index → Path}
    n_blocks: int,
    registry,
    raw_opts: dict,
    fcs,              # FCSFile — used to get per-block bin counts
) -> dict:            # ch_id → (ChannelData, list[str])
    """
    Stage 0.3: Read and bin all RAW files for ALL channels together.

    Two-stage length enforcement (matches Matlab FRET_FCS_ANALYSIS logic):

    1. Cross-channel min-truncation per R file — faithfully replicates:
           len = min(size(s1,1), size(s2,1));
           s1 = s1(1:len,1); s2 = s2(1:len,1);
       This handles the case where one channel has 1 extra stray photon.

    2. FCS-block truncation — for R files where BOTH channels have the same
       extra bin, the corresponding FCS block's bin count is used as the hard
       ceiling.  This removes the cumulative timing drift completely.
    """
    bits          = raw_opts.get('bits',          32)
    n_header      = raw_opts.get('n_header',      32)
    clock_rate_hz = raw_opts.get('clock_rate_hz', 15_000_000)
    dt_min_ms     = registry.dt_min

    channel_ids = sorted(raw_map.keys())
    # Only process R files present in ALL channels
    all_rs = sorted(set.intersection(*(set(raw_map[ch].keys())
                                       for ch in channel_ids)))

    # Build per-R, per-channel FCS block sizes (at 1ms resolution).
    # FCS blocks are interleaved: even = ch0, odd = ch1.
    # For R file r (1-indexed): ch_k block index = 2*(r-1) + k.
    # Converting 1ms FCS bins → dt_min bins: multiply by dt_ratio.
    fcs_blocks = fcs.blocks
    dt_ratio   = round(1.0 / dt_min_ms)   # e.g. 10 for dt_min=0.1ms

    def _fcs_bins_for_channel(r: int, k_ch: int) -> int:
        """Return the FCS bin count (at dt_min) for channel k_ch of R file r.
        Returns 0 if no FCS block is available (skip ceiling for this file)."""
        blk_idx = 2 * (r - 1) + k_ch
        if blk_idx < len(fcs_blocks) and fcs_blocks[blk_idx].count_rates:
            return fcs_blocks[blk_idx].count_rates[0].shape[0] * dt_ratio
        return 0

    def _load_r(r: int) -> dict:
        """Bin all channels for one repetition; return {ch_id: segment}."""
        result = {}
        for ch in channel_ids:
            arrivals   = read_arrivals(raw_map[ch][r], bits=bits,
                                       n_header=n_header,
                                       clock_rate_hz=clock_rate_hz)
            result[ch] = bin_arrivals(arrivals,
                                      bin_ms=dt_min_ms).astype(np.float64)
        return result

    n_workers = min(8, len(all_rs))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        per_r = list(pool.map(_load_r, all_rs))

    total_mismatch = 0
    segs = {ch: [] for ch in channel_ids}
    srcs = {ch: [] for ch in channel_ids}

    for r, ch_segs in zip(all_rs, per_r):
        # Stage 1: per-channel FCS block ceiling — each channel is trimmed
        # to the bin count of its own FCS block (removes the stray photon
        # that falls just past the measurement window).
        trimmed = {}
        for k_ch, ch in enumerate(channel_ids):
            lim = _fcs_bins_for_channel(r, k_ch)
            trimmed[ch] = ch_segs[ch][:lim] if lim > 0 else ch_segs[ch]

        # Stage 2: cross-channel min (Matlab fix) — align any residual
        # length difference between channels.
        min_len = min(len(trimmed[ch]) for ch in channel_ids)
        max_len = max(len(trimmed[ch]) for ch in channel_ids)
        total_mismatch += max_len - min_len

        for ch in channel_ids:
            segs[ch].append(trimmed[ch][:min_len])
            srcs[ch].append(str(raw_map[ch][r]))

    if total_mismatch:
        log.debug(f'  RAW inter-channel length mismatch: '
                  f'{total_mismatch} bins discarded across all R files')

    out = {}
    for ch in channel_ids:
        cd            = ChannelData(channel_id=ch)
        cd.i_min_full = np.concatenate(segs[ch]) if segs[ch] else np.array([])
        out[ch]       = (cd, srcs[ch])
    return out


def _derive_and_validate(
    ch:       ChannelData,
    fcs_i100: np.ndarray,     # full-trace 1ms FCS intensity (kHz)
    fcs:      FCSFile,
    registry,
) -> Tuple[ChannelData, int]:
    """
    Stage 0.5: Derive i050 and i100 from i_min by block-averaging,
    then validate against FCS trace.

    Returns (updated ChannelData, n_failed_submeasurements).
    """
    dt_min   = registry.dt_min          # e.g. 0.1 ms
    B050     = round(0.5 / dt_min)      # bins per 0.5ms bin  (e.g. 5)
    B100     = round(1.0 / dt_min)      # bins per 1.0ms bin  (e.g. 10)

    i_min = ch.i_min_full

    # Trim to a multiple of B100 (needed for clean block-averaging)
    trim = (len(i_min) // B100) * B100
    if trim < len(i_min):
        log.debug(f'  Trimming i_min from {len(i_min)} to {trim} bins')
    i_min = i_min[:trim]

    # Convert counts/bin → kHz count rate.
    # bin_arrivals returns integer counts per dt_min bin.
    # 1 count per dt_min ms = 1/(dt_min × 1e-3) Hz = 1/(dt_min × 1e-3 × 1e3) kHz
    khz_factor = 1.0 / (dt_min * 1e-3) / 1e3   # = 10.0 for dt_min = 0.1 ms
    i_min_khz  = i_min * khz_factor              # (trim,) kHz

    # Block-average to coarser resolutions.
    # mean() over N bins of a count-rate signal preserves the count rate in kHz.
    i100 = i_min_khz.reshape(-1, B100).mean(axis=1)   # (n_bins_100,) kHz
    i050 = (i_min_khz.reshape(-1, B050).mean(axis=1)
            if B050 > 0 else i100.copy())               # kHz

    ch.i_min_full = i_min_khz   # store in kHz (was counts/bin)
    ch.i100_full  = i100
    ch.i050_full  = i050

    # ── Validate against FCS ─────────────────────────────────────────────────
    bins_per_block = fcs.blocks[0].count_rates[0].shape[0] if (
        fcs.blocks and fcs.blocks[0].count_rates
    ) else 2000   # default 2s at 1ms

    log.info(
        f'  Validation: len(i100)={len(i100)}, len(fcs_i100)={len(fcs_i100)}, '
        f'bins_per_block={bins_per_block}, fcs.n_blocks={fcs.n_blocks}'
    )

    # Pass criterion: total photon counts within 5% per sub-measurement.
    # Per-bin metrics (Pearson r, MARD) are unreliable for SM-FCS data because
    # most bins are background (few photons → shot noise ≫ signal → r≈0 even
    # for identical traces).  We log r for diagnostics only.
    # Sub-measurements for which no RAW data was loaded (i100k empty) are
    # skipped rather than counted as failures.
    n_failed   = 0
    n_evaluated = 0
    for k in range(fcs.n_blocks):
        sl    = slice(k * bins_per_block, (k + 1) * bins_per_block)
        i100k = i100[sl] if sl.stop <= len(i100) else np.array([])
        fcsk  = fcs_i100[sl] if sl.stop <= len(fcs_i100) else np.array([])

        if len(i100k) == 0:
            continue   # no RAW data for this sub-measurement — skip

        if len(fcsk) == 0 or len(i100k) != len(fcsk):
            ch.validation['failed_submeasurements'].append(k)
            n_failed   += 1
            n_evaluated += 1
            continue

        # Total photon count match (primary criterion)
        sum_fcs  = float(fcsk.sum())
        sum_raw  = float(i100k.sum())
        cnt_diff = abs(sum_raw - sum_fcs) / (abs(sum_fcs) + 1e-9)

        # Pearson r — logged for diagnostics, not used for pass/fail
        if fcsk.std() > 0 and i100k.std() > 0:
            r = float(np.corrcoef(i100k, fcsk)[0, 1])
        else:
            r = 1.0 if np.allclose(i100k, fcsk) else 0.0

        ch.validation['r'].append(r)
        ch.validation['counts_match'].append(cnt_diff <= 0.05)

        n_evaluated += 1
        if cnt_diff > 0.05:
            ch.validation['failed_submeasurements'].append(k)
            n_failed += 1
            log.warning(
                f'  Sub-measurement k={k}: cnt_diff={cnt_diff:.4f} (r={r:.3f})'
            )
        else:
            log.debug(
                f'  Sub-measurement k={k}: cnt_diff={cnt_diff:.4f} (r={r:.3f}) OK'
            )

    return ch, n_failed, n_evaluated


def _strip_raw_channels(meas: Measurement) -> None:
    """
    Remove i_min_full and i050_full when has_raw has been downgraded to False.
    """
    for ch in meas.channels.values():
        ch.i_min_full = None
        ch.i050_full  = None
