"""
fcs_io.py — Zeiss FCS text-file parser
═══════════════════════════════════════
Translated from compile_FCS_file.m and read_FCS_file.m (Matlab).

Quickstart
----------
    from fcs_io import parse_fcs_file

    fcs = parse_fcs_file('sample.fcs')
    print(fcs.name, fcs.n_blocks, 'sub-measurements')

    # Concatenated 1ms-binned intensity trace (kHz) for the primary channel:
    trace = fcs.intensity_trace(channel=0)     # shape (N,)

File format (Zeiss confocal FCS export)
----------------------------------------
The FCS file is a text file containing one or more sub-measurement blocks.
Each block contains:
  - AcquisitionTime   — timestamp string
  - RawData           — relative path to the associated .raw file
  - CountRateArray    — N rows × 2 cols:  [time_s, count_rate_kHz]
  - CorrelationArray  — M rows × 2 cols:  [lag_s,  g_tau]
  - PhotonCountHistogramArray / PulseDistanceHistogramArray (optional)
  - MeasurementTime   — block duration in seconds
  - Channels          — number of detector channels
  - DetectorWavelengthRangeStart/End — wavelength range per channel (nm)

A sub-measurement boundary is detected when all channel wavelength ranges
for the current block have been read, mirroring the Matlab behaviour.
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class FCSBlock:
    """One sub-measurement block parsed from an FCS file."""
    acq_time_str:      Optional[str]       = None   # raw timestamp string
    duration_s:        Optional[float]     = None   # measurement duration (s)
    n_channels:        int                 = 0
    wavelength_ranges: list                = field(default_factory=list)
    # [(start_nm, end_nm), ...]  — one entry per channel

    raw_file:          Optional[str]       = None
    # relative filename of the associated .raw file (no path prefix)

    count_rates:       list                = field(default_factory=list)
    # list of np.ndarray shape (N, 2):  [time_s, count_rate_kHz]
    # one entry per channel (or one for single-channel instruments)

    acfs:              list                = field(default_factory=list)
    # list of np.ndarray shape (M, 2):  [lag_s, g_tau]

    photon_hists:      list                = field(default_factory=list)
    pulse_dist_hists:  list                = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        return None if self.duration_s is None else self.duration_s * 1e3


@dataclass
class FCSFile:
    """Parsed FCS file — a list of sub-measurement blocks."""
    path:   Path
    name:   str        # stem without extension
    blocks: list       # list[FCSBlock]

    # ── convenience properties ───────────────────────────────────────────────

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_channels(self) -> int:
        """Detector channel count inferred from the first informative block."""
        for b in self.blocks:
            if b.n_channels > 0:
                return b.n_channels
        # fallback: guess from number of count-rate arrays in first block
        if self.blocks and self.blocks[0].count_rates:
            return len(self.blocks[0].count_rates)
        return 1

    @property
    def total_duration_ms(self) -> float:
        return sum(b.duration_ms for b in self.blocks if b.duration_ms is not None)

    def intensity_trace(self, channel: int = 0) -> np.ndarray:
        """
        Concatenate the count-rate intensity values across all blocks for one
        detector channel and return a 1-D array in **kHz**.

        Two FCS block layouts are handled automatically:

        (A) Multi-channel-per-block: each FCSBlock contains one CountRateArray
            per channel.  E.g. a 2-channel file has blocks with 2 arrays each.
            → Use direct indexing: arr = b.count_rates[channel].

        (B) Interleaved (one channel per block): blocks alternate channels.
            block 0 = R1/Ch0, block 1 = R1/Ch1, block 2 = R2/Ch0, ...
            → Use modulo skip: skip blocks where i % nc != channel % nc.

        The file stores count rates in Hz (e.g. 135000.0 = 135 kHz).
        This method divides by 1000 to return kHz.
        """
        nc    = max(self.n_channels, 1)
        parts = []

        # Detect layout from first block that has data.
        first_b = next((b for b in self.blocks if b.count_rates), None)
        # If the first block already has >= nc count-rate arrays, assume layout A.
        multi_ch_per_block = (
            first_b is not None and nc > 1 and len(first_b.count_rates) >= nc
        )

        for i, b in enumerate(self.blocks):
            if not b.count_rates:
                continue
            if multi_ch_per_block:
                # Layout A: all channels in each block — pick by index.
                arr = b.count_rates[channel % len(b.count_rates)]
            else:
                # Layout B: one channel per block — skip non-matching blocks.
                if i % nc != channel % nc:
                    continue
                arr = b.count_rates[0]

            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
                parts.append(arr[:, 1] / 1000.0)   # Hz → kHz
            elif arr.ndim == 1 and arr.shape[0] > 0:
                parts.append(arr / 1000.0)
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    def acf(self, channel: int = 0, block: int = 0) -> Optional[np.ndarray]:
        """Return the ACF array [lag_s, g_tau] for a given channel and block."""
        if block >= len(self.blocks):
            return None
        b = self.blocks[block]
        idx = channel % len(b.acfs) if b.acfs else -1
        return b.acfs[idx] if b.acfs else None

    def raw_file_for(self, block: int) -> Optional[str]:
        """Return the raw filename associated with a given sub-measurement index."""
        if block < len(self.blocks):
            return self.blocks[block].raw_file
        return None


# ── Parser ───────────────────────────────────────────────────────────────────

def parse_fcs_file(filepath, verbose: bool = False) -> FCSFile:
    """
    Parse a Zeiss FCS text file.

    Parameters
    ----------
    filepath : str or Path
    verbose  : bool — print each matched keyword line (for debugging)

    Returns
    -------
    FCSFile
    """
    filepath = Path(filepath)

    with open(filepath, 'r', errors='replace') as fh:
        lines = fh.readlines()

    blocks:     list[FCSBlock] = []
    current                    = FCSBlock()
    n_lambda_seen              = 0  # count of wavelength range endpoints seen

    def _flush():
        nonlocal current, n_lambda_seen
        # only keep blocks that have at least some useful data
        if (current.count_rates or current.duration_s is not None
                or current.raw_file is not None):
            # convert wavelength_ranges to (start, end) tuples
            current.wavelength_ranges = [
                tuple(wr) for wr in current.wavelength_ranges
            ]
            blocks.append(current)
        current       = FCSBlock()
        n_lambda_seen = 0

    def _read_table(start_line: int, n_rows: int) -> tuple[np.ndarray, int]:
        """
        Read up to n_rows data rows from lines[start_line:].
        Returns (array of shape (found_rows, 2), next_line_index).
        """
        rows = []
        j    = start_line
        while j < len(lines) and len(rows) < n_rows:
            parts = lines[j].split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                    j += 1
                    continue
                except ValueError:
                    pass
            # blank or non-numeric line — skip but do not stop reading
            # (some FCS files have occasional blank lines mid-block)
            if parts:          # non-blank but unparseable → stop
                break
            j += 1
        arr = np.array(rows, dtype=np.float64) if rows else np.empty((0, 2))
        return arr, j

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── AcquisitionTime ───────────────────────────────────────────────────
        if 'AcquisitionTime =' in line:
            if verbose:
                print(f'[acqtime]  {line.rstrip()}')
            m = re.search(r'AcquisitionTime\s*=\s*(\S+)', line)
            if m:
                current.acq_time_str = m.group(1)
            i += 1

        # ── RawData ───────────────────────────────────────────────────────────
        elif 'RawData =' in line:
            if verbose:
                print(f'[rawdata]  {line.rstrip()}')
            # Extract filename — keep only the part after the last '/'
            m = re.search(r'RawData\s*=\s*(\S+)', line)
            if m:
                raw_path = m.group(1)
                # strip any directory prefix
                current.raw_file = raw_path.split('/')[-1].split('\\')[-1]
            i += 1

        # ── MeasurementTime ───────────────────────────────────────────────────
        elif 'MeasurementTime =' in line:
            if verbose:
                print(f'[duration] {line.rstrip()}')
            m = re.search(r'MeasurementTime\s*=\s*([\d.eE+\-]+)', line)
            if m:
                current.duration_s = float(m.group(1))
            i += 1

        # ── Channels ──────────────────────────────────────────────────────────
        elif re.search(r'\bChannels\s*=\s*\d', line):
            if verbose:
                print(f'[channels] {line.rstrip()}')
            m = re.search(r'Channels\s*=\s*(\d+)', line)
            if m:
                current.n_channels = int(m.group(1))
            i += 1

        # ── DetectorWavelengthRange ───────────────────────────────────────────
        elif 'DetectorWavelengthRangeStart' in line:
            if verbose:
                print(f'[wl_start] {line.rstrip()}')
            m = re.search(r'DetectorWavelengthRangeStart\d*\s*=\s*([\d.]+)', line)
            if m:
                current.wavelength_ranges.append([float(m.group(1)), None])
                n_lambda_seen += 1
            i += 1

        elif 'DetectorWavelengthRangeEnd' in line:
            if verbose:
                print(f'[wl_end]   {line.rstrip()}')
            m = re.search(r'DetectorWavelengthRangeEnd\d*\s*=\s*([\d.]+)', line)
            if m:
                # fill the last (start, None) entry
                for wr in reversed(current.wavelength_ranges):
                    if wr[1] is None:
                        wr[1] = float(m.group(1))
                        break
                n_lambda_seen += 1
            # flush when all wavelength ranges for all channels are complete
            if (current.n_channels > 0
                    and n_lambda_seen >= 2 * current.n_channels):
                _flush()
            i += 1

        # ── CountRateArray ────────────────────────────────────────────────────
        elif 'CountRateArray =' in line:
            if verbose:
                print(f'[countrate] {line.rstrip()}')
            m = re.search(r'CountRateArray\s*=\s*(\d+)\s+(\d+)', line)
            if m:
                n_rows = int(m.group(1))
                arr, i = _read_table(i + 1, n_rows)
                if arr.size:
                    current.count_rates.append(arr)
            else:
                i += 1

        # ── CorrelationArray ──────────────────────────────────────────────────
        elif 'CorrelationArray =' in line:
            if verbose:
                print(f'[acf]      {line.rstrip()}')
            m = re.search(r'CorrelationArray\s*=\s*(\d+)\s+(\d+)', line)
            if m:
                n_rows = int(m.group(1))
                arr, i = _read_table(i + 1, n_rows)
                if arr.size:
                    current.acfs.append(arr)
            else:
                i += 1

        # ── PhotonCountHistogramArray ─────────────────────────────────────────
        elif 'PhotonCountHistogramArray =' in line:
            m = re.search(r'PhotonCountHistogramArray\s*=\s*(\d+)\s+(\d+)', line)
            if m:
                n_rows = int(m.group(1))
                arr, i = _read_table(i + 1, n_rows)
                if arr.size:
                    current.photon_hists.append(arr)
            else:
                i += 1

        # ── PulseDistanceHistogramArray ───────────────────────────────────────
        elif 'PulseDistanceHistogramArray =' in line:
            m = re.search(r'PulseDistanceHistogramArray\s*=\s*(\d+)\s+(\d+)', line)
            if m:
                n_rows = int(m.group(1))
                arr, i = _read_table(i + 1, n_rows)
                if arr.size:
                    current.pulse_dist_hists.append(arr)
            else:
                i += 1

        else:
            i += 1

    _flush()   # handle any final block not terminated by wavelength-range lines

    return FCSFile(path=filepath, name=filepath.stem, blocks=blocks)
