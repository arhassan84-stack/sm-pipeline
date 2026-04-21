"""
raw_io.py — Sony/Zeiss RAW photon-arrival-time file reader
═══════════════════════════════════════════════════════════
Translated from read_PAT.m (Matlab) by Syuan-Ming Guo, MIT 2012.

Quickstart
----------
    from raw_io import read_raw, parse_raw_header

    # Auto-detect header and clock rate, bin into 0.1ms bins:
    trace = read_raw('sample.raw', bin_ms=0.1)   # uses auto-detected parameters

    # Or in two steps (keep raw arrival times for re-binning):
    arrivals = read_arrivals('sample.raw')         # → arrival times in seconds
    trace_1ms  = bin_arrivals(arrivals, bin_ms=1.0)
    trace_01ms = bin_arrivals(arrivals, bin_ms=0.1)

    # Inspect header metadata:
    hdr = parse_raw_header('sample.raw')
    print(hdr)   # {'clock_rate_hz': 15000000, 'header_bytes': 96, ...}

File format (confirmed from 01082026 dataset)
---------------------------------------------
The .raw file is a binary stream of uint16 little-endian integers.
Each value is a photon inter-arrival time in hardware clock ticks.
Cumulative summation converts inter-arrival intervals to absolute arrival
timestamps, which are then divided by the bin size to produce a histogram
(photon count per bin).

Binary layout:
  Bytes   0–67  : ASCII text "Carl Zeiss ConfoCor3 - raw data file - version 3.000 - META N   "
  Bytes  68–79  : 12-byte UUID / acquisition identifier
  Bytes  80–91  : 12 null bytes
  Bytes  92–95  : uint32 little-endian clock rate in Hz (e.g. 0x00E4E1C0 = 15,000,000)
  Bytes  96+    : null padding (variable length), then uint16 inter-arrival times
  → skip first 48 uint16 values (96 bytes) to reach photon data

Confirmed parameters (01082026_itga5_296i dataset):
  bits          = 16
  n_header      = 48   (uint16 values = 96 bytes)
  clock_rate_hz = 15_000_000  (15 MHz, same for ChS1 and ChS2)
  unit: kHz via 15e6 / mean_iat

The hardware clock rate (Hz) and any header convention depend on the Sony
detector model used with the Zeiss system.  These are supplied as parameters
rather than hard-coded (see Stage 0.3 of algorithm_sm_inference.md).

Known / assumed defaults (to be confirmed from instrument spec):
  bits=16                — 16-bit unsigned integers
  n_header=0             — no header (adjust when format is confirmed)
  clock_rate_hz=1e8      — 100 MHz clock (10 ns/tick) — VERIFY WITH USER

Two-channel acquisition
-----------------------
For two-channel instruments the data is split between two separate files,
conventionally named 'prefixA.raw' and 'prefixB.raw'.  Use read_raw_pair().
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ── Constants (confirmed from 01082026 dataset) ──────────────────────────────

DEFAULT_BITS         = 32           # uint32 inter-arrival times (4-byte LE values)
DEFAULT_N_HEADER     = 32           # uint32 values to skip (= 128 bytes header)
DEFAULT_CLOCK_HZ     = 15_000_000   # 15 MHz (embedded in header bytes 92-95)

# Header field offsets (bytes)
_HEADER_CLOCK_OFFSET = 92          # uint32 LE clock rate starts here
_HEADER_DATA_OFFSET  = 96          # photon data starts here (96 bytes fixed header)


# ── Header parsing ───────────────────────────────────────────────────────────

def parse_raw_header(filepath) -> dict:
    """
    Parse the binary header of a Carl Zeiss ConfoCor3 .raw file.

    Returns
    -------
    dict with keys:
        'version_str'    : str  — ASCII text line (e.g. "Carl Zeiss ConfoCor3 ...")
        'clock_rate_hz'  : int  — hardware clock rate in Hz (from bytes 92-95)
        'header_bytes'   : int  — number of bytes to skip before photon data (96)
        'n_header_uint16': int  — header_bytes / 2  (n_header for read_arrivals)
    """
    with open(filepath, 'rb') as f:
        hdr = f.read(96)

    if len(hdr) < 96:
        return {
            'version_str':     '',
            'clock_rate_hz':   DEFAULT_CLOCK_HZ,
            'header_bytes':    0,
            'n_header_uint16': 0,
        }

    version_str = hdr[:68].decode('ascii', errors='replace').strip()

    # Clock rate at bytes 92-95 as uint32 little-endian
    import struct
    clock_rate_hz = struct.unpack_from('<I', hdr, 92)[0]
    if clock_rate_hz == 0:
        clock_rate_hz = DEFAULT_CLOCK_HZ

    return {
        'version_str':     version_str,
        'clock_rate_hz':   clock_rate_hz,
        'header_bytes':    128,               # 128 bytes total header (confirmed)
        'n_header_uint32': 32,                # 32 uint32 values to skip (confirmed)
    }


# ── Core I/O ─────────────────────────────────────────────────────────────────

def read_arrivals(
    filepath,
    bits:         int   = DEFAULT_BITS,
    n_header:     int   = DEFAULT_N_HEADER,
    clock_rate_hz: float = DEFAULT_CLOCK_HZ,
) -> np.ndarray:
    """
    Read photon inter-arrival times from a .raw file and return absolute
    arrival timestamps in seconds.

    The ConfoCor3 .raw format stores each inter-arrival time (IAT) as one
    uint32 little-endian value.  The file begins with a 128-byte header
    (= 32 uint32 values) followed by the photon IAT stream.

    Faithful to read_PAT.m (Syuan-Ming Guo, MIT 2012):
        patA(1:opt.hdr) = [];
        patA_c = cumsum(double(patA));   % NO [::2] striding

    Parameters
    ----------
    filepath      : str or Path
    bits          : dtype width in bits; 32 for this format (uint32 IATs)
    n_header      : number of dtype-width values to skip for the header
                    (default 32 = 128 bytes / 4 bytes per uint32)
    clock_rate_hz : hardware clock frequency in Hz (ticks/second)

    Returns
    -------
    arrivals : np.ndarray, float64, shape (n_photons,)
        Absolute photon arrival times in seconds.
    """
    filepath = Path(filepath)
    dtype    = np.dtype(f'<u{bits // 8}')   # little-endian unsigned

    raw = np.fromfile(filepath, dtype=dtype)
    if n_header > 0:
        raw = raw[n_header:]

    # Skip variable-length null-padding zeros that follow the fixed header.
    first_nonzero = int(np.argmax(raw != 0))
    raw = raw[first_nonzero:]

    # cumsum of uint32 IATs → absolute arrival ticks
    arrival_ticks = np.cumsum(raw.astype(np.int64))

    # convert ticks → seconds
    return arrival_ticks / clock_rate_hz


def bin_arrivals(
    arrivals:   np.ndarray,
    bin_ms:     float,
    t_end_s:    Optional[float] = None,
) -> np.ndarray:
    """
    Bin photon arrival timestamps into a photon-count histogram.

    Translated from the binning step of read_PAT.m:
        patA_c = ceil(patA_c / opt.bin)   [1-indexed bin assignment]
        imA(patA_c(j)) += 1

    Parameters
    ----------
    arrivals : np.ndarray — photon arrival times in seconds
    bin_ms   : float      — bin width in milliseconds
    t_end_s  : float, optional — total trace duration in seconds.
        If None, inferred from the last arrival time.

    Returns
    -------
    trace : np.ndarray, uint32, shape (n_bins,)
        Photon count per bin.  Bin k covers [k*bin_s, (k+1)*bin_s).
    """
    bin_s = bin_ms * 1e-3

    # Assign each photon to a bin (0-indexed).
    # Faithful to read_PAT.m: patA_c = ceil(patA_c / opt.bin)  [1-indexed]
    # Convert to 0-indexed by subtracting 1.
    bin_indices = np.ceil(arrivals / bin_s).astype(np.int64) - 1

    # Guard against -1 from photons at exactly t=0 (floating-point edge case).
    bin_indices = np.maximum(bin_indices, 0)

    if t_end_s is not None:
        n_bins = max(int(np.ceil(t_end_s / bin_s)), int(bin_indices[-1]) + 1)
    else:
        n_bins = int(bin_indices[-1]) + 1

    trace = np.bincount(bin_indices, minlength=n_bins).astype(np.uint32)
    return trace


def read_raw(
    filepath,
    bits:          int   = DEFAULT_BITS,
    n_header:      int   = DEFAULT_N_HEADER,
    clock_rate_hz: float = DEFAULT_CLOCK_HZ,
    bin_ms:        float = 0.1,
    t_end_s:       Optional[float] = None,
) -> np.ndarray:
    """
    Read a .raw file and return a binned photon-count trace.

    This is the one-shot convenience wrapper combining read_arrivals() and
    bin_arrivals().

    Parameters
    ----------
    filepath      : str or Path — path to the .raw file
    bits          : 8 or 16 — binary dtype
    n_header      : number of values to discard at the start
    clock_rate_hz : hardware clock in Hz (ticks → seconds conversion)
    bin_ms        : bin width in milliseconds for the output trace
    t_end_s       : optional total duration (seconds); inferred if None

    Returns
    -------
    trace : np.ndarray, uint32 — photon count per bin
    """
    arrivals = read_arrivals(filepath, bits=bits, n_header=n_header,
                             clock_rate_hz=clock_rate_hz)
    return bin_arrivals(arrivals, bin_ms=bin_ms, t_end_s=t_end_s)


def read_raw_pair(
    prefix:        str,
    bits:          int   = DEFAULT_BITS,
    n_header:      int   = DEFAULT_N_HEADER,
    clock_rate_hz: float = DEFAULT_CLOCK_HZ,
    bin_ms:        float = 0.1,
    t_end_s:       Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a two-channel acquisition from prefixA.raw and prefixB.raw.

    Translated from the n_trc==2 branch of read_PAT.m.
    Both channels are binned to the same time axis (length = max of both).

    Parameters
    ----------
    prefix        : str — common filename prefix (without 'A.raw' / 'B.raw')
    (other params same as read_raw)

    Returns
    -------
    trace_a, trace_b : two np.ndarray uint32, padded to the same length
    """
    arr_a = read_arrivals(f'{prefix}A.raw', bits=bits, n_header=n_header,
                          clock_rate_hz=clock_rate_hz)
    arr_b = read_arrivals(f'{prefix}B.raw', bits=bits, n_header=n_header,
                          clock_rate_hz=clock_rate_hz)

    # joint time axis
    if t_end_s is None:
        t_end_s = max(arr_a[-1], arr_b[-1])

    trace_a = bin_arrivals(arr_a, bin_ms=bin_ms, t_end_s=t_end_s)
    trace_b = bin_arrivals(arr_b, bin_ms=bin_ms, t_end_s=t_end_s)

    # pad to same length
    n = max(len(trace_a), len(trace_b))
    if len(trace_a) < n:
        trace_a = np.pad(trace_a, (0, n - len(trace_a)))
    if len(trace_b) < n:
        trace_b = np.pad(trace_b, (0, n - len(trace_b)))

    return trace_a, trace_b


# ── Filename parsing helpers ──────────────────────────────────────────────────

def parse_raw_filename(filename: str) -> Optional[dict]:
    """
    Parse a RAW filename of the form:
        {name}_{hash}_R{r}_P{p}_K{k}_Ch{channel_id}.raw

    Returns a dict with keys: name, hash, r, p, k, channel_id
    or None if the filename does not match.

    Example:
        parse_raw_filename('lateralDorsal_nt_1_6e127ff3_R2_P1_K1_ChS2.raw')
        → {'name': 'lateralDorsal_nt_1', 'hash': '6e127ff3',
           'r': 2, 'p': 1, 'k': 1, 'channel_id': 'S2'}
    """
    stem = Path(filename).stem  # strip .raw
    # pattern: anything_R{int}_P{int}_K{int}_Ch{label}
    m = re.match(
        r'^(.+?)_([0-9a-fA-F]{8,64})_R(\d+)_P(\d+)_K(\d+)_Ch(.+)$',
        stem
    )
    if not m:
        return None
    return {
        'name':       m.group(1),
        'hash':       m.group(2),
        'r':          int(m.group(3)),
        'p':          int(m.group(4)),
        'k':          int(m.group(5)),
        'channel_id': m.group(6),
    }


import re  # needed by parse_raw_filename
