#!/usr/bin/env python3
"""
aggregate_events.py
-------------------
Reads all *_events.csv files from processed_on_* folders in experiment
directories, enriches every event row with structured metadata parsed from
the experiment name and measurement (FCS) name, and writes one flat CSV.

No third-party dependencies — uses stdlib only (csv, json, pathlib, re).

Usage
-----
    python aggregate_events.py \\
        --experiments-dir /path/to/experiments \\
        [--version v4.6]      # default: latest completed version per experiment
        [--output aggregated_events.csv]

If --version is omitted the latest successfully-processed version is used for
each experiment independently.  If a requested version is not available for a
given experiment a warning is printed and that experiment is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ══  CONFIGURATION — edit this section for your project  ══════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── Fluorescent protein aliases ───────────────────────────────────────────────
# Map every lowercase alias that might appear in a filename to a canonical name.
# Longer aliases are checked before shorter ones to prevent partial-match errors.
FP_ALIASES: dict[str, str] = {
    # mCherry2 (must come before mcherry)
    'mcherry2':  'mCherry2',
    'ch2':       'mCherry2',
    # mCherry
    'mcherry':   'mCherry',
    'cherry':    'mCherry',
    # GFP / emerald GFP
    'emgfp':     'emGFP',
    'egfp':      'GFP',
    'gfp':       'GFP',
    # mRFP
    'mrfp':      'mRFP',
    'rfp':       'mRFP',
    # mScarletI  (must come before shorter sub-matches)
    'mscarletI': 'mScarletI',
    'mscarleti': 'mScarletI',
    'mscarlet':  'mScarletI',
    'mscarli':   'mScarletI',
    'scarli':    'mScarletI',
    # NeonGreen — 'ng' is short; safe here because 'ng' only appears as a
    # standalone suffix (e.g. itgb1bNG) not inside background tokens which
    # are parsed separately.
    'neongreen': 'NeonGreen',
    'ng':        'NeonGreen',
    # Citrine / mGold
    'citrine':   'Citrine',
    'mgold':     'mGold',
}

# ── Fluorescent protein → detector channel ────────────────────────────────────
# S1 = green (short-wavelength) channel; S2 = red (long-wavelength) channel.
FP_CHANNEL: dict[str, str] = {
    'GFP':       'S1',
    'emGFP':     'S1',
    'NeonGreen': 'S1',
    'Citrine':   'S1',
    'mGold':     'S1',
    'mCherry':   'S2',
    'mCherry2':  'S2',
    'mRFP':      'S2',
    'mScarletI': 'S2',
}

# ── Molecule display names (for figures / publications) ───────────────────────
# Key = abbreviated molecule name as it appears in the experiment folder name.
# Add entries as needed; unknown molecules fall back to the raw abbreviation.
MOLECULE_DISPLAY: dict[str, str] = {
    'itgb1a':       'Integrin β1a',
    'itgb1b':       'Integrin β1b',
    'itga5':        'Integrin α5',
    'itga5FYLDD':   'Integrin α5-FYLDD',
    'a5FYLDD':      'Integrin α5-FYLDD',
    'FYLDDGFP':     'Integrin α5-FYLDD',   # older naming style
    'itga5_296i':   'Integrin α5-296i',
    'fbn2b':        'Fibrillin-2b',
    'itgaVb3b':     'Integrin αVβ3b',
    # add more as needed
}

# ── Background / genetic context display names ────────────────────────────────
BACKGROUND_DISPLAY: dict[str, str] = {
    'fn1aNG':   'fn1a-NeonGreen',
    'fn1b':     'fn1b',
    'TLF':      'Wild-type (TLF)',
    'MZHLO30':  'MZH&L (O30)',
    # add more as needed
}

# ─────────────────────────────────────────────────────────────────────────────
# ══  MEASUREMENT NAME PARSING RULES  ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# Tissues: checked in priority order (first match wins).
TISSUE_PATTERNS: list[tuple[str, str]] = [
    ('psm',    'PSM'),
    ('somite', 'somite'),
    ('nt',     'NT'),
]

# Directions: checked longest-first to avoid partial matches.
DIRECTION_PATTERNS: list[tuple[str, str]] = [
    ('interiorlateral',  'interior_lateral'),
    ('lateraldorsal',    'lateral_dorsal'),
    ('dorsallateral',    'dorsal_lateral'),
    ('dorsalmedial',     'dorsal_medial'),
    ('verydorsal',       'very_dorsal'),
    ('interior',         'interior'),
    ('lateral',          'lateral'),
    ('ventral',          'ventral'),
    ('dorsal',           'dorsal'),
    ('medial',           'medial'),
]

# Cell types
CELLTYPE_PATTERNS: list[tuple[str, str]] = [
    ('boundary',   'boundary'),
    ('mesenchyme', 'mesenchyme'),
]

# Fallback rules applied when a field could not be detected.
# Key = detected tissue (or None); value = defaults for missing fields.
FALLBACK_RULES: dict = {
    'somite': {'direction': 'dorsal', 'cell_type': 'boundary'},
    'NT':     {'direction': 'dorsal', 'cell_type': None},
    'PSM':    {'direction': 'dorsal', 'cell_type': None},
    None:     {'direction': None,     'cell_type': None},
}

# Measurement stems to skip entirely (case-insensitive prefix match).
SKIP_PREFIXES: tuple[str, ...] = ('solu', 'cell_')

# ─────────────────────────────────────────────────────────────────────────────
# ══  EXPERIMENT NAME TOKENS  ══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

POSITION_TOKENS: set[str] = {'posterior', 'anterior'}
MISC_TOKENS:     set[str] = {
    'transp', 'transplant',
    'red', 'redonly',
    'yellow', 'yellowonly',
    '2ex', '1ex', '3ex',
    'fret',
    'rotenone',
}

# ── Whole-token compound overrides ────────────────────────────────────────────
# Some tokens contain both a targeting prefix and an FP fused without a
# separator, where the FP name starts with 'm' (e.g. memCherry2 → mem + mCherry2).
# The suffix algorithm can't recover the 'm' that's shared, so declare these
# explicitly.  Key = lowercase whole token; value = (mol_part, fp_canonical).
COMPOUND_TOKENS: dict[str, tuple[str, str]] = {
    'memcherry2':   ('mem', 'mCherry2'),
    'memcherry':    ('mem', 'mCherry'),
    'memrfp':       ('mem', 'mRFP'),
    'memgfp':       ('mem', 'GFP'),
    'memng':        ('mem', 'NeonGreen'),
    'memgold':      ('mem', 'mGold'),
    'memscarli':    ('mem', 'mScarletI'),
    # add more compound fusions as needed
}

# ─────────────────────────────────────────────────────────────────────────────
# ══  PARSING HELPERS  ═════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# Pre-sort aliases by descending length so longest match is always tried first.
_FP_ALIASES_SORTED: list[tuple[str, str]] = sorted(
    FP_ALIASES.items(), key=lambda kv: -len(kv[0])
)


def _detect_fp(token: str):
    """
    Return (molecule_part, fp_canonical) if *token* is or ends with a known FP.
    Return (None, None) otherwise.
    Handles: 'mCherry2' → ('', 'mCherry2'), 'itgb1bGFP' → ('itgb1b', 'GFP'),
             'memCherry2' → ('mem', 'mCherry2')  [via COMPOUND_TOKENS override].
    """
    tl = token.lower()
    # Check compound overrides first (handles mem+FP where 'm' is shared)
    if tl in COMPOUND_TOKENS:
        return COMPOUND_TOKENS[tl]
    # Exact match
    for alias, canonical in _FP_ALIASES_SORTED:
        if tl == alias:
            return ('', canonical)
        if tl.endswith(alias) and len(tl) > len(alias):
            return (token[: len(token) - len(alias)], canonical)
    return (None, None)


def _parse_date(token: str):
    """Convert DDMMYYYY → ISO date string YYYY-MM-DD, or return None."""
    m = re.fullmatch(r'(\d{2})(\d{2})(\d{4})', token)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    try:
        return date(int(yyyy), int(mm), int(dd)).isoformat()
    except ValueError:
        return token  # malformed — keep raw


def _parse_molecules(mol_tokens: list[str]) -> list[dict]:
    """
    Parse tokens between the date and 'in_' into molecule dicts.
    Each dict: {molecule, fp, molecule_display, channel}.
    Strategy: use FP aliases as anchors; tokens before each FP form the molecule name.
    """
    # Expand tokens that have an FP as a suffix (e.g. 'itgb1bGFP')
    expanded = []   # list of (text, fp_canonical | None)
    for tok in mol_tokens:
        mol_part, fp_can = _detect_fp(tok)
        if fp_can is not None:
            if mol_part:
                expanded.append((mol_part, None))
            expanded.append(('', fp_can))
        else:
            expanded.append((tok, None))

    # Group mol-part tokens before each FP marker
    molecules = []
    current_parts: list[str] = []
    for text, fp_can in expanded:
        if fp_can is None:
            if text:
                current_parts.append(text)
        else:
            mol_abbrev = '_'.join(current_parts) if current_parts else 'unknown'
            molecules.append({
                'molecule':         mol_abbrev,
                'fp':               fp_can,
                'molecule_display': MOLECULE_DISPLAY.get(mol_abbrev, mol_abbrev),
                'channel':          FP_CHANNEL.get(fp_can, 'unknown'),
            })
            current_parts = []

    # Leftover tokens with no trailing FP (single molecule, FP undetected)
    if current_parts and not molecules:
        mol_abbrev = '_'.join(current_parts)
        molecules.append({
            'molecule':         mol_abbrev,
            'fp':               None,
            'molecule_display': MOLECULE_DISPLAY.get(mol_abbrev, mol_abbrev),
            'channel':          'unknown',
        })

    return molecules


def parse_experiment_name(exp_name: str) -> dict:
    """
    Parse a folder name such as:
        01082026_itga5_296i_mCherry2_itgb1a_GFP_in_TLF
        07222025_itgb1a_mCherry2_in_fn1aNG_posterior
        08282025_itgb1bGFP_transp_in_TLF
        07182025_fbn2b_mScarlI
    Returns dict with: date, background, background_display, position, misc, molecules.
    """
    tokens = exp_name.split('_')
    if not tokens:
        return {}

    date_str = _parse_date(tokens[0])

    # Locate the 'in' separator token
    in_idx = None
    for i, t in enumerate(tokens[1:], start=1):
        if t.lower() == 'in':
            in_idx = i
            break

    post_in = tokens[in_idx + 1:] if in_idx is not None else []

    # Background = post-'in' tokens that are not position/misc keywords
    bg_tokens = [t for t in post_in
                 if t.lower() not in POSITION_TOKENS and t.lower() not in MISC_TOKENS]
    background = '_'.join(bg_tokens) if bg_tokens else None

    # Position
    all_lower = [t.lower() for t in tokens[1:]]
    position = 'posterior' if 'posterior' in all_lower else 'anterior'

    # Misc
    misc_found = [t for t in tokens[1:] if t.lower() in MISC_TOKENS]
    misc = '_'.join(misc_found) if misc_found else None

    # Molecule section = tokens between date and 'in' (or end), minus misc/position
    mol_end = in_idx if in_idx is not None else len(tokens)
    mol_tokens_raw = tokens[1:mol_end]
    mol_tokens = [t for t in mol_tokens_raw
                  if t.lower() not in POSITION_TOKENS and t.lower() not in MISC_TOKENS]

    molecules = _parse_molecules(mol_tokens)

    return {
        'date':               date_str,
        'background':         background,
        'background_display': BACKGROUND_DISPLAY.get(background, background),
        'position':           position,
        'misc':               misc,
        'molecules':          molecules,
    }


def parse_measurement_name(stem: str) -> dict:
    """
    Parse a measurement stem (FCS filename without channel suffix and extension).
    Uses case-insensitive keyword matching on the whole normalised string.
    Returns dict with: tissue, direction, cell_type,
                       direction_fallback_applied, cell_type_fallback_applied.
    """
    # Normalise: lowercase, collapse underscores
    norm = stem.lower().replace('_', '')

    tissue = None
    for pattern, canonical in TISSUE_PATTERNS:
        if pattern in norm:
            tissue = canonical
            break

    direction = None
    for pattern, canonical in DIRECTION_PATTERNS:
        if pattern.replace('_', '') in norm:
            direction = canonical
            break

    cell_type = None
    for pattern, canonical in CELLTYPE_PATTERNS:
        if pattern in norm:
            cell_type = canonical
            break

    fallback = FALLBACK_RULES.get(tissue, FALLBACK_RULES[None])
    dir_fallback = direction is None
    ct_fallback  = cell_type is None
    if dir_fallback:
        direction = fallback['direction']
    if ct_fallback:
        cell_type = fallback['cell_type']

    return {
        'tissue':                     tissue,
        'direction':                  direction,
        'cell_type':                  cell_type,
        'direction_fallback_applied': dir_fallback,
        'cell_type_fallback_applied': ct_fallback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ══  VERSION / PROCESSED FOLDER SELECTION  ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

_VERSION_RE = re.compile(r'v(\d+)\.(\d+)')


def _version_tuple(ver_str: str) -> tuple:
    m = _VERSION_RE.search(ver_str or '')
    if not m:
        return (0, 0)
    return tuple(int(x) for x in m.groups())


def _read_log(folder: Path):
    log_path = folder / 'pipeline_log.json'
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            return json.load(f)
    except Exception:
        return None


def _has_events(folder: Path) -> bool:
    return any(folder.glob('*_events.csv'))


def find_processed_folder(exp_dir: Path, target_version):
    """
    Return (folder, pipeline_version, status_msg).
    status_msg: 'ok' | 'not_found' | 'not_completed' | 'no_events'
    """
    candidates = sorted(exp_dir.glob('processed_on_*'))
    if not candidates:
        return None, '', 'not_found'

    annotated = []
    for folder in candidates:
        log = _read_log(folder)
        if log:
            ver  = log.get('pipeline_version', 'unknown')
            done = log.get('status', '') == 'completed'
        else:
            ver  = 'unknown'
            done = _has_events(folder)
        annotated.append((folder, ver, done))

    if target_version is None:
        completed = [(p, v, d) for p, v, d in annotated if d]
        if not completed:
            return None, '', 'not_completed'
        best = max(completed, key=lambda x: (_version_tuple(x[1]), x[0].name))
        folder, ver, _ = best
    else:
        matches = [(p, v, d) for p, v, d in annotated if v == target_version]
        if not matches:
            return None, '', 'not_found'
        completed = [(p, v, d) for p, v, d in matches if d]
        pool = completed if completed else matches
        folder, ver, done = max(pool, key=lambda x: x[0].name)
        if not done:
            return folder, ver, 'not_completed'

    if not _has_events(folder):
        return folder, ver, 'no_events'

    return folder, ver, 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# ══  MAIN AGGREGATION  ════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(experiments_dir: Path, target_version, output: Path) -> None:
    exp_dirs = sorted(
        d for d in experiments_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith('.')
        and not d.name.startswith('processed_on_')
    )

    writer      = None
    out_f       = None
    n_rows      = 0
    n_exp_ok    = 0
    n_skip_exp  = 0
    n_skip_meas = 0

    META_COLS = [
        'experiment', 'measurement', 'channel', 'pipeline_version', 'processed_folder',
        'date', 'background', 'background_display', 'position', 'misc',
        'molecule', 'fp', 'molecule_display',
        'tissue', 'direction', 'cell_type', 'direction_fallback', 'cell_type_fallback',
    ]

    try:
        out_f = open(output, 'w', newline='', encoding='utf-8')

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name

            folder, ver, status = find_processed_folder(exp_dir, target_version)

            if status == 'not_found':
                msg = f'no processed folder' + (f' for version {target_version}' if target_version else '')
                print(f'SKIP  [{exp_name}] {msg}', file=sys.stderr)
                n_skip_exp += 1
                continue
            if status == 'not_completed':
                print(f'SKIP  [{exp_name}] version {ver} did not complete'
                      + (f' ({folder.name})' if folder else ''), file=sys.stderr)
                n_skip_exp += 1
                continue
            if status == 'no_events':
                print(f'SKIP  [{exp_name}] {folder.name} has no *_events.csv', file=sys.stderr)
                n_skip_exp += 1
                continue

            exp_meta = parse_experiment_name(exp_name)
            ch_to_mol: dict[str, dict] = {m['channel']: m for m in exp_meta['molecules']}

            print(f'  {exp_name}  [{ver}]  {folder.name}')

            csv_files = sorted(folder.glob('*_events.csv'))
            for csv_path in csv_files:
                stem = re.sub(r'_events$', '', csv_path.stem)

                ch_match = re.search(r'_(S[12])$', stem)
                if ch_match:
                    channel   = ch_match.group(1)
                    meas_stem = stem[: ch_match.start()]
                else:
                    channel   = 'unknown'
                    meas_stem = stem

                if any(meas_stem.lower().startswith(p) for p in SKIP_PREFIXES):
                    n_skip_meas += 1
                    continue

                meas_meta = parse_measurement_name(meas_stem)
                mol_info  = ch_to_mol.get(channel, {})

                try:
                    with open(csv_path, newline='', encoding='utf-8') as cf:
                        reader = csv.DictReader(cf)
                        if reader.fieldnames is None:
                            continue
                        data_cols = list(reader.fieldnames)

                        # Initialise writer on first file so we know all columns
                        if writer is None:
                            all_cols = META_COLS + data_cols
                            writer = csv.DictWriter(out_f, fieldnames=all_cols,
                                                    extrasaction='ignore')
                            writer.writeheader()

                        meta_values = {
                            'experiment':          exp_name,
                            'measurement':         meas_stem,
                            'channel':             channel,
                            'pipeline_version':    ver,
                            'processed_folder':    folder.name,
                            'date':                exp_meta.get('date', ''),
                            'background':          exp_meta.get('background', ''),
                            'background_display':  exp_meta.get('background_display', ''),
                            'position':            exp_meta.get('position', ''),
                            'misc':                exp_meta.get('misc', ''),
                            'molecule':            mol_info.get('molecule', ''),
                            'fp':                  mol_info.get('fp', ''),
                            'molecule_display':    mol_info.get('molecule_display', ''),
                            'tissue':              meas_meta['tissue'] or '',
                            'direction':           meas_meta['direction'] or '',
                            'cell_type':           meas_meta['cell_type'] or '',
                            'direction_fallback':  meas_meta['direction_fallback_applied'],
                            'cell_type_fallback':  meas_meta['cell_type_fallback_applied'],
                        }

                        for row in reader:
                            out_row = dict(meta_values)
                            out_row.update(row)
                            writer.writerow(out_row)
                            n_rows += 1

                except Exception as e:
                    print(f'  WARN  cannot read {csv_path.name}: {e}', file=sys.stderr)
                    continue

            n_exp_ok += 1

    finally:
        if out_f:
            out_f.close()

    if n_rows == 0:
        print('No data collected — nothing written.', file=sys.stderr)
        return

    print(f'\nDone: {n_rows} events from {n_exp_ok} experiments → {output}')
    if n_skip_exp:
        print(f'  ({n_skip_exp} experiment(s) skipped — see warnings above)')
    if n_skip_meas:
        print(f'  ({n_skip_meas} measurement file(s) skipped by prefix filter)')


# ─────────────────────────────────────────────────────────────────────────────
# ══  CLI  ═════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate pipeline events CSVs into one flat table.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--experiments-dir', '-e',
        default='experiments',
        help='Path to the experiments directory (default: ./experiments)',
    )
    parser.add_argument(
        '--version', '-v',
        default=None,
        help='Pipeline version to use, e.g. v4.6  (default: latest per experiment)',
    )
    parser.add_argument(
        '--output', '-o',
        default='aggregated_events.csv',
        help='Output CSV path (default: aggregated_events.csv)',
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir).expanduser().resolve()
    if not exp_dir.is_dir():
        sys.exit(f'ERROR: experiments dir not found: {exp_dir}')

    output = Path(args.output).expanduser().resolve()

    print(f'Experiments dir : {exp_dir}')
    print(f'Version filter  : {args.version or "latest (per experiment)"}')
    print(f'Output          : {output}')
    print()

    aggregate(exp_dir, args.version, output)


if __name__ == '__main__':
    main()
