#!/usr/bin/env python3
"""
make_architecture_diagram.py — white-background version, wider & taller spacing.
"""

import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = 'results_01082026/figures/architecture_multidt_B.png'
pathlib.Path('results_01082026/figures').mkdir(parents=True, exist_ok=True)

BG    = '#FFFFFF'
BLACK = '#111111'
MGRAY = '#555555'
LGRAY = '#888888'

C_IN   = '#1565C0'
C_POOL = '#00838F'
C_PROJ = '#00695C'
C_RES  = '#6A1B9A'
C_GAP  = '#2E7D32'
C_FEAT = '#E65100'
C_HEAD = '#B71C1C'
C_OUT2 = '#1B5E20'

FS_TITLE = 16
FS_BL    = 9.5
FS_SB    = 8.0
FS_LEG   = 8.5

# ── Figure: wide and tall ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 17), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-1.5, 25)
ax.set_ylim(0, 18)
ax.axis('off')

# ── Column centres (wider spacing) ────────────────────────────────────────────
X_B1 = 3.0
X_B2 = 8.5
X_B3 = 14.0
X_FT = 20.5
BW   = 3.4   # box width
BH   = 0.60  # box height

# ── Vertical positions (more spread) ─────────────────────────────────────────
Y_IN   = 15.5
Y_AP   = 13.9
Y_PROJ = 12.3
Y_RES_TOP = 11.0
Y_RES_BOT = 8.2
Y_GAP  = 6.8
Y_CAT  = 5.4
Y_H1   = 4.2
Y_H2   = 3.1
Y_H3   = 2.0


def rbox(cx, cy, w, h, label, sublabel='', color=None, fs=FS_BL, fs_sub=FS_SB, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.10',
        facecolor=color, edgecolor='#DDDDDD', linewidth=1.4, zorder=3))
    wt = 'bold' if bold else 'normal'
    yo = cy + (0.13 if sublabel else 0)
    ax.text(cx, yo, label, ha='center', va='center', color='#FFFFFF',
            fontsize=fs, fontweight=wt, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.20, sublabel, ha='center', va='center',
                color='#EEEEEE', fontsize=fs_sub, style='italic', zorder=4)


def arr(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=LGRAY, lw=1.8))


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(12.0, 17.5, 'WaveNetMultiDT_B  —  Architecture',
        ha='center', va='center', color=BLACK,
        fontsize=FS_TITLE, fontweight='bold')
ax.text(12.0, 16.8,
        '3 independent WaveNet backbones  +  hand-crafted feature branch  →  D̂ prediction',
        ha='center', va='center', color=MGRAY, fontsize=FS_BL+1, style='italic')

# ── Column headers ────────────────────────────────────────────────────────────
for X, lbl in [(X_B1, 'Branch  dt=0.1ms'), (X_B2, 'Branch  dt=0.5ms'),
               (X_B3, 'Branch  dt=1.0ms'), (X_FT, 'Feature branch')]:
    ax.text(X, Y_IN + BH/2 + 0.30, lbl, ha='center', va='bottom',
            color=BLACK, fontsize=FS_BL+0.5, fontweight='bold')

# ── Input boxes ───────────────────────────────────────────────────────────────
rbox(X_B1, Y_IN, BW, BH, 'dt = 0.1 ms  input',  '40 960 bins', C_IN, bold=True)
rbox(X_B2, Y_IN, BW, BH, 'dt = 0.5 ms  input',  '8 192 bins',  C_IN, bold=True)
rbox(X_B3, Y_IN, BW, BH, 'dt = 1.0 ms  input',  '4 096 bins',  C_IN, bold=True)
rbox(X_FT, Y_IN, BW, BH, '906 ACF/PSD features', '302 features / dt × 3', C_FEAT, bold=True)

# ×1000 annotation
for X in [X_B1, X_B2, X_B3]:
    ax.text(X + BW/2 + 0.15, Y_IN, '×1 000\n(kHz→cts/s)',
            ha='left', va='center', color='#FF6F00',
            fontsize=FS_SB, style='italic', fontweight='bold')

# ── AvgPool row ───────────────────────────────────────────────────────────────
for X in [X_B1, X_B2, X_B3, X_FT]:
    arr(X, Y_IN - BH/2, X, Y_AP + BH/2)

rbox(X_B1, Y_AP, BW, BH, 'AvgPool1d  (×16)', '40 960  →  2 560 bins', C_POOL)
rbox(X_B2, Y_AP, BW, BH, '(no pooling)',     'native resolution',      C_POOL)
rbox(X_B3, Y_AP, BW, BH, '(no pooling)',     'native resolution',      C_POOL)
rbox(X_FT, Y_AP, BW, BH, 'Standard Scaler',  'z-score all features',   C_FEAT)

# ── Input projection conv ─────────────────────────────────────────────────────
for X in [X_B1, X_B2, X_B3, X_FT]:
    arr(X, Y_AP - BH/2, X, Y_PROJ + BH/2)

for X in [X_B1, X_B2, X_B3]:
    rbox(X, Y_PROJ, BW, BH, 'Conv1d(1→256, k=16, s=4)', 'BatchNorm · ReLU', C_PROJ)
    ax.text(X, Y_PROJ - BH/2 - 0.20, 'z-score + augment',
            ha='center', va='top', color='#1565C0', fontsize=FS_SB - 0.5, style='italic')

rbox(X_FT, Y_PROJ, BW, BH, 'Linear(906 → 256)', 'BatchNorm · ReLU · Dropout(0.1)', C_FEAT)

# ── 8 × DilatedResBlock (tall stacked box) ────────────────────────────────────
RH = Y_RES_TOP - Y_RES_BOT
for X in [X_B1, X_B2, X_B3]:
    arr(X, Y_PROJ - BH/2, X, Y_RES_TOP + 0.1)
    ax.add_patch(FancyBboxPatch(
        (X - BW/2, Y_RES_BOT), BW, RH,
        boxstyle='round,pad=0.10',
        facecolor=C_RES, edgecolor='#DDDDDD', linewidth=1.4, zorder=3))
    yc = (Y_RES_TOP + Y_RES_BOT) / 2
    ax.text(X, yc + 0.45, '8 × DilatedResBlock',
            ha='center', va='center', color='#FFFFFF',
            fontsize=FS_BL, fontweight='bold', zorder=4)
    ax.text(X, yc + 0.00, 'Conv1d(256→256, k=3)',
            ha='center', va='center', color='#EEEEEE',
            fontsize=FS_SB, zorder=4)
    ax.text(X, yc - 0.38, 'dilation ∈ {1, 2, 4, 8, 16, 32, 64, 128}',
            ha='center', va='center', color='#EEEEEE',
            fontsize=FS_SB, style='italic', zorder=4)
    ax.text(X, yc - 0.75, 'BatchNorm · ReLU · Residual skip',
            ha='center', va='center', color='#EEEEEE',
            fontsize=FS_SB, style='italic', zorder=4)

# Feature branch lower half (Linear 256→128)
arr(X_FT, Y_PROJ - BH/2, X_FT, Y_RES_TOP + 0.1)
ax.add_patch(FancyBboxPatch(
    (X_FT - BW/2, Y_RES_BOT), BW, RH,
    boxstyle='round,pad=0.10',
    facecolor=C_FEAT, edgecolor='#DDDDDD', linewidth=1.4, zorder=3))
yc = (Y_RES_TOP + Y_RES_BOT) / 2
ax.text(X_FT, yc + 0.25, 'Linear(256 → 128)',
        ha='center', va='center', color='#FFFFFF',
        fontsize=FS_BL, fontweight='bold', zorder=4)
ax.text(X_FT, yc - 0.25, 'BatchNorm · ReLU',
        ha='center', va='center', color='#EEEEEE',
        fontsize=FS_SB, zorder=4)

# ── GAP / embedding ───────────────────────────────────────────────────────────
for X in [X_B1, X_B2, X_B3, X_FT]:
    arr(X, Y_RES_BOT, X, Y_GAP + BH/2)

for X in [X_B1, X_B2, X_B3]:
    rbox(X, Y_GAP, BW, BH, 'AdaptiveAvgPool1d(1)', '→  256-dim embedding', C_GAP)
rbox(X_FT, Y_GAP, BW, BH, 'Output', '128-dim embedding', C_GAP)

# ── Concatenate ───────────────────────────────────────────────────────────────
X_CAT = (X_B1 + X_FT) / 2
for X in [X_B1, X_B2, X_B3, X_FT]:
    ax.annotate('', xy=(X_CAT, Y_CAT + BH/2), xytext=(X, Y_GAP - BH/2),
                arrowprops=dict(arrowstyle='->', color=LGRAY, lw=1.6,
                                connectionstyle='arc3,rad=0'))

rbox(X_CAT, Y_CAT, 6.0, BH,
     'Concatenate  [ 256 + 256 + 256 + 128 ]  =  896-dim',
     '', C_HEAD, bold=True, fs=FS_BL+1)

# ── Head ──────────────────────────────────────────────────────────────────────
arr(X_CAT, Y_CAT - BH/2, X_CAT, Y_H1 + BH/2)
rbox(X_CAT, Y_H1, 6.0, BH,
     'Linear(896 → 512)  ·  BatchNorm  ·  ReLU  ·  Dropout(0.1)', '', C_HEAD)

arr(X_CAT, Y_H1 - BH/2, X_CAT, Y_H2 + BH/2)
rbox(X_CAT, Y_H2, 6.0, BH, 'Linear(512 → 128)  ·  ReLU', '', C_HEAD)

arr(X_CAT, Y_H2 - BH/2, X_CAT, Y_H3 + BH/2)
ax.add_patch(FancyBboxPatch(
    (X_CAT - 3.2, Y_H3 - BH/2), 6.4, BH,
    boxstyle='round,pad=0.10',
    facecolor=C_OUT2, edgecolor='#A5D6A7', linewidth=2.0, zorder=3))
ax.text(X_CAT, Y_H3, 'Linear(128 → 1)  →  log D̂  →  exp  →  D̂  (µm²/s)',
        ha='center', va='center', color='#FFFFFF',
        fontsize=FS_BL+1.5, fontweight='bold', zorder=4)

# ── Legend (far left, clear of branches) ─────────────────────────────────────
leg_items = [
    ('Input data',          C_IN),
    ('Pooling / normalise', C_POOL),
    ('Projection conv',     C_PROJ),
    ('Dilated residual',    C_RES),
    ('Global avg pool',     C_GAP),
    ('Feature branch',      C_FEAT),
    ('Regression head',     C_HEAD),
    ('Output',              C_OUT2),
]
lx0 = -1.35; ly0 = 7.0; ldy = 0.90
ax.text(lx0 + 0.20, ly0 + len(leg_items)*ldy + 0.35, 'Legend',
        color=BLACK, fontsize=FS_LEG+1, fontweight='bold')
for i, (lbl, col) in enumerate(leg_items):
    yleg = ly0 + (len(leg_items)-1-i) * ldy
    ax.add_patch(FancyBboxPatch((lx0, yleg - 0.20), 0.60, 0.48,
                                boxstyle='round,pad=0.06',
                                facecolor=col, edgecolor='#CCCCCC', lw=1.2))
    ax.text(lx0 + 0.80, yleg + 0.04, lbl,
            color=BLACK, fontsize=FS_SB + 0.5, va='center')

fig.savefig(OUT, dpi=180, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT}')
