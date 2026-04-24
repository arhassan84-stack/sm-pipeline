#!/usr/bin/env python3
"""
make_algorithm_flowchart.py
Two-column landscape layout:
  Left  — Phase 1: Peak Detection
  Right — Phase 2: D-guided Expansion
White background, black arrows.
"""

import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = 'results_01082026/figures/algorithm_flowchart.png'
pathlib.Path('results_01082026/figures').mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = '#FFFFFF'
BLACK = '#111111'
MGRAY = '#555555'

C_IO    = '#1565C0'   # blue
C_PRE   = '#2E7D32'   # green
C_PEAK  = '#BF360C'   # deep orange
C_DEC   = '#880E4F'   # deep pink / magenta
C_EXP   = '#4A148C'   # deep purple
C_MODEL = '#004D40'   # dark teal
C_OUT   = '#0D47A1'   # indigo

FS_MAIN  = 10.0
FS_SUB   = 8.0
FS_PHASE = 11.0
FS_TITLE = 15.0
FS_LEG   = 9.0

# ── Figure (landscape 20×11) ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 11), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.axis('off')

# ── Geometry ──────────────────────────────────────────────────────────────────
CX_L = 5.8        # left-column centre
CX_R = 15.5       # right-column centre
W    = 5.8        # main box width
Ws   = 2.7        # side box (Cat1 / Cat2)
H    = 0.55       # box height
DH   = 0.62       # diamond half-height
G    = 1.10       # gap between box centres

# ── Helpers ───────────────────────────────────────────────────────────────────
ARROW_KW = dict(arrowstyle='->', color=BLACK, lw=1.5)

def rbox(cx, cy, w, h, label, sublabel='', color=None):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=color, edgecolor='#CCCCCC', linewidth=1.2, zorder=3))
    yo = cy + (0.11 if sublabel else 0)
    ax.text(cx, yo, label, ha='center', va='center', color='#FFFFFF',
            fontsize=FS_MAIN, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.15, sublabel, ha='center', va='center',
                color='#F0F0F0', fontsize=FS_SUB, style='italic', zorder=4)

def diamond(cx, cy, w, h, label, color):
    xs = [cx, cx+w/2, cx, cx-w/2, cx]
    ys = [cy+h/2, cy, cy-h/2, cy, cy+h/2]
    ax.fill(xs, ys, color=color, zorder=3)
    ax.plot(xs, ys, color='#CCCCCC', lw=1.2, zorder=4)
    ax.text(cx, cy, label, ha='center', va='center', color='#FFFFFF',
            fontsize=FS_MAIN, fontweight='bold', zorder=5)

def varrow(x, y0, y1, label='', side='right'):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(**ARROW_KW))
    if label:
        lx = x + (0.18 if side == 'right' else -0.18)
        ax.text(lx, (y0+y1)/2, label, ha='left' if side == 'right' else 'right',
                va='center', color=MGRAY, fontsize=FS_SUB, style='italic')

def harrow(x0, y0, x1, y1, label='', va_pos='top'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(**ARROW_KW))
    if label:
        my = max(y0, y1) + (0.12 if va_pos == 'top' else 0)
        ax.text((x0+x1)/2, my, label, ha='center', va='bottom',
                color=MGRAY, fontsize=FS_SUB, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(11.0, 11.60, 'Event-Detection Algorithm  —  Stage 3 v2',
        ha='center', va='center', color=BLACK,
        fontsize=FS_TITLE, fontweight='bold')

# Dividing line between phases
ax.axvline(10.0, ymin=0.04, ymax=0.93,
           color='#CCCCCC', lw=1.2, ls='--', zorder=1)

# Phase labels
ax.text(CX_L, 11.05, 'Phase 1  —  Peak Detection',
        ha='center', va='center', color=C_PEAK,
        fontsize=FS_PHASE, fontweight='bold')
ax.text(CX_R + 0.7, 11.05, 'Phase 2  —  D-guided Expansion',
        ha='center', va='center', color=C_EXP,
        fontsize=FS_PHASE, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN  (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
y = 10.40

def nxt(cur, gap=G): return cur - gap

y0 = y
rbox(CX_L, y0, W, H, 'Raw Trace  (multi-resolution)',
     'i_1ms  |  i_0.5ms  |  i_0.1ms', C_IO)

y1 = nxt(y0)
rbox(CX_L, y1, W, H, 'Baseline Estimation',
     'Noise model  →  per-bin background (kHz)', C_PRE)
varrow(CX_L, y0-H/2, y1+H/2)

y2 = nxt(y1)
rbox(CX_L, y2, W, H, 'Baseline Subtraction  (all resolutions)',
     'max(0, i − baseline)  |  0.5ms & 0.1ms upsampled via np.repeat', C_PRE)
varrow(CX_L, y1-H/2, y2+H/2)

y3 = nxt(y2)
rbox(CX_L, y3, W, H, 'Box-smooth i_1ms  (50 ms window)',
     'Used only for half-width measurement', C_PEAK)
varrow(CX_L, y2-H/2, y3+H/2)

y4 = nxt(y3)
rbox(CX_L, y4, W, H, 'Peak Finding  (scipy.signal.find_peaks)',
     'Min distance 50 ms  |  amplitude ≥ 50 % of global max', C_PEAK)
varrow(CX_L, y3-H/2, y4+H/2)

y5 = nxt(y4)
rbox(CX_L, y5, W, H, 'Measure half-width  w½  at 60 % of peak',
     'Measured on smoothed trace', C_PEAK)
varrow(CX_L, y4-H/2, y5+H/2)

# Category diamond
y6 = nxt(y5, G + 0.05)
diamond(CX_L, y6, W - 0.3, DH, '2·w½  ≥  200 ms ?', C_DEC)
varrow(CX_L, y5-H/2, y6+DH)

# Cat 1 / Cat 2
y7 = y6 - DH - 0.58
XL2 = CX_L - Ws/2 - 0.15
XR2 = CX_L + Ws/2 + 0.15
rbox(XL2, y7, Ws, H, 'Cat 1  (broad)',
     'seed = [tp−w½, tp+w½]', C_PEAK)
rbox(XR2, y7, Ws, H, 'Cat 2  (narrow)',
     'seed = 64 ms each side', C_PEAK)
harrow(CX_L-0.2, y6, XL2+Ws/2-0.01, y7, 'Yes')
harrow(CX_L+0.2, y6, XR2-Ws/2+0.01, y7, 'No')

# Claiming loop
y8 = nxt(y7, G - 0.05)
rbox(CX_L, y8, W, H, 'Claiming Loop  (descending peak height)',
     'Each peak claims its seed  |  overlapping peaks excluded', C_PEAK)
varrow(XL2, y7-H/2, y8+H/2)
varrow(XR2, y7-H/2, y8+H/2)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN  (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
y9 = 10.40   # align top with left column
rbox(CX_R, y9, W, H, 'Initial D Prediction  (WaveNet model)',
     'Fixed 4 096 ms window centred on peak  |  multi-dt branches', C_MODEL)

# D > 1 diamond
y10 = nxt(y9, G + 0.05)
diamond(CX_R, y10, W - 0.3, DH, 'D̂  >  1  µm²/s ?', C_DEC)
varrow(CX_R, y9-H/2, y10+DH)

# Fast group
X_FG = CX_R + W/2 + 1.80
Y_FG = y10
rbox(X_FG, Y_FG, 2.5, H, 'Fast group',
     'Keep seed, no expansion', C_EXP)
harrow(CX_R + (W-0.3)/2, y10, X_FG-1.25+0.01, Y_FG, 'Yes')

# Tau lookup
y11 = y10 - DH - 0.70
rbox(CX_R, y11, W, H,
     'Look up  τmax  from  D̂',
     'D ∈ [0.02..1.00] µm²/s  →  τ ∈ [2048..128] ms', C_EXP)
varrow(CX_R, y10-DH, y11+H/2, 'No')

# Double window
y12 = nxt(y11)
rbox(CX_R, y12, W, H, 'Double window:  w  →  2w',
     'Stop if: 2w ≥ 4096 ms  |  overlap claimed  |  2w ≥ τmax', C_EXP)
varrow(CX_R, y11-H/2, y12+H/2)

# Re-predict
y13 = nxt(y12)
rbox(CX_R, y13, W, H, 'Re-predict D  (WaveNet)',
     'Update τmax  |  repeat until stop condition', C_MODEL)
varrow(CX_R, y12-H/2, y13+H/2)

# Loop-back arrow (right side of right column)
lx = CX_R + W/2 + 0.25
ax.annotate('', xy=(lx, y11+H/2), xytext=(lx, y13-H/2),
            arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.8))
ax.text(lx + 0.08, (y11+y13)/2, 'expand\nagain?',
        ha='left', va='center', color='#1B5E20',
        fontsize=FS_SUB, style='italic', fontweight='bold')

# Event output
y14 = nxt(y13)
rbox(CX_R, y14, W, H, 'Event Output',
     't_peak  |  t_left  |  t_right  |  D̂  |  log₁₀D  |  category', C_OUT)
varrow(CX_R, y13-H/2, y14+H/2)
# Fast group merges into event output
ax.annotate('', xy=(CX_R+W/2+0.01, y14), xytext=(X_FG, Y_FG-H/2),
            arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.5))

# CSV
y15 = nxt(y14, G*0.75)
rbox(CX_R, y15, W*0.70, H, '{stem}_{ch}_events.csv', '', C_IO)
varrow(CX_R, y14-H/2, y15+H/2)

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTOR: Phase 1 → Phase 2
# Z-shaped arrow: claiming-loop → right → up → initial-D-pred
# ─────────────────────────────────────────────────────────────────────────────
CX_mid = 10.0
# Horizontal leg 1: claiming loop right edge → mid
ax.annotate('', xy=(CX_mid, y8), xytext=(CX_L+W/2, y8),
            arrowprops=dict(arrowstyle='-', color=BLACK, lw=1.8))
# Vertical leg: mid bottom → mid top
ax.annotate('', xy=(CX_mid, y9+H/2+0.02), xytext=(CX_mid, y8),
            arrowprops=dict(arrowstyle='-', color=BLACK, lw=1.8))
# Horizontal leg 2: mid → right column left edge (with arrow head)
ax.annotate('', xy=(CX_R-W/2, y9+H/2*0.0), xytext=(CX_mid, y9+H/2*0.0),
            arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.8))

# Label on connector
ax.text(CX_mid + 0.15, (y8 + y9)/2 + 0.3,
        'per-event\nprocessing',
        ha='left', va='center', color=BLACK,
        fontsize=FS_SUB, style='italic', fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND  (upper-left, above the left column, vertical)
# ─────────────────────────────────────────────────────────────────────────────
leg_items = [
    ('Input / Output',    C_IO),
    ('Pre-processing',    C_PRE),
    ('Peak detection',    C_PEAK),
    ('Decision',          C_DEC),
    ('Window expansion',  C_EXP),
    ('Neural model',      C_MODEL),
    ('Output',            C_OUT),
]
lx0 = 0.30; ly_top = 10.82; ldy = 0.58
ax.text(lx0, ly_top + 0.30, 'Legend',
        color=BLACK, fontsize=FS_LEG+0.5, fontweight='bold')
for i, (lbl, col) in enumerate(leg_items):
    yleg = ly_top - i * ldy
    ax.add_patch(FancyBboxPatch(
        (lx0, yleg - 0.18), 0.60, 0.40,
        boxstyle='round,pad=0.05',
        facecolor=col, edgecolor='#CCCCCC', lw=1.0, zorder=3))
    ax.text(lx0 + 0.78, yleg + 0.02, lbl,
            color=BLACK, fontsize=FS_SUB+0.5, va='center')

fig.savefig(OUT, dpi=180, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT}')
