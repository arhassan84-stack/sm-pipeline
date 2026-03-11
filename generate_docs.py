"""
Generate:
  1. Project_Timeline.docx / .pdf  — no FCS references
  2. Model_Comparison_Table.docx / .pdf
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import reportlab.lib.pagesizes as pagesizes
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    # Remove any existing w:shd elements first so they don't conflict
    for existing in tcPr.findall(qn('w:shd')):
        tcPr.remove(existing)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1, color=None):
    h = doc.add_heading(text, level=level)
    if color:
        for run in h.runs:
            run.font.color.rgb = RGBColor(*color)
    return h

def add_para(doc, text, bold=False, size=10):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PROJECT TIMELINE
# ═════════════════════════════════════════════════════════════════════════════

STATUS_COLORS = {
    'complete':  ('2E7D32', colors.Color(0.18, 0.49, 0.20)),
    'running':   ('E65100', colors.Color(0.90, 0.32, 0.00)),
    'abandoned': ('B71C1C', colors.Color(0.72, 0.11, 0.11)),
}

PHASES = [
    {
        'phase': 1,
        'title': 'Data Ingestion & Feature Engineering',
        'status': 'complete',
        'description': (
            'Loaded the MATLAB intensity trace dataset (unrollIntensity.mat, ~600k readings). '
            'Filtered to noise=0 traces (~100k). Computed a comprehensive 302-dimensional feature '
            'vector per trace:\n'
            '  • Intensity statistics (mean, variance, Mandel Q, skewness, kurtosis)\n'
            '  • ACF at 32 log-spaced lags (via FFT)\n'
            '  • Scattering transform: J=8, Q=2 Morlet wavelets → 136 coefficients\n'
            '  • PSD at 32 log-spaced bins\n'
            '  • Transit-time features (5 threshold-crossing stats)\n'
            '  • 71 expanded features: exponential/stretched-exp ACF fits, ACF decay '
            'percentiles, threshold crossings, multi-scale segment stats, higher-order '
            'cumulants, multi-tau variance, non-stationarity metrics\n'
            '  • 19 ACF ratio features\n'
            'Performed 90/10 train-test split (85,698 train / 9,502 test after d≤10 filter). '
            'Built disk caches (numpy .npy) for fast repeated loading.'
        ),
        'key_outputs': [
            'cache_X_train_90pct.npy, cache_X_test_90pct.npy (302 features)',
            'cache_i_train_90pct.npy, cache_i_test_90pct.npy (raw traces)',
            'build_cache_90pct.py, add_expanded_features.py',
        ],
    },
    {
        'phase': 2,
        'title': 'Sklearn Baseline Models',
        'status': 'complete',
        'description': (
            'Trained sklearn models on 302 features (no raw traces). Target: log(d), '
            'evaluated on d≤10 test samples (9,502).\n'
            '  • HistGradientBoostingRegressor (HGBR): Best R²=0.9115, MAPE=16.1%\n'
            '  • MLP baseline (512-256-128): MAPE=15.9%\n'
            '  • MLP wider/deeper variants (512→2048 widths): MAPE=15.8–16.3%\n'
            'HGBR established the feature-only baseline. sklearn MLPs showed limited '
            'improvement with more capacity due to CPU-only training and slow convergence.'
        ),
        'key_outputs': [
            'results_mlp_90pct_*.txt (5 sklearn MLP variants)',
            'predict_d_90pct.py, predict_d_mlp_*.py',
        ],
    },
    {
        'phase': 3,
        'title': 'First PyTorch GPU Models',
        'status': 'complete',
        'description': (
            'Moved to GPU-accelerated PyTorch. Introduced dual-branch architecture for '
            'CNN-based models (raw trace branch + feature branch).\n'
            '  • MLP+BN+Dropout (2048-1024-512-256-128): MAPE=15.2%\n'
            '  • ResNet-MLP (512-dim, 6 blocks): MAPE=15.6%\n'
            '  • CNN-Fusion (4096-pt raw trace + 207 features): MAPE=14.9%\n'
            '  • Patch Transformer (64×64 patches + 207 features): MAPE=15.9%\n'
            '  • Multi-scale CNN (kernels k=8,32,128 + 207 features): MAPE=14.7%\n'
            'CNN-Fusion was the first model to exploit raw trace information alongside '
            'hand-crafted features. Training time dropped from hours (sklearn) to <3 min.'
        ),
        'key_outputs': [
            'pt_mlp_bn_gpu.py, pt_resnet_mlp_gpu.py, pt_cnn_fusion_gpu.py',
            'pt_transformer_gpu.py, pt_cnn_multiscale_gpu.py',
        ],
    },
    {
        'phase': 4,
        'title': 'FT-Transformer & Cosine Annealing',
        'status': 'complete',
        'description': (
            'Explored feature-only transformer architectures and improved training schedules.\n'
            '  • FT-Transformer (E=64, 8H, 4L, 283f): MAPE=14.7%, d<1=12.1%, d≥1=20.1%\n'
            '  • FT-Transformer Large (E=128, 8H, 6L, 283f): MAPE=15.0%\n'
            '  • FT-Transformer v2 (E=64, 4H, 6L, 283f): MAPE=14.8%\n'
            '  • MLP+Cosine (CosineAnnealingWarmRestarts, 302f): MAPE=14.6%\n'
            '  • ResNet+Cosine (302f): MAPE=14.7%\n'
            '  • CNN (90k traces, 302f): MAPE=14.2%\n'
            'CosineAnnealingWarmRestarts became the standard scheduler going forward. '
            'FT-Transformer did not outperform CNNs on this task.'
        ),
        'key_outputs': [
            'pt_fttransformer_gpu.py, pt_fttransformer_large_gpu.py, pt_fttransformer_v2_gpu.py',
            'pt_mlp_cosine_gpu.py, pt_resnet_cosine_gpu.py, pt_cnn_90pct_gpu.py',
        ],
    },
    {
        'phase': 5,
        'title': 'Ensemble Strategies & Meta-Learner',
        'status': 'complete',
        'description': (
            'Combined multiple models via geometric mean ensembles and learned meta-learners.\n'
            '  • 3-model ensemble (MLP+ResNet+CNN geometric mean): MAPE=13.9%\n'
            '  • 4-model ensemble (+HGBR): MAPE=14.2% (HGBR hurt)\n'
            '  • Balanced MLP/ResNet (d<1 downsampled): MAPE=15.4/15.2% (no gain)\n'
            '  • NNLS meta-learner 13 models, 5-fold CV: MAPE=13.07% (best ensemble)\n'
            'NNLS-13 assigned zero weight to all transformer models, confirming CNNs '
            'dominated. Per-regime routing yielded 13.01% but no reliable improvement.'
        ),
        'key_outputs': [
            'pt_ensemble_gpu.py, meta_learner_5fold.py',
            'results_meta_learner_5fold.txt (MAPE=13.07%)',
        ],
    },
    {
        'phase': 6,
        'title': 'Specialist Routing Experiments',
        'status': 'abandoned',
        'description': (
            'Attempted separate specialist models for slow (d<1) and fast (d≥1) diffusers, '
            'with a learned router to dispatch predictions.\n'
            '  • Specialist slow (trained d≤1.5): MAPE=23.1% on full test (46% on d≥1 — failed)\n'
            '  • Specialist fast (trained d≥0.5): MAPE=125.7% on full test (catastrophic on d<1)\n'
            '  • Router ensemble (soft gating): MAPE=14.9% — worse than single model\n'
            'Specialist models fail catastrophically on out-of-distribution samples. '
            'The router approach was abandoned.'
        ),
        'key_outputs': [
            'pt_specialist_slow_gpu.py, pt_specialist_fast_gpu.py',
            'results_pt_router_all.txt',
        ],
    },
    {
        'phase': 7,
        'title': 'WaveNet Architecture — Exploration',
        'status': 'complete',
        'description': (
            'Introduced WaveNet1D: dilated causal convolutions with receptive field covering '
            'the full trace. Architecture: 8 DilatedResBlocks, dilations [1,2,4,8,16,32,64,128], '
            'channels=128 (or 256 for wide variant). Dual-branch: trace (dilated conv) + '
            'features (MLP→128). Fused via head: Linear(384→256→64→1).\n'
            '  • WaveNet (ch=128, RF=2040): MAPE=13.4%, d<1=10.8%, d≥1=18.8%\n'
            '  • WaveNet seeds s2–s5: MAPE=13.2–13.6%\n'
            '  • WaveNet alt (10 blocks, RF=8184): MAPE=13.5%\n'
            '  • WaveNet stride2 (RF=4092): MAPE=12.8%, d<1=10.1%, d≥1=18.2%\n'
            '  • WaveNet wide (ch=256): MAPE=13.6%\n'
            '  • WaveNet+FTT hybrid: MAPE=14.5% (transformer branch did not help)\n'
            '  • CNN with augmentation: MAPE=16.5% (augmentation hurt)\n'
            '  • CNN with weighted loss: MAPE=14.8%\n'
            '  • CNN peakcrop (512pt): MAPE=15.7%\n'
            'WaveNet stride2 achieved 12.8% — best single model to that point. '
            'The d≥1 regime remained the bottleneck across all variants (~18–20%).'
        ),
        'key_outputs': [
            'pt_wavenet_gpu.py, pt_wavenet_wide_gpu.py (and 8 variant scripts)',
            'results_pt_wavenet_stride2.txt (MAPE=12.8%)',
        ],
    },
    {
        'phase': 8,
        'title': 'WaveNet Wide + Data Augmentation',
        'status': 'complete',
        'description': (
            'Trained WaveNet wide (ch=256) with trace augmentation: random amplitude scaling '
            'U[0.85,1.15], Gaussian noise (σ=0.03), and random shift (±128 bins). '
            'This achieved the best single-model result.\n'
            '  • wavenet_wide_aug: MAPE=12.34%, d<1=9.79%, d≥1=17.60% ← PROJECT BEST\n'
            'Follow-up experiments:\n'
            '  • Specialist WaveNet wide (trained d≥0.8, ~230k samples): used as router\n'
            '  • Fine-tune from aug checkpoint (LR=5e-5): MAPE=13.1% (slight regression)\n'
            '  • Router ensemble (aug routes d≥threshold to specialist): MAPE≈12.35% '
            '(no significant gain over single model)\n'
            'Key insight: augmentation on raw traces is the most effective regularization '
            'found so far, improving d≥1 by 2 percentage points vs. non-aug variants.'
        ),
        'key_outputs': [
            'pt_wavenet_wide_aug_gpu.py → model_wavenet_wide_aug.pt',
            'results: MAPE=12.34%, d<1=9.79%, d≥1=17.60%',
        ],
    },
    {
        'phase': 9,
        'title': 'Smoothed Trace WaveNet (W=3,5,10,20,50)',
        'status': 'complete',
        'description': (
            'Built smoothed trace caches (moving-average W=3,5,10,20,50 bins) and '
            'recomputed all 302 features on smoothed traces. Each model receives:\n'
            '  • Smoothed 4096-pt trace + 302 smoothed features\n'
            'Hypothesis: smoothing might reveal slower timescales relevant to the d≥1 regime.\n'
            'Results — all worse than wavenet_wide_aug baseline (12.34%):\n'
            '  • W=3:  MAPE=13.0%  d<1=10.8%  d≥1=17.7%  (best of group)\n'
            '  • W=5:  MAPE=13.5%  d<1=11.1%  d≥1=18.6%\n'
            '  • W=10: MAPE=13.8%  d<1=11.6%  d≥1=18.3%\n'
            '  • W=20: MAPE=14.3%  d<1=12.3%  d≥1=18.6%\n'
            '  • W=50: MAPE=14.6%  d<1=12.5%  d≥1=18.8%\n'
            'Clear trend: heavier smoothing progressively hurts. Smoothing alone does not '
            'improve on the raw-trace aug model.'
        ),
        'key_outputs': [
            'cache_i_{train,test}_smooth{W}.npy, cache_X_{train,test}_smooth{W}.npy',
            'pt_wavenet_wide_smooth_gpu.py',
            'results_pt_wavenet_wide_smooth{W}.txt  (W=3,5,10,20,50)',
        ],
    },
    {
        'phase': 10,
        'title': 'Fused Raw+Smooth WaveNet (W=3,5,10,20,50)',
        'status': 'complete',
        'description': (
            'Each training sample contains BOTH raw and smoothed representations simultaneously:\n'
            '  • Trace input: raw (4096 pts) + smooth (4096 pts) concatenated → 8192 pts\n'
            '  • Feature input: raw (302) + smooth (302) concatenated → 604 dims\n'
            'Same 390k training samples — not doubled. AMP (mixed precision) training.\n'
            'Results:\n'
            '  • W=3:  MAPE=14.0%  d<1=11.1%  d≥1=20.0%\n'
            '  • W=5:  MAPE=12.6%  d<1=10.1%  d≥1=17.8%  (best of group)\n'
            '  • W=10: MAPE=13.5%  d<1=11.2%  d≥1=18.2%\n'
            '  • W=20: MAPE=13.8%  d<1=11.9%  d≥1=17.7%\n'
            '  • W=50: MAPE=13.0%  d<1=10.6%  d≥1=17.9%\n'
            'Fused W=5 is the closest competitor to wavenet_wide_aug (12.6% vs 12.34%). '
            'The added information from smoothing does not consistently outperform the '
            'pure raw-trace augmented model.'
        ),
        'key_outputs': [
            'cache_X_{train,test}_fused{W}.npy (604-dim)',
            'pt_wavenet_wide_fused_gpu.py (AMP, 8192-pt input, feat_dim=604)',
            'results_pt_wavenet_wide_fused{W}.txt  (W=3,5,10,20,50)',
        ],
    },
    {
        'phase': 11,
        'title': 'Online Background Augmentation (wn_online)',
        'status': 'complete',
        'description': (
            'Added online Poisson background noise to the trace branch during training '
            'to improve robustness to real experimental conditions. The simulation data '
            'already contains photon shot noise (B=50 photons/bin at dt=1ms); "noise0" '
            'refers to zero background only. Per batch: b_pct ~ Uniform[0, 20%] of '
            'MAX_RATE = 0–10 photons/bin background added before z-score normalisation.\n'
            'Feature branch received clean pre-computed features (known limitation).\n'
            'Results at four fixed background levels on clean test set:\n'
            '  • b=0%  (clean):  MAPE=19.5%  d<1=18.3%  d≥1=21.9%\n'
            '  • b=5%  (2.5 ph/bin): MAPE=19.4%  d<1=17.8%  d≥1=22.6%\n'
            '  • b=10% (5.0 ph/bin): MAPE=20.0%  d<1=18.3%  d≥1=23.6%\n'
            '  • b=20% (10.0 ph/bin): MAPE=21.5%  d<1=19.6%  d≥1=25.6%\n'
            'Outcome: training with 0–20% background range severely degraded performance '
            'even on clean data (19.5% vs 12.34% baseline). The augmentation range was '
            'too wide — heavy background corrupted traces enough to push the model away '
            'from the optimal clean-data solution. Approach needs revisiting with a '
            'narrower background range (e.g. 0–5%).'
        ),
        'key_outputs': [
            'pt_wavenet_wide_aug_online_gpu.py, job_wavenet_wide_aug_online.sh',
            'model_pt_wavenet_wide_aug_online.pt',
            'pred_logd_pt_wavenet_wide_aug_online_test_b{00,05,10,20}pct.npy',
            'results_pt_wavenet_wide_aug_online.txt',
        ],
    },
    {
        'phase': 12,
        'title': 'Variable dt Simulations & Training (dt=0.25ms, 0.5ms)',
        'status': 'running',
        'description': (
            'Generated new simulation datasets with finer temporal resolution. Fixed '
            'tMax=4096ms; bin count scales with dt:\n'
            '  • dt=0.25ms → 16,384 bins/trace (tag: dt025)\n'
            '  • dt=0.5ms  →  8,192 bins/trace (tag: dt050)\n'
            '  • dt=1.0ms  →  4,096 bins/trace (existing)\n'
            'Feature pipeline generalised via make_lag_constants(N) — all lag arrays '
            'computed from actual trace length N.\n\n'
            'dt050 pipeline:\n'
            '  • Simulation (job 55532119_1): COMPLETED (59 s)\n'
            '  • Cache build (job 55614563): RUNNING\n'
            '  • Training pt_wavenet_wide_aug_dt050 (job 55614564): PENDING (afterok)\n'
            '    WaveNet wide + online augmentation (shift ±256 bins), 8192-pt input\n\n'
            'dt025 pipeline:\n'
            '  • Simulation (job 55612970): COMPLETED (resubmitted at 64G/4h after OOM)\n'
            '  • Cache build (job 55614569): RUNNING — auto-submitted by watcher\n'
            '  • Smooth cache (W=2,4,6,12,20,40): PENDING — watcher will auto-submit\n'
            '  • Training: not yet planned'
        ),
        'key_outputs': [
            'sims_dt025_noise0_{i,d}.npy  (16,384 bins, ~25 GB)',
            'sims_dt050_noise0_{i,d}.npy  (8,192 bins, 9.2 GB) — complete',
            'cache_{i,X,d}_{train,test}_dt050.npy — building',
            'pt_wavenet_wide_aug_dt050_gpu.py, job_wavenet_wide_aug_dt050.sh',
            'model_pt_wavenet_wide_aug_dt050.pt — pending',
            '/tmp/watch_dt025_pipeline.sh (auto-chains dt025 steps 2 & 3)',
        ],
    },
]


def build_timeline_docx(out_path):
    doc = Document()

    # Page margins
    for sec in doc.sections:
        sec.left_margin   = Inches(1.0)
        sec.right_margin  = Inches(1.0)
        sec.top_margin    = Inches(1.0)
        sec.bottom_margin = Inches(1.0)

    # Title
    title = doc.add_heading('Diffusion Coefficient Prediction — Project Timeline', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
        run.font.size = Pt(18)

    doc.add_paragraph('Simulation-based machine learning pipeline for predicting '
                       'diffusion coefficients from fluorescence intensity traces.')
    doc.add_paragraph('')

    # Legend
    leg = doc.add_paragraph()
    leg.add_run('Status:  ').bold = True
    leg.add_run(' Complete ')
    run = leg.add_run(' ■ ')
    run.font.color.rgb = RGBColor(0x2E, 0x7D, 0x32)
    leg.add_run('  Running ')
    run2 = leg.add_run(' ■ ')
    run2.font.color.rgb = RGBColor(0xE6, 0x51, 0x00)
    leg.add_run('  Abandoned ')
    run3 = leg.add_run(' ■ ')
    run3.font.color.rgb = RGBColor(0xB7, 0x1C, 0x1C)

    doc.add_paragraph('')

    for ph in PHASES:
        hex_c, _ = STATUS_COLORS[ph['status']]
        rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

        # Phase heading
        heading = doc.add_heading(f"Phase {ph['phase']}: {ph['title']}", level=2)
        for run in heading.runs:
            run.font.color.rgb = RGBColor(*rgb)

        # Status badge
        sp = doc.add_paragraph()
        badge = sp.add_run(f"  Status: {ph['status'].upper()}  ")
        badge.bold = True
        badge.font.color.rgb = RGBColor(*rgb)

        # Description
        doc.add_paragraph(ph['description'])

        # Key outputs
        out_para = doc.add_paragraph()
        out_para.add_run('Key files / outputs:').bold = True
        for item in ph['key_outputs']:
            doc.add_paragraph(f'    • {item}', style='List Bullet')

        doc.add_paragraph('')

    doc.save(out_path)
    print(f'Saved timeline docx → {out_path}')


def build_timeline_pdf(out_path):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=pagesizes.letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                  fontSize=18, textColor=colors.HexColor('#1A237E'),
                                  spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'],
                                     fontSize=11, spaceAfter=12, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=13, spaceBefore=14, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=9, leading=13, spaceAfter=6)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
                                   fontSize=9, leading=12, leftIndent=16,
                                   bulletIndent=8, spaceAfter=2)
    bold_style = ParagraphStyle('Bold', parent=styles['Normal'],
                                 fontSize=9, leading=12)

    story = []

    story.append(Paragraph('Diffusion Coefficient Prediction — Project Timeline', title_style))
    story.append(Paragraph(
        'Simulation-based machine learning pipeline for predicting diffusion coefficients '
        'from fluorescence intensity traces.',
        subtitle_style))
    story.append(Spacer(1, 12))

    for ph in PHASES:
        hex_c, rl_color = STATUS_COLORS[ph['status']]
        h2 = ParagraphStyle(f'H2_{ph["phase"]}', parent=h2_style,
                             textColor=rl_color)
        blocks = []
        blocks.append(Paragraph(f"Phase {ph['phase']}: {ph['title']}", h2))

        status_style = ParagraphStyle('StatusStyle', parent=bold_style,
                                       textColor=rl_color, spaceBefore=0, spaceAfter=4)
        blocks.append(Paragraph(f"Status: {ph['status'].upper()}", status_style))

        # Description — split on newlines for bullets
        for line in ph['description'].split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('•'):
                blocks.append(Paragraph(stripped, bullet_style))
            else:
                blocks.append(Paragraph(stripped, body_style))

        blocks.append(Paragraph('<b>Key files / outputs:</b>', bold_style))
        for item in ph['key_outputs']:
            blocks.append(Paragraph(f'• {item}', bullet_style))
        blocks.append(Spacer(1, 10))

        story.append(KeepTogether(blocks))

    doc.build(story)
    print(f'Saved timeline pdf → {out_path}')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – MODEL COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

MODELS = [
    # ── Phase 1: Early sklearn exploration (50% split, ~10k–50k train) ───
    ('GBR baseline\n(39f, 10k train)', 'Phase 1',
     'sklearn GradientBoostingRegressor; 39 features (intensity stats + ACF@29 lags). '
     'Predicts raw d (no log transform). First end-to-end run on MATLAB data.',
     'None', '39 features', '39', '10,000',
     '54.7%', '—', '—', 'Complete'),
    ('HistGBR\n(175f, 50k, log d)', 'Phase 1',
     'HistGradientBoostingRegressor; 175 features (+scattering J=8,Q=2). '
     'First log(d) target; 50k training samples. Training: 3.6 s vs 3 min for GBR.',
     'None', '175 features', '175', '50,000',
     '17.5%', '—', '—', 'Complete'),
    ('HistGBR tuned\n(207f, 50k)', 'Phase 1',
     'HistGBR with hyperparameter search; 207 features (+PSD at 32 log-spaced bins). '
     'Best params: max_iter=800, max_depth=5, LR=0.08, min_leaf=80, max_leaves=127.',
     'None', '207 features', '207', '50,000',
     '16.8%', '—', '—', 'Complete'),
    ('MLP 512-256-128\n(207f, 50k)', 'Phase 1',
     'sklearn MLPRegressor, 207 features, 50k training samples, log(d) target. '
     'Early stopping at 29 iterations. Comparable to tuned HistGBR.',
     'None', '207 features', '207', '50,000',
     '16.4%', '—', '—', 'Complete'),
    # ── sklearn baselines ─────────────────────────────────────────────────
    ('HGBR (283f)', 'Phase 2',
     'sklearn HistGradientBoosting Regressor',
     'None', '283 features', '283', '85,698 simulated traces',
     '16.1%', '—', '—', 'Complete'),
    ('MLP baseline\n(512-256-128)', 'Phase 2',
     'sklearn MLPRegressor',
     'None', '302 features', '302', '85,698',
     '15.9%', '—', '—', 'Complete'),
    ('MLP wider\n(1024-512-256-128)', 'Phase 2',
     'sklearn MLPRegressor',
     'None', '302 features', '302', '85,698',
     '15.8%', '—', '—', 'Complete'),
    # ── First PyTorch models ───────────────────────────────────────────────
    ('PT MLP+BN\n(2048-1024-512-256-128)', 'Phase 3',
     'PyTorch MLP + BatchNorm + Dropout(0.15)',
     'None', '302 features', '302', '85,698',
     '15.2%', '—', '—', 'Complete'),
    ('PT ResNet-MLP\n(dim=512, 6 blocks)', 'Phase 3',
     'PyTorch ResNet-style MLP, 6 residual blocks',
     'None', '302 features', '302', '85,698',
     '15.6%', '—', '—', 'Complete'),
    ('CNN-Fusion\n(v1)', 'Phase 3',
     '1D CNN on raw trace + FC feature branch → head',
     'Raw 4096-pt trace', '207 features', '207', '47,600 (50% split)',
     '14.9%', '—', '—', 'Complete'),
    ('Patch Transformer', 'Phase 3',
     'Patch Transformer (64×64-pt patches, d=128, 4H, 4L) + FC feature branch',
     'Raw 4096-pt trace\n(as patches)', '207 features', '207', '47,600',
     '15.9%', '—', '—', 'Complete'),
    ('Multi-scale CNN\n(v1)', 'Phase 3',
     '3-branch CNN (k=8,32,128) + feature branch → head',
     'Raw 4096-pt trace', '207 features', '207', '47,600',
     '14.7%', '—', '—', 'Complete'),
    # ── FT-Transformer ────────────────────────────────────────────────────
    ('FT-Transformer\n(E=64, 8H, 4L)', 'Phase 4',
     'Feature Tokenization Transformer, 4 layers, 8 heads, d=64',
     'None', '283 features', '283', '85,698',
     '14.7%', '12.1%', '20.1%', 'Complete'),
    ('FT-Transformer\nLarge (E=128, 8H, 6L)', 'Phase 4',
     'FT-Transformer, 6 layers, 8 heads, d=128',
     'None', '283 features', '283', '85,698',
     '15.0%', '12.3%', '20.5%', 'Complete'),
    ('FT-Transformer v2\n(E=64, 4H, 6L)', 'Phase 4',
     'FT-Transformer, 6 layers, 4 heads, d=64',
     'None', '283 features', '283', '85,698',
     '14.8%', '12.2%', '20.2%', 'Complete'),
    # ── Cosine / improved ─────────────────────────────────────────────────
    ('MLP+Cosine', 'Phase 4',
     'PyTorch MLP+BN+Dropout + CosineAnnealingWarmRestarts',
     'None', '302 features', '302', '85,698',
     '14.6%', '—', '—', 'Complete'),
    ('ResNet+Cosine', 'Phase 4',
     'PyTorch ResNet-MLP + CosineAnnealingWarmRestarts',
     'None', '302 features', '302', '85,698',
     '14.7%', '—', '—', 'Complete'),
    ('CNN 90k\n(v2)', 'Phase 4',
     '1D CNN + 302-dim feature branch, CosineAnnealing',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '14.2%', '—', '—', 'Complete'),
    # ── Ensembles ─────────────────────────────────────────────────────────
    ('3-model Ensemble\n(geom. mean)', 'Phase 5',
     'Geometric mean of MLP+Cosine, ResNet+Cosine, CNN-90k',
     'Ensemble of above', 'Ensemble of above', '302', '85,698',
     '13.9%', '—', '—', 'Complete'),
    ('NNLS Meta-learner\n(13 models, 5-fold)', 'Phase 5',
     'Non-negative least squares combining 13 model predictions, 5-fold CV',
     'Ensemble of all', 'Ensemble of all', 'N/A', '85,698',
     '13.07%', '—', '—', 'Complete'),
    ('Balanced MLP', 'Phase 5',
     'MLP+BN+Dropout trained on class-balanced data (d<1 downsampled)',
     'None', '283 features', '283', '57,190 (balanced)',
     '15.4%', '12.9%', '20.5%', 'Complete'),
    ('Balanced ResNet', 'Phase 5',
     'ResNet-MLP trained on class-balanced data',
     'None', '283 features', '283', '57,190 (balanced)',
     '15.2%', '12.5%', '20.6%', 'Complete'),
    # ── Specialist routing ────────────────────────────────────────────────
    ('Specialist Slow\n(d≤1.5 filter)', 'Phase 6',
     'MLP+BN+Dropout trained only on slow diffusers (d≤1.5)',
     'None', '302 features', '302', '62,113 (filtered)',
     '23.1%\n(full test)', '11.9%', '46.2%', 'Abandoned'),
    ('Specialist Fast\n(d≥0.5 filter)', 'Phase 6',
     'MLP+BN+Dropout trained only on fast diffusers (d≥0.5)',
     'None', '302 features', '302', '37,211 (filtered)',
     '125.7%\n(full test)', '176.6%', '20.7%', 'Abandoned'),
    ('Router Ensemble\n(soft gating)', 'Phase 6',
     'Learned router (sigmoid) gating between specialist models',
     'None', '302 features', '302', '85,698',
     '14.9%', '12.1%', '20.6%', 'Abandoned'),
    # ── WaveNet variants ──────────────────────────────────────────────────
    ('WaveNet\n(ch=128, RF=2040)', 'Phase 7',
     'WaveNet1D: 8 dilated residual blocks (dils=[1..128]), ch=128 + feature MLP',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '13.4%', '10.8%', '18.8%', 'Complete'),
    ('WaveNet seeds\ns2–s5', 'Phase 7',
     'WaveNet1D ch=128 retrained with seeds 2–5 for ensemble diversity',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '13.2–13.6%', '10.6–10.9%', '18.7–19.2%', 'Complete'),
    ('WaveNet alt\n(RF=8184)', 'Phase 7',
     'WaveNet1D: 10 blocks, dilations [1..512], RF=8184 lags (full trace)',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '13.5%', '10.9%', '18.8%', 'Complete'),
    ('WaveNet stride2\n(RF=4092)', 'Phase 7',
     'WaveNet1D: 10 blocks, initial stride=2, RF=4092 lags',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '12.8%', '10.1%', '18.2%', 'Complete'),
    ('WaveNet wide\n(ch=256)', 'Phase 7',
     'WaveNet1D: 8 blocks, channels=256 (wider feature maps)',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '13.6%', '10.7%', '19.5%', 'Complete'),
    ('WaveNet+FTT\nhybrid', 'Phase 7',
     'WaveNet trace branch + FTT feature branch (E=64, 8H, 4L) → fusion',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '14.5%', '12.2%', '19.2%', 'Complete'),
    ('CNN + augmentation', 'Phase 7',
     'CNN-90k + trace augmentation: scale U[0.85,1.15] + noise σ=0.03 + shift ±128',
     'Raw 4096-pt trace\n(augmented)', '302 features', '302', '85,698',
     '16.5%', '13.2%', '23.3%', 'Complete'),
    ('CNN + weighted loss', 'Phase 7',
     'CNN-90k with sample weight w=max(1, sqrt(d)) in MSE loss',
     'Raw 4096-pt trace', '302 features', '302', '85,698',
     '14.8%', '12.0%', '20.7%', 'Complete'),
    ('CNN peakcrop\n(W=512)', 'Phase 7',
     'CNN on 512-pt peak-centered window (discards tails)',
     '512-pt peak crop', '302 features', '302', '85,698',
     '15.7%', '13.0%', '21.3%', 'Complete'),
    # ── Best model + phase 8 ──────────────────────────────────────────────
    ('WaveNet wide + aug\n★ PROJECT BEST ★', 'Phase 8',
     'WaveNet1D ch=256, 8 blocks + trace augmentation (scale+noise+shift). '
     'Trained on 390k simulated traces (dt=1ms, tMax=4096ms).',
     'Raw 4096-pt trace\n(augmented training)', '302 features', '302',
     '390,000 simulated',
     '12.34%', '9.79%', '17.60%', 'Best'),
    ('WaveNet wide\nspecialist (d≥0.8)', 'Phase 8',
     'WaveNet wide trained only on d≥0.8 samples from 390k simulation set; '
     'used as specialist in router ensemble.',
     'Raw 4096-pt trace', '302 features', '302', '~230k simulated\n(d≥0.8 filter)',
     '~17.5%\n(full test)', '—', '17.5%', 'Complete'),
    ('WaveNet wide aug v2\n(fine-tune LR=5e-5)', 'Phase 8',
     'Fine-tuned from wavenet_wide_aug checkpoint; LR=5e-5, CosineAnnealing '
     'T_max=100, patience=20. Same 390k simulated training set.',
     'Raw 4096-pt trace', '302 features', '302', '390,000 simulated',
     '13.1%', '—', '—', 'Complete'),
    ('Router aug\n(aug→specialist)', 'Phase 8',
     'wavenet_wide_aug routes samples with predicted D≥threshold to specialist. '
     'Uses predictions from both 390k-trained models.',
     'Raw 4096-pt trace', '302 features', '302', '390,000 simulated',
     '≈12.35%', '—', '—', 'Complete'),
    # ── Phase 9: smooth ───────────────────────────────────────────────────
    ('WaveNet smooth W=3', 'Phase 9',
     'WaveNet wide on moving-avg smoothed trace (W=3) + 302 smoothed features.',
     'Smooth 4096-pt (W=3)', '302 smooth features', '302', '390,000 simulated',
     '13.0%', '10.8%', '17.7%', 'Complete'),
    ('WaveNet smooth W=5', 'Phase 9',
     'WaveNet wide on moving-avg smoothed trace (W=5) + 302 smoothed features.',
     'Smooth 4096-pt (W=5)', '302 smooth features', '302', '390,000 simulated',
     '13.5%', '11.1%', '18.6%', 'Complete'),
    ('WaveNet smooth W=10', 'Phase 9',
     'WaveNet wide on moving-avg smoothed trace (W=10) + 302 smoothed features.',
     'Smooth 4096-pt (W=10)', '302 smooth features', '302', '390,000 simulated',
     '13.8%', '11.6%', '18.3%', 'Complete'),
    ('WaveNet smooth W=20', 'Phase 9',
     'WaveNet wide on moving-avg smoothed trace (W=20) + 302 smoothed features.',
     'Smooth 4096-pt (W=20)', '302 smooth features', '302', '390,000 simulated',
     '14.3%', '12.3%', '18.6%', 'Complete'),
    ('WaveNet smooth W=50', 'Phase 9',
     'WaveNet wide on moving-avg smoothed trace (W=50) + 302 smoothed features.',
     'Smooth 4096-pt (W=50)', '302 smooth features', '302', '390,000 simulated',
     '14.6%', '12.5%', '18.8%', 'Complete'),
    # ── Phase 10: fused ───────────────────────────────────────────────────
    ('WaveNet fused W=3', 'Phase 10',
     'WaveNet wide: raw+smooth3 trace (8192-pt concat) + raw+smooth3 features (604-dim). AMP.',
     'Raw+Smooth3\n8192-pt concat', 'Raw 302 + Smooth 302\n= 604-dim', '604',
     '390,000 simulated', '14.0%', '11.1%', '20.0%', 'Complete'),
    ('WaveNet fused W=5', 'Phase 10',
     'WaveNet wide: raw+smooth5 trace (8192-pt concat) + raw+smooth5 features (604-dim). AMP.',
     'Raw+Smooth5\n8192-pt concat', 'Raw 302 + Smooth 302\n= 604-dim', '604',
     '390,000 simulated', '12.6%', '10.1%', '17.8%', 'Complete'),
    ('WaveNet fused W=10', 'Phase 10',
     'WaveNet wide: raw+smooth10 trace (8192-pt concat) + raw+smooth10 features (604-dim). AMP.',
     'Raw+Smooth10\n8192-pt concat', 'Raw 302 + Smooth 302\n= 604-dim', '604',
     '390,000 simulated', '13.5%', '11.2%', '18.2%', 'Complete'),
    ('WaveNet fused W=20', 'Phase 10',
     'WaveNet wide: raw+smooth20 trace (8192-pt concat) + raw+smooth20 features (604-dim). AMP.',
     'Raw+Smooth20\n8192-pt concat', 'Raw 302 + Smooth 302\n= 604-dim', '604',
     '390,000 simulated', '13.8%', '11.9%', '17.7%', 'Complete'),
    ('WaveNet fused W=50', 'Phase 10',
     'WaveNet wide: raw+smooth50 trace (8192-pt concat) + raw+smooth50 features (604-dim). AMP.',
     'Raw+Smooth50\n8192-pt concat', 'Raw 302 + Smooth 302\n= 604-dim', '604',
     '390,000 simulated', '13.0%', '10.6%', '17.9%', 'Complete'),
    # ── Phase 11: online background augmentation ──────────────────────────
    ('WaveNet wide\naug online\n(b=0–20%)', 'Phase 11',
     'WaveNet wide aug + online Poisson background: b_pct~Uniform[0,20%] of MAX_RATE '
     'per batch. Feature branch still uses clean pre-computed features. '
     'Evaluated at 4 fixed b levels; b=0% shown here (clean test).',
     'Raw 4096-pt trace\n(+Poisson bg\nduring training)', '302 clean features', '302',
     '390,000 simulated',
     '19.5% (b=0)\n19.4% (b=5%)\n20.0% (b=10%)\n21.5% (b=20%)',
     '18.3%', '21.9%', 'Complete'),
    # ── Phase 12: dt050 training ──────────────────────────────────────────
    ('WaveNet wide\naug dt050\n(dt=0.5ms)', 'Phase 12',
     'WaveNet wide + online augmentation (shift ±256 bins) trained on dt050 data '
     '(8192 bins/trace, tMax=4096ms). AdaptiveAvgPool1d handles 8192-pt input. '
     '302 features computed by make_lag_constants(8192).',
     'Raw 8192-pt trace\n(dt=0.5ms)', '302 dt050 features', '302',
     '~194k simulated\n(dt050, d≤10)',
     'Pending', 'Pending', 'Pending', 'Running'),
]

TABLE_COLS = ['Model', 'Phase', 'Architecture / Description', 'Trace Input',
              'Feature Input', 'Feat Dim', 'Train Samples',
              'MAPE (overall)', 'd<1 MAPE', 'd≥1 MAPE', 'Status']


def build_table_docx(out_path):
    doc = Document()
    for sec in doc.sections:
        sec.orientation = 1  # landscape
        sec.page_width  = Inches(14)
        sec.page_height = Inches(8.5)
        sec.left_margin = sec.right_margin = Inches(0.5)
        sec.top_margin  = sec.bottom_margin = Inches(0.5)

    t = doc.add_heading('Model Comparison Table', 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in t.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
        run.font.size = Pt(16)

    p = doc.add_paragraph(
        'Phase 1: early exploration, 10k\u201350k training samples (50% split). '
        'Phases 2\u20137: MATLAB-derived data (85,698 train / 9,502 test, d\u226410 filter). '
        'Phases 8\u201311: 390,000 simulated traces (dt=1ms, tMax=4096ms). '
        'Phase 12: dt050 data (~194k, dt=0.5ms, 8192 bins). '
        'Test set: 9,502 samples (d\u226410). MAPE on log-transformed d. '
        '\u2605 Gold highlight = current project best model.')
    p.runs[0].font.size = Pt(9)

    # Build table
    tbl = doc.add_table(rows=1, cols=len(TABLE_COLS))
    tbl.style = 'Table Grid'

    # Header row
    hdr = tbl.rows[0].cells
    for i, col in enumerate(TABLE_COLS):
        cell = hdr[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(col)
        run.bold = True
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(cell, '1A237E')

    STATUS_HEX = {'Complete': 'C8E6C9', 'Running': 'FFE0B2', 'Abandoned': 'FFCDD2',
                  'Best': 'FFD740'}

    for row_data in MODELS:
        row = tbl.add_row().cells
        for i, val in enumerate(row_data):
            row[i].text = val
            for par in row[i].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(8)
        # Color by status (last column)
        status = row_data[-1]
        bg = STATUS_HEX.get(status, 'FFFFFF')
        for cell in row:
            set_cell_bg(cell, bg)

    # Column widths (approx)
    widths = [1.3, 0.6, 2.8, 1.3, 1.5, 0.6, 1.0, 0.9, 0.8, 0.8, 0.8]
    for i, w in enumerate(widths):
        for cell in tbl.columns[i].cells:
            cell.width = Inches(w)

    doc.save(out_path)
    print(f'Saved model table docx → {out_path}')


def build_table_pdf(out_path):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=pagesizes.landscape(pagesizes.letter),
        leftMargin=0.4*inch, rightMargin=0.4*inch,
        topMargin=0.5*inch,  bottomMargin=0.5*inch,
    )
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle('T', parent=styles['Title'],
                              fontSize=15, textColor=colors.HexColor('#1A237E'),
                              spaceAfter=4, alignment=TA_CENTER)
    sub_s   = ParagraphStyle('S', parent=styles['Normal'],
                              fontSize=8, spaceAfter=8, alignment=TA_CENTER)
    cell_s  = ParagraphStyle('C', parent=styles['Normal'],
                              fontSize=7, leading=9, wordWrap='CJK')

    story = []
    story.append(Paragraph('Model Comparison Table', title_s))
    story.append(Paragraph(
        'Phase 1: early exploration, 10k\u201350k training samples (50% split). '
        'Phases 2\u20137: MATLAB-derived data (85,698 train / 9,502 test). '
        'Phases 8\u201311: 390,000 simulated traces (dt=1ms). '
        'Phase 12: dt050 data (~194k, dt=0.5ms, 8192 bins). '
        'Test set: 9,502 samples (d\u226410). MAPE on log-transformed d. '
        '\u2605 Gold highlight = current project best model.',
        sub_s))

    # Build table data
    header = [Paragraph(f'<b>{c}</b>', cell_s) for c in TABLE_COLS]
    data = [header]
    for row_data in MODELS:
        data.append([Paragraph(str(v), cell_s) for v in row_data])

    STATUS_RL = {
        'Complete':  colors.HexColor('#C8E6C9'),
        'Running':   colors.HexColor('#FFE0B2'),
        'Abandoned': colors.HexColor('#FFCDD2'),
        'Best':      colors.HexColor('#FFD740'),
    }

    # Row background colors — explicit per-row BACKGROUND commands (avoids cycling issues)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A237E')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTSIZE',   (0, 0), (-1, -1), 7),
        ('GRID',       (0, 0), (-1, -1), 0.3, colors.grey),
        ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
    ]
    for r_idx, row_data in enumerate(MODELS):
        bg = STATUS_RL.get(row_data[-1], colors.white)
        style_cmds.append(('BACKGROUND', (0, r_idx + 1), (-1, r_idx + 1), bg))

    # Col widths (landscape letter = 11in usable)
    col_widths = [1.1, 0.5, 2.3, 1.0, 1.2, 0.5, 0.85, 0.75, 0.65, 0.65, 0.7]
    col_widths = [w * inch for w in col_widths]

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)

    doc.build(story)
    print(f'Saved model table pdf → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    build_timeline_docx(os.path.join(BASE, 'Project_Timeline_v2.docx'))
    build_timeline_pdf(os.path.join(BASE,  'Project_Timeline_v2.pdf'))
    build_table_docx(os.path.join(BASE,    'Model_Comparison_Table_v2.docx'))
    build_table_pdf(os.path.join(BASE,     'Model_Comparison_Table_v2.pdf'))
    print('Done.')
