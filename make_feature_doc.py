from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin   = Inches(1.1)
    section.right_margin  = Inches(1.1)

# ── Styles ────────────────────────────────────────────────────────────────────
style_normal = doc.styles['Normal']
style_normal.font.name = 'Calibri'
style_normal.font.size = Pt(10.5)

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    run = p.runs[0] if p.runs else p.add_run(text)
    run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p

def para(text='', bold=False, italic=False, size=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    # Header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shading = OxmlElement('w:shd')
        shading.set(qn('w:val'), 'clear')
        shading.set(qn('w:color'), 'auto')
        shading.set(qn('w:fill'), 'D9E2F3')
        cell._tc.get_or_add_tcPr().append(shading)
    # Data rows
    for ri, row in enumerate(rows):
        tr = t.rows[ri + 1]
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(10)
    # Column widths
    if col_widths:
        for ci, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[ci].width = Inches(w)
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
title = doc.add_heading('FCS Diffusion Prediction — Feature Pipeline', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

p = doc.add_paragraph()
run = p.add_run(
    'All 302 features extracted per intensity trace by build_cache_dt.py. '
    'Features are computed independently for each dt value. '
    'For multi-dt models (dt=0.1, 0.5, 1.0 ms), the 302-dim vectors are '
    'concatenated to form a 906-dim feature vector.'
)
run.font.size = Pt(10.5)
run.italic = True
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
heading('Feature Groups Summary', level=1)
doc.add_paragraph()

summary_rows = [
    ('0–7',    'Basic statistics',              '8',  'Mean, variance, Mandel Q, skewness, kurtosis, kappa1, kappa2, kappa2_norm'),
    ('8–9',    'G0 + half-decay',               '2',  'G0 = var/mean²; lag where ACF < G0/2'),
    ('10–41',  'ACF log-spaced lags',           '32', 'Autocorrelation at ~32 log-spaced lags (lag 1 to N/2) via FFT'),
    ('42–57',  'Scattering S1',                 '16', '1st-order scattering: mean |u₁| for each of 16 Morlet filters (J=8, Q=2)'),
    ('58–177', 'Scattering S2',                 '120','2nd-order scattering: mean |u₂| for 120 filter pairs'),
    ('178–209','PSD log-spaced bins',           '32', 'Power spectral density in 32 log-spaced frequency bins'),
    ('210–214','Transit fractions',             '5',  'Fraction of trace ≥ {80,60,40,20,10}% of peak'),
    ('215–217','Exponential ACF fit',           '3',  'A, tau_D, RMSE  (log-space OLS fit to G(τ) = A·exp(−τ/τ_D))'),
    ('218–220','Stretched-exp ACF fit',         '3',  'tau_D, beta, RMSE  (log-log OLS fit to G(τ) = G0·exp(−(τ/τ_D)^β))'),
    ('221–224','ACF decay percentiles',         '4',  'Lag where G = {90,75,25,10}% of G0 (linear interpolation)'),
    ('225–234','Short linear lags',             '10', 'ACF at lags {5,8,10,11,13,14,16,17,18,20} — lags not in log-spaced set'),
    ('235–254','Threshold crossing stats',      '20', '4 stats × 5 thresholds: n_bursts/N, max_burst/N, std_burst/N, mean_inter/N'),
    ('255–269','Multi-scale segment stats',     '15', '3 stats × 5 windows [32,64,128,512,1024]: mean_var, std_mean, cv_var'),
    ('270–271','Higher-order cumulants',        '2',  'kappa3_norm = ⟨δI³⟩/mean³;  kappa4_norm = ⟨δI⁴⟩/mean⁴'),
    ('272–281','Multi-tau variance',            '10', 'kappa2_norm(T) at bin times T = {2,4,8,16,32,64,128,256,512,1024}'),
    ('282–285','Non-stationarity',              '4',  'mean_ratio, var_ratio, slope_norm, corr_halves (first vs second half)'),
    ('286–301','ACF log-ratio features',        '19', 'log(G(τ+1)/G(τ)) for τ = 1…19  — local slope of ACF decay'),
]

add_table(
    headers=['Columns', 'Group', 'Dim', 'Description'],
    rows=summary_rows,
    col_widths=[0.75, 1.75, 0.5, 3.6]
)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# Detailed sections
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Basic statistics ───────────────────────────────────────────────────────
heading('1. Basic Statistics  (cols 0–7,  8 features)', level=2)
add_table(
    headers=['Col', 'Name', 'Formula'],
    rows=[
        ('0', 'mean',        'μ = (1/N) Σ I(t)'),
        ('1', 'variance',    'σ² = (1/N) Σ (I−μ)²'),
        ('2', 'Mandel Q',    'Q = (σ²−μ)/μ'),
        ('3', 'skewness',    'κ₃ / σ³'),
        ('4', 'kurtosis',    'excess: κ₄/σ⁴ − 3'),
        ('5', 'kappa1',      'κ₁ = μ  (1st factorial cumulant)'),
        ('6', 'kappa2',      'κ₂ = σ²−μ  (2nd factorial cumulant)'),
        ('7', 'kappa2_norm', 'κ₂/μ²'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 2. G0 + half-decay ───────────────────────────────────────────────────────
heading('2. G0 + Half-Decay  (cols 8–9,  2 features)', level=2)
add_table(
    headers=['Col', 'Name', 'Description'],
    rows=[
        ('8',  'G0',         'G0 = σ²/μ² — amplitude of autocorrelation at lag 0'),
        ('9',  'half_decay', 'Smallest lag τ where ACF(τ) < G0/2; capped at last lag if never reached'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 3. ACF ────────────────────────────────────────────────────────────────────
heading('3. Autocorrelation Function — Log-Spaced Lags  (cols 10–41,  32 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Computed via FFT: G(τ) = ⟨δI(t)·δI(t+τ)⟩/(N−τ)/μ²  '
    'where δI = I − μ.  '
    'Evaluated at ~32 log-spaced integer lags from τ=1 to τ=N/2, '
    'generated by np.unique(np.round(np.logspace(0, log₁₀(N/2), 32))). '
    'Exact count may be slightly less than 32 after deduplication at short lags.'
).font.size = Pt(10.5)
doc.add_paragraph()

# ── 4. Scattering ─────────────────────────────────────────────────────────────
heading('4. Scattering Transform  (cols 42–177,  136 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Morlet wavelet filterbank with J=8 octaves and Q=2 filters per octave '
    '(16 filters total). Each filter ψ_{j,q} is a Gaussian in frequency space '
    'centred at ξ_{j,q} = 0.5/2^(j+q/Q).'
).font.size = Pt(10.5)
add_table(
    headers=['Group', 'Dim', 'Formula'],
    rows=[
        ('S1 (cols 42–57)', '16', 'S1[j,q] = mean_t |I * ψ_{j,q}|  — 1st-order mean modulus'),
        ('S2 (cols 58–177)', '120', 'S2[k1,k2] = mean_t ||I*ψ_{k1}|*ψ_{k2}|  — 2nd-order, all k2>k1 pairs'),
    ],
    col_widths=[1.6, 0.55, 4.45]
)
doc.add_paragraph()

# ── 5. PSD ────────────────────────────────────────────────────────────────────
heading('5. Power Spectral Density  (cols 178–209,  32 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'PSD(f) = |FFT(δI)|²/N.  '
    'Binned into 32 log-spaced frequency bins from f_min to f_Nyquist; '
    'each feature is the mean PSD within that bin.'
).font.size = Pt(10.5)
doc.add_paragraph()

# ── 6. Transit ────────────────────────────────────────────────────────────────
heading('6. Transit Fractions  (cols 210–214,  5 features)', level=2)
add_table(
    headers=['Col', 'Threshold', 'Definition'],
    rows=[
        ('210', '≥ 80% of peak', 'Fraction of bins where I(t) ≥ 0.80 · max(I)'),
        ('211', '≥ 60% of peak', 'Fraction of bins where I(t) ≥ 0.60 · max(I)'),
        ('212', '≥ 40% of peak', 'Fraction of bins where I(t) ≥ 0.40 · max(I)'),
        ('213', '≥ 20% of peak', 'Fraction of bins where I(t) ≥ 0.20 · max(I)'),
        ('214', '≥ 10% of peak', 'Fraction of bins where I(t) ≥ 0.10 · max(I)'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 7. Exp ACF fit ────────────────────────────────────────────────────────────
heading('7. Exponential ACF Fit  (cols 215–217,  3 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Vectorised OLS fit of  log G(τ) = log A − τ/τ_D  '
    'over all log-spaced ACF lags simultaneously.'
).font.size = Pt(10.5)
add_table(
    headers=['Col', 'Name', 'Description'],
    rows=[
        ('215', 'A',      'Amplitude at lag 0 from exponential fit'),
        ('216', 'tau_D',  'Diffusion time τ_D = 1/(−slope)  [bins]'),
        ('217', 'RMSE',   'Root-mean-square residual of fit vs measured ACF'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 8. Stretched-exp ACF ─────────────────────────────────────────────────────
heading('8. Stretched-Exponential ACF Fit  (cols 218–220,  3 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Vectorised log-log OLS fit of  '
    'log(−log(G/G0)) = β·log(τ) + c  '
    '(equivalent to G(τ) = G0·exp(−(τ/τ_D)^β)).'
).font.size = Pt(10.5)
add_table(
    headers=['Col', 'Name', 'Description'],
    rows=[
        ('218', 'tau_D', 'Characteristic decay time  [bins]'),
        ('219', 'beta',  'Stretching exponent β (clipped to [0.05, 5])'),
        ('220', 'RMSE',  'Root-mean-square residual of stretched-exp fit'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 9. ACF decay percentiles ─────────────────────────────────────────────────
heading('9. ACF Decay Percentiles  (cols 221–224,  4 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Linear interpolation between adjacent log-spaced lags to find τ such that '
    'G(τ) = frac × G0.  Capped at the last available lag if the ACF never decays '
    'to that level.'
).font.size = Pt(10.5)
add_table(
    headers=['Col', 'Fraction', 'Description'],
    rows=[
        ('221', '90%', 'Lag where G(τ) = 0.90 · G0'),
        ('222', '75%', 'Lag where G(τ) = 0.75 · G0'),
        ('223', '25%', 'Lag where G(τ) = 0.25 · G0'),
        ('224', '10%', 'Lag where G(τ) = 0.10 · G0'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 10. Short lags ────────────────────────────────────────────────────────────
heading('10. Short Linear Lags  (cols 225–234,  10 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'ACF values at short integer lags that are NOT already included in the '
    'log-spaced set (lags 1–20 minus those in LAGS_CURR). '
    'For dt=1ms (N=4096): lags {5, 8, 10, 11, 13, 14, 16, 17, 18, 20}.'
).font.size = Pt(10.5)
doc.add_paragraph()

# ── 11. Threshold crossing ────────────────────────────────────────────────────
heading('11. Threshold Crossing Statistics  (cols 235–254,  20 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'For each of 5 amplitude thresholds {80,60,40,20,10}% of peak, '
    '4 statistics are computed from the burst structure above that threshold:'
).font.size = Pt(10.5)
add_table(
    headers=['Stat', 'Formula', 'Meaning'],
    rows=[
        ('n_bursts/N',    '(number of threshold crossings) / N', 'Crossing rate'),
        ('max_burst/N',   '(longest contiguous run above threshold) / N', 'Longest burst, normalised'),
        ('std_burst/N',   'std(burst durations) / N', 'Burst duration variability'),
        ('mean_inter/N',  'mean(inter-burst intervals) / N', 'Mean waiting time, normalised'),
    ],
    col_widths=[1.5, 2.5, 2.6]
)
doc.add_paragraph()

# ── 12. Segment stats ─────────────────────────────────────────────────────────
heading('12. Multi-Scale Segment Statistics  (cols 255–269,  15 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Trace split into non-overlapping windows of size w ∈ {32, 64, 128, 512, 1024}. '
    '3 statistics per window size:'
).font.size = Pt(10.5)
add_table(
    headers=['Stat', 'Formula', 'Meaning'],
    rows=[
        ('mean_var',  'mean over segments of within-segment variance', 'Average local variability'),
        ('std_mean',  'std over segments of within-segment mean',      'Non-uniformity of mean across time'),
        ('cv_var',    'std(segment variances) / mean(segment variances)', 'Coefficient of variation of variance'),
    ],
    col_widths=[1.2, 2.9, 2.5]
)
doc.add_paragraph()

# ── 13. Higher-order cumulants ────────────────────────────────────────────────
heading('13. Higher-Order Cumulants  (cols 270–271,  2 features)', level=2)
add_table(
    headers=['Col', 'Name', 'Formula'],
    rows=[
        ('270', 'kappa3_norm', '⟨(I−μ)³⟩ / μ³  — normalised 3rd cumulant (skewness of fluctuations)'),
        ('271', 'kappa4_norm', '⟨(I−μ)⁴⟩ / μ⁴  — normalised 4th cumulant (excess kurtosis proxy)'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 14. Multi-tau variance ────────────────────────────────────────────────────
heading('14. Multi-Tau Variance  (cols 272–281,  10 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'Intensity binned into T-bin intervals; variance of binned signal normalised '
    'by mean². Bin times T ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}. '
    'Equivalent to kappa2_norm at progressively coarser time resolution — '
    'captures how photon statistics change with averaging time.'
).font.size = Pt(10.5)
doc.add_paragraph()

# ── 15. Non-stationarity ─────────────────────────────────────────────────────
heading('15. Non-Stationarity  (cols 282–285,  4 features)', level=2)
add_table(
    headers=['Col', 'Name', 'Formula / Meaning'],
    rows=[
        ('282', 'mean_ratio',   'μ(first half) / μ(second half) — drift in mean intensity'),
        ('283', 'var_ratio',    'σ²(first half) / σ²(second half) — drift in variance'),
        ('284', 'slope_norm',   'Linear slope across 16 equal segments, normalised by μ — long-term trend'),
        ('285', 'corr_halves',  'Pearson correlation between first and second half fluctuations — temporal coherence'),
    ],
    col_widths=[0.45, 1.3, 4.85]
)
doc.add_paragraph()

# ── 16. ACF log-ratio ─────────────────────────────────────────────────────────
heading('16. ACF Log-Ratio Features  (cols 286–301 (approx),  19 features)', level=2)
p = doc.add_paragraph()
p.add_run(
    'For consecutive lag index pairs (τ, τ+1) with τ = 1…19 '
    '(referencing whichever column stores ACF at that lag):\n'
    '    log_ratio(τ) = log( G(τ+1) / G(τ) )\n'
    'Captures the local decay rate of the ACF. Negative for a decaying ACF; '
    'magnitude encodes curvature (fast vs slow diffusion).'
).font.size = Pt(10.5)

# ── Footer note ───────────────────────────────────────────────────────────────
doc.add_paragraph()
doc.add_paragraph()
hr = doc.add_paragraph('─' * 80)
hr.runs[0].font.size = Pt(8)
hr.runs[0].font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)

note = doc.add_paragraph()
run = note.add_run(
    'Feature implementation: build_cache_dt.py  (functions: extract_base_features, '
    'compute_transit_features, compute_expanded_features, compute_acf_ratio_features).\n'
    'For multi-dt models (wavenet_multidt_A/B/C/D): features extracted independently '
    'per dt and concatenated → 906-dim vector.\n'
    'All features are float64; traces are float32 on disk.'
)
run.font.size = Pt(9)
run.font.italic = True
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'FCS_Feature_Pipeline.docx'
doc.save(out)
print(f"Saved {out}")
