"""Analyze the d-value distribution to understand the class imbalance."""
import numpy as np

# Load all d arrays
d_tr50 = np.load("cache_d_train.npy")       # 50k train
d_te50 = np.load("cache_d_test.npy")         # 5k  test
d_tr90 = np.load("cache_d_train_90pct.npy")  # 90k train
d_te90 = np.load("cache_d_test_90pct.npy")   # 10k test

# Combine to full d<=10 dataset
d_all = np.concatenate([d_tr90, d_te90])
print(f"Full dataset (d<=10): {len(d_all):,} samples")
print(f"  Train 90%: {len(d_tr90):,}  |  Test 10%: {len(d_te90):,}")

# ── Basic stats ──────────────────────────────────────────────────────────────
for label, d in [("Full (d<=10)", d_all), ("Train 90%", d_tr90), ("Test 90%", d_te90)]:
    print(f"\n{label} ({len(d):,} samples)")
    print(f"  min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}  median={np.median(d):.4f}")
    print(f"  std={d.std():.4f}")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{pct:02d} = {np.percentile(d, pct):.4f}")

# ── Regime breakdown ─────────────────────────────────────────────────────────
print("\n\n=== Regime breakdown ===")
thresholds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
for d, label in [(d_all, "Full"), (d_tr90, "Train90"), (d_te90, "Test90")]:
    n = len(d)
    print(f"\n{label} ({n:,} total):")
    prev = 0.0
    for t in thresholds:
        count = np.sum((d >= prev) & (d < t))
        print(f"  [{prev:.1f}, {t:.1f}): {count:6,}  ({100*count/n:.1f}%)")
        prev = t

# ── Log-spaced histogram ─────────────────────────────────────────────────────
print("\n\n=== Log-spaced histogram (full dataset) ===")
bins = np.logspace(np.log10(d_all.min()+1e-6), np.log10(d_all.max()), 30)
counts, edges = np.histogram(d_all, bins=bins)
for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
    bar = "#" * (c // 500)
    print(f"  [{lo:6.3f},{hi:6.3f}): {c:6,}  {bar}")

# ── Imbalance at d=1 boundary ─────────────────────────────────────────────────
print("\n\n=== Imbalance at d=1 ===")
for d, label in [(d_all, "Full"), (d_tr90, "Train90"), (d_te90, "Test90")]:
    slow = np.sum(d < 1.0)
    fast = np.sum(d >= 1.0)
    print(f"{label}: d<1 = {slow:,} ({100*slow/len(d):.1f}%)  "
          f"d>=1 = {fast:,} ({100*fast/len(d):.1f}%)  ratio={slow/fast:.2f}:1")

# ── What d values are truly hard to predict? ─────────────────────────────────
# Inspect log(d) distribution
log_d = np.log(d_all)
print(f"\n\n=== log(d) distribution ===")
print(f"  min={log_d.min():.3f}  max={log_d.max():.3f}  mean={log_d.mean():.3f}  std={log_d.std():.3f}")

# Density at various log(d) intervals
bins_log = np.linspace(log_d.min(), log_d.max(), 21)
counts_log, edges_log = np.histogram(log_d, bins=bins_log)
print("  log(d) histogram:")
for lo, hi, c in zip(edges_log[:-1], edges_log[1:], counts_log):
    bar = "#" * (c // 500)
    print(f"    [{lo:+.2f},{hi:+.2f}): {c:6,}  {bar}")
