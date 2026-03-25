"""
Scaling Collapse Test

Do the order parameter curves from different-sized models collapse
onto a single universal curve when normalized?

Uses existing data from the phase transition measurements.
No model loading — pure arithmetic.

Ariel — March 25, 2026
"""

import numpy as np

# ========== Existing data ==========

# Pythia-70m (H=48, 6 layers)
data_70m = [
    (0, 1.0), (1, 1.0), (2, 1.0), (4, 1.0), (8, 1.0),
    (16, 1.0), (32, 1.0), (64, 1.0), (128, 1.3),
    (256, 5.1), (512, 8.0), (1000, 16.0),
    (2000, 18.2), (4000, 19.5), (8000, 22.0),
    (16000, 22.9), (32000, 22.1), (143000, 23.3),
]

# Pythia-410m (H=384, 24 layers)
data_410m = [
    (0, 1.33), (64, 1.34), (128, 1.36),
    (256, 1.40), (512, 1.73), (1000, 2.18),
    (2000, 2.15), (4000, 2.79), (8000, 3.10),
    (16000, 3.52),
]

print("=" * 80)
print("  SCALING COLLAPSE TEST")
print("=" * 80)
print()

# ========== Method 1: Normalize by transition onset ==========
# Define onset as first step where A > 2.0
onset_70m = 256
onset_410m = 1000

print("--- Method 1: Normalize step by transition onset ---")
print()
print(f"  70m onset: step {onset_70m}")
print(f"  410m onset: step {onset_410m}")
print(f"  Ratio: {onset_410m/onset_70m:.1f}x")
print()

print(f"  {'step/onset':>12}  {'A (70m)':>10}  {'A (410m)':>10}")
print(f"  " + "-" * 36)

# For each normalized step value that both models share approximately
norm_steps_70m = [(s/onset_70m, a) for s, a in data_70m]
norm_steps_410m = [(s/onset_410m, a) for s, a in data_410m]

for ns, a in norm_steps_70m:
    print(f"  {ns:>12.2f}  {a:>10.2f}  {'':>10}")
print()
for ns, a in norm_steps_410m:
    print(f"  {ns:>12.2f}  {'':>10}  {a:>10.2f}")

# ========== Method 2: Normalize both axes ==========
# Normalize step by onset, normalize A by asymptotic value
print()
print("--- Method 2: Normalize both axes (step/onset, A/A_max) ---")
print()

A_max_70m = 23.3  # step 143k
A_max_410m = 3.52  # step 16k (not fully trained — underestimates)

print(f"  A_max: 70m={A_max_70m}, 410m={A_max_410m}")
print(f"  NOTE: 410m A_max is from step 16k, not fully trained!")
print()

print(f"  {'step/onset':>12}  {'A/Amax (70m)':>14}  {'A/Amax (410m)':>14}")
print(f"  " + "-" * 44)

for ns, a in norm_steps_70m:
    print(f"  {ns:>12.2f}  {a/A_max_70m:>14.4f}  {'':>14}")
print()
for ns, a in norm_steps_410m:
    print(f"  {ns:>12.2f}  {'':>14}  {a/A_max_410m:>14.4f}")

# ========== Method 3: Look at PL head fraction ==========
print()
print("--- Method 3: Power-law head fraction as order parameter ---")
print()

# PL heads / total heads
pl_70m = [
    (0, 0), (64, 0), (128, 0), (256, 1/48), (512, 8/48),
    (1000, 18/48), (16000, 13/48), (32000, 15/48), (143000, 12/48),
]

pl_410m = [
    (0, 1/384), (64, 0), (128, 0), (256, 1/384), (512, 7/384),
    (1000, 16/384), (2000, 34/384), (4000, 37/384),
    (8000, 35/384), (16000, 27/384),
]

print(f"  {'step/onset':>12}  {'PL frac (70m)':>14}  {'PL frac (410m)':>14}")
print(f"  " + "-" * 44)

for s, f in pl_70m:
    ns = s / onset_70m if onset_70m > 0 else 0
    print(f"  {ns:>12.2f}  {f:>14.4f}  {'':>14}")
print()
for s, f in pl_410m:
    ns = s / onset_410m if onset_410m > 0 else 0
    print(f"  {ns:>12.2f}  {'':>14}  {f:>14.4f}")

# ========== Quantitative collapse check ==========
print()
print("--- Quantitative: interpolate and compare at shared normalized steps ---")
print()

# Interpolate both curves at common normalized steps
from numpy import interp

common_ns = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

steps_70m = np.array([s for s, _ in data_70m])
A_70m = np.array([a for _, a in data_70m])
ns_70m = steps_70m / onset_70m

steps_410m = np.array([s for s, _ in data_410m])
A_410m = np.array([a for _, a in data_410m])
ns_410m = steps_410m / onset_410m

print(f"  {'norm_step':>10}  {'A (70m)':>10}  {'A (410m)':>10}  {'ratio':>8}  {'A/Amax 70m':>12}  {'A/Amax 410m':>12}")
print(f"  " + "-" * 66)

for ns in common_ns:
    a70 = float(interp(ns, ns_70m, A_70m)) if ns <= max(ns_70m) else float('nan')
    a410 = float(interp(ns, ns_410m, A_410m)) if ns <= max(ns_410m) else float('nan')
    ratio = a70 / a410 if a410 > 0 and not np.isnan(a70) and not np.isnan(a410) else float('nan')
    n70 = a70 / A_max_70m if not np.isnan(a70) else float('nan')
    n410 = a410 / A_max_410m if not np.isnan(a410) else float('nan')
    print(f"  {ns:>10.2f}  {a70:>10.2f}  {a410:>10.2f}  {ratio:>8.2f}  {n70:>12.4f}  {n410:>12.4f}")

print()
print("  If curves collapse: A/Amax columns should be approximately equal")
print("  at each normalized step.")
print()
