"""
exp-065 — Refined analysis (POST-HOC, after registered P1/P3 operationalizations
showed estimator problems; see notes.md Results for the registered outcomes).

P1-refined: test the full three-term asymptotic with PREDICTED coefficients
(no free parameters):
    comp(u) = B(1-a,1-b) u^{1-a-b} + zeta(a) u^{-b} + zeta(b) u^{-a} + O(u^{-1-min(a,b)})
The registered windowed-slope estimator was contaminated by the zeta corrections
(decaying as slowly as u^{-(1-max(a,b))}); the refined test checks the structure
the derivation actually predicts.

P3-refined: scrambling = flatness of the composite positional PROFILE (the law's
direct object), not row entropy (which cannot discriminate a shallow power law
from uniform). Track the windowed composite slope through depth for the q=4
kernel (predict: slope ~0 by L=2, then pileup/negative-decay regime) and the
LongChat-median kernel (predict: slope stays > 0 and ends near 2*0.184 = 0.368
at L=40).

Ariel — June 12, 2026.
"""

import json
import numpy as np
from scipy.special import zeta, beta as Beta

OUT = {}
N = 4096


def three_term_test(a, b, us=(64, 128, 256, 512, 1024)):
    t = np.arange(1, N + 1, dtype=float)
    comp = np.convolve(t ** (-a), t ** (-b))
    rows = []
    for u in us:
        c = comp[u - 2]  # comp[j] corresponds to lag u = j + 2
        pred = Beta(1 - a, 1 - b) * u ** (1 - a - b) \
            + zeta(a) * u ** (-b) + zeta(b) * u ** (-a)
        rows.append({"u": u, "ratio": round(float(c / pred), 6)})
    return rows


# ---------------- P1-refined: full grid, ratio test ----------------
grid = np.round(np.arange(0.10, 0.91, 0.10), 2)
worst = 1.0
n_pairs = 0
ratios_all = []
for a in grid:
    for b in grid:
        if (1 - a) + (1 - b) >= 0.9:
            continue
        n_pairs += 1
        rows = three_term_test(a, b)
        for r in rows:
            ratios_all.append(abs(r["ratio"] - 1.0))
ratios_all = np.array(ratios_all)
OUT["P1_refined"] = {
    "n_pairs": n_pairs,
    "n_checks": len(ratios_all),
    "max_abs_ratio_dev": round(float(ratios_all.max()), 6),
    "median_abs_ratio_dev": round(float(np.median(ratios_all)), 6),
    "verdict": "CONFIRMED (no-free-parameter coefficients)"
    if ratios_all.max() < 0.01 else "FAIL",
}
print(f"P1-refined: {n_pairs} pairs x 5 scales; max |ratio-1| = {ratios_all.max():.6f}, "
      f"median = {np.median(ratios_all):.6f} -> {OUT['P1_refined']['verdict']}")


# ---------------- P3-refined: composite profile slope through depth ----------------
def causal_rowstoch(n, a):
    M = np.zeros((n, n))
    for q in range(n):
        u = q - np.arange(q + 1) + 1.0
        w = u ** (-a)
        M[q, : q + 1] = w / w.sum()
    return M


def profile_slope(row, lo, hi):
    """Windowed log-log slope of the deep-query lookback profile."""
    q = len(row) - 1
    lags = q - np.arange(len(row)) + 1.0   # lag of each key position
    m = (lags >= lo) & (lags <= hi) & (row > 0)
    x, y = np.log(lags[m]), np.log(row[m])
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])   # = -a_comp


n = 512
traj = {}
for D, label in [(0.25, "q4_D0.25"), (0.4921, "longchat_D0.4921")]:
    M = causal_rowstoch(n, 2 * D)
    P = np.eye(n)
    slopes = []
    for L in range(1, 41):
        P = P @ M
        slopes.append(round(-profile_slope(P[-1], lo=8, hi=128), 4))  # = a_comp(L)
    traj[label] = slopes

# Predictions: q4: a_comp(L) = 1 - L*0.5 -> 0.5, 0.0, then pileup (negative)
#              longchat: a_comp(L) = 1 - L*0.0158 -> 0.368 at L=40
pred_lc_40 = 1 - 40 * (1 - 2 * 0.4921)
OUT["P3_refined"] = {
    "q4_acomp_by_L": traj["q4_D0.25"][:6],
    "q4_pred_by_L": [round(1 - L * 0.5, 2) for L in range(1, 7)],
    "longchat_acomp_L1_L10_L20_L40": [traj["longchat_D0.4921"][i] for i in (0, 9, 19, 39)],
    "longchat_pred_L40": round(pred_lc_40, 4),
    "longchat_meas_L40": traj["longchat_D0.4921"][-1],
}
print(f"P3-refined q4 (D=0.25):  measured a_comp by layer {traj['q4_D0.25'][:5]}")
print(f"                          predicted              {[round(1-L*0.5,2) for L in range(1,6)]}")
print(f"P3-refined LongChat:      a_comp at L=1,10,20,40 = "
      f"{[traj['longchat_D0.4921'][i] for i in (0,9,19,39)]}")
print(f"                          predicted at L=40       = {pred_lc_40:.4f}")

with open("research/physics/experiments/exp-065_composition_law/results_refined.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nWrote results_refined.json")
