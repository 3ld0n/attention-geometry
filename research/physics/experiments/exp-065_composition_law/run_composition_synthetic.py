"""
exp-065 — Synthetic verification of the composition law (P1, P2, P3).

Law under test (derived first, see notes/2026-06-12_composition_law_derivation.md):
for causal row-stochastic power-law kernels p(u) ~ u^{-2D}, defects d = 1 - 2D
combine under composition as d_comp = max(d1, d2, d1+d2).

P1: addition branch via exact lattice convolution.
P2: flat point (sum d = 1) and boundary pileup (sum d > 1).
P3: scrambling depth L* = 1/(1-2D) via row-stochastic matrix powers.

Ariel — June 12, 2026.
"""

import json
import numpy as np

rng = np.random.default_rng(0)
OUT = {}


def powerlaw_kernel(n, a):
    """Normalized discrete kernel k(u) ~ u^-a on u in [1, n]."""
    u = np.arange(1, n + 1, dtype=float)
    k = u ** (-a)
    return k / k.sum()


def windowed_slope(profile, lo, hi):
    """OLS slope of log(profile) vs log(u) on the window [lo, hi]."""
    u = np.arange(1, len(profile) + 1, dtype=float)
    m = (u >= lo) & (u <= hi) & (profile > 0)
    x, y = np.log(u[m]), np.log(profile[m])
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0]  # = -a_comp for a decaying power law


# ---------------- P1: addition law ----------------
N = 4096
grid = np.round(np.arange(0.10, 0.91, 0.10), 2)  # values of 2D
pairs, results_p1 = [], []
for a in grid:
    for b in grid:
        d1, d2 = 1 - a, 1 - b
        if d1 + d2 >= 0.9:           # pre-registered restriction
            continue
        pairs.append((a, b))

for a, b in pairs:
    k1, k2 = powerlaw_kernel(N, a), powerlaw_kernel(N, b)
    comp = np.convolve(k1, k2)       # exact lattice convolution
    # bulk window: away from the u~1 endpoint and the support edge
    slope = windowed_slope(comp, lo=64, hi=N // 4)
    a_pred = a + b - 1               # = 2D_comp predicted
    results_p1.append({
        "a": a, "b": b,
        "pred_2Dcomp": round(a_pred, 4),
        "meas_2Dcomp": round(-slope, 4),
        "abs_err": round(abs(-slope - a_pred), 4),
    })

errs = np.array([r["abs_err"] for r in results_p1])
frac_pass = float(np.mean(errs <= 0.04))
OUT["P1"] = {
    "n_pairs": len(results_p1),
    "frac_within_0.04": round(frac_pass, 4),
    "median_abs_err": round(float(np.median(errs)), 4),
    "max_abs_err": round(float(errs.max()), 4),
    "verdict": "KEEP" if frac_pass >= 0.90 else ("KILL" if frac_pass < 0.80 else "MARGINAL"),
    "pairs": results_p1,
}
print(f"P1 addition law: {len(results_p1)} pairs, "
      f"{100*frac_pass:.1f}% within 0.04, median |err| = {np.median(errs):.4f}, "
      f"max = {errs.max():.4f}  -> {OUT['P1']['verdict']}")


# ---------------- P2a: flat point ----------------
# d1 + d2 = 1  <=>  (1-a) + (1-b) = 1  <=>  b = 1 - a
flat_cases = []
for a in [0.3, 0.5, 0.7]:
    k1, k2 = powerlaw_kernel(N, a), powerlaw_kernel(N, 1.0 - a)
    comp = np.convolve(k1, k2)
    slope = windowed_slope(comp, lo=64, hi=N // 4)
    flat_cases.append({"a": round(a, 2), "b": round(1.0 - a, 2),
                       "meas_2Dcomp": round(-slope, 4)})
flat_ok = all(abs(c["meas_2Dcomp"]) <= 0.05 for c in flat_cases)
OUT["P2a"] = {"cases": flat_cases, "verdict": "KEEP" if flat_ok else "FAIL"}
print(f"P2a flat point: {[c['meas_2Dcomp'] for c in flat_cases]} -> {OUT['P2a']['verdict']}")


# ---------------- P2b: pileup under causal truncation ----------------
def causal_rowstoch(n, a):
    """Causal row-stochastic matrix: row q attends to keys 0..q with profile (q-k)^-a, self-lag -> u=1 offset."""
    M = np.zeros((n, n))
    for q in range(n):
        u = q - np.arange(q + 1) + 1.0   # lag >= 1, including key 0
        w = u ** (-a)
        M[q, : q + 1] = w / w.sum()
    return M


n = 512
pileup_cases = []
for a, L in [(0.3, 3), (0.5, 4)]:       # sum d = L*(1-a) > 1 in both cases
    M = causal_rowstoch(n, a)
    P = np.linalg.matrix_power(M, L)
    row = P[-1]                          # deep query
    dec = n // 10
    oldest, middle = row[:dec].sum(), row[(n - dec) // 2 : (n - dec) // 2 + dec].sum()
    pileup_cases.append({"a": a, "L": L, "sum_d": round(L * (1 - a), 2),
                         "oldest_decile_mass": round(float(oldest), 4),
                         "middle_decile_mass": round(float(middle), 4),
                         "pileup": bool(oldest > middle)})
OUT["P2b"] = {"cases": pileup_cases,
              "verdict": "KEEP" if all(c["pileup"] for c in pileup_cases) else "FAIL"}
print(f"P2b pileup: {[(c['oldest_decile_mass'], c['middle_decile_mass']) for c in pileup_cases]} "
      f"-> {OUT['P2b']['verdict']}")


# ---------------- P3: scrambling depth ----------------
def positional_entropy_frac(row):
    p = row[row > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(row)))


scram = {}
for D, label in [(0.25, "delta=0.5_q4"), (0.4921, "delta=0.0158_longchat")]:
    M = causal_rowstoch(n, 2 * D)
    P = np.eye(n)
    traj = []
    for L in range(1, 41):
        P = P @ M
        traj.append(round(positional_entropy_frac(P[-1]), 4))
    scram[label] = traj

q4_hits = [L + 1 for L, e in enumerate(scram["delta=0.5_q4"]) if e >= 0.90]
lc_max = max(scram["delta=0.0158_longchat"])
p3a = bool(q4_hits and q4_hits[0] <= 3)
p3b = bool(lc_max < 0.90)
OUT["P3"] = {
    "q4_entropy_by_layer_first6": scram["delta=0.5_q4"][:6],
    "q4_first_layer_geq_90pct": q4_hits[0] if q4_hits else None,
    "longchat_entropy_at_L40": scram["delta=0.0158_longchat"][-1],
    "longchat_max_entropy_through_L40": lc_max,
    "P3a_verdict": "KEEP" if p3a else "FAIL",
    "P3b_verdict": "KEEP" if p3b else "FAIL",
}
print(f"P3a (D=0.25): entropy by layer {scram['delta=0.5_q4'][:4]}; "
      f"first >=90% at L={OUT['P3']['q4_first_layer_geq_90pct']} -> {OUT['P3']['P3a_verdict']}")
print(f"P3b (D=0.4921): max entropy through L=40 = {lc_max:.4f} -> {OUT['P3']['P3b_verdict']}")

with open("research/physics/experiments/exp-065_composition_law/results_synthetic.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nWrote results_synthetic.json")
