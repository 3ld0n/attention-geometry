"""
exp-065 — P4: two-level composition from measured LongChat-13B per-head (C, Delta).

Structure: within layer l, heads act in parallel (additive mixture, C-weighted);
across layers, serially (matrix product of row-stochastic kernels); residual
connection = lazy walk K_l = mu*I + (1-mu)*M_l, mu scanned.

Measured per the pre-registration (notes.md P4):
  - windowed local exponent Delta_comp(u) of the deep-query composite profile
  - mu-robustness, C-weights vs uniform
  - boundary (primacy) pileup mass
  - descriptive comparison: composite profile shape vs Liu et al. accuracy U-shape

Data: archived exp-024 census (1,343 conformal heads, 40 layers, seq_len 512).

Ariel — June 12, 2026.
"""

import json
import numpy as np
from scipy.stats import spearmanr

N = 512
DATA = "research/physics/experiments/exp-024_bcft_litm/results/bcft_longchat_measurements.json"
OUT = {"N": N}

with open(DATA) as f:
    census = json.load(f)
heads = census["heads"]
n_layers = census["n_layers"]

# Liu et al. accuracy for LongChat-13B-16K, 20 documents (key at position 0..19)
LIU20_idx = np.array([0, 4, 9, 14, 19])
LIU20_acc = np.array([68.6, 57.4, 55.3, 52.5, 55.0])


def layer_mixture_profile(layer, weighting="C"):
    """Additive within-layer mixture profile m(u), u = 1..N."""
    u = np.arange(1, N + 1, dtype=float)
    m = np.zeros(N)
    hs = [h for h in heads if h["layer"] == layer]
    if not hs:
        return None
    for h in hs:
        w = h["C"] if weighting == "C" else 1.0
        m += w * u ** (-2 * h["delta"])
    return m / m.sum()


def causal_rowstoch_from_profile(m):
    """Causal row-stochastic matrix with row q given by truncated/renormalized m."""
    M = np.zeros((N, N))
    for q in range(N):
        w = m[: q + 1][::-1]          # key j at lag q-j+1 -> m[q-j]
        s = w.sum()
        if s > 0:
            M[q, : q + 1] = w / s
        else:
            M[q, q] = 1.0
    return M


def local_exponent_curve(row, windows):
    """Windowed log-log slope of the deep-query lookback profile, per window."""
    q = len(row) - 1
    lags = q - np.arange(len(row)) + 1.0
    out = {}
    for lo, hi in windows:
        msk = (lags >= lo) & (lags <= hi) & (row > 0)
        x, y = np.log(lags[msk]), np.log(row[msk])
        A = np.column_stack([x, np.ones(len(x))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[f"{lo}-{hi}"] = round(-float(coef[0]) / 2, 4)   # Delta_comp = a_comp/2
    return out


# Precompute per-layer matrices (C-weighted and uniform)
print("Building per-layer mixtures...")
mats = {}
for weighting in ["C", "uniform"]:
    Ms = []
    for l in range(n_layers):
        m = layer_mixture_profile(l, weighting)
        Ms.append(causal_rowstoch_from_profile(m) if m is not None else np.eye(N))
    mats[weighting] = Ms

# Reference: pure additive mixture across ALL heads (the April-style k=1 term)
u = np.arange(1, N + 1, dtype=float)
m_all = np.zeros(N)
for h in heads:
    m_all += h["C"] * u ** (-2 * h["delta"])
m_all /= m_all.sum()
add_curve = local_exponent_curve(
    np.concatenate([m_all[::-1], [0]])[: N], [(8, 32), (16, 64), (32, 128), (64, 256)]
)
# build a "row" for the additive profile: deep query attending back with profile m_all
row_add = np.zeros(N)
row_add[: N - 1] = m_all[: N - 1][::-1]   # key j at lag N-1-j... construct directly:
q = N - 1
row_add = np.zeros(N)
for j in range(N):
    lag = q - j + 1
    row_add[j] = m_all[lag - 1]
row_add /= row_add.sum()
add_curve = local_exponent_curve(row_add, [(8, 32), (16, 64), (32, 128), (64, 256)])
OUT["additive_only_Delta_eff"] = add_curve
print(f"Additive-only (k=1 reference) Delta_eff by window: {add_curve}")

# ---------------- Composition scan ----------------
windows = [(8, 32), (16, 64), (32, 128), (64, 256)]
scan = {}
for weighting in ["C", "uniform"]:
    for mu in [0.0, 0.25, 0.5, 0.75, 0.9]:
        P = np.eye(N)
        for l in range(n_layers):
            K = mu * np.eye(N) + (1 - mu) * mats[weighting][l]
            P = P @ K
        row = P[-1]
        curve = local_exponent_curve(row, windows)
        dec = N // 10
        oldest = float(row[:dec].sum())
        middle = float(row[(N - dec) // 2: (N - dec) // 2 + dec].sum())
        # document-binned profile for U-shape comparison (20 bins)
        bins = np.array_split(row, 20)
        doc_profile = np.array([b.sum() for b in bins])
        rho, p = spearmanr(doc_profile[LIU20_idx], LIU20_acc)
        scan[f"{weighting}_mu{mu}"] = {
            "Delta_comp_by_window": curve,
            "oldest_decile_mass": round(oldest, 4),
            "middle_decile_mass": round(middle, 4),
            "primacy_ratio": round(oldest / max(middle, 1e-12), 2),
            "doc_profile_20bins": [round(float(x), 5) for x in doc_profile],
            "spearman_vs_liu20": (round(float(rho), 3), round(float(p), 4)),
        }
        print(f"{weighting:>7s} mu={mu:.2f}: Delta_comp {curve}  "
              f"primacy_ratio={scan[f'{weighting}_mu{mu}']['primacy_ratio']:>8.2f}  "
              f"rho_vs_acc={rho:+.3f}")

OUT["scan"] = scan

# ---------------- Registered P4 evaluation ----------------
# Primary: mu=0, C-weights, Delta_comp(u) in [0.12, 0.22] across u in [32, 256]
prim = scan["C_mu0.0"]["Delta_comp_by_window"]
prim_vals = [prim["32-128"], prim["64-256"]]
p4_primary = all(0.12 <= v <= 0.22 for v in prim_vals)
# Robustness: max-min over mu at the 32-128 window < 0.05
mu_vals = [scan[f"C_mu{m}"]["Delta_comp_by_window"]["32-128"] for m in [0.0, 0.25, 0.5, 0.75]]
mu_spread = max(mu_vals) - min(mu_vals)
OUT["P4_registered"] = {
    "primary_window_values": prim_vals,
    "primary_pass": bool(p4_primary),
    "mu_values_32-128": mu_vals,
    "mu_spread": round(float(mu_spread), 4),
    "mu_robust": bool(mu_spread < 0.05),
    "references": {"accuracy_fitted": 0.1711, "april_additive_dx20": 0.1655},
}
print(f"\nP4 registered: primary (mu=0, u in 32-256) = {prim_vals}, pass={p4_primary}")
print(f"               mu spread at 32-128 = {mu_spread:.4f}, robust={mu_spread < 0.05}")

with open("research/physics/experiments/exp-065_composition_law/results_longchat.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nWrote results_longchat.json")
