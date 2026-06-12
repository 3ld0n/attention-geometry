"""
exp-066 — P3: depth-accumulated primacy in the MEASURED LongChat-13B kernels.

Compose the per-layer additive mixtures (built from measured per-head (C, Delta)
of exp-024) for the first k layers, k=1..40. Test:
  P3a: prim_decile(k) grows monotonically with k (Spearman >= 0.90 at mu=0).
  P3b: document-binned rank-match with Liu-20 accuracy emerges with depth
       (report the k at which rho first exceeds +0.5; not thresholded, n=5).
  P3c: effective-depth echo of the collapse: prim_decile(mu=0.5, k=40)
       ~ prim_decile(mu=0, k=20) (same d_eff=20), within the P1b band.

Composition primitives mirror exp-065 run_composition_longchat.py.
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

LIU20_idx = np.array([0, 4, 9, 14, 19])
LIU20_acc = np.array([68.6, 57.4, 55.3, 52.5, 55.0])


def layer_mixture_profile(layer, weighting="C"):
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
    M = np.zeros((N, N))
    for q in range(N):
        w = m[: q + 1][::-1]
        s = w.sum()
        if s > 0:
            M[q, : q + 1] = w / s
        else:
            M[q, q] = 1.0
    return M


def prim_decile(row):
    return float(row[: N // 10].sum())


def doc_profile_rho(row):
    bins = np.array_split(row, 20)
    prof = np.array([b.sum() for b in bins])
    rho, _ = spearmanr(prof[LIU20_idx], LIU20_acc)
    return float(rho)


# Build per-layer matrices once (C-weighted and uniform)
print("Building per-layer mixtures...")
mats = {}
for weighting in ["C", "uniform"]:
    Ms = []
    for l in range(n_layers):
        m = layer_mixture_profile(l, weighting)
        Ms.append(causal_rowstoch_from_profile(m) if m is not None else np.eye(N))
    mats[weighting] = Ms


# ---------------- P3a / P3b: layer-truncation sweep ----------------
p3 = {}
for weighting in ["C", "uniform"]:
    for mu in [0.0, 0.5]:
        prim_by_k, rho_by_k = [], []
        P = np.eye(N)
        for k in range(1, n_layers + 1):
            K = mu * np.eye(N) + (1 - mu) * mats[weighting][k - 1]
            P = P @ K
            row = P[-1]
            prim_by_k.append(prim_decile(row))
            rho_by_k.append(doc_profile_rho(row))
        ks = np.arange(1, n_layers + 1)
        rho_mono, _ = spearmanr(ks, prim_by_k)
        # depth at which U-shape rho first exceeds +0.5
        first_k = next((int(k) for k, r in zip(ks, rho_by_k) if r > 0.5), None)
        p3[f"{weighting}_mu{mu}"] = {
            "prim_decile_by_k": [round(x, 5) for x in prim_by_k],
            "rho_N_prim_decile": round(float(rho_mono), 4),
            "P3a_pass": bool(rho_mono >= 0.90),
            "doc_rho_by_k": [round(x, 3) for x in rho_by_k],
            "first_k_rho_gt_0.5": first_k,
            "prim_decile_at_k40": round(prim_by_k[-1], 5),
        }
        print(f"P3 {weighting:>7s} mu={mu}: rho(k,prim)={rho_mono:+.4f} "
              f"(P3a {'KEEP' if rho_mono>=0.90 else 'FAIL'}); "
              f"U-shape rho>+0.5 first at k={first_k}; prim@40={prim_by_k[-1]:.4f}")
OUT["P3"] = p3

# ---------------- P3c: effective-depth echo ----------------
# prim_decile(mu=0.5, k=40) [d_eff=20] vs prim_decile(mu=0, k=20) [d_eff=20]
a = p3["C_mu0.5"]["prim_decile_by_k"][39]   # mu=0.5, k=40
b = p3["C_mu0.0"]["prim_decile_by_k"][19]   # mu=0,   k=20
OUT["P3c_effective_depth_echo"] = {
    "mu0.5_k40_deff20": round(a, 5),
    "mu0_k20_deff20": round(b, 5),
    "abs_delta": round(abs(a - b), 5),
    "within_P1b_band_0.06": bool(abs(a - b) <= 0.06),
}
print(f"P3c effective-depth echo: mu=.5/k=40 -> {a:.4f}  vs  mu=0/k=20 -> {b:.4f}  "
      f"|delta|={abs(a-b):.4f} (band 0.06: {'YES' if abs(a-b)<=0.06 else 'NO'})")

with open("research/physics/experiments/exp-066_primacy_from_depth/results_longchat.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nWrote results_longchat.json")
