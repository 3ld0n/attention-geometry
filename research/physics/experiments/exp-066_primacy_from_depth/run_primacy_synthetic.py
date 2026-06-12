"""
exp-066 — Synthetic verification of depth-accumulated primacy (P1, P2).

Mechanism (derived first; see notes/2026-06-12_primacy_from_depth_derivation.md):
position 0 is an absorbing boundary of any causal row-stochastic kernel.
Composition is a backward walk into it; primacy = absorbed mass.

P1: Law A. Lazy walk K = mu*I + (1-mu)*M over L layers. Primacy depends on
    (mu, L) only through effective depth d_eff = (1-mu)*L.
    P1a: prim_decile monotone increasing in d_eff (Spearman >= 0.95).
    P1b: curves at different mu collapse onto the mu=0 curve vs d_eff
         (median |delta| <= 0.03, p90 <= 0.06).
P2: Law B. At fixed d_eff, absolute start-window mass decreases with context N.

Composition primitives mirror exp-065. Exact lattice arithmetic.
Ariel — June 12, 2026.
"""

import json
import numpy as np
from scipy.stats import spearmanr

OUT = {}


def causal_rowstoch(n, two_delta):
    """Causal row-stochastic matrix; row q ~ (lag)^-two_delta over keys 0..q, self-lag=1."""
    M = np.zeros((n, n))
    for q in range(n):
        u = q - np.arange(q + 1) + 1.0
        w = u ** (-two_delta)
        M[q, : q + 1] = w / w.sum()
    return M


def prim_decile(row):
    n = len(row)
    return float(row[: max(1, n // 10)].sum())


def prim_window(row, W):
    return float(row[:W].sum())


# ---------------- P1: Law A (collapse + monotonicity) ----------------
N = 512
mus = [0.0, 0.25, 0.5, 0.75]
Ls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40]
p1 = {}
for D in [0.25, 0.49]:
    M = causal_rowstoch(N, 2 * D)
    # cache M^L for the integer powers we need (max 40)
    powers = {0: np.eye(N)}
    Pm = np.eye(N)
    for k in range(1, max(Ls) + 1):
        Pm = Pm @ M
        powers[k] = Pm
    pts = []  # (mu, L, d_eff, primacy)
    for mu in mus:
        K = mu * np.eye(N) + (1 - mu) * M
        P = np.eye(N)
        Kp = {0: np.eye(N)}
        Acc = np.eye(N)
        for k in range(1, max(Ls) + 1):
            Acc = Acc @ K
            Kp[k] = Acc
        for L in Ls:
            row = Kp[L][-1]
            pts.append((mu, L, (1 - mu) * L, prim_decile(row)))
    pts = np.array(pts)  # columns: mu, L, d_eff, prim
    # P1a monotonicity in d_eff
    rho_a, _ = spearmanr(pts[:, 2], pts[:, 3])
    # P1b collapse: reference = mu=0 curve (d_eff == L there)
    ref_mask = pts[:, 0] == 0.0
    ref_deff = pts[ref_mask, 2]
    ref_prim = pts[ref_mask, 3]
    order = np.argsort(ref_deff)
    ref_deff, ref_prim = ref_deff[order], ref_prim[order]
    deltas = []
    for mu, L, deff, prim in pts:
        if mu == 0.0:
            continue
        if deff < ref_deff.min() or deff > ref_deff.max():
            continue
        pred = float(np.interp(deff, ref_deff, ref_prim))
        deltas.append(abs(prim - pred))
    deltas = np.array(deltas)
    p1[f"D={D}"] = {
        "rho_deff_primacy": round(float(rho_a), 4),
        "P1a_pass": bool(rho_a >= 0.95),
        "n_collapse_points": int(len(deltas)),
        "median_abs_delta": round(float(np.median(deltas)), 4),
        "p90_abs_delta": round(float(np.percentile(deltas, 90)), 4),
        "max_abs_delta": round(float(deltas.max()), 4),
        "P1b_pass": bool(np.median(deltas) <= 0.03 and np.percentile(deltas, 90) <= 0.06),
        "primacy_vs_deff_mu0": [
            [round(float(d), 2), round(float(p), 4)]
            for d, p in zip(ref_deff, ref_prim)
        ],
    }
    print(f"P1 D={D}: rho(d_eff,prim)={rho_a:.4f} (P1a {'KEEP' if rho_a>=0.95 else 'FAIL'}); "
          f"collapse median|d|={np.median(deltas):.4f} p90={np.percentile(deltas,90):.4f} "
          f"(P1b {'KEEP' if (np.median(deltas)<=0.03 and np.percentile(deltas,90)<=0.06) else 'FAIL'})")
OUT["P1"] = p1


# ---------------- P2: Law B (context-length dilution) ----------------
W = 8
p2 = {}
for D, thresholded in [(0.49, True), (0.25, False)]:
    by_depth = {}
    for L in [8, 20]:
        Ns, win_mass, dec_mass = [], [], []
        for n in [128, 256, 512, 1024]:
            M = causal_rowstoch(n, 2 * D)
            P = np.linalg.matrix_power(M, L)
            row = P[-1]
            Ns.append(n)
            win_mass.append(prim_window(row, W))
            dec_mass.append(prim_decile(row))
        rho_w, _ = spearmanr(Ns, win_mass)
        rho_d, _ = spearmanr(Ns, dec_mass)
        by_depth[f"L={L}"] = {
            "N": Ns,
            "start_window_mass": [round(x, 5) for x in win_mass],
            "rho_N_window": round(float(rho_w), 4),
            "oldest_decile_mass": [round(x, 5) for x in dec_mass],
            "rho_N_decile": round(float(rho_d), 4),
            "pass_window_decreasing": bool(rho_w <= -0.90),
        }
    p2[f"D={D}"] = {"thresholded": thresholded, **by_depth}
    flags = [v["pass_window_decreasing"] for k, v in by_depth.items()]
    print(f"P2 D={D} (thresholded={thresholded}): "
          + "; ".join(f"{k} rho_N(window)={v['rho_N_window']:+.3f}" for k, v in by_depth.items())
          + f"  -> {'KEEP' if all(flags) else 'FAIL'}")
OUT["P2"] = p2

with open("research/physics/experiments/exp-066_primacy_from_depth/results_synthetic.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nWrote results_synthetic.json")
