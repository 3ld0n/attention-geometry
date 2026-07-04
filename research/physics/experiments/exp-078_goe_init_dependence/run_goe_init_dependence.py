"""
exp-078 — GOE dependence on initialization scheme and head dimension.

See notes.md for pre-stated hypotheses (H1 moment universality, H2 orthogonal,
H3 sparsity, H4 dimension crossover, H0 null). Pure numpy, no models.

Usage:
  .venv/bin/python3 research/physics/experiments/exp-078_goe_init_dependence/run_goe_init_dependence.py

Ariel — July 4, 2026, night session.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GOE_R = 0.536
POISSON_R = 0.386
TOL = 0.02
HIDDEN = 768
N_MATRICES = 144
SEED = 42

OUT = Path(__file__).resolve().parent / "results.json"


def r_ratio(eigvals: np.ndarray) -> float:
    s = np.diff(eigvals)
    if len(s) < 3:
        return float("nan")
    lo, hi = s[:-1], s[1:]
    return float(np.mean(np.minimum(lo, hi) / (np.maximum(lo, hi) + 1e-30)))


def sample_factor(rng: np.random.Generator, scheme: str, d_k: int) -> np.ndarray:
    if scheme == "gaussian_0.02":
        return rng.normal(0, 0.02, (HIDDEN, d_k))
    if scheme == "gaussian_1.0":
        return rng.normal(0, 1.0, (HIDDEN, d_k))
    if scheme == "xavier_uniform":
        a = np.sqrt(6.0 / (HIDDEN + d_k))
        return rng.uniform(-a, a, (HIDDEN, d_k))
    if scheme == "orthogonal":
        A = rng.normal(0, 1.0, (HIDDEN, d_k))
        Q, _ = np.linalg.qr(A)
        return Q  # (HIDDEN, d_k), orthonormal columns
    if scheme == "student_t_nu3":
        return rng.standard_t(3, (HIDDEN, d_k)) * 0.02
    if scheme == "sparse_gaussian_p0.1":
        W = rng.normal(0, 0.02, (HIDDEN, d_k))
        mask = rng.uniform(size=(HIDDEN, d_k)) < 0.1
        return W * mask
    raise ValueError(scheme)


def condition_stats(rng, scheme: str, d_k: int) -> dict:
    rs = []
    for _ in range(N_MATRICES):
        WQ = sample_factor(rng, scheme, d_k)
        WK = sample_factor(rng, scheme, d_k)
        WQK = WQ.T @ WK
        M = (WQK + WQK.T) / 2.0
        rs.append(r_ratio(np.linalg.eigvalsh(M)))
    r = np.array(rs)
    mean = float(np.nanmean(r))
    dist_goe = abs(mean - GOE_R)
    dist_poi = abs(mean - POISSON_R)
    verdict = ("GOE-like" if dist_goe < TOL else
               "Poisson-like" if dist_poi < TOL else
               "GOE-tendency" if mean > (GOE_R + POISSON_R) / 2 else "Poisson-tendency")
    return {"r_mean": round(mean, 6), "r_std": round(float(np.nanstd(r)), 6),
            "se": round(float(np.nanstd(r) / np.sqrt(len(r))), 6),
            "n": len(r), "verdict": verdict,
            "dist_to_goe": round(dist_goe, 6), "dist_to_poisson": round(dist_poi, 6)}


def goe_control(rng, size: int) -> dict:
    rs = []
    for _ in range(N_MATRICES):
        A = rng.normal(0, 1.0, (size, size))
        M = (A + A.T) / np.sqrt(2 * size)
        rs.append(r_ratio(np.linalg.eigvalsh(M)))
    r = np.array(rs)
    return {"r_mean": round(float(r.mean()), 6), "r_std": round(float(r.std()), 6)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("exp-078: GOE init-scheme and dimension dependence")
    print(f"  hidden={HIDDEN}  n_matrices/condition={N_MATRICES}  seed={SEED}")
    print(f"  refs: GOE={GOE_R}  Poisson={POISSON_R}  tol={TOL}\n")

    ctrl = goe_control(rng, 64)
    print(f"estimator control (direct GOE 64x64): r={ctrl['r_mean']:.4f} ± {ctrl['r_std']:.4f}\n")

    schemes = ["gaussian_0.02", "gaussian_1.0", "xavier_uniform",
               "orthogonal", "student_t_nu3", "sparse_gaussian_p0.1"]
    print("── init schemes at d_k=64 ──")
    scheme_results = {}
    for s in schemes:
        st = condition_stats(rng, s, 64)
        scheme_results[s] = st
        print(f"  {s:22s} r={st['r_mean']:.4f} ± {st['se']:.4f}  {st['verdict']}")

    print("\n── d_k sweep (gaussian_0.02) ──")
    dk_results = {}
    for d_k in [4, 8, 16, 32, 64, 128, 256]:
        st = condition_stats(rng, "gaussian_0.02", d_k)
        dk_results[str(d_k)] = st
        print(f"  d_k={d_k:4d}  r={st['r_mean']:.4f} ± {st['se']:.4f} "
              f"(per-matrix std {st['r_std']:.4f})  {st['verdict']}")

    # ── verdicts ──
    print("\n=== HYPOTHESIS VERDICTS ===")
    h1 = all(scheme_results[s]["dist_to_goe"] < TOL
             for s in ["gaussian_0.02", "gaussian_1.0", "xavier_uniform", "student_t_nu3"])
    print(f"H1 (moment universality: gaussian x2, xavier, student-t all GOE): "
          f"{'CONFIRMED' if h1 else 'NOT CONFIRMED'}")
    h2 = scheme_results["orthogonal"]["dist_to_goe"] < TOL
    print(f"H2 (orthogonal GOE-like, committed prediction): "
          f"{'CONFIRMED' if h2 else 'NOT CONFIRMED'} "
          f"(r={scheme_results['orthogonal']['r_mean']:.4f})")
    h3 = scheme_results["sparse_gaussian_p0.1"]["r_mean"] < 0.516
    print(f"H3 (sparse p=0.1 degrades below 0.516): "
          f"{'CONFIRMED' if h3 else 'NOT CONFIRMED'} "
          f"(r={scheme_results['sparse_gaussian_p0.1']['r_mean']:.4f})")
    means = {k: v["r_mean"] for k, v in dk_results.items()}
    drift = max(means.values()) - min(means.values())
    stds = [dk_results[k]["r_std"] for k in sorted(dk_results, key=int)]
    h4_mean_flat = all(abs(v - GOE_R) < TOL for k, v in means.items() if int(k) >= 8)
    h4_std_grows = stds[0] > stds[-1]
    h4 = h4_mean_flat and h4_std_grows
    print(f"H4 (mean flat at d_k>=8: {h4_mean_flat}; per-matrix std grows as d_k drops: "
          f"{h4_std_grows}) → {'CONFIRMED' if h4 else 'NOT CONFIRMED'} "
          f"(mean drift over sweep: {drift:.4f})")

    result = {
        "experiment": "exp-078",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "protocol": {"hidden": HIDDEN, "n_matrices": N_MATRICES, "seed": SEED,
                     "matrix": "sym W_Q^T W_K, factors (768, d_k)",
                     "references": {"GOE": GOE_R, "Poisson": POISSON_R, "tol": TOL}},
        "estimator_control_goe64": ctrl,
        "schemes_dk64": scheme_results,
        "dk_sweep_gaussian002": dk_results,
        "verdicts": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
