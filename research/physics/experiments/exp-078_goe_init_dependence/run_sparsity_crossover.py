"""
exp-078 addendum — sparsity crossover sweep for GOE level statistics.

Pre-stated hypotheses in notes.md (HA1 monotone decrease, HA2 crossover at
p* ~ 0.003-0.01, HA0 null). Seed 43 (fresh, not the main run's 42).

Usage:
  .venv/bin/python3 research/physics/experiments/exp-078_goe_init_dependence/run_sparsity_crossover.py
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
D_K = 64
N_MATRICES = 144
SEED = 43

OUT = Path(__file__).resolve().parent / "results_sparsity.json"


def r_ratio(eigvals: np.ndarray) -> float:
    s = np.diff(eigvals)
    if len(s) < 3:
        return float("nan")
    lo, hi = s[:-1], s[1:]
    return float(np.mean(np.minimum(lo, hi) / (np.maximum(lo, hi) + 1e-30)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    ps = [0.1, 0.03, 0.01, 0.003, 0.001]
    print(f"exp-078 addendum: sparsity crossover (d_k={D_K}, hidden={HIDDEN}, "
          f"n={N_MATRICES}/condition, seed={SEED})")
    results = {}
    for p in ps:
        rs = []
        for _ in range(N_MATRICES):
            WQ = rng.normal(0, 0.02, (HIDDEN, D_K)) * (rng.uniform(size=(HIDDEN, D_K)) < p)
            WK = rng.normal(0, 0.02, (HIDDEN, D_K)) * (rng.uniform(size=(HIDDEN, D_K)) < p)
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
        results[str(p)] = {
            "r_mean": round(mean, 6), "r_std": round(float(np.nanstd(r)), 6),
            "se": round(float(np.nanstd(r) / np.sqrt(len(r))), 6),
            "nonzeros_per_factor_row": round(HIDDEN * p, 1),
            "verdict": verdict}
        print(f"  p={p:6.3f} (~{HIDDEN*p:6.1f} nz/col) r={mean:.4f} ± "
              f"{results[str(p)]['se']:.4f}  {verdict}")

    means = [results[str(p)]["r_mean"] for p in ps]
    ha1 = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    crossover = next((p for p in ps if results[str(p)]["r_mean"] < 0.516), None)
    print(f"\nHA1 (monotone decrease with p): {'CONFIRMED' if ha1 else 'NOT CONFIRMED'}")
    print(f"HA2 (crossover below 0.516 in range): "
          f"{'CONFIRMED at p=' + str(crossover) if crossover else 'NOT CONFIRMED (HA0 territory)'}")

    with open(OUT, "w") as f:
        json.dump({
            "experiment": "exp-078-addendum",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
            "protocol": {"hidden": HIDDEN, "d_k": D_K, "n_matrices": N_MATRICES,
                         "seed": SEED, "fill_probs": ps},
            "results": results,
            "verdicts": {"HA1": ha1, "HA2_crossover_p": crossover},
        }, f, indent=1)
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    main()
