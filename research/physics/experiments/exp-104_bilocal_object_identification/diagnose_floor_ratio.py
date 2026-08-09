"""exp-104 diagnostic — how badly does the constant term dominate?

Not a hypothesis test. Characterizes the saved lag profiles to determine whether
a floor-aware estimator for Delta_G is feasible at the available SNR, which is
what exp-105's design has to be built around.

For each object and head:
  floor  := median profile over the far tail [400, 511]
  signal := profile(FIT_LO) - profile(FIT_HI)   (the variation carrying Delta)
  ratio  := floor / signal

A ratio of R means the Delta-bearing variation is ~1/R of the total profile
amplitude, so relative noise must be well below 1/R for Delta to be recoverable.
Noise is estimated from the high-lag scatter of the profile about a local median.

Ariel — August 8, 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIT_LO, FIT_HI = 8, 256
TAIL_LO, TAIL_HI = 400, 511
OBJECTS = ("A", "G_out", "G_K", "G_cos")


def local_noise(p: np.ndarray) -> float:
    """Relative scatter of the profile about a 9-point running median, high lags."""
    seg = p[TAIL_LO:TAIL_HI + 1].astype(float)
    if len(seg) < 12 or not np.all(np.isfinite(seg)):
        return np.nan
    k = 9
    med = np.array([np.median(seg[max(0, i - k // 2):i + k // 2 + 1])
                    for i in range(len(seg))])
    resid = seg - med
    scale = np.abs(seg).mean()
    return float(np.std(resid) / scale) if scale > 0 else np.nan


def main() -> None:
    prof = np.load(HERE / "profiles_gpt2.npz")
    heads = json.loads((HERE / "results_gpt2.json").read_text())["heads"]

    conformal = [h for h in heads
                 if h["r2_A"] is not None and h["r2_A"] >= 0.90
                 and h["delta_A"] is not None and h["delta_A"] >= 0.05]
    syk_near = [h for h in conformal if abs(h["delta_A"] - 0.25) <= 0.05]

    print("Floor domination and SNR, per object")
    print("ratio = floor / (Delta-bearing variation across the fit window)")
    print("noise = relative scatter of the profile at high lag")
    print("recoverable if noise << 1/ratio\n")

    for label, subset in (("ALL", heads), ("CONFORMAL", conformal), ("SYK-NEAR", syk_near)):
        if not subset:
            continue
        print(f"[{label}] n={len(subset)}")
        print(f"  {'object':8s} {'floor/signal':>13s} {'noise':>10s} "
              f"{'1/ratio':>10s} {'verdict':>14s}")
        for name in OBJECTS:
            ratios, noises = [], []
            for h in subset:
                p = prof[name][h["layer"], h["head"]].astype(float)
                floor = float(np.median(p[TAIL_LO:TAIL_HI + 1]))
                signal = float(p[FIT_LO] - p[FIT_HI])
                if signal != 0 and np.isfinite(floor):
                    ratios.append(abs(floor / signal))
                noises.append(local_noise(p))
            r = np.median([x for x in ratios if np.isfinite(x)]) if ratios else np.nan
            nz = np.nanmedian(noises)
            inv = 1.0 / r if r and np.isfinite(r) and r > 0 else np.nan
            if not np.isfinite(r) or not np.isfinite(nz):
                verdict = "—"
            elif nz < 0.1 * inv:
                verdict = "recoverable"
            elif nz < inv:
                verdict = "marginal"
            else:
                verdict = "NOT recoverable"
            print(f"  {name:8s} {r:>13.1f} {nz:>10.2e} {inv:>10.2e} {verdict:>14s}")
        print()

    # How much averaging would a marginal case need?
    print("Required n_inputs scaling (noise ~ 1/sqrt(n), currently n=50):")
    for label, subset in (("SYK-NEAR", syk_near),):
        for name in ("G_out", "G_K"):
            ratios, noises = [], []
            for h in subset:
                p = prof[name][h["layer"], h["head"]].astype(float)
                floor = float(np.median(p[TAIL_LO:TAIL_HI + 1]))
                signal = float(p[FIT_LO] - p[FIT_HI])
                if signal != 0:
                    ratios.append(abs(floor / signal))
                noises.append(local_noise(p))
            r, nz = np.median(ratios), np.nanmedian(noises)
            target = 0.1 / r          # want noise <= 0.1 * (1/ratio)
            factor = (nz / target) ** 2 if target > 0 else np.inf
            print(f"  [{label}] {name:6s} need ~{factor * 50:,.0f} inputs "
                  f"(x{factor:,.0f} more than now)")


if __name__ == "__main__":
    main()
