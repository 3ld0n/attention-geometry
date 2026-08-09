"""exp-105 validation gate — run and judged BEFORE the estimators touch model data.

Pass criteria V1-V4 are committed in notes.md (commit 6a5b73c). This script
only evaluates them; it does not choose them.

M1 is validated on synthetic PROFILES: P(dx) = c + b*dx^(-2*Delta) + noise.
M2 is validated on synthetic MATRICES: H(i,j) = c + f(i) + f(j) + b*|i-j|^(-2*Delta),
which is the structure melonic eq. (2.1) predicts, so the test is whether double
centering recovers Delta from exactly the contamination the derivation claims to
annihilate.

Ariel — August 8, 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from estimator import (FIT_HI, FIT_LO, Fit, double_center, kit, m0_census,
                       m1_three_param, m2_centered)

HERE = Path(__file__).resolve().parent
SEQ_LEN = kit.SEQ_LEN
DEEP_LO = kit.DEEP_LO

DELTAS = (0.10, 0.25, 0.375, 0.50, 0.75, 1.00)
RATIOS = (0.0, 0.5, 1.0, 4.0, 10.0, 50.0)
NOISES = (0.0, 3e-3, 6e-3, 1e-2, 2e-2, 3e-2)
N_REAL = 20
SEED = 105
DETAIL_NOISE = 6e-3          # noise level shown in the per-cell table

# The regime the real data occupies (exp-104 diagnostic: ratio 0.6-6,
# noise ~6e-3), widened to 1e-2 so the envelope covers the data with margin.
IN_REGIME = lambda ratio, noise: ratio <= 10.0 and noise <= 6e-3  # noqa: E731

V1_TOL, V2_DRIFT, V3_FRAC, V4_TOL = 0.03, 0.03, 0.80, 0.01


def synth_profile(delta: float, ratio: float, noise: float,
                  rng: np.random.Generator) -> np.ndarray:
    """P(dx) = c + b*dx^(-2*delta), floor set so floor/signal == ratio."""
    dx = np.arange(SEQ_LEN, dtype=float)
    dx[0] = 1.0
    b = 1.0
    signal = b * (FIT_LO ** (-2 * delta) - FIT_HI ** (-2 * delta))
    c = ratio * signal
    p = c + b * dx ** (-2 * delta)
    if noise > 0:
        p = p * (1.0 + noise * rng.standard_normal(SEQ_LEN))
    return p


def synth_matrix(delta: float, ratio: float, noise: float,
                 rng: np.random.Generator) -> np.ndarray:
    """H(i,j) = c + f(i) + f(j) + b*|i-j|^(-2*delta) + noise.

    f is a smooth random per-index term: exactly the separable contamination
    (terms 2 and 3 of eq. 2.1) that double centering must annihilate.
    """
    n = SEQ_LEN
    idx = np.arange(n)
    lag = np.abs(idx[:, None] - idx[None, :]).astype(float)
    np.fill_diagonal(lag, 1.0)

    b = 1.0
    signal = b * (FIT_LO ** (-2 * delta) - FIT_HI ** (-2 * delta))
    c = ratio * signal

    # separable term, same order as the floor so it is a real contaminant
    t = idx / n
    f = c * (0.5 + 0.5 * np.sin(2 * np.pi * (t + rng.random())))

    H = c + f[:, None] + f[None, :] + b * lag ** (-2 * delta)
    if noise > 0:
        H = H * (1.0 + noise * rng.standard_normal((n, n)))
    return 0.5 * (H + H.T)


def cell_key(d, r, nz):
    return f"d{d}_r{r}_n{nz}"


def run() -> dict:
    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}

    for delta in DELTAS:
        for ratio in RATIOS:
            for noise in NOISES:
                m1_d, m1_ok, m0_d, m2_d, m2_ok = [], [], [], [], []
                for _ in range(N_REAL):
                    p = synth_profile(delta, ratio, noise, rng)
                    f1 = m1_three_param(p)
                    m1_d.append(f1.delta if f1.delta is not None else np.nan)
                    m1_ok.append(bool(f1.ok))
                    f0 = m0_census(p)
                    m0_d.append(f0.delta if f0.delta is not None else np.nan)

                    H = synth_matrix(delta, ratio, noise, rng)
                    Hc = double_center(H, DEEP_LO, SEQ_LEN)
                    prof = kit.lag_profile(Hc[None, ...])[0]
                    f2 = m2_centered(prof)
                    m2_d.append(f2.delta if f2.delta is not None else np.nan)
                    m2_ok.append(bool(f2.ok))

                def med_where(vals, flags):
                    sel = [v for v, f in zip(vals, flags)
                           if f and v is not None and np.isfinite(v)]
                    return float(np.median(sel)) if sel else None

                results[cell_key(delta, ratio, noise)] = {
                    "delta_true": delta, "ratio": ratio, "noise": noise,
                    "m1_median": float(np.nanmedian(m1_d)) if np.any(~np.isnan(m1_d)) else None,
                    "m1_ok_frac": float(np.mean(m1_ok)),
                    "m1_median_ok": med_where(m1_d, m1_ok),
                    "m2_median": float(np.nanmedian(m2_d)) if np.any(~np.isnan(m2_d)) else None,
                    "m2_ok_frac": float(np.mean(m2_ok)),
                    "m2_median_ok": med_where(m2_d, m2_ok),
                    "m0_median": float(np.nanmedian(m0_d)) if np.any(~np.isnan(m0_d)) else None,
                }
    return results


def judge(res: dict) -> dict:
    verdict = {}

    for m in ("m1", "m2"):
        # V1 accuracy in-regime
        errs, fails = [], []
        for cell in res.values():
            if not IN_REGIME(cell["ratio"], cell["noise"]):
                continue
            est = cell[f"{m}_median"]
            if est is None:
                fails.append(cell)
                continue
            err = abs(est - cell["delta_true"])
            errs.append(err)
            if err > V1_TOL:
                fails.append(cell)
        v1 = bool(errs) and float(np.median(errs)) <= V1_TOL
        v1_max = float(np.max(errs)) if errs else None

        # V2 no floor-induced drift
        drifts = []
        for delta in DELTAS:
            for noise in NOISES:
                if noise > 6e-3:
                    continue
                seq = [res[cell_key(delta, r, noise)][f"{m}_median"]
                       for r in RATIOS if r <= 10.0]
                seq = [s for s in seq if s is not None]
                if len(seq) >= 2:
                    drifts.append(max(seq) - min(seq))
        v2 = bool(drifts) and float(np.max(drifts)) <= V2_DRIFT

        # V3 calibrated failure: cells failing V1 must report not-ok
        loud = []
        for cell in res.values():
            est = cell[f"{m}_median"]
            bad = est is None or abs(est - cell["delta_true"]) > V1_TOL
            if bad:
                loud.append(1.0 - cell[f"{m}_ok_frac"])
        v3 = (float(np.mean(loud)) >= V3_FRAC) if loud else True

        # Supplementary, NOT pre-registered: accuracy and drift restricted to
        # cells the estimator itself accepts. This is the operationally relevant
        # question ("where it says it works, does it work?") and is reported
        # alongside the pre-registered criteria rather than in place of them.
        errs_ok = []
        for cell in res.values():
            if not IN_REGIME(cell["ratio"], cell["noise"]):
                continue
            est = cell[f"{m}_median_ok"]
            if est is not None and cell[f"{m}_ok_frac"] > 0:
                errs_ok.append(abs(est - cell["delta_true"]))
        drifts_ok = []
        for delta in DELTAS:
            for noise in NOISES:
                if noise > 6e-3:
                    continue
                seq = [res[cell_key(delta, r, noise)][f"{m}_median_ok"]
                       for r in RATIOS if r <= 10.0]
                seq = [s for s in seq if s is not None]
                if len(seq) >= 2:
                    drifts_ok.append(max(seq) - min(seq))

        verdict[m] = {"V1": v1, "V1_max_err": v1_max, "V2": v2,
                      "V2_max_drift": float(np.max(drifts)) if drifts else None,
                      "V3": v3, "V3_loud_frac": float(np.mean(loud)) if loud else None,
                      "n_fail_cells": len(fails),
                      "V1c_max_err": float(np.max(errs_ok)) if errs_ok else None,
                      "V1c_median_err": float(np.median(errs_ok)) if errs_ok else None,
                      "V2c_max_drift": float(np.max(drifts_ok)) if drifts_ok else None,
                      "n_accepted_cells": len(errs_ok)}

    # V4: with zero floor, M1 must match the census estimator
    diffs = []
    for delta in DELTAS:
        for noise in NOISES:
            if noise > 6e-3:
                continue
            cell = res[cell_key(delta, 0.0, noise)]
            if cell["m1_median"] is not None and cell["m0_median"] is not None:
                diffs.append(abs(cell["m1_median"] - cell["m0_median"]))
    verdict["m1"]["V4"] = bool(diffs) and float(np.max(diffs)) <= V4_TOL
    verdict["m1"]["V4_max_diff"] = float(np.max(diffs)) if diffs else None
    verdict["m2"]["V4"] = None      # M2 operates on centered data; V4 N/A

    verdict["m1"]["PASS"] = all(verdict["m1"][k] for k in ("V1", "V2", "V3", "V4"))
    verdict["m2"]["PASS"] = all(verdict["m2"][k] for k in ("V1", "V2", "V3"))
    return verdict


def main() -> None:
    print("exp-105 validation gate — synthetic ground truth\n")
    res = run()
    ver = judge(res)
    (HERE / "validation_results.json").write_text(
        json.dumps({"cells": res, "verdict": ver}, indent=1))

    def num(x, nd=4):
        return "—" if x is None else f"{x:.{nd}f}"

    def flag(b):
        return "PASS" if b else "FAIL"

    for m, name in (("m1", "M1 conditioned 3-param"), ("m2", "M2 double centering")):
        v = ver[m]
        print(f"[{name}]")
        print(f"  V1 accuracy in-regime (|err| <= {V1_TOL}):     "
              f"{flag(v['V1'])}   max err {num(v['V1_max_err'])}")
        print(f"  V2 no floor-induced drift (<= {V2_DRIFT}):     "
              f"{flag(v['V2'])}   max drift {num(v['V2_max_drift'])}")
        print(f"  V3 calibrated failure (>= {V3_FRAC}):          "
              f"{flag(v['V3'])}   loud frac {num(v['V3_loud_frac'], 3)}")
        if v.get("V4") is not None:
            print(f"  V4 matches census at zero floor (<= {V4_TOL}): "
                  f"{flag(v['V4'])}   max diff {num(v['V4_max_diff'])}")
        print(f"  ==> pre-registered verdict: {flag(v['PASS'])}")
        print(f"  [supplementary, not pre-registered] restricted to cells the "
              f"estimator accepts ({v['n_accepted_cells']} cells):")
        print(f"      median err {num(v['V1c_median_err'])}   "
              f"max err {num(v['V1c_max_err'])}   "
              f"max drift {num(v['V2c_max_drift'])}\n")

    print("in-regime cell detail (ratio <= 10, noise <= 6e-3):")
    print(f"  {'Δ_true':>7s} {'ratio':>6s} {'noise':>8s} {'M1':>8s} {'M2':>8s} {'census':>8s}")
    for delta in DELTAS:
        for ratio in RATIOS:
            for noise in NOISES:
                if not IN_REGIME(ratio, noise) or noise != DETAIL_NOISE:
                    continue
                c = res[cell_key(delta, ratio, noise)]
                f = lambda x: "—" if x is None else f"{x:.4f}"   # noqa: E731
                print(f"  {delta:>7.3f} {ratio:>6.1f} {noise:>8.0e} "
                      f"{f(c['m1_median']):>8s} {f(c['m2_median']):>8s} "
                      f"{f(c['m0_median']):>8s}")


if __name__ == "__main__":
    main()
