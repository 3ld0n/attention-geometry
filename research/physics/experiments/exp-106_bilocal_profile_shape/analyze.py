"""exp-106 analysis: per-head detail, kill-condition checks, and the direct test of
Proposition 1's floor prediction.

Proposition 1 [EXACT]: G = ||vbar||^2 * 11^T + A Ktilde A^T, so G's lag profile has
an exactly constant additive term equal to ||vbar||^2 -- the squared norm of the
head's mean value vector. That is measurable independently of any fit, and this is
the first time the program has checked it.

Run: python analyze.py
"""
from __future__ import annotations

import json
import numpy as np

FIT_LO, FIT_HI = 8, 256
W = slice(FIT_LO, FIT_HI + 1)
R2_MIN, DELTA_MIN = 0.90, 0.05


def census_fit(profile: np.ndarray):
    lags = np.arange(FIT_LO, FIT_HI + 1)
    y = profile[FIT_LO:FIT_HI + 1]
    ok = y > 1e-15
    if ok.sum() < 5:
        return None, None
    lx, ly = np.log(lags[ok].astype(float)), np.log(y[ok])
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ c
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return float(-c[1] / 2), 1 - float(np.sum(resid ** 2)) / ss_tot


def main() -> None:
    applied = json.loads(open("applied_gpt2.json").read())
    heads = applied["heads"]
    z = np.load("profiles_forward_gpt2.npz")
    G, vbar_sq = z["G_out"], z["vbar_sq"]
    n_inputs = applied["n_inputs"]
    # profiles were accumulated (summed) over inputs; vbar_sq is a per-input mean
    Gm = G / n_inputs

    nm = json.loads(open("numerical_map.json").read())
    map_dA = np.array([r["delta_A_measured"] for r in nm["map_cb0"]])
    map_dG = np.array([r["delta_G"] for r in nm["map_cb0"]])

    conformal = [h for h in heads if h["r2_A"] and h["r2_A"] >= R2_MIN
                 and h["delta_A"] and h["delta_A"] >= DELTA_MIN]
    syk = [h for h in conformal if abs(h["delta_A"] - 0.25) <= 0.05]

    # ---------------- K3: sign of alpha, per head ----------------
    print("=== K3 check: sign of alpha in P_meas ~ alpha*P_pred + beta ===")
    for arm in ("arm1a", "arm1b", "arm2"):
        for label, subset in (("all", heads), ("conformal", conformal), ("SYK", syk)):
            a = [h[arm]["alpha"] for h in subset if h[arm]["alpha"] is not None]
            neg = sum(1 for x in a if x < 0)
            print(f"  {arm:6s} {label:10s} negative alpha: {neg}/{len(a)}"
                  f"{'   <-- K3 FIRES' if neg > len(a) / 2 else ''}")

    # ---------------- Proposition 1: the floor ----------------
    print("\n=== Proposition 1 (EXACT): G's floor should equal ||vbar||^2 ===")
    print("  L/H    Delta_A   P_G(8)      ||vbar||^2   frac_floor  "
          "P_G(256)-||vbar||^2   connected decays?")
    prop1 = []
    for h in sorted(heads, key=lambda r: (r["layer"], r["head"])):
        ell, hh = h["layer"], h["head"]
        p = Gm[ell, hh]
        mu = float(vbar_sq[ell, hh])
        frac = mu / p[FIT_LO] if p[FIT_LO] != 0 else np.nan
        conn = p - mu
        # the connected part must be non-negative-ish and decaying
        dec = bool(conn[FIT_LO] > conn[FIT_HI])
        rec = {"layer": ell, "head": hh, "delta_A": h["delta_A"],
               "P_G_8": float(p[FIT_LO]), "vbar_sq": mu,
               "frac_floor_at_lag8": float(frac),
               "connected_at_256": float(conn[FIT_HI]),
               "connected_decays": dec,
               "connected_min_in_window": float(conn[W].min())}
        prop1.append(rec)
        if h in syk or (ell, hh) in {(0, 6), (9, 0), (10, 0), (10, 6), (11, 2)}:
            tag = "SYK" if h in syk else "e105"
            print(f"  {ell:2d}/{hh:<3d} {h['delta_A']:.4f}  {p[FIT_LO]:.4e}  "
                  f"{mu:.4e}  {frac:8.4f}   {conn[FIT_HI]:+.4e}        "
                  f"{dec}   [{tag}]")

    fr = np.array([r["frac_floor_at_lag8"] for r in prop1])
    print(f"\n  fraction of P_G(8) explained by ||vbar||^2, over 144 heads:")
    print(f"    median {np.median(fr):.4f}   IQR [{np.percentile(fr,25):.4f}, "
          f"{np.percentile(fr,75):.4f}]   min {fr.min():.4f}  max {fr.max():.4f}")
    n_neg = sum(1 for r in prop1 if r["connected_min_in_window"] < 0)
    print(f"    heads where the connected part goes negative in the window: "
          f"{n_neg}/144")

    # after removing the exact floor, refit the connected part with the census
    print("\n=== Refit of the CONNECTED part (P_G - ||vbar||^2) with the census ===")
    print("  This is Proposition 1 used as an estimator: the floor is not fitted,")
    print("  it is computed. Compare against the numerical forward-model map.")
    print("  L/H    Delta_A   Delta_G(conn)  R2     map(Delta_A)  measured-map")
    refit = []
    for h in sorted(conformal, key=lambda r: (r["layer"], r["head"])):
        ell, hh = h["layer"], h["head"]
        conn = Gm[ell, hh] - float(vbar_sq[ell, hh])
        d, r2 = census_fit(conn)
        pred = float(np.interp(h["delta_A"], map_dA, map_dG))
        rec = {"layer": ell, "head": hh, "delta_A": h["delta_A"],
               "delta_G_connected": d, "r2_connected": r2,
               "map_prediction": pred,
               "residual": (d - pred) if d is not None else None}
        refit.append(rec)
        mark = " [SYK]" if h in syk else ""
        if d is None:
            print(f"  {ell:2d}/{hh:<3d} {h['delta_A']:.4f}  (no fit: profile "
                  f"non-positive){mark}")
        else:
            print(f"  {ell:2d}/{hh:<3d} {h['delta_A']:.4f}  {d:+.4f}       "
                  f"{r2:.4f}  {pred:.4f}       {d - pred:+.4f}{mark}")

    ok = [r for r in refit if r["delta_G_connected"] is not None]
    if ok:
        res = np.array([r["residual"] for r in ok])
        r2s = np.array([r["r2_connected"] for r in ok])
        print(f"\n  n fitted {len(ok)}/{len(refit)};  median R2 {np.median(r2s):.4f}")
        print(f"  median(measured - map) = {np.median(res):+.4f}   "
              f"IQR [{np.percentile(res,25):+.4f}, {np.percentile(res,75):+.4f}]")

    # ---------------- H4 (POST-HOC): exp-105's five accepted heads -------------
    print("\n=== H4 [POST-HOC, exponents already seen before the derivation] ===")
    e105 = {(0, 6): 0.2446, (9, 0): 0.4322, (10, 0): 0.2112,
            (10, 6): 0.0238, (11, 2): 0.0425}
    print("  L/H   Delta_A   e105 Delta_G   closed-form   numerical map")
    for (ell, hh), dg in e105.items():
        h = next(r for r in heads if r["layer"] == ell and r["head"] == hh)
        dA = h["delta_A"]
        closed = max(0.0, min(2 * dA - 0.5, dA))
        num = float(np.interp(dA, map_dA, map_dG))
        print(f"  {ell:2d}/{hh:<2d} {dA:.4f}   {dg:.4f}         {closed:.4f}"
              f"        {num:.4f}")

    # ---------------- residual structure ----------------
    print("\n=== Residual structure (the 'high R2, structured residual' watch) ===")
    for arm in ("arm1a", "arm1b"):
        for label, subset in (("conformal", conformal), ("SYK", syk)):
            runs = [h[arm]["resid_run_log"] for h in subset
                    if h[arm]["resid_run_log"] is not None]
            r2 = [h[arm]["r2_log"] for h in subset if h[arm]["r2_log"] is not None]
            print(f"  {arm} {label:10s} median R2_log {np.median(r2):.4f}  "
                  f"median longest same-sign residual run "
                  f"{np.median(runs):.0f} of 249 lags")

    json.dump({"prop1": prop1, "refit_connected": refit},
              open("analysis_gpt2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
