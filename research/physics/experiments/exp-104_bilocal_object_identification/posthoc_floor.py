"""exp-104 POST-HOC (exploratory, NOT pre-registered).

The pre-registered fit is a 2-parameter log-log OLS, identical to the census
estimator. But melonic-note eq. (2.2) says E[H(1,2)] carries a constant term
(the bare propagator G_0 = w sum_ab K_ab) plus the query-query dependent piece.
A log-log fit on a profile with a nonzero floor is dragged toward Delta ~ 0 by
the floor regardless of the underlying decay, so the pre-registered Delta_G may
be measuring G_0 rather than the correlator.

This script asks whether the CONNECTED part has a power law, by fitting the
3-parameter model

    prof(dx) = c + b * dx^(-2*Delta)

and, separately, by subtracting a floor estimated from the far tail and refitting
with the census's own 2-parameter estimator.

This is exploratory. It does not amend the pre-registration and its numbers are
not a test of anything. Its purpose is to say whether a properly designed
measurement of H's exponent is worth pre-registering.

Ariel — August 8, 2026.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"

spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)

FIT_LO, FIT_HI = kit.FIT_LO, kit.FIT_HI
TAIL_LO, TAIL_HI = 400, 511          # far-tail window for the floor estimate
OBJECTS = ("A", "G_out", "G_K", "G_cos")


def fit_three_param(profile: np.ndarray):
    """prof(dx) = c + b*dx^(-2*Delta). Returns (Delta, c_fraction, R2)."""
    lags = np.arange(FIT_LO, FIT_HI + 1, dtype=float)
    y = profile[FIT_LO:FIT_HI + 1].astype(float)
    if not np.all(np.isfinite(y)) or y.max() <= 0:
        return None, None, None

    def model(dx, c, b, delta):
        return c + b * dx ** (-2.0 * delta)

    y0 = float(np.median(profile[TAIL_LO:TAIL_HI + 1]))
    p0 = [y0, max(y[0] - y0, 1e-12), 0.25]
    try:
        popt, _ = curve_fit(model, lags, y, p0=p0, maxfev=20000)
    except Exception:
        return None, None, None
    c, b, delta = popt
    resid = y - model(lags, *popt)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None
    # fraction of the in-window signal carried by the constant term
    c_frac = float(abs(c) / abs(y).mean()) if abs(y).mean() > 0 else None
    return float(delta), c_frac, r2


def fit_floor_subtracted(profile: np.ndarray):
    """Subtract a far-tail floor, then use the census's own 2-param estimator."""
    floor = float(np.median(profile[TAIL_LO:TAIL_HI + 1]))
    sub = profile - floor
    return kit.fit_head(sub)


def main() -> None:
    prof = np.load(HERE / "profiles_gpt2.npz")
    prereg = json.loads((HERE / "results_gpt2.json").read_text())
    heads = prereg["heads"]

    conformal = [h for h in heads
                 if h["r2_A"] is not None and h["r2_A"] >= kit.R2_MIN
                 and h["delta_A"] is not None and h["delta_A"] >= kit.DELTA_MIN]
    syk_near = [h for h in conformal if abs(h["delta_A"] - 0.25) <= 0.05]

    rows = []
    for h in heads:
        ell, hd = h["layer"], h["head"]
        rec = {"layer": ell, "head": hd, "delta_A": h["delta_A"]}
        for name in OBJECTS:
            p = prof[name][ell, hd]
            d3, cfrac, r23 = fit_three_param(p)
            dsub, r2sub = fit_floor_subtracted(p)
            rec[f"{name}_delta3"] = d3
            rec[f"{name}_cfrac"] = cfrac
            rec[f"{name}_r2_3p"] = r23
            rec[f"{name}_delta_sub"] = None if dsub is None else float(dsub)
            rec[f"{name}_r2_sub"] = None if r2sub is None else float(r2sub)
        rows.append(rec)

    (HERE / "posthoc_floor_gpt2.json").write_text(json.dumps(rows, indent=1))

    def med(rows_, key):
        v = np.array([r[key] for r in rows_ if r[key] is not None], dtype=float)
        return np.median(v) if len(v) else None

    def pick(subset):
        keys = {(s["layer"], s["head"]) for s in subset}
        return [r for r in rows if (r["layer"], r["head"]) in keys]

    print("POST-HOC (exploratory, not pre-registered)")
    print(f"floor estimated as median of profile over lags [{TAIL_LO},{TAIL_HI}]\n")

    for label, subset in (("ALL", heads), ("CONFORMAL", conformal), ("SYK-NEAR", syk_near)):
        sel = pick(subset)
        if not sel:
            continue
        print(f"[{label}] n={len(sel)}")
        print(f"  {'object':8s} {'Δ prereg':>9s} {'Δ 3-param':>10s} {'Δ floor-sub':>12s} "
              f"{'R² 3p':>7s} {'R² sub':>7s} {'const/mean':>11s}")
        for name in OBJECTS:
            dp = med(sel, "delta_A") if name == "A" else None
            pre = np.median([h[f"delta_{name}"] for h in subset
                             if h[f"delta_{name}"] is not None])
            d3 = med(sel, f"{name}_delta3")
            ds = med(sel, f"{name}_delta_sub")
            r3 = med(sel, f"{name}_r2_3p")
            rs = med(sel, f"{name}_r2_sub")
            cf = med(sel, f"{name}_cfrac")
            f = lambda x, n=4: ("—" if x is None else f"{x:.{n}f}")
            print(f"  {name:8s} {f(pre):>9s} {f(d3):>10s} {f(ds):>12s} "
                  f"{f(r3, 3):>7s} {f(rs, 3):>7s} {f(cf, 3):>11s}")
        print()


if __name__ == "__main__":
    main()
