"""Why did V1 fail? Test the scale-separation explanation.

Proposition 2 is a leading asymptotic valid for 1 << s << U, where U = i - s is the
number of key positions summed over. Under the census protocol (n = 512, queries
i >= 256, fit lags [8, 256]) we have U in [256 - s, 511 - s], so at the top of the
fit window U collapses toward zero and there is no scale separation at all.

Prediction if that is the explanation: growing n while holding the fit window fixed
should drive the measured Delta_G toward Proposition 2's map.

Run: python diagnose_v1_failure.py
"""
from __future__ import annotations

import json
import numpy as np

FIT_LO, FIT_HI = 8, 256


def lag_profile(M: np.ndarray, deep_lo: int) -> np.ndarray:
    L = M.shape[0]
    prof = np.zeros(L)
    for dx in range(L):
        diag = np.diagonal(M, offset=-dx)
        k_lo = max(deep_lo, dx) - dx
        if k_lo < diag.shape[-1]:
            prof[dx] = diag[k_lo:].mean()
    return prof


def census_fit(profile: np.ndarray, lo: int = FIT_LO, hi: int = FIT_HI):
    lags = np.arange(lo, hi + 1)
    y = profile[lo:hi + 1]
    ok = y > 1e-15
    if ok.sum() < 5:
        return None, None
    lx, ly = np.log(lags[ok].astype(float)), np.log(y[ok])
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ c
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return float(-c[1] / 2), 1 - float(np.sum(resid ** 2)) / ss_tot


def predicted(dA: float) -> float:
    return max(0.0, min(2.0 * dA - 0.5, dA))


def build_A(dA: float, n: int) -> np.ndarray:
    u = np.arange(n, dtype=float)
    f = np.maximum(u, 1.0) ** (-2.0 * dA)
    idx = np.arange(n)
    lag = idx[:, None] - idx[None, :]
    A = np.where(lag >= 0, f[np.clip(lag, 0, n - 1)], 0.0)
    A /= A.sum(axis=1, keepdims=True)
    return A


def main() -> None:
    grid = [0.10, 0.20, 0.25, 0.30, 0.375, 0.45, 0.55, 0.75]
    sizes = [512, 1024, 2048, 4096]
    out = {"fit_window": [FIT_LO, FIT_HI], "rows": []}

    hdr = "Delta_A   pred   " + "".join(f"n={n:<6}" for n in sizes)
    print(hdr)
    print("-" * len(hdr))
    for dA in grid:
        row = {"delta_A": dA, "pred": predicted(dA), "measured": {}}
        cells = []
        for n in sizes:
            A = build_A(dA, n)
            # queries start at n/2 so U = i - s is O(n) across the whole fit window
            dG, r2 = census_fit(lag_profile(A @ A.T, deep_lo=n // 2))
            row["measured"][str(n)] = {"delta_G": dG, "r2": r2}
            cells.append(f"{dG:.4f} ")
        out["rows"].append(row)
        print(f"{dA:<9.3f} {predicted(dA):<6.3f} " + "".join(f"{c:<8}" for c in cells))

    # Does the residual shrink with n? That is the scale-separation signature.
    print("\n|measured - predicted| by n:")
    for r in out["rows"]:
        errs = [abs(r["measured"][str(n)]["delta_G"] - r["pred"]) for n in sizes]
        trend = "shrinking" if errs[-1] < errs[0] - 1e-4 else "NOT shrinking"
        print(f"  Delta_A={r['delta_A']:<6.3f} " +
              " ".join(f"{e:.4f}" for e in errs) + f"   {trend}")
        r["abs_err_by_n"] = dict(zip(map(str, sizes), errs))

    with open("diagnose_v1_failure.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
