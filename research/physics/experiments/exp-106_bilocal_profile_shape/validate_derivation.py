"""exp-106 synthetic gate — does the derived exponent map survive at finite n?

Tests the DERIVATION, not an estimator. There is no noise model and no acceptance
envelope here: every profile is computed exactly from a constructed matrix, so the
only thing this can detect is an algebra error in
`notes/2026-08-08_bilocal_from_attention_derivation.md` §3.

Pass criteria V1-V4 are as pre-registered in notes.md and are evaluated here.

Run: python validate_derivation.py
"""
from __future__ import annotations

import json
import numpy as np

# --- census protocol constants, frozen (replication/measure_conformal_heads.py) ---
SEQ_LEN = 512
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256


def lag_profile(M: np.ndarray) -> np.ndarray:
    """Census lag_profile for a single (L, L) matrix. Queries i >= max(DEEP_LO, dx)."""
    L = M.shape[0]
    prof = np.zeros(L)
    for dx in range(L):
        diag = np.diagonal(M, offset=-dx)
        k_lo = max(DEEP_LO, dx) - dx
        if k_lo < diag.shape[-1]:
            prof[dx] = diag[k_lo:].mean()
    return prof


def census_fit(profile: np.ndarray):
    """Census 2-parameter log-log OLS. Returns (Delta, R2)."""
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
    r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-30 else 0.0
    return float(-c[1] / 2), r2


def predicted_delta_G(delta_A: float) -> float:
    """Proposition 2: Delta_G = max(0, min(2*Delta_A - 1/2, Delta_A))."""
    return max(0.0, min(2.0 * delta_A - 0.5, delta_A))


def build_causal_A(delta_A: float, c_over_b: float, n: int = SEQ_LEN) -> np.ndarray:
    """Causal row-stochastic A with lag profile f(u) = c + b*u^(-2*delta_A)."""
    u = np.arange(n, dtype=float)
    f = c_over_b + np.maximum(u, 1.0) ** (-2.0 * delta_A)
    idx = np.arange(n)
    lag = idx[:, None] - idx[None, :]
    A = np.where(lag >= 0, f[np.clip(lag, 0, n - 1)], 0.0)
    A /= A.sum(axis=1, keepdims=True)
    return A


def K_uniform(n: int, mu: float) -> np.ndarray:
    return mu * np.ones((n, n))


def K_powerlaw(n: int, q: float) -> np.ndarray:
    idx = np.arange(n)
    return (1.0 + np.abs(idx[:, None] - idx[None, :])) ** (-q)


def main() -> None:
    out: dict = {"protocol": {"seq_len": SEQ_LEN, "deep_lo": DEEP_LO,
                              "fit_lags": [FIT_LO, FIT_HI]}}

    # ---------------- V1: the exponent map, K = I, c = 0 ----------------
    grid = [0.10, 0.20, 0.25, 0.30, 0.375, 0.45, 0.55, 0.75]
    v1 = []
    for dA in grid:
        A = build_causal_A(dA, 0.0)
        p = lag_profile(A @ A.T)
        dG, r2 = census_fit(p)
        pred = predicted_delta_G(dA)
        # marginal cell: prediction is logarithmic, criterion is dG <= 0.08
        if abs(dA - 0.25) < 1e-9:
            ok = dG <= 0.08
        else:
            ok = abs(dG - pred) <= 0.05
        v1.append({"delta_A": dA, "delta_G_pred": pred, "delta_G_measured": dG,
                   "r2": r2, "pass": bool(ok)})
        print(f"V1  Delta_A={dA:.3f}  pred={pred:.4f}  measured={dG:.4f}  "
              f"R2={r2:.4f}  {'PASS' if ok else 'FAIL'}")
    out["V1"] = {"cells": v1, "pass": all(c["pass"] for c in v1)}

    # sanity: the census estimator recovers Delta_A on A itself
    ctrl = []
    for dA in grid:
        A = build_causal_A(dA, 0.0)
        dhat, r2 = census_fit(lag_profile(A))
        ctrl.append({"delta_A": dA, "delta_A_recovered": dhat, "r2": r2})
        print(f"    control on A: Delta_A={dA:.3f} -> {dhat:.4f} (R2={r2:.4f})")
    out["control_on_A"] = ctrl

    # ---------------- V2: Proposition 1 is exact ----------------
    v2 = []
    for dA in (0.25, 0.45):
        A = build_causal_A(dA, 0.1)
        base = lag_profile(A @ A.T)
        for mu in (0.1, 1.0, 10.0):
            shifted = lag_profile(A @ (np.eye(SEQ_LEN) + K_uniform(SEQ_LEN, mu)) @ A.T)
            resid = np.abs(shifted[FIT_LO:FIT_HI + 1] - (base[FIT_LO:FIT_HI + 1] + mu))
            rel = float(np.max(resid) / mu)
            v2.append({"delta_A": dA, "mu": mu, "max_rel_dev": rel,
                       "pass": bool(rel < 1e-10)})
            print(f"V2  Delta_A={dA:.2f} mu={mu:<5} max relative deviation "
                  f"from exact shift = {rel:.3e}  "
                  f"{'PASS' if rel < 1e-10 else 'FAIL'}")
    out["V2"] = {"cells": v2, "pass": all(c["pass"] for c in v2)}

    # ---------------- V3: q-dependence of the correlated-K map ----------------
    v3 = []
    for dA in (0.30, 0.45):
        row = []
        for q in (0.5, 1.0, 2.0):
            A = build_causal_A(dA, 0.0)
            dG, r2 = census_fit(lag_profile(A @ K_powerlaw(SEQ_LEN, q) @ A.T))
            row.append({"q": q, "delta_G": dG, "r2": r2})
            print(f"V3  Delta_A={dA:.2f} q={q:<4} -> Delta_G={dG:.4f} (R2={r2:.4f})")
        mono = all(row[i + 1]["delta_G"] >= row[i]["delta_G"] - 1e-6
                   for i in range(len(row) - 1))
        v3.append({"delta_A": dA, "cells": row, "monotone_in_q": bool(mono)})
        print(f"    monotone in q: {'PASS' if mono else 'FAIL'}")
    out["V3"] = {"cells": v3, "pass": all(c["monotone_in_q"] for c in v3)}

    # ---------------- V4: arm 2 reproduces arm 1 on a TI-built A ----------------
    # arm 2 rebuilds A from its own measured lag profile, then row-normalizes.
    v4 = []
    for dA in (0.25, 0.45):
        A = build_causal_A(dA, 0.1)
        p_arm1 = lag_profile(A @ A.T)
        prof_A = lag_profile(A)
        n = SEQ_LEN
        idx = np.arange(n)
        lag = idx[:, None] - idx[None, :]
        A_ti = np.where(lag >= 0, prof_A[np.clip(lag, 0, n - 1)], 0.0)
        A_ti /= A_ti.sum(axis=1, keepdims=True)
        p_arm2 = lag_profile(A_ti @ A_ti.T)
        w = slice(FIT_LO, FIT_HI + 1)
        rel = float(np.max(np.abs(p_arm2[w] - p_arm1[w]) / np.abs(p_arm1[w])))
        v4.append({"delta_A": dA, "max_rel_diff_arm2_vs_arm1": rel,
                   "pass": bool(rel < 1e-10)})
        print(f"V4  Delta_A={dA:.2f}  arm2 vs arm1 max rel diff = {rel:.3e}  "
              f"{'PASS' if rel < 1e-10 else 'FAIL'}")
    out["V4"] = {"cells": v4, "pass": all(c["pass"] for c in v4)}

    # ---------------- shape check: can c + b*s^(-2D) fit the forward model? ------
    # Proposition 3 says no. Recorded as the R2 of the census 2-param fit and of a
    # 3-param fit on the exactly-computed forward-model profile (no noise at all).
    shape = []
    for dA in (0.25, 0.30, 0.375, 0.45):
        for cb in (0.0, 0.1):
            A = build_causal_A(dA, cb)
            p = lag_profile(A @ A.T)
            _, r2_2p = census_fit(p)
            shape.append({"delta_A": dA, "c_over_b": cb, "r2_census_2param": r2_2p})
            print(f"P3  Delta_A={dA:.3f} c/b={cb:.1f} -> census R2 on the exact "
                  f"forward model = {r2_2p:.4f}")
    out["prop3_shape"] = shape

    gate = all(out[k]["pass"] for k in ("V1", "V2", "V3", "V4"))
    out["gate_pass"] = bool(gate)
    print("\nGATE:", "PASS" if gate else "FAIL",
          "  (V1 %s, V2 %s, V3 %s, V4 %s)" % tuple(
              "pass" if out[k]["pass"] else "FAIL" for k in ("V1", "V2", "V3", "V4")))

    with open("validation_derivation.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
