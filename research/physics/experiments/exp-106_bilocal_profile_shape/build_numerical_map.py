"""Two jobs: verify why V4 failed, and build the numerical Delta_A -> Delta_G map.

V4 diagnosis. Arm 2 rebuilds a TI matrix from the census lag profile of A. For
lags <= 256 the census averages over a fixed query block (i in [256, 511]), so the
profile is exactly proportional to f there. For lags > 256 the query block shrinks,
so the reconstructed profile is NOT proportional to f - and those large-lag entries
feed each row's normalization. Predicted signature: prof_A(u)/f(u) constant for
u <= 256 and drifting for u > 256.

Numerical map. Proposition 2 is a leading asymptotic and the census window is not
asymptotic (see diagnose_v1_failure.py). The usable object is therefore the map
computed numerically at the census's own protocol.

Run: python build_numerical_map.py
"""
from __future__ import annotations

import json
import numpy as np

SEQ_LEN, DEEP_LO = 512, 256
FIT_LO, FIT_HI = 8, 256


def lag_profile(M: np.ndarray) -> np.ndarray:
    L = M.shape[0]
    prof = np.zeros(L)
    for dx in range(L):
        diag = np.diagonal(M, offset=-dx)
        k_lo = max(DEEP_LO, dx) - dx
        if k_lo < diag.shape[-1]:
            prof[dx] = diag[k_lo:].mean()
    return prof


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


def causal_from_profile(prof: np.ndarray) -> np.ndarray:
    """Causal row-stochastic matrix whose lag structure is `prof`. Arm 2's builder."""
    n = len(prof)
    idx = np.arange(n)
    lag = idx[:, None] - idx[None, :]
    A = np.where(lag >= 0, prof[np.clip(lag, 0, n - 1)], 0.0)
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)


def build_A(dA: float, c_over_b: float = 0.0, n: int = SEQ_LEN) -> np.ndarray:
    u = np.arange(n, dtype=float)
    f = c_over_b + np.maximum(u, 1.0) ** (-2.0 * dA)
    return causal_from_profile(f)


def main() -> None:
    out: dict = {}

    # ---- V4 diagnosis -------------------------------------------------------
    print("V4 diagnosis: prof_A(u) / f(u), normalized to its value at u=8")
    dA = 0.45
    u = np.arange(SEQ_LEN, dtype=float)
    f = np.maximum(u, 1.0) ** (-2.0 * dA)
    prof = lag_profile(build_A(dA))
    ratio = prof / f
    ratio = ratio / ratio[8]
    probe = [8, 64, 128, 200, 250, 256, 260, 300, 400, 500]
    for p in probe:
        print(f"   u={p:<4} ratio={ratio[p]:.6f}")
    drift_le_256 = float(np.max(np.abs(ratio[8:257] - 1.0)))
    drift_gt_256 = float(np.max(np.abs(ratio[257:] - 1.0)))
    print(f"   max |ratio-1| for u<=256: {drift_le_256:.3e}")
    print(f"   max |ratio-1| for u>256 : {drift_gt_256:.3e}")
    out["v4_diagnosis"] = {"max_dev_lags_le_256": drift_le_256,
                           "max_dev_lags_gt_256": drift_gt_256,
                           "explanation_supported":
                               bool(drift_le_256 < 1e-12 < drift_gt_256)}

    # confirm: feeding the TRUE f (correct at all lags) makes arm2 == arm1 exactly
    A_true = build_A(dA)
    A_rebuilt_true = causal_from_profile(f)
    exact = float(np.max(np.abs(A_true - A_rebuilt_true)))
    print(f"   arm2 rebuilt from true f vs original A: max abs diff = {exact:.3e}")
    out["v4_diagnosis"]["rebuild_from_true_f_max_abs_diff"] = exact

    # ---- the numerical map --------------------------------------------------
    print("\nNumerical map at the census protocol (n=512, i>=256, lags [8,256])")
    print("  c/b = 0 (pure power law), K = I")
    grid = np.round(np.arange(0.05, 0.951, 0.025), 4)
    rows = []
    for d in grid:
        A = build_A(float(d))
        dA_meas, _ = census_fit(lag_profile(A))
        dG, r2 = census_fit(lag_profile(A @ A.T))
        rows.append({"delta_A": float(d), "delta_A_measured": dA_meas,
                     "delta_G": dG, "r2": r2})
    out["map_cb0"] = rows
    for r in rows:
        if abs(r["delta_A"] * 40 % 2) < 1e-9:  # print every other point
            print(f"   Delta_A={r['delta_A']:.3f} -> Delta_G={r['delta_G']:.4f} "
                  f"(R2={r['r2']:.4f})")

    # invert: what Delta_A puts G at the SYK value 1/4? Use the *measured* Delta_A
    # so the map is expressed in the same units the census reports.
    dA_arr = np.array([r["delta_A_measured"] for r in rows])
    dG_arr = np.array([r["delta_G"] for r in rows])
    dA_for_syk = float(np.interp(0.25, dG_arr, dA_arr))
    dG_at_quarter = float(np.interp(0.25, dA_arr, dG_arr))
    print(f"\n   Delta_G = 1/4  <=  Delta_A = {dA_for_syk:.4f}")
    print(f"   Delta_A = 1/4  =>  Delta_G = {dG_at_quarter:.4f}")
    out["inversion"] = {"delta_A_giving_delta_G_quarter": dA_for_syk,
                        "delta_G_at_delta_A_quarter": dG_at_quarter}

    # sensitivity to the sink constant c
    print("\n  sensitivity to the attention-sink constant c/b:")
    sens = {}
    for cb in (0.0, 0.01, 0.1, 0.5):
        vals = []
        for d in (0.25, 0.375, 0.50):
            A = build_A(d, cb)
            dA_meas, _ = census_fit(lag_profile(A))
            dG, r2 = census_fit(lag_profile(A @ A.T))
            vals.append((d, dA_meas, dG, r2))
        sens[str(cb)] = [{"delta_A_nominal": d, "delta_A_measured": m,
                          "delta_G": g, "r2": r} for d, m, g, r in vals]
        print("   c/b=%-5s " % cb + "  ".join(
            f"[nom {d:.3f} meas {m:.4f}]->{g:.4f}(R2={r:.3f})"
            for d, m, g, r in vals))
    out["sink_sensitivity"] = sens

    with open("numerical_map.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
