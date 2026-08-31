"""exp-129 — Theory-of-A Level 3: Protocol Comparison for σ_delta Measurement.

Pre-registration committed to attention-geometry at 8246e6a before this script
was written. Analysis-only: loads positional_means.npz from exp-128.

Tests whether Frobenius normalization vs per-row cosine similarity accounts for
the discrepancy between exp-128's σ_delta = 0.189 and exp-117's σ_delta = 0.249.

Ariel — 2026-08-31.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP128 = HERE.parent / "exp-128_multilayer_residual_decomposition"
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"

spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
sys.path.insert(0, str(EXP112.parent / "exp-107_natural_text_bilocal"))
exp112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp112)

pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW = exp112.WINDOW

PREREG_COMMIT = "8246e6a"


def r2_of_slope(profile: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, profile, rcond=None)
    y_pred = X @ c
    ss_tot = float(((profile - profile.mean()) ** 2).sum())
    ss_res = float(((profile - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def sigma_frob(mat: np.ndarray) -> tuple[float, float]:
    """Frobenius normalization: mat / ||mat||_F, then outer-product cosine."""
    frob = np.linalg.norm(mat)
    M = mat.astype(np.float64) / frob        # (512, 768)
    C = M @ M.T                               # (512, 512)
    profile = pooled_window_profile(C)
    sigma = -ols_slope(profile, WINDOW)
    r2 = r2_of_slope(profile, WINDOW)
    return float(sigma), float(r2), profile


def sigma_rowcos(mat: np.ndarray) -> tuple[float, float]:
    """Per-row cosine similarity (exp-128 protocol)."""
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = mat.astype(np.float64) / norms       # (512, 768)
    C = M @ M.T                              # (512, 512)
    profile = pooled_window_profile(C)
    sigma = -ols_slope(profile, WINDOW)
    r2 = r2_of_slope(profile, WINDOW)
    return float(sigma), float(r2), profile


def main() -> None:
    print("exp-129: Protocol Comparison for σ_delta Measurement", flush=True)
    print(f"Pre-registration commit: {PREREG_COMMIT}\n", flush=True)

    # Load saved positional means from exp-128
    data = np.load(EXP128 / "positional_means.npz")
    h0 = data["h0"].astype(np.float64)
    attn0 = data["attn0"].astype(np.float64)
    mlp0 = data["mlp0"].astype(np.float64)
    attn1 = data["attn1"].astype(np.float64)
    mlp1 = data["mlp1"].astype(np.float64)
    h2 = data["h2"].astype(np.float64)

    delta_total = attn0 + mlp0 + attn1 + mlp1

    print("=== Protocol A: Frobenius normalization ===", flush=True)
    sigma_A, r2_A, profile_A = sigma_frob(delta_total)
    print(f"  σ_A(Δ_total):  {sigma_A:.4f}  R²={r2_A:.4f}", flush=True)

    sigma_C, r2_C, _ = sigma_frob(h2)
    print(f"  σ_C(h^(2)):   {sigma_C:.4f}  R²={r2_C:.4f}", flush=True)

    print("\n=== Protocol B: Per-row cosine (exp-128 verification) ===", flush=True)
    sigma_B, r2_B, _ = sigma_rowcos(delta_total)
    print(f"  σ_B(Δ_total):  {sigma_B:.4f}  R²={r2_B:.4f}  "
          f"(exp-128 reported: 0.1892)", flush=True)

    print("\n=== Per-component Frobenius slopes ===", flush=True)
    comp_frob = {}
    for name, arr in [("h0", h0), ("attn0", attn0), ("mlp0", mlp0),
                      ("attn1", attn1), ("mlp1", mlp1), ("h2", h2),
                      ("delta_total", delta_total)]:
        s, r, _ = sigma_frob(arr)
        comp_frob[name] = {"sigma_frob": s, "r2_frob": r}
        print(f"  {name:15s}  σ_frob={s:+.4f}  R²={r:.4f}", flush=True)

    print("\n=== Registered predictions ===", flush=True)
    P1_ok = 0.22 <= sigma_A <= 0.28
    P2_ok = sigma_A > sigma_B
    P3_ok = abs(sigma_C - sigma_A) < 0.03
    K1_fired = sigma_A < 0.20

    print(f"  P1 (σ_A ∈ [0.22, 0.28]):      {sigma_A:.4f}  → {'OK' if P1_ok else 'FAIL'}", flush=True)
    print(f"  P2 (σ_A > σ_B):               {sigma_A:.4f} vs {sigma_B:.4f}  → {'OK' if P2_ok else 'FAIL'}", flush=True)
    print(f"  P3 (|σ_C - σ_A| < 0.03):      |{sigma_C:.4f}-{sigma_A:.4f}|={abs(sigma_C-sigma_A):.4f}  → {'OK' if P3_ok else 'FAIL'}", flush=True)
    print(f"  K1 (σ_A < 0.20):              {'FIRED' if K1_fired else 'ok'}", flush=True)

    if K1_fired:
        overall = "inconclusive"
    elif P1_ok and P2_ok:
        overall = "confirmed"
    elif P1_ok:
        overall = "partial"
    else:
        overall = "falsified"

    print(f"\n  Overall verdict: {overall.upper()}", flush=True)

    results = {
        "exp": "exp-129",
        "date": "2026-08-31",
        "prereg_commit": PREREG_COMMIT,
        "source_data": "exp-128 positional_means.npz",
        "n_seqs": 100,
        "seq_len": 512,
        "protocols": {
            "A_frobenius": {
                "description": "Frobenius normalization of delta_total",
                "sigma": sigma_A,
                "r2": r2_A,
            },
            "B_rowcos": {
                "description": "Per-row cosine similarity (exp-128 protocol)",
                "sigma": sigma_B,
                "r2": r2_B,
                "exp128_reported": 0.1892,
            },
            "C_frob_h2": {
                "description": "Frobenius normalization of h^(2)",
                "sigma": sigma_C,
                "r2": r2_C,
            },
        },
        "per_component_frobenius": comp_frob,
        "registered_verdicts": {
            "P1": {"ok": P1_ok, "sigma_A": sigma_A,
                   "criterion": "sigma_A in [0.22, 0.28]"},
            "P2": {"ok": P2_ok, "sigma_A": sigma_A, "sigma_B": sigma_B,
                   "criterion": "sigma_A > sigma_B"},
            "P3": {"ok": P3_ok, "diff": abs(sigma_C - sigma_A),
                   "criterion": "|sigma_C - sigma_A| < 0.03"},
        },
        "kill_conditions": {
            "K1": {"fired": K1_fired, "criterion": "sigma_A < 0.20"},
        },
        "overall_verdict": overall,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
