"""
exp-067 — v2 pre-registration: Test A (joint (Δ,λ) → implied valley).

Pre-registration committed: research/physics/notes/2026-06-14_v2_preregistration.md

Test A uses archived BCFT functional-form fit data (exp-026) to compute the
implied valley v̂(Δ̂, λ̂) on fresh models and compare with measured valley.

Thresholds (from pre-registration):
  Keep A1: ρ(v̂, valley_meas) ≥ 0.60 AND p ≤ 1e-5 AND n ≥ 50
  Keep A2 (improvement): ρ(v̂, valley_meas) > ρ(Δ̂, valley_meas)
  Kill A1: ρ < 0.50
  Partial: ρ ∈ [0.50, 0.60)

Diagnostic models (NOT confirmatory — used to set thresholds):
  Pythia-2.8B: ρ = +0.683
  GPT-Neo-2.7B: ρ = +0.906

Fresh models in archived data:
  Pythia-410m (exp-026 T101027Z file — not used in exp-061)

For models not yet in archived exp-026 data, a live measurement section follows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]  # .../ariel
EXP026_DIR = ROOT / "research/physics/experiments/exp-026_bcft_functional_form/results"

# Pipeline geometry (exact, matches exp-025/exp-026/exp-061)
L = 512
Q = np.arange(384, 512)
THIRD = L // 3
J_START = np.arange(1, THIRD)
J_MID = np.arange(THIRD, 2 * THIRD)
J_END = np.arange(2 * THIRD, 384)

CONF_R2 = 0.85
CONF_DELTA = 0.05

# Pre-registration thresholds
KEEP_A1_RHO = 0.60
KILL_A1_RHO = 0.50
KEEP_P = 1e-5
MIN_N = 50


def masses(delta: float) -> dict:
    """Pipeline-exact bulk (X0) and image (X1) third-masses for given Δ."""
    out = {}
    for name, jr in [("S", J_START), ("M", J_MID), ("E", J_END)]:
        ii = Q[:, None].astype(float)
        jj = jr[None, :].astype(float)
        bulk = ((ii - jj) ** (-2.0 * delta)).mean()
        image = ((ii + jj) ** (-2.0 * delta)).mean()
        out[name + "0"] = float(bulk)
        out[name + "1"] = float(image)
    denom = out["S1"] - out["E1"]
    out["lambda_star"] = (out["E0"] - out["S0"]) / denom if denom > 0 else float("inf")
    return out


def model_valley(delta: float, lam: float, m: dict | None = None) -> float:
    m = m or masses(delta)
    S = m["S0"] + lam * m["S1"]
    M = m["M0"] + lam * m["M1"]
    E = m["E0"] + lam * m["E1"]
    peak = max(S, E)
    if peak <= 0:
        return float("nan")
    return 1.0 - M / peak


def verdict_a1(rho: float, p: float, n: int) -> str:
    if n < MIN_N:
        return f"INSUFFICIENT ({n} < {MIN_N} required)"
    if rho >= KEEP_A1_RHO and p <= KEEP_P:
        return "KEEP (A1 confirmed)"
    if rho >= KILL_A1_RHO and rho < KEEP_A1_RHO:
        return "PARTIAL (clears April bar, below v2 threshold)"
    if rho < KILL_A1_RHO:
        return "KILL (below April bar)"
    return f"AMBIGUOUS (ρ={rho:.3f}, p={p:.2e})"


def run_test_a_from_archived(exp026_file: Path, model_name: str) -> dict:
    """Run Test A on a model from archived exp-026 data."""
    data = json.loads(exp026_file.read_text())
    model_data = data["models"].get(model_name)
    if model_data is None:
        raise KeyError(f"{model_name} not in {exp026_file.name}")

    all_heads = model_data["heads"]
    heads = [
        h for h in all_heads
        if h.get("R2_1d", 0) >= CONF_R2
        and h.get("delta_1d", 0) >= CONF_DELTA
        and h.get("delta_BCFT") is not None
        and h.get("lambda_BCFT") is not None
        and h.get("valley_depth") is not None
        and 0 < h["delta_BCFT"] < 3
    ]
    n = len(heads)
    print(f"  {model_name}: {n} conformal heads (of {len(all_heads)} total)")

    mass_cache: dict = {}
    v_hat, v_meas, delta_vals = [], [], []
    for h in heads:
        dl = round(h["delta_BCFT"], 4)
        if dl not in mass_cache:
            mass_cache[dl] = masses(dl)
        vh = model_valley(h["delta_BCFT"], h["lambda_BCFT"], mass_cache[dl])
        v_hat.append(vh)
        v_meas.append(h["valley_depth"])
        delta_vals.append(h["delta_BCFT"])

    v_hat = np.array(v_hat)
    v_meas = np.array(v_meas)
    delta_vals = np.array(delta_vals)

    # Test A1: ρ(v̂, valley_meas)
    rho_joint, p_joint = spearmanr(v_hat, v_meas)
    # Comparison: ρ(Δ̂, valley_meas)
    rho_delta, p_delta = spearmanr(delta_vals, v_meas)
    improvement = rho_joint > rho_delta

    v_a1 = verdict_a1(float(rho_joint), float(p_joint), n)

    print(f"    ρ(v̂, valley_meas) = {rho_joint:+.4f}  p = {p_joint:.2e}  → {v_a1}")
    print(f"    ρ(Δ̂, valley_meas) = {rho_delta:+.4f}  p = {p_delta:.2e}")
    print(f"    Improvement (A2): {'YES' if improvement else 'NO'}  (Δρ = {rho_joint - rho_delta:+.4f})")

    return {
        "model": model_name,
        "source": str(exp026_file.name),
        "n_conformal_heads": n,
        "n_total_heads": len(all_heads),
        "rho_joint_implied": {"rho": float(rho_joint), "p": float(p_joint)},
        "rho_delta_only": {"rho": float(rho_delta), "p": float(p_delta)},
        "improvement_a2": bool(improvement),
        "delta_rho": float(rho_joint - rho_delta),
        "verdict_a1": v_a1,
    }


def main() -> None:
    results = {
        "experiment": "exp-067-test-a",
        "preregistration": "research/physics/notes/2026-06-14_v2_preregistration.md",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "keep_a1_rho": KEEP_A1_RHO,
            "kill_a1_rho": KILL_A1_RHO,
            "keep_p": KEEP_P,
            "min_n": MIN_N,
        },
        "diagnostic_models": {
            "EleutherAI/pythia-2.8b": {
                "rho_joint_implied": 0.683,
                "note": "threshold-setter — not a confirmatory test"
            },
            "EleutherAI/gpt-neo-2.7B": {
                "rho_joint_implied": 0.906,
                "note": "threshold-setter — not a confirmatory test"
            }
        },
        "confirmatory_tests": {}
    }

    # --- Test A: Pythia-410m from archived data ---
    print("\n=== Test A: Pythia-410m (archived exp-026 data) ===")
    f410 = EXP026_DIR / "bcft_functional_form_fit_2026-04-17T101027Z.json"
    r410 = run_test_a_from_archived(f410, "EleutherAI/pythia-410m")
    results["confirmatory_tests"]["EleutherAI/pythia-410m"] = r410

    # --- Live measurement section (for Pythia-1.4b, GPT-2-medium) ---
    # These require loading models and running BCFT functional-form fits.
    # They follow the same protocol: run_bcft_functional_form_fit.py on the
    # target model, save to exp-026 format, then call run_test_a_from_archived.
    print("\n=== Live model section ===")
    print("Pythia-1.4b and GPT-2-medium require BCFT functional-form fit runs.")
    print("Run the fit script first; results appended here when available.")
    results["confirmatory_tests"]["pending"] = [
        "EleutherAI/pythia-1.4b",
        "openai-community/gpt2-medium",
        "mistralai/Mistral-7B-v0.3",
    ]

    # Save results
    out_path = Path(__file__).parent / "results_test_a.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for model, r in results["confirmatory_tests"].items():
        if model == "pending":
            continue
        rho_j = r["rho_joint_implied"]["rho"]
        rho_d = r["rho_delta_only"]["rho"]
        imp = r["improvement_a2"]
        v = r["verdict_a1"]
        print(f"  {model}:")
        print(f"    ρ(v̂,valley) = {rho_j:+.4f}  ρ(Δ,valley) = {rho_d:+.4f}  improvement={imp}")
        print(f"    → {v}")


if __name__ == "__main__":
    main()
