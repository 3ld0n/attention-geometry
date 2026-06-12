"""
exp-061 — λ-sign derivation re-test (Phase 0 item 0.4, anomaly A1, EVALUATION W4).

Derivation committed first in notes/2026-06-10_phase0_0.4_lambda_sign_derivation.md:
under the image form A(i,j) = C[(i−j)^{−2Δ} + λ(i+j)^{−2Δ}] and the EXACT pipeline
valley statistic (deep queries i ≥ 384, key thirds, valley = 1 − M/max(S,E)), the
masses are linear in λ (X = X₀ + λX₁) with E₀ > M₀ > S₀ and S₁ > M₁ > E₁, hence:

  Regime 1 (end-dominated, λ < λ*(Δ)):  ∂valley/∂λ < 0   (unconditional)
  Regime 2 (start-dominated, λ > λ*):    ∂valley/∂λ > 0   (unconditional)
  λ*(Δ) = (E₀ − S₀)/(S₁ − E₁)

COMMITTED PREDICTION: measured per-head λ sits in Regime 1 ⟹ ρ(λ, valley) negative.

This script:
  1. Computes pipeline-exact X₀(Δ), X₁(Δ) and λ*(Δ) on a Δ grid.
  2. Re-test (pre-stated): for every conformal head in the exp-026 functional-form
     fits (Pythia-2.8B, GPT-Neo-2.7B):
       - regime classification at (Δ̂_BCFT, λ̂_BCFT),
       - model-implied valley v̂(Δ̂, λ̂)  →  Spearman ρ(v̂, valley_measured),
       - per-layer ρ(λ̂, valley_measured) sign tally vs the predicted negative sign.
  3. SECONDARY (labeled, run after step 2): the SCT-breaking reinterpretation
     (PAPER_BRIEF_NULL_CONE §5: high-λ = bulk-like spread heads) —
     ρ(λ̂, middle-mass fraction) per model.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"
EXP026 = (OUT_DIR.parent / "exp-026_bcft_functional_form/results/"
          "bcft_functional_form_fit_2026-04-17T102458Z.json")

# exact pipeline geometry (exp-025 lineage, L = 512)
L = 512
Q = np.arange(384, 512)            # deep queries
THIRD = L // 3                     # 170
J_START = np.arange(1, THIRD)      # 1..169
J_MID = np.arange(THIRD, 2 * THIRD)    # 170..339
J_END = np.arange(2 * THIRD, 384)      # 340..383 (capped at Q_LO)

CONF_R2 = 0.85
CONF_DELTA = 0.05


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
    # for large Δ the image kernels underflow; S1 > E1 analytically, so λ* → ∞
    out["lambda_star"] = (out["E0"] - out["S0"]) / denom if denom > 0 else float("inf")
    return out


def model_valley(delta: float, lam: float, m: dict | None = None) -> float:
    m = m or masses(delta)
    S = m["S0"] + lam * m["S1"]
    M = m["M0"] + lam * m["M1"]
    E = m["E0"] + lam * m["E1"]
    return 1.0 - M / max(S, E)


def dvalley_sign(delta: float, lam: float, m: dict | None = None) -> int:
    m = m or masses(delta)
    return -1 if lam < m["lambda_star"] else +1


def main() -> None:
    # 1) λ*(Δ) table
    grid = np.round(np.arange(0.05, 1.01, 0.05), 2)
    lam_star_table = {f"{d:.2f}": round(masses(float(d))["lambda_star"], 3) for d in grid}
    print("λ*(Δ):", json.dumps(lam_star_table, indent=1), flush=True)

    # 2) re-test against exp-026 archived fits
    d26 = json.loads(EXP026.read_text())
    retest = {}
    for model_name, m in d26["models"].items():
        heads = [h for h in m["heads"]
                 if h.get("R2_1d") is not None and h["R2_1d"] >= CONF_R2
                 and h.get("delta_1d") is not None and h["delta_1d"] >= CONF_DELTA
                 and h.get("delta_BCFT") is not None and h.get("lambda_BCFT") is not None
                 and h.get("valley_depth") is not None
                 and 0 < h["delta_BCFT"] < 3]
        v_hat, v_meas, lam, regime1 = [], [], [], 0
        per_layer = {}
        mass_cache = {}
        for h in heads:
            dl = round(h["delta_BCFT"], 4)
            if dl not in mass_cache:
                mass_cache[dl] = masses(dl)
            mm = mass_cache[dl]
            vh = model_valley(h["delta_BCFT"], h["lambda_BCFT"], mm)
            v_hat.append(vh)
            v_meas.append(h["valley_depth"])
            lam.append(h["lambda_BCFT"])
            if h["lambda_BCFT"] < mm["lambda_star"]:
                regime1 += 1
            per_layer.setdefault(h["layer"], []).append(
                (h["lambda_BCFT"], h["valley_depth"]))

        rho_vv, p_vv = spearmanr(v_hat, v_meas)
        rho_lv, p_lv = spearmanr(lam, v_meas)

        layer_rows, neg, pos = [], 0, 0
        for lay in sorted(per_layer):
            pairs = per_layer[lay]
            if len(pairs) < 5:
                continue
            r, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            if np.isnan(r):
                continue
            layer_rows.append({"layer": lay, "n": len(pairs), "rho_lambda_valley": float(r),
                               "p": float(p)})
            neg += r < 0
            pos += r > 0

        retest[model_name] = {
            "n_conformal_heads_used": len(heads),
            "frac_regime1_end_dominated": regime1 / len(heads) if heads else None,
            "predicted_sign_dvalley_dlambda": "negative (Regime 1)",
            "rho_model_implied_valley_vs_measured": {"rho": float(rho_vv), "p": float(p_vv)},
            "rho_lambda_vs_measured_valley_pooled": {"rho": float(rho_lv), "p": float(p_lv)},
            "per_layer_lambda_valley": layer_rows,
            "per_layer_sign_tally": {"negative": int(neg), "positive": int(pos),
                                     "n_layers": len(layer_rows)},
        }
        print(f"\n[{model_name}] n={len(heads)} regime1={regime1/len(heads):.3f} "
              f"ρ(v̂,v)={rho_vv:+.3f} ρ(λ,v)={rho_lv:+.3f} "
              f"layers neg/pos = {neg}/{pos}", flush=True)

    # 3) SECONDARY (labeled): SCT-breaking reinterpretation — λ vs middle-mass fraction
    secondary = {}
    for model_name, m in d26["models"].items():
        heads = [h for h in m["heads"]
                 if h.get("R2_1d") is not None and h["R2_1d"] >= CONF_R2
                 and h.get("delta_1d") is not None and h["delta_1d"] >= CONF_DELTA
                 and h.get("lambda_BCFT") is not None
                 and all(h.get(k) is not None for k in ("start", "middle", "end"))]
        lam = [h["lambda_BCFT"] for h in heads]
        mfrac = [h["middle"] / (h["start"] + h["middle"] + h["end"]) for h in heads]
        rho, p = spearmanr(lam, mfrac)
        secondary[model_name] = {"rho_lambda_middle_mass_fraction": float(rho),
                                 "p": float(p), "n": len(heads)}
        print(f"[secondary {model_name}] ρ(λ, middle-frac) = {rho:+.3f} (p={p:.2e})", flush=True)

    result = {
        "experiment": "exp-061",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "phase0_item": "0.4",
        "derivation_note": "notes/2026-06-10_phase0_0.4_lambda_sign_derivation.md (committed first)",
        "derived_result": ("valley(λ) is V-shaped: ∂valley/∂λ < 0 for λ < λ*(Δ) "
                           "(end-dominated), > 0 above; sign result is unconditional in Δ "
                           "given the pipeline geometry"),
        "lambda_star_of_delta": lam_star_table,
        "retest": retest,
        "secondary_sct_reinterpretation": {
            "label": "secondary analysis, run after the primary re-test, as pre-stated",
            "hypothesis": "PAPER_BRIEF_NULL_CONE §5: high-λ heads are bulk-like spread heads "
                          "→ λ should correlate positively with middle-mass fraction",
            "results": secondary,
        },
        "archived_inputs": {"exp026": str(EXP026.name)},
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUT_DIR).decode().strip()
    except Exception:
        result["git_commit"] = "unknown"
    RESULTS_PATH.write_text(json.dumps(result, indent=1))
    print(f"\nWrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
