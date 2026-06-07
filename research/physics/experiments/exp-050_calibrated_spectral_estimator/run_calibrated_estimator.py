"""
exp-050 — Calibrated spectral estimator for BCFT conformal heads

Hypothesis (pre-stated, 2026-06-07):

  H1 (CALIBRATION IMPROVES CONSISTENCY): After applying synthetic calibration
     to invert finite-DFT bias, corrected Δ_freq values are closer to Δ_pos
     than raw Δ_freq values.
     Measured as: mean |Δ_freq_corrected − Δ_pos| < mean |Δ_freq_raw − Δ_pos|
     across the 44 conformal heads from exp-045.

  H2 (CFT KINEMATICS HOLDS AFTER CALIBRATION): For SYK-near heads (Δ_pos ≈ 0.25),
     the calibration-corrected Δ_freq_corrected ≈ 0.25. The DFT was measuring
     the right physics but with a known finite-window bias.

  H3 (BIAS IS MONOTONE AND INVERTIBLE): The calibration curve α_measured(Δ_true)
     is monotone increasing in Δ_true. Larger Δ_true (faster decay) → less
     negative α_measured (smaller finite-window distortion). Guarantees unique
     inversion.

  H4 (NULL): Even after calibration, Δ_freq_corrected systematically disagrees
     with Δ_pos. This would indicate a genuine departure from CFT kinematics
     in the spectral domain — not a methodological artifact.

Physical background:
  - exp-045 found: finite DFT of G(r) ~ r^{-2Δ} on 55 lags gives α ≈ −0.82
    for conformal heads, while CFT kinematics predicts α = 2Δ−1 ≈ −0.49.
  - Bias mechanism: G_true(r) is nonzero at the truncation boundary (r=55)
    for slowly-decaying profiles (small Δ). The sharp cutoff introduces
    additional low-frequency power, making the measured spectral exponent
    more negative than the continuum formula.
  - Calibration strategy: generate synthetic G_true(r) = r^{−2Δ} for known
    Δ, apply identical DFT protocol as exp-045, measure resulting α_measured(Δ).
    Build calibration curve; invert to correct real data.

Method:
  1. Calibration curve: Δ_true ∈ [0.02, 0.60] (81 points, fine grid)
     For each: G_true(r) = r^{-2Δ} for r=1..55, apply exp-045 DFT protocol,
     record α_measured.
  2. Polynomial fit to calibration curve: α_measured = P(Δ_true).
  3. For each conformal head from exp-045: given α_measured, solve for Δ_corrected.
  4. Test H1–H4.

Deviating from queue Option A description:
  Queue says "analysis-only if limited to cached GPT-2 conformal heads."
  This is exactly that: pure numpy analysis on exp-045 results.json.
  No model loading, no forward passes.

References:
  exp-045: G_>/G_< spectral function (GPT-2)
           raw result: α_median = -0.820, Δ_freq_median = 0.090
           vs Δ_pos_median = 0.255, α_pred_median = -0.489
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import interpolate

# ── constants (must match exp-045 protocol exactly) ────────────────────────────
MAX_DX   = 56        # r = 0..55; use r=1..55 for DFT (skip r=0)
N_LAG    = MAX_DX - 1  # 55 points in the lag profile
N_PAD    = 256       # zero-pad to this length
K_LOW    = 2         # spectral fit lower bound (k index)
K_HIGH   = 18        # spectral fit upper bound (k index)

# Calibration grid
DELTA_MIN   = 0.02
DELTA_MAX   = 0.60
N_DELTA_CAL = 81     # fine grid for calibration curve

# Directories
EXP045_RESULTS = Path("research/physics/experiments/exp-045_gpt2_spectral_function/results.json")
OUT_DIR        = Path("research/physics/experiments/exp-050_calibrated_spectral_estimator")
RESULTS_FILE   = OUT_DIR / "results.json"


# ── DFT protocol (identical to exp-045) ────────────────────────────────────────

def apply_dft_protocol(G_pos: np.ndarray) -> tuple[float, float]:
    """
    Given lag profile G_pos[0..MAX_DX-1], apply exp-045 DFT protocol.
    Returns (alpha_measured, r2_freq).
    G_pos[0] is self-attention weight (r=0); we skip it.
    """
    profile = G_pos[1:].astype(np.float64)  # r=1..55, shape (55,)
    padded = np.zeros(N_PAD, dtype=np.float64)
    padded[:len(profile)] = profile
    spectrum = np.abs(np.fft.rfft(padded))  # shape (N_PAD//2+1,) = (129,)

    k_arr = np.arange(len(spectrum))
    mask = (k_arr >= K_LOW) & (k_arr < K_HIGH) & (spectrum > 1e-30)
    if mask.sum() < 4:
        return float("nan"), 0.0

    log_k = np.log(k_arr[mask].astype(float))
    log_s = np.log(spectrum[mask])
    A_mat = np.column_stack([np.ones_like(log_k), log_k])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_s, rcond=None)
    pred = A_mat @ coeffs
    ss_res = np.sum((log_s - pred) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    alpha = float(coeffs[1])
    return alpha, r2


# ── calibration curve ──────────────────────────────────────────────────────────

def build_calibration_curve(delta_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each Δ in delta_grid:
      - Generate G_true(r) = r^{-2Δ} for r = 0..55 (G[0] = 1 by convention,
        but we skip it in DFT; G[r] = r^{-2Δ} for r >= 1)
      - Apply exp-045 DFT protocol
      - Record α_measured

    Returns (alpha_measured_arr, continuum_pred_arr) both shape (len(delta_grid),).
    """
    n = len(delta_grid)
    alpha_measured = np.zeros(n)
    alpha_continuum = 2.0 * delta_grid - 1.0  # the theoretical prediction

    # Build lag array once
    r_arr = np.arange(MAX_DX, dtype=np.float64)  # r = 0..55

    for i, delta in enumerate(delta_grid):
        G = np.zeros(MAX_DX)
        G[0] = 1.0  # r=0 placeholder (will be skipped by DFT protocol)
        # G(r) = r^{-2Δ} for r >= 1
        for r in range(1, MAX_DX):
            G[r] = float(r) ** (-2.0 * delta)

        alpha, _ = apply_dft_protocol(G)
        alpha_measured[i] = alpha

    return alpha_measured, alpha_continuum


# ── inversion: α_measured → Δ_corrected ───────────────────────────────────────

def build_inverter(delta_grid: np.ndarray, alpha_measured: np.ndarray):
    """
    Build an interpolation function: α_measured → Δ_corrected.
    The calibration curve should be monotone (H3); if so, we can invert.
    Returns: inverter function, is_monotone (bool), and direction (ascending/descending).
    """
    # Sort by alpha_measured (x-axis for inversion)
    sort_idx = np.argsort(alpha_measured)
    alpha_sorted = alpha_measured[sort_idx]
    delta_sorted = delta_grid[sort_idx]

    # Check monotonicity: since larger Δ → less negative α (larger α), should be ascending
    diffs = np.diff(alpha_sorted)
    is_monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    # Build interpolation function
    inverter = interpolate.interp1d(
        alpha_sorted, delta_sorted,
        kind='cubic',
        bounds_error=False,
        fill_value=(delta_sorted[0], delta_sorted[-1])  # extrapolate flat
    )

    return inverter, is_monotone, alpha_sorted, delta_sorted


# ── main analysis ───────────────────────────────────────────────────────────────

def main():
    print("exp-050: Calibrated spectral estimator for BCFT conformal heads", flush=True)
    print("=" * 60, flush=True)
    print(f"Calibration grid: Δ ∈ [{DELTA_MIN}, {DELTA_MAX}], N={N_DELTA_CAL}", flush=True)
    print(f"DFT protocol: N_LAG={N_LAG}, N_PAD={N_PAD}, k∈[{K_LOW},{K_HIGH})", flush=True)

    # ── Step 1: Build calibration curve ───────────────────────────────────────
    print("\n[1] Building calibration curve...", flush=True)
    delta_grid = np.linspace(DELTA_MIN, DELTA_MAX, N_DELTA_CAL)
    alpha_measured_cal, alpha_continuum_cal = build_calibration_curve(delta_grid)

    # Print calibration summary
    print(f"  Δ_true | α_measured | α_continuum | bias", flush=True)
    for delta, alpha_m, alpha_c in zip(delta_grid[::10], alpha_measured_cal[::10], alpha_continuum_cal[::10]):
        bias = alpha_m - alpha_c
        print(f"  {delta:.3f}  | {alpha_m:9.4f}  | {alpha_c:11.4f}  | {bias:+.4f}", flush=True)

    # Check H3: monotonicity
    diffs_cal = np.diff(alpha_measured_cal)
    is_monotone_ascending = bool(np.all(diffs_cal >= -1e-6))
    is_monotone_descending = bool(np.all(diffs_cal <= 1e-6))
    is_monotone = is_monotone_ascending or is_monotone_descending
    print(f"\nH3 (monotone calibration curve): {'CONFIRMED' if is_monotone else 'VIOLATED'}", flush=True)
    print(f"  Direction: {'ascending' if is_monotone_ascending else 'descending' if is_monotone_descending else 'non-monotone'}", flush=True)

    # ── Step 2: Build inverter ─────────────────────────────────────────────────
    print("\n[2] Building calibration inverter...", flush=True)
    inverter, is_monotone_check, alpha_sorted, delta_sorted = build_inverter(
        delta_grid, alpha_measured_cal
    )

    # Validate inverter on calibration points
    print("  Validation on calibration grid:", flush=True)
    for delta_true in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        # Find the measured alpha for this delta_true
        idx = np.argmin(np.abs(delta_grid - delta_true))
        alpha_m = alpha_measured_cal[idx]
        delta_recovered = float(inverter(alpha_m))
        print(f"    Δ_true={delta_true:.2f} → α_measured={alpha_m:.4f} → Δ_recovered={delta_recovered:.4f} "
              f"(error={delta_recovered-delta_true:+.4f})", flush=True)

    # ── Step 3: Load exp-045 data and apply calibration ───────────────────────
    print(f"\n[3] Loading exp-045 data: {EXP045_RESULTS}", flush=True)
    with open(EXP045_RESULTS) as f:
        exp045 = json.load(f)

    conformal_heads = [h for h in exp045["per_head"] if h.get("conformal")]
    syk_near_heads  = [h for h in exp045["per_head"] if h.get("syk_near")]
    print(f"  Conformal heads: {len(conformal_heads)}", flush=True)
    print(f"  SYK-near heads:  {len(syk_near_heads)}", flush=True)

    # Apply calibration to each conformal head
    per_head_calibrated = []
    for h in conformal_heads:
        alpha_obs = h["alpha"]
        delta_pos = h["delta_pos"]
        delta_freq_raw = h["delta_freq"]

        if alpha_obs is None or math.isnan(alpha_obs):
            per_head_calibrated.append({**h, "delta_freq_corrected": None, "correction_applied": False})
            continue

        delta_freq_corrected = float(inverter(alpha_obs))
        per_head_calibrated.append({
            **h,
            "delta_freq_corrected": round(delta_freq_corrected, 6),
            "correction_applied": True,
        })

    # ── Step 4: Test hypotheses ────────────────────────────────────────────────
    print("\n[4] Testing hypotheses...", flush=True)

    valid = [h for h in per_head_calibrated if h.get("correction_applied")]
    syk_valid = [h for h in valid if h.get("syk_near")]

    delta_pos_vals       = [h["delta_pos"] for h in valid]
    delta_freq_raw_vals  = [h["delta_freq"] for h in valid]
    delta_freq_corr_vals = [h["delta_freq_corrected"] for h in valid]
    alpha_vals           = [h["alpha"] for h in valid]

    # H1: calibration reduces mean absolute error
    mae_raw  = float(np.mean([abs(c - p) for c, p in zip(delta_freq_raw_vals, delta_pos_vals)]))
    mae_corr = float(np.mean([abs(c - p) for c, p in zip(delta_freq_corr_vals, delta_pos_vals)]))
    h1_confirmed = mae_corr < mae_raw
    print(f"\nH1 (calibration improves consistency):", flush=True)
    print(f"  MAE(Δ_freq_raw  − Δ_pos) = {mae_raw:.4f}", flush=True)
    print(f"  MAE(Δ_freq_corr − Δ_pos) = {mae_corr:.4f}", flush=True)
    print(f"  → H1 {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'}", flush=True)

    # H2: SYK-near heads → Δ_freq_corrected ≈ 0.25
    syk_delta_pos  = [h["delta_pos"] for h in syk_valid]
    syk_delta_corr = [h["delta_freq_corrected"] for h in syk_valid]
    syk_delta_raw  = [h["delta_freq"] for h in syk_valid]
    syk_alpha      = [h["alpha"] for h in syk_valid]

    if syk_delta_corr:
        syk_corr_med = statistics.median(syk_delta_corr)
        syk_raw_med  = statistics.median(syk_delta_raw)
        syk_pos_med  = statistics.median(syk_delta_pos)
        h2_confirmed = abs(syk_corr_med - 0.25) < 0.05
        print(f"\nH2 (CFT kinematics holds after calibration, SYK-near heads):", flush=True)
        print(f"  n(SYK-near) = {len(syk_valid)}", flush=True)
        print(f"  Δ_pos   median = {syk_pos_med:.4f}  (position-space, ground truth)", flush=True)
        print(f"  Δ_freq_raw median   = {syk_raw_med:.4f}  (before calibration)", flush=True)
        print(f"  Δ_freq_corr median  = {syk_corr_med:.4f}  (after calibration)", flush=True)
        print(f"  α_measured median   = {statistics.median(syk_alpha):.4f}  (SYK pred: -0.50)", flush=True)
        print(f"  → H2 {'CONFIRMED' if h2_confirmed else 'NOT CONFIRMED'} "
              f"(|{syk_corr_med:.4f} − 0.25| = {abs(syk_corr_med - 0.25):.4f})", flush=True)
    else:
        h2_confirmed = False
        syk_corr_med = float("nan")
        syk_raw_med  = float("nan")
        syk_pos_med  = float("nan")
        print(f"\nH2: No SYK-near heads with valid correction.", flush=True)

    # H4 (null): check overall residual after calibration
    mean_residual_corr = float(np.mean([c - p for c, p in zip(delta_freq_corr_vals, delta_pos_vals)]))
    std_residual_corr  = float(np.std([c - p for c, p in zip(delta_freq_corr_vals, delta_pos_vals)]))
    print(f"\nH4 (null — systematic disagreement remains):", flush=True)
    print(f"  Residual after calibration: {mean_residual_corr:+.4f} ± {std_residual_corr:.4f}", flush=True)
    h4_confirmed = abs(mean_residual_corr) > 0.05
    print(f"  → H4 {'CONFIRMED (genuine discrepancy)' if h4_confirmed else 'NOT CONFIRMED (bias was methodological)'}", flush=True)

    # Full conformal head summary
    all_delta_corr_med = statistics.median(delta_freq_corr_vals) if delta_freq_corr_vals else float("nan")
    all_delta_pos_med  = statistics.median(delta_pos_vals) if delta_pos_vals else float("nan")
    all_delta_raw_med  = statistics.median(delta_freq_raw_vals) if delta_freq_raw_vals else float("nan")

    print(f"\nAll conformal heads (n={len(valid)}):", flush=True)
    print(f"  Δ_pos       median = {all_delta_pos_med:.4f}  (position-space)", flush=True)
    print(f"  Δ_freq_raw  median = {all_delta_raw_med:.4f}  (before calibration)", flush=True)
    print(f"  Δ_freq_corr median = {all_delta_corr_med:.4f}  (after calibration)", flush=True)
    print(f"  Pearson r (corrected vs pos):  {float(np.corrcoef([h['delta_pos'] for h in valid], delta_freq_corr_vals)[0,1]):.4f}", flush=True)

    # ── Step 5: Physical interpretation ───────────────────────────────────────
    print("\n[5] Physical interpretation:", flush=True)
    bias_at_025 = float(alpha_measured_cal[np.argmin(np.abs(delta_grid - 0.25))] - (2*0.25 - 1))
    print(f"  Bias at Δ=0.25: α_measured − α_continuum = {bias_at_025:+.4f}", flush=True)
    print(f"  Source: finite window G(r) nonzero at r=55, sharp cutoff adds low-k power", flush=True)
    print(f"  Calibration corrects this systematic effect.", flush=True)

    # Per-head table (conformal heads, sorted by δ_pos)
    print(f"\nPer-head calibration results (conformal heads):", flush=True)
    print(f"  {'L':>3} {'H':>3}  {'Δ_pos':>6}  {'α':>7}  {'Δ_raw':>6}  {'Δ_corr':>6}  {'err_raw':>7}  {'err_corr':>7}  SYK", flush=True)
    sorted_heads = sorted(valid, key=lambda h: h["delta_pos"])
    for h in sorted_heads:
        err_raw  = h["delta_freq"] - h["delta_pos"]
        err_corr = h["delta_freq_corrected"] - h["delta_pos"]
        syk_flag = "★" if h.get("syk_near") else " "
        print(f"  L{h['layer']:2d} H{h['head']:2d}  "
              f"{h['delta_pos']:6.4f}  "
              f"{h['alpha']:7.4f}  "
              f"{h['delta_freq']:6.4f}  "
              f"{h['delta_freq_corrected']:6.4f}  "
              f"{err_raw:+7.4f}  {err_corr:+7.4f}  {syk_flag}", flush=True)

    # ── Save results ───────────────────────────────────────────────────────────
    calibration_curve = [
        {
            "delta_true": round(float(d), 4),
            "alpha_measured": round(float(a), 6),
            "alpha_continuum": round(float(c), 6),
            "bias": round(float(a - c), 6),
        }
        for d, a, c in zip(delta_grid, alpha_measured_cal, alpha_continuum_cal)
    ]

    result = {
        "experiment": "exp-050",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": {
            "H1_calibration_improves_consistency": {
                "statement": "MAE(Δ_freq_corrected − Δ_pos) < MAE(Δ_freq_raw − Δ_pos)",
                "result": "CONFIRMED" if h1_confirmed else "NOT CONFIRMED",
                "mae_raw": round(mae_raw, 6),
                "mae_corrected": round(mae_corr, 6),
            },
            "H2_cft_kinematics_after_calibration": {
                "statement": "SYK-near heads: Δ_freq_corrected ≈ 0.25 (|diff| < 0.05)",
                "result": "CONFIRMED" if h2_confirmed else "NOT CONFIRMED",
                "syk_near_delta_pos_median": round(syk_pos_med, 6) if not math.isnan(syk_pos_med) else None,
                "syk_near_delta_freq_raw_median": round(syk_raw_med, 6) if not math.isnan(syk_raw_med) else None,
                "syk_near_delta_freq_corrected_median": round(syk_corr_med, 6) if not math.isnan(syk_corr_med) else None,
                "n_syk_near": len(syk_valid),
            },
            "H3_monotone_calibration": {
                "statement": "α_measured(Δ_true) is monotone in Δ_true",
                "result": "CONFIRMED" if is_monotone else "VIOLATED",
                "ascending": is_monotone_ascending,
            },
            "H4_null_genuine_discrepancy": {
                "statement": "Residual |mean(Δ_freq_corr − Δ_pos)| > 0.05 after calibration",
                "result": "CONFIRMED" if h4_confirmed else "NOT CONFIRMED",
                "mean_residual": round(mean_residual_corr, 6),
                "std_residual": round(std_residual_corr, 6),
            },
        },
        "protocol": {
            "source_data": str(EXP045_RESULTS),
            "n_lag": N_LAG,
            "n_pad": N_PAD,
            "k_fit_range": [K_LOW, K_HIGH],
            "calibration_delta_range": [DELTA_MIN, DELTA_MAX],
            "n_calibration_points": N_DELTA_CAL,
            "interpolation_method": "cubic",
        },
        "calibration": {
            "description": "Synthetic calibration: G_true(r)=r^{-2Δ}, apply exp-045 DFT, measure α",
            "bias_at_delta_025": round(bias_at_025, 6),
            "curve": calibration_curve,
        },
        "summary": {
            "conformal_heads_analyzed": len(valid),
            "syk_near_heads": len(syk_valid),
            "all_conformal_delta_pos_median": round(all_delta_pos_med, 6),
            "all_conformal_delta_freq_raw_median": round(all_delta_raw_med, 6),
            "all_conformal_delta_freq_corrected_median": round(all_delta_corr_med, 6) if not math.isnan(all_delta_corr_med) else None,
        },
        "per_head": [
            {
                "layer": h["layer"],
                "head": h["head"],
                "delta_pos": h["delta_pos"],
                "alpha": h["alpha"],
                "delta_freq_raw": h["delta_freq"],
                "delta_freq_corrected": h["delta_freq_corrected"],
                "error_raw": round(h["delta_freq"] - h["delta_pos"], 6) if h["delta_freq"] else None,
                "error_corrected": round(h["delta_freq_corrected"] - h["delta_pos"], 6) if h["delta_freq_corrected"] else None,
                "syk_near": h.get("syk_near", False),
            }
            for h in valid
        ],
        "reference": {
            "exp045": {
                "conformal_heads": 44,
                "delta_pos_median": 0.255258,
                "alpha_median_raw": -0.819916,
                "delta_freq_median_raw": 0.090042,
                "pearson_r_pos_vs_freq_raw": 0.93967,
            }
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
