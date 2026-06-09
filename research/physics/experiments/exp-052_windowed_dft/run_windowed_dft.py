"""
exp-052 — Windowed DFT spectral estimator (GPT-2)

Pre-stated hypotheses (2026-06-08):

  H1 (windowing collapses gap): Applying Hanning window + floor subtraction
     before DFT brings Δ_freq within 0.05 of Δ_pos for SYK-near heads.
     From exp-050: raw Δ_freq ≈ 0.083 for SYK-near, Δ_pos ≈ 0.234.
     H1 requires Δ_freq_corrected ≥ 0.184 (within 0.05 of 0.234).

  H2 (floor subtraction independently matters): Windowed+floor Δ_freq
     differs from windowed-only Δ_freq by ≥ 0.03 (floor subtraction
     makes a measurable independent contribution).

  H0 (gap persists): The gap between Δ_freq and Δ_pos remains ≥ 0.10
     even after windowing and floor subtraction. The spectral method
     measures fundamentally different structure from position-space Δ.

Physical background (from exp-050 analysis):
  Two-component bias identified:
  (1) Finite-window DFT bias at Δ=0.25: −0.147 in α units (correctable
      by Hanning window, which tapers the profile to zero at the edges).
  (2) Structural bias ~−0.186 in α units (from attention floor at large r,
      normalization constraints, and imperfect power-law form — partial
      treatment via floor subtraction).

  Hanning window expected to address component (1).
  Floor subtraction expected to address part of component (2).
  Normalization constraint and power-law imperfection remain.

  Honest prediction before running: H0 likely (gap persists), H2 possibly
  confirmed, H1 unlikely given multiple sources of structural bias.

Protocol:
  - Model: gpt2 (cached, 12 layers × 12 heads = 144 heads)
  - Same forward passes as exp-045: SEQ_LEN=256, N_INPUTS=50, RNG_SEED=42,
    MIN_POS=32, MAX_DX=56
  - Per head: save full G_pos(r) lag profile (r=0..55)
  - Three DFT conditions:
    (A) raw: same as exp-045 (no windowing, no floor)
    (B) windowed: Hanning window applied to G_pos[1..55] before DFT
    (C) windowed+floor: floor subtraction then Hanning window then DFT
  - Floor estimation: mean of G_pos[r ≥ FLOOR_CUTOFF]
  - Compare Δ_freq for each condition to Δ_pos

Compare to:
  exp-045  raw DFT:     SYK-near Δ_freq median ≈ 0.083, r=0.940 (pos vs freq ordering)
  exp-050  calibrated:  SYK-near Δ_freq_corr   ≈ 0.135  (finite-window corrected)
  exp-007  pos-space:   SYK-near Δ_pos median  ≈ 0.249
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

# ── constants (same as exp-045/007 where applicable) ──────────────────────────
SEQ_LEN       = 256
N_INPUTS      = 50
MAX_DX        = 56
MIN_POS       = 32
FIT_LOW       = 3
FIT_HIGH      = 50
R2_THRESHOLD  = 0.90
RNG_SEED      = 42
SYK_NEAR_TOL  = 0.05

# Spectral fit range
K_LOW  = 2
K_HIGH = 18
N_PAD  = 256

# Floor estimation: use lags ≥ this threshold for floor estimate
FLOOR_CUTOFF = 28   # lags 28..55 → ~28 points for floor estimate

OUT_DIR      = Path("research/physics/experiments/exp-052_windowed_dft")
RESULTS_FILE = OUT_DIR / "results.json"


# ── position-space fitting (exp-007 method) ────────────────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    """Mean attention weight G_>(r) as a function of lag r."""
    seq    = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A.astype(np.float64)


def fit_power_law_pos(dx_arr: np.ndarray, y_arr: np.ndarray,
                      low: int, high: int) -> tuple[float, float]:
    """Fit G(r) ~ r^{-2Δ} in position space. Returns (Δ, R²)."""
    mask = (dx_arr >= low) & (dx_arr < high) & (y_arr > 1e-20)
    if mask.sum() < 5:
        return float("nan"), 0.0
    log_x = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask].astype(float))
    A_mat = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    pred   = A_mat @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    delta  = float(-coeffs[1] / 2.0)
    return delta, r2


# ── spectral fitting helpers ───────────────────────────────────────────────────

def estimate_floor(G_pos: np.ndarray, cutoff: int) -> float:
    """Estimate floor as mean of G_pos[r >= cutoff]."""
    vals = G_pos[cutoff:]
    vals = vals[vals > 0]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def hanning_window(n: int) -> np.ndarray:
    """Hanning window of length n."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def compute_dft_and_fit(G_pos: np.ndarray, n_pad: int,
                        k_low: int, k_high: int,
                        apply_window: bool = False,
                        subtract_floor: bool = False,
                        floor_cutoff: int = 28
                        ) -> tuple[float, float, float]:
    """
    Compute spectral exponent from G_pos lag profile.

    Returns (alpha, r2_freq, delta_freq).
    alpha: spectral exponent (log|Ĝ| ~ α log(k))
    delta_freq = (alpha + 1) / 2
    """
    # Use r=1..MAX_DX-1 (skip r=0)
    profile = G_pos[1:].astype(np.float64).copy()
    n       = len(profile)

    if subtract_floor:
        # Use G_pos (full, including r=0 start) for floor estimation
        floor_val = estimate_floor(G_pos, floor_cutoff)
        profile = profile - floor_val
        # Clip to avoid negative values from noise
        profile = np.maximum(profile, 0.0)

    if apply_window:
        w       = hanning_window(n)
        profile = profile * w

    # Zero-pad to N_PAD
    padded  = np.zeros(n_pad, dtype=np.float64)
    padded[:n] = profile
    spectrum = np.abs(np.fft.rfft(padded))

    # Fit log|Ĝ(k)| ~ α log(k) in [k_low, k_high)
    k_arr  = np.arange(len(spectrum))
    mask   = (k_arr >= k_low) & (k_arr < k_high) & (spectrum > 1e-30)
    if mask.sum() < 4:
        return float("nan"), 0.0, float("nan")
    log_k  = np.log(k_arr[mask].astype(float))
    log_s  = np.log(spectrum[mask])
    A_mat  = np.column_stack([np.ones_like(log_k), log_k])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_s, rcond=None)
    pred   = A_mat @ coeffs
    ss_res = np.sum((log_s - pred) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    alpha  = float(coeffs[1])
    delta  = (alpha + 1.0) / 2.0
    return alpha, r2, delta


# ── main ───────────────────────────────────────────────────────────────────────

def median_or_nan(vals: list[float]) -> float:
    finite = [v for v in vals if not math.isnan(v)]
    return float(statistics.median(finite)) if finite else float("nan")


def pearson_r(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 4:
        return float("nan")
    xv = np.array([p[0] for p in pairs])
    yv = np.array([p[1] for p in pairs])
    return float(np.corrcoef(xv, yv)[0, 1])


def main() -> None:
    print("exp-052: Windowed DFT spectral estimator (GPT-2)", flush=True)
    print(f"  Pre-stated H1: windowed+floor Δ_freq within 0.05 of Δ_pos", flush=True)
    print(f"  Pre-stated H2: floor subtraction independently contributes ≥ 0.03", flush=True)
    print(f"  Pre-stated H0: gap persists ≥ 0.10 despite windowing+floor", flush=True)
    print(f"  Floor cutoff: r ≥ {FLOOR_CUTOFF}", flush=True)
    print(flush=True)

    print("Loading GPT-2 (eager)...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    n_layers    = model.config.n_layer
    n_heads     = model.config.n_head
    vocab_size  = model.config.vocab_size
    total_heads = n_layers * n_heads
    print(f"  {n_layers} layers × {n_heads} heads = {total_heads} heads", flush=True)

    rng    = np.random.default_rng(RNG_SEED)
    dx_arr = np.arange(MAX_DX)

    # Accumulate lag profiles
    A_heads: dict[int, dict[int, np.ndarray]] = {
        l: {h: np.zeros(MAX_DX, dtype=np.float64) for h in range(n_heads)}
        for l in range(n_layers)
    }

    print(f"Running {N_INPUTS} forward passes...", flush=True)
    for inp_idx in range(N_INPUTS):
        token_ids = torch.tensor(
            rng.integers(0, vocab_size, size=(1, SEQ_LEN)), dtype=torch.long
        )
        with torch.no_grad():
            outputs = model(token_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].float().numpy()  # (n_heads, seq, seq)
            for h in range(n_heads):
                A_heads[l][h] += compute_lag_profile(attn[h], MIN_POS, MAX_DX)

        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS} done", flush=True)

    # Average over inputs
    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    print("Computing position-space fits + three DFT conditions...", flush=True)
    per_head: list[dict] = []

    for l in range(n_layers):
        for h in range(n_heads):
            G = A_heads[l][h]

            # Position-space
            delta_pos, r2_pos = fit_power_law_pos(dx_arr, G, FIT_LOW, FIT_HIGH)

            # Floor value (for all conditions)
            floor_val = estimate_floor(G, FLOOR_CUTOFF)

            # (A) Raw DFT — replicates exp-045
            alpha_raw, r2_raw, df_raw = compute_dft_and_fit(
                G, N_PAD, K_LOW, K_HIGH,
                apply_window=False, subtract_floor=False
            )

            # (B) Windowed only
            alpha_win, r2_win, df_win = compute_dft_and_fit(
                G, N_PAD, K_LOW, K_HIGH,
                apply_window=True, subtract_floor=False
            )

            # (C) Floor subtraction + Hanning window
            alpha_wf, r2_wf, df_wf = compute_dft_and_fit(
                G, N_PAD, K_LOW, K_HIGH,
                apply_window=True, subtract_floor=True,
                floor_cutoff=FLOOR_CUTOFF
            )

            is_conformal = (r2_pos >= R2_THRESHOLD
                            and not math.isnan(delta_pos)
                            and delta_pos > 0)
            is_syk_near  = (is_conformal
                            and abs(delta_pos - 0.25) <= SYK_NEAR_TOL)

            per_head.append({
                "layer": l, "head": h,
                "delta_pos":    round(delta_pos, 6) if not math.isnan(delta_pos) else None,
                "r2_pos":       round(r2_pos, 6),
                "floor_val":    round(floor_val, 8),
                # raw
                "alpha_raw":    round(alpha_raw, 6) if not math.isnan(alpha_raw) else None,
                "delta_freq_raw": round(df_raw, 6) if not math.isnan(df_raw) else None,
                # windowed
                "alpha_win":    round(alpha_win, 6) if not math.isnan(alpha_win) else None,
                "delta_freq_win": round(df_win, 6) if not math.isnan(df_win) else None,
                # windowed + floor
                "alpha_wf":     round(alpha_wf, 6) if not math.isnan(alpha_wf) else None,
                "delta_freq_wf": round(df_wf, 6) if not math.isnan(df_wf) else None,
                "conformal": is_conformal,
                "syk_near":  is_syk_near,
            })

    # ── summaries ──────────────────────────────────────────────────────────────
    conformal = [r for r in per_head if r["conformal"]]
    syk_near  = [r for r in per_head if r["syk_near"]]

    n_conformal = len(conformal)
    n_syk       = len(syk_near)

    def med(lst: list[dict], key: str) -> float:
        return median_or_nan([r[key] for r in lst if r[key] is not None])

    def pr(lst: list[dict], k1: str, k2: str) -> float:
        return pearson_r(
            [r[k1] for r in lst if r[k1] is not None and r[k2] is not None],
            [r[k2] for r in lst if r[k1] is not None and r[k2] is not None],
        )

    # ── print results ──────────────────────────────────────────────────────────
    print(f"\n=== RESULTS ===", flush=True)
    print(f"Conformal heads (pos R²>{R2_THRESHOLD}):      {n_conformal}/{total_heads}", flush=True)
    print(f"SYK-near heads (|Δ_pos−0.25|≤{SYK_NEAR_TOL}): {n_syk}/{total_heads}", flush=True)
    print(flush=True)

    for label, subset in [("Conformal heads", conformal), ("SYK-near heads", syk_near)]:
        n_sub = len(subset)
        dp  = med(subset, "delta_pos")
        raw = med(subset, "delta_freq_raw")
        win = med(subset, "delta_freq_win")
        wf  = med(subset, "delta_freq_wf")
        gap_raw = dp - raw if not math.isnan(dp) and not math.isnan(raw) else float("nan")
        gap_win = dp - win if not math.isnan(dp) and not math.isnan(win) else float("nan")
        gap_wf  = dp - wf  if not math.isnan(dp) and not math.isnan(wf)  else float("nan")
        h2_delta = win - raw if not math.isnan(win) and not math.isnan(raw) else float("nan")  # windowed only
        floor_contribution = wf - win if not math.isnan(wf) and not math.isnan(win) else float("nan")
        print(f"{label} (n={n_sub}):", flush=True)
        print(f"  Δ_pos  median          = {dp:.4f}", flush=True)
        print(f"  Δ_freq raw             = {raw:.4f}   (gap = {gap_raw:.4f}) [replicates exp-045]", flush=True)
        print(f"  Δ_freq windowed        = {win:.4f}   (gap = {gap_win:.4f}) [Hanning only]", flush=True)
        print(f"  Δ_freq windowed+floor  = {wf:.4f}   (gap = {gap_wf:.4f}) [Hanning + floor sub]", flush=True)
        print(f"  Windowing contribution  = {h2_delta:+.4f} (win − raw)", flush=True)
        print(f"  Floor contribution      = {floor_contribution:+.4f} (wf  − win)", flush=True)
        print(flush=True)

    # H-assessment
    syk_dp = med(syk_near, "delta_pos")
    syk_wf = med(syk_near, "delta_freq_wf")
    gap_wf = syk_dp - syk_wf if not math.isnan(syk_dp) and not math.isnan(syk_wf) else float("nan")
    win_contrib = med(syk_near, "delta_freq_win") - med(syk_near, "delta_freq_raw")
    floor_contrib = med(syk_near, "delta_freq_wf") - med(syk_near, "delta_freq_win")

    print("=== HYPOTHESIS VERDICTS ===", flush=True)
    if not math.isnan(gap_wf):
        h1_verdict = "CONFIRMED" if gap_wf <= 0.05 else "NOT CONFIRMED"
        print(f"H1 (gap < 0.05): {h1_verdict}   (gap = {gap_wf:.4f})", flush=True)
    if not math.isnan(floor_contrib):
        h2_verdict = "CONFIRMED" if abs(floor_contrib) >= 0.03 else "NOT CONFIRMED"
        print(f"H2 (floor ≥ 0.03): {h2_verdict}  (floor contribution = {floor_contrib:+.4f})", flush=True)
    if not math.isnan(gap_wf):
        h0_verdict = "CONFIRMED" if gap_wf >= 0.10 else "NOT CONFIRMED"
        print(f"H0 (gap persists ≥ 0.10): {h0_verdict}   (gap = {gap_wf:.4f})", flush=True)

    # Correlation (pos vs windowed+floor freq)
    r_raw = pr(conformal, "delta_pos", "delta_freq_raw")
    r_wf  = pr(conformal, "delta_pos", "delta_freq_wf")
    print(f"\nPearson r (Δ_pos vs Δ_freq_raw):   {r_raw:.4f}  [exp-045 baseline]", flush=True)
    print(f"Pearson r (Δ_pos vs Δ_freq_wf):    {r_wf:.4f}  [after windowing+floor]", flush=True)

    # ── save results ───────────────────────────────────────────────────────────
    result = {
        "experiment": "exp-052",
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": {
            "H1": "Windowed+floor Δ_freq within 0.05 of Δ_pos for SYK-near heads.",
            "H2": "Floor subtraction contributes ≥ 0.03 to Δ_freq beyond windowing alone.",
            "H0": "Gap persists ≥ 0.10 even after windowing+floor — spectral measures differ.",
        },
        "protocol": {
            "model": "gpt2", "attn_implementation": "eager",
            "seq_len": SEQ_LEN, "n_inputs": N_INPUTS, "rng_seed": RNG_SEED,
            "min_pos": MIN_POS, "max_dx": MAX_DX,
            "pos_fit_range": [FIT_LOW, FIT_HIGH], "r2_threshold": R2_THRESHOLD,
            "fft_n_pad": N_PAD, "spectral_fit_range": [K_LOW, K_HIGH],
            "floor_cutoff": FLOOR_CUTOFF,
            "conditions": [
                "raw: no windowing, no floor (replicates exp-045)",
                "windowed: Hanning window before zero-padding",
                "windowed+floor: floor subtraction then Hanning then DFT",
            ],
        },
        "references": {
            "exp045_raw_syk_near_alpha_median": -0.833,
            "exp045_raw_syk_near_delta_freq_median": 0.083,
            "exp050_calibrated_syk_near_delta_freq": 0.135,
            "exp007_syk_near_delta_pos_median": 0.249,
        },
        "summary": {
            "total_heads":    total_heads,
            "conformal_heads": n_conformal,
            "syk_near_heads":  n_syk,
            "conformal": {
                "delta_pos_med":      med(conformal, "delta_pos"),
                "delta_freq_raw_med": med(conformal, "delta_freq_raw"),
                "delta_freq_win_med": med(conformal, "delta_freq_win"),
                "delta_freq_wf_med":  med(conformal, "delta_freq_wf"),
                "pearson_r_pos_raw":  r_raw,
                "pearson_r_pos_wf":   r_wf,
            },
            "syk_near": {
                "delta_pos_med":       med(syk_near, "delta_pos"),
                "delta_freq_raw_med":  med(syk_near, "delta_freq_raw"),
                "delta_freq_win_med":  med(syk_near, "delta_freq_win"),
                "delta_freq_wf_med":   med(syk_near, "delta_freq_wf"),
                "gap_raw":             med(syk_near, "delta_pos") - med(syk_near, "delta_freq_raw"),
                "gap_win":             med(syk_near, "delta_pos") - med(syk_near, "delta_freq_win"),
                "gap_wf":              gap_wf,
                "windowing_contribution": win_contrib,
                "floor_contribution":  floor_contrib,
            },
            "verdicts": {
                "H1_gap_lt_005": gap_wf <= 0.05 if not math.isnan(gap_wf) else None,
                "H2_floor_ge_003": abs(floor_contrib) >= 0.03 if not math.isnan(floor_contrib) else None,
                "H0_gap_ge_010": gap_wf >= 0.10 if not math.isnan(gap_wf) else None,
            },
        },
        "per_head": per_head,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
