"""
exp-045 — G_> vs G_< spectral function (GPT-2, cached)

Hypothesis (pre-stated, 2026-06-02):

  1. CONSISTENCY: For GPT-2 conformal heads (R²>0.90, from exp-007), the
     frequency-space spectral density |Ĝ_>(k)| ~ k^α satisfies α ≈ 2Δ_pos − 1.
     For Δ_pos ≈ 0.25 (SYK q=4), predicted spectral exponent α_pred ≈ −0.50.

  2. CORRELATION: Δ_freq = (α+1)/2 correlates with Δ_pos across the 44 conformal
     heads (Pearson r ≥ 0.7). If the position-space measurement is genuine
     conformal scaling, both routes to Δ should agree.

  3. ZERO-TEMPERATURE: G_< = 0 (causal attention, no backward connections).
     This corresponds to the β→∞ (T=0) SYK ground state, not a finite-temperature
     thermal state. Finite-T SYK would require G_< ≠ 0 — impossible for causal
     attention by construction.

  4. NULL: If position-space power-law is a fitting artifact, α will be
     uncorrelated with Δ_pos and the spectral density will not follow a power law.

Physical background:
  - G_>(r) = mean A_h(i, i-r) over sequence positions i — the forward/causal
    Wightman function in discrete position space.
  - G_<(r) = 0 for causal transformers (no future attention). Spectral function
    ρ(ω) = G_>(ω) − G_<(ω) = G_>(ω).
  - CFT kinematics: if G_>(r) ~ r^{−2Δ}, then |Ĝ_>(k)| ~ k^{2Δ−1} (continuum).
  - Zero-temperature SYK: spectral weight supported on ω > 0 only.

Protocol:
  - Model: gpt2 (cached, 12 layers × 12 heads = 144 heads)
  - Same as exp-007: SEQ_LEN=256, N_INPUTS=50, RNG_SEED=42, MIN_POS=32, MAX_DX=56
  - For each head: compute G_>(r) (lag profile), fit position-space power law
    (exp-007 method), then take DFT and fit spectral exponent α.
  - DFT: numpy.fft.rfft on the lag profile G_>(r=1..MAX_DX-1). Zero-pad to
    improve frequency resolution.
  - Spectral fit: log|Ĝ(k)| ~ α log(k) for k in [K_LOW, K_HIGH].
  - Report: per-head (Δ_pos, Δ_freq, α, r² for each fit), summary statistics,
    Pearson correlation between Δ_pos and Δ_freq for conformal heads.

Compare to:
  exp-007  GPT-2 softmax:  44/144 conformal, Δ_med=0.2493, Δ_SYK=0.2493
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

# ── constants (same as exp-007) ────────────────────────────────────────────────
SEQ_LEN   = 256
N_INPUTS  = 50
MAX_DX    = 56
MIN_POS   = 32
FIT_LOW   = 3
FIT_HIGH  = 50
R2_THRESHOLD = 0.90
RNG_SEED  = 42
SYK_NEAR_TOL = 0.05

# Spectral fit range: k indices into the rfft output (k=0 is DC, skip)
K_LOW  = 2   # skip DC (k=0) and first mode (k=1, strongly influenced by mean)
K_HIGH = 18  # stop before high-k lattice artifacts

# Zero-padding factor: rfft on N_PAD points gives N_PAD//2+1 frequency bins
N_PAD  = 256  # pad the MAX_DX-length profile to this length

OUT_DIR      = Path("research/physics/experiments/exp-045_gpt2_spectral_function")
RESULTS_FILE = OUT_DIR / "results.json"


# ── position-space power-law fitting (exp-007 method) ─────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    """Mean attention weight G_>(r) as a function of lag r."""
    seq = attn_head.shape[0]
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
    return A.astype(np.float32)


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
    pred = A_mat @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2    = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    delta = float(-coeffs[1] / 2.0)
    return delta, r2


# ── frequency-space spectral function fitting ──────────────────────────────────

def compute_spectral_function(G_pos: np.ndarray, n_pad: int) -> np.ndarray:
    """
    Compute |Ĝ_>(k)| via zero-padded real FFT.

    G_pos: lag profile G_>(r), r=0..MAX_DX-1.
    Returns amplitudes |Ĝ(k)| for k=0..n_pad//2.
    """
    # Use r=1..MAX_DX-1 (skip r=0, which is self-attention / diagonal)
    profile = G_pos[1:].astype(np.float64)
    # Zero-pad
    padded = np.zeros(n_pad, dtype=np.float64)
    padded[: len(profile)] = profile
    spectrum = np.abs(np.fft.rfft(padded))
    return spectrum


def fit_spectral_exponent(spectrum: np.ndarray, k_low: int, k_high: int) -> tuple[float, float]:
    """
    Fit log|Ĝ(k)| ~ α log(k) in [k_low, k_high].
    Returns (α, R²).  Δ_freq = (α + 1) / 2.
    """
    k_arr  = np.arange(len(spectrum))
    mask   = (k_arr >= k_low) & (k_arr < k_high) & (spectrum > 1e-30)
    if mask.sum() < 4:
        return float("nan"), 0.0
    log_k  = np.log(k_arr[mask].astype(float))
    log_s  = np.log(spectrum[mask])
    A_mat  = np.column_stack([np.ones_like(log_k), log_k])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_s, rcond=None)
    pred   = A_mat @ coeffs
    ss_res = np.sum((log_s - pred) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    alpha  = float(coeffs[1])
    return alpha, r2


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("exp-045: G_> vs G_< spectral function (GPT-2)", flush=True)
    print(f"  Position fit: r∈[{FIT_LOW},{FIT_HIGH}), R²>{R2_THRESHOLD}", flush=True)
    print(f"  Spectral fit: k∈[{K_LOW},{K_HIGH}), N_PAD={N_PAD}", flush=True)
    print(f"  SYK prediction: Δ=0.25 → α=2Δ−1=−0.50", flush=True)

    print("Loading GPT-2 (eager)...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    n_layers   = model.config.n_layer    # 12
    n_heads    = model.config.n_head     # 12
    vocab_size = model.config.vocab_size
    total_heads = n_layers * n_heads     # 144
    print(f"  {n_layers} layers × {n_heads} heads = {total_heads} heads", flush=True)

    rng    = np.random.default_rng(RNG_SEED)
    dx_arr = np.arange(MAX_DX)

    # Accumulate lag profiles: [layer][head] → sum over inputs
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

    # Per-head analysis
    print("Computing spectral functions and fitting...", flush=True)
    per_head_results: list[dict] = []

    for l in range(n_layers):
        for h in range(n_heads):
            G = A_heads[l][h]

            # Position-space fit
            delta_pos, r2_pos = fit_power_law_pos(dx_arr, G, FIT_LOW, FIT_HIGH)

            # Spectral function
            spectrum = compute_spectral_function(G, N_PAD)
            alpha, r2_freq = fit_spectral_exponent(spectrum, K_LOW, K_HIGH)

            delta_freq = float((alpha + 1.0) / 2.0) if not math.isnan(alpha) else float("nan")
            alpha_pred = float(2.0 * delta_pos - 1.0) if not math.isnan(delta_pos) else float("nan")

            per_head_results.append({
                "layer":      l,
                "head":       h,
                "delta_pos":  round(delta_pos, 6) if not math.isnan(delta_pos) else None,
                "r2_pos":     round(r2_pos, 6),
                "alpha":      round(alpha, 6) if not math.isnan(alpha) else None,
                "r2_freq":    round(r2_freq, 6),
                "delta_freq": round(delta_freq, 6) if not math.isnan(delta_freq) else None,
                "alpha_pred": round(alpha_pred, 6) if not math.isnan(alpha_pred) else None,
                "conformal":  (r2_pos >= R2_THRESHOLD and not math.isnan(delta_pos) and delta_pos > 0),
                "syk_near":   (
                    r2_pos >= R2_THRESHOLD and not math.isnan(delta_pos) and delta_pos > 0
                    and abs(delta_pos - 0.25) <= SYK_NEAR_TOL
                ),
            })

    # Summary over conformal heads
    conformal_heads = [r for r in per_head_results if r["conformal"]]
    syk_near_heads  = [r for r in per_head_results if r["syk_near"]]

    n_conformal = len(conformal_heads)
    n_syk_near  = len(syk_near_heads)

    # Filter to heads with valid spectral fits
    valid_both = [
        r for r in conformal_heads
        if r["delta_pos"] is not None and r["delta_freq"] is not None
    ]

    delta_pos_vals  = [r["delta_pos"] for r in valid_both]
    delta_freq_vals = [r["delta_freq"] for r in valid_both]
    alpha_vals      = [r["alpha"] for r in valid_both]
    alpha_pred_vals = [r["alpha_pred"] for r in valid_both]

    # Pearson correlation between Δ_pos and Δ_freq
    pearson_r = float("nan")
    if len(delta_pos_vals) >= 4:
        dp = np.array(delta_pos_vals)
        df = np.array(delta_freq_vals)
        pearson_r = float(np.corrcoef(dp, df)[0, 1])

    # Median values
    delta_pos_med  = float(statistics.median(delta_pos_vals))  if delta_pos_vals  else float("nan")
    delta_freq_med = float(statistics.median(delta_freq_vals)) if delta_freq_vals else float("nan")
    alpha_med      = float(statistics.median(alpha_vals))      if alpha_vals      else float("nan")
    alpha_pred_med = float(statistics.median(alpha_pred_vals)) if alpha_pred_vals else float("nan")

    # For SYK-near heads specifically
    syk_valid = [r for r in syk_near_heads if r["delta_freq"] is not None]
    syk_alpha_vals = [r["alpha"] for r in syk_valid]
    syk_alpha_med  = float(statistics.median(syk_alpha_vals)) if syk_alpha_vals else float("nan")
    syk_dfq_vals   = [r["delta_freq"] for r in syk_valid]
    syk_dfq_med    = float(statistics.median(syk_dfq_vals)) if syk_dfq_vals else float("nan")

    # Print results
    print(f"\n=== RESULTS ===", flush=True)
    print(f"Conformal heads (pos-space R²>{R2_THRESHOLD}): {n_conformal}/{total_heads}", flush=True)
    print(f"SYK-near heads (|Δ_pos−0.25|≤{SYK_NEAR_TOL}): {n_syk_near}/{total_heads}", flush=True)
    print(f"Heads with valid spectral fit (both):           {len(valid_both)}/{n_conformal}", flush=True)
    print(flush=True)
    print(f"Conformal heads (valid both fits):", flush=True)
    print(f"  Δ_pos  median = {delta_pos_med:.4f}  (position-space, exp-007 method)", flush=True)
    print(f"  α      median = {alpha_med:.4f}       (spectral exponent)", flush=True)
    print(f"  Δ_freq median = {delta_freq_med:.4f}  (= (α+1)/2)", flush=True)
    print(f"  α_pred median = {alpha_pred_med:.4f}  (= 2Δ_pos−1, CFT prediction)", flush=True)
    print(flush=True)
    print(f"SYK-near heads (Δ_pos≈0.25, with spectral fit):", flush=True)
    print(f"  n = {len(syk_valid)}", flush=True)
    print(f"  α  median = {syk_alpha_med:.4f}  (predicted: −0.50)", flush=True)
    print(f"  Δ_freq median = {syk_dfq_med:.4f}  (predicted: 0.25)", flush=True)
    print(flush=True)
    print(f"Correlation (Δ_pos vs Δ_freq) across conformal heads: r = {pearson_r:.4f}", flush=True)
    print(flush=True)

    # Physical interpretation
    print("Physical interpretation:", flush=True)
    print(f"  G_<: identically zero (causal attention → zero-temperature SYK).", flush=True)
    print(f"  Spectral function ρ(ω) = G_>(ω) supported on ω>0 only.", flush=True)
    if not math.isnan(syk_alpha_med) and abs(syk_alpha_med - (-0.5)) < 0.15:
        print(f"  ✓ α ≈ −0.50 for SYK-near heads: spectral density consistent with Δ=0.25.", flush=True)
    elif not math.isnan(syk_alpha_med):
        print(f"  ✗ α = {syk_alpha_med:.3f} ≠ −0.50: spectral density inconsistent with position-space Δ.", flush=True)
    if not math.isnan(pearson_r):
        if pearson_r >= 0.7:
            print(f"  ✓ High correlation (r={pearson_r:.3f}): position- and frequency-space Δ agree.", flush=True)
        elif pearson_r >= 0.4:
            print(f"  ~ Moderate correlation (r={pearson_r:.3f}): partial consistency.", flush=True)
        else:
            print(f"  ✗ Low correlation (r={pearson_r:.3f}): measurements disagree — possible artifact.", flush=True)

    print(flush=True)
    print("Per-head breakdown (conformal heads with spectral fit):", flush=True)
    print(f"  {'L':>3} {'H':>3}  {'Δ_pos':>7}  {'R²_p':>6}  {'α':>7}  {'α_pred':>7}  {'Δ_freq':>7}  {'R²_f':>6}  SYK", flush=True)
    for r in valid_both:
        syk_flag = "★" if r["syk_near"] else " "
        print(f"  L{r['layer']:2d} H{r['head']:2d}  "
              f"{r['delta_pos']:7.4f}  "
              f"{r['r2_pos']:6.4f}  "
              f"{r['alpha']:7.4f}  "
              f"{r['alpha_pred']:7.4f}  "
              f"{r['delta_freq']:7.4f}  "
              f"{r['r2_freq']:6.4f}  {syk_flag}", flush=True)

    result = {
        "experiment": "exp-045",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": {
            "consistency": (
                "For GPT-2 conformal heads (R²>0.90, exp-007), spectral exponent α ≈ 2Δ_pos−1. "
                "SYK-near heads (Δ≈0.25) predicted α≈−0.50."
            ),
            "correlation": (
                "Δ_freq = (α+1)/2 correlates with Δ_pos across conformal heads (Pearson r≥0.7)."
            ),
            "zero_temperature": (
                "G_< = 0 (causal attention). Spectral function supported on ω>0 only (T_eff=0 SYK)."
            ),
            "null": (
                "If position-space power-law is artifact: α uncorrelated with Δ_pos, no power-law spectrum."
            ),
        },
        "protocol": {
            "model": "gpt2",
            "attn_implementation": "eager",
            "seq_len": SEQ_LEN,
            "n_inputs": N_INPUTS,
            "rng_seed": RNG_SEED,
            "min_pos": MIN_POS,
            "max_dx": MAX_DX,
            "pos_fit_range": [FIT_LOW, FIT_HIGH],
            "r2_threshold": R2_THRESHOLD,
            "fft_n_pad": N_PAD,
            "spectral_fit_range": [K_LOW, K_HIGH],
            "note": (
                "Same forward-pass protocol as exp-007. Spectral function computed via "
                "zero-padded rfft of G_>(r=1..MAX_DX-1). G_< = 0 by causal attention construction."
            ),
        },
        "syk_prediction": {
            "delta": 0.25,
            "alpha_pred": -0.50,
            "note": "CFT kinematics: Ĝ(k) ~ k^{2Δ-1} for G(r) ~ r^{-2Δ}",
        },
        "summary": {
            "total_heads": total_heads,
            "conformal_heads": n_conformal,
            "syk_near_heads": n_syk_near,
            "valid_spectral_fits": len(valid_both),
            "conformal_delta_pos_median": round(delta_pos_med, 6) if not math.isnan(delta_pos_med) else None,
            "conformal_alpha_median": round(alpha_med, 6) if not math.isnan(alpha_med) else None,
            "conformal_alpha_pred_median": round(alpha_pred_med, 6) if not math.isnan(alpha_pred_med) else None,
            "conformal_delta_freq_median": round(delta_freq_med, 6) if not math.isnan(delta_freq_med) else None,
            "syk_near_alpha_median": round(syk_alpha_med, 6) if not math.isnan(syk_alpha_med) else None,
            "syk_near_delta_freq_median": round(syk_dfq_med, 6) if not math.isnan(syk_dfq_med) else None,
            "pearson_r_pos_vs_freq": round(pearson_r, 6) if not math.isnan(pearson_r) else None,
        },
        "physical_interpretation": {
            "G_less": "identically zero — causal attention enforces zero-temperature SYK",
            "spectral_support": "ω > 0 only (no backward excitations)",
            "temperature": "T_eff = 0, equivalent to β→∞ SYK ground state",
        },
        "per_head": per_head_results,
        "reference": {
            "exp007": {
                "conformal_heads": 44, "total": 144,
                "delta_median": 0.2493, "syk_near": 44, "syk_near_median": 0.2493,
            }
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
