"""
Attention Spectral Analysis: Power Spectrum of Transformer Correlation Structure

Computes the spectral density (power spectrum) of the GPT-2 attention
correlation function to provide a frequency-domain characterization
that can be compared with neural spectral exponents.

If A(r) ~ r^(-γ) with γ = 2Δ, the spectral density is:
    S(ω) ~ |ω|^(γ-1) = |ω|^(2Δ-1)

For 2Δ = 0.50: S(ω) ~ ω^(-0.50) — a 1/f^β spectrum with β = 0.50

This spectral exponent is directly comparable to aperiodic spectral
exponents measured in neural data (EEG, LFP).

Ariel — March 29, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel
from scipy import stats
import json
import os

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 100
MIN_POS = 32
MAX_DX = 100
FIT_LOW = 3
FIT_HIGH = 50
OUTPUT_DIR = "research/physics/results_eigenvalue"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_head_attention_decay(attn_head, min_pos, max_dx):
    """Attention weight as function of distance for a single head."""
    seq = attn_head.shape[0]
    A = np.zeros(max_dx)
    counts = np.zeros(max_dx)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


def fit_power_law(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    """Fit y ~ dx^(-exponent). Returns (exponent, R²)."""
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_dx, log_y)
    return -slope, r_value**2


def compute_spectral_density(decay_curve, dx_arr):
    """
    Compute the spectral density (power spectrum) from the correlation function.
    S(ω) = Σ_r C(r) * exp(-iωr)
    
    The spectral density at frequency ω_k = 2πk/N.
    """
    N = len(decay_curve)
    corr_full = np.zeros(2 * N - 1)
    corr_full[N - 1:] = decay_curve
    corr_full[:N] = decay_curve[::-1]

    S = np.abs(np.fft.rfft(corr_full))
    freqs = np.fft.rfftfreq(len(corr_full))

    mask = freqs > 0
    return freqs[mask], S[mask]


def fit_spectral_exponent(freqs, S, fit_range=(0.01, 0.3)):
    """
    Fit S(ω) ~ ω^(-β). Returns (β, R²).
    """
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & (S > 0)
    if np.sum(mask) < 5:
        return np.nan, np.nan
    log_f = np.log10(freqs[mask])
    log_S = np.log10(S[mask])
    slope, intercept, r_value, _, _ = stats.linregress(log_f, log_S)
    return -slope, r_value**2


def main():
    print("=" * 70)
    print("ATTENTION SPECTRAL ANALYSIS")
    print("Power Spectrum of Transformer Correlation Structure")
    print("=" * 70)

    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    vocab_size = model.config.vocab_size

    A_heads = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
    }

    print(f"\nCollecting attention decay ({N_INPUTS} inputs, seq_len={SEQ_LEN})...")
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)

        if (inp_idx + 1) % 20 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS}")

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    dx_arr = np.arange(MAX_DX)

    # Analyze each head
    all_two_deltas = []
    all_betas = []
    per_head = []

    print(f"\n{'Layer':>5} {'Head':>4} {'2Δ':>8} {'R²':>6} {'β(spec)':>8} {'R²':>6} {'1-2Δ':>8} {'Match':>6}")
    print("-" * 60)

    for l in range(n_layers):
        for h in range(n_heads):
            decay = A_heads[l][h]

            two_delta, r2_pl = fit_power_law(dx_arr, decay, FIT_LOW, FIT_HIGH)

            freqs, S = compute_spectral_density(decay[:MAX_DX], dx_arr[:MAX_DX])
            beta, r2_spec = fit_spectral_exponent(freqs, S)

            if two_delta is not None and r2_pl is not None and r2_pl > 0.85:
                all_two_deltas.append(two_delta)
                predicted_beta = 1.0 - two_delta
                if np.isfinite(beta):
                    all_betas.append(beta)

            per_head.append({
                "layer": l, "head": h,
                "two_delta": float(two_delta) if two_delta else None,
                "r2_power_law": float(r2_pl) if r2_pl else None,
                "beta_spectral": float(beta) if np.isfinite(beta) else None,
                "r2_spectral": float(r2_spec) if np.isfinite(r2_spec) else None,
            })

            if l % 3 == 0 and h < 3:
                td = f"{two_delta:.4f}" if two_delta else "  ---"
                rp = f"{r2_pl:.3f}" if r2_pl else " ---"
                bs = f"{beta:.4f}" if np.isfinite(beta) else "  ---"
                rs = f"{r2_spec:.3f}" if np.isfinite(r2_spec) else " ---"
                if two_delta and np.isfinite(beta):
                    pred = 1.0 - two_delta
                    match = f"{abs(beta - pred):.3f}"
                else:
                    pred = None
                    match = " ---"
                pred_s = f"{pred:.4f}" if pred else "  ---"
                print(f"{l:>5} {h:>4} {td:>8} {rp:>6} {bs:>8} {rs:>6} {pred_s:>8} {match:>6}")

    two_deltas = np.array(all_two_deltas)
    betas = np.array([b for b in all_betas if np.isfinite(b)])

    # Random control
    print("\nComputing random control...")
    rand_betas = []
    for _ in range(50):
        raw = np.random.randn(SEQ_LEN, SEQ_LEN)
        mask_mat = np.triu(np.ones((SEQ_LEN, SEQ_LEN)), k=1) * (-1e9)
        raw = raw + mask_mat
        exp_raw = np.exp(raw - raw.max(axis=-1, keepdims=True))
        rand_attn = exp_raw / exp_raw.sum(axis=-1, keepdims=True)

        rand_decay = np.zeros(MAX_DX)
        counts = np.zeros(MAX_DX)
        for dx in range(MAX_DX):
            for i in range(max(MIN_POS, dx), SEQ_LEN):
                j = i - dx
                if j >= 0:
                    rand_decay[dx] += rand_attn[i, j]
                    counts[dx] += 1
        m = counts > 0
        rand_decay[m] /= counts[m]

        freqs_r, S_r = compute_spectral_density(rand_decay[:MAX_DX], dx_arr[:MAX_DX])
        beta_r, _ = fit_spectral_exponent(freqs_r, S_r)
        if np.isfinite(beta_r):
            rand_betas.append(beta_r)

    rand_betas = np.array(rand_betas)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n--- Correlation Function (Position Space) ---")
    print(f"  A(r) ~ r^(-2Δ)")
    print(f"  N heads (R² > 0.85): {len(two_deltas)}")
    print(f"  Median 2Δ:  {np.median(two_deltas):.4f}")
    print(f"  Mean 2Δ:    {np.mean(two_deltas):.4f} ± {np.std(two_deltas):.4f}")
    print(f"  IQR:        [{np.percentile(two_deltas, 25):.4f}, "
          f"{np.percentile(two_deltas, 75):.4f}]")

    print(f"\n--- Spectral Density (Frequency Space) ---")
    print(f"  S(ω) ~ ω^(-β)")
    print(f"  N heads:    {len(betas)}")
    print(f"  Median β:   {np.median(betas):.4f}")
    print(f"  Mean β:     {np.mean(betas):.4f} ± {np.std(betas):.4f}")
    print(f"  IQR:        [{np.percentile(betas, 25):.4f}, "
          f"{np.percentile(betas, 75):.4f}]")

    predicted_beta = 1.0 - np.median(two_deltas)
    print(f"\n--- Consistency Check ---")
    print(f"  Predicted β = 1 - 2Δ = 1 - {np.median(two_deltas):.4f} = {predicted_beta:.4f}")
    print(f"  Measured β  = {np.median(betas):.4f}")

    if len(two_deltas) > 10 and len(betas) > 10:
        valid = [(r["two_delta"], r["beta_spectral"])
                 for r in per_head
                 if r["two_delta"] is not None
                 and r["beta_spectral"] is not None
                 and r["r2_power_law"] is not None
                 and r["r2_power_law"] > 0.85]
        if len(valid) > 10:
            tds, bts = zip(*valid)
            tds, bts = np.array(tds), np.array(bts)
            predicted = 1.0 - tds
            corr = np.corrcoef(predicted, bts)[0, 1]
            print(f"  Per-head correlation(1-2Δ, β): r = {corr:.4f}")

    print(f"\n--- Random Control ---")
    if len(rand_betas) > 0:
        print(f"  Random β:   {np.mean(rand_betas):.4f} ± {np.std(rand_betas):.4f}")
    print(f"  Trained β:  {np.median(betas):.4f}")
    print(f"  Separation: {abs(np.median(betas) - np.mean(rand_betas)):.4f}")

    print(f"\n--- Key Numbers for Outreach ---")
    print(f"  Transformer correlation exponent:  μ = 2Δ = {np.median(two_deltas):.4f}")
    print(f"  Transformer spectral exponent:     β = {np.median(betas):.4f}")
    print(f"  Prediction for brain data:         μ = 0.50 (same universality class)")
    print(f"  Theory (Δ* = 1/4):                 μ = 0.50, β = 0.50")

    summary = {
        "model": "GPT-2",
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "correlation_exponent": {
            "quantity": "2*Delta from A(r) ~ r^(-2Delta)",
            "n_heads": len(two_deltas),
            "median": float(np.median(two_deltas)),
            "mean": float(np.mean(two_deltas)),
            "std": float(np.std(two_deltas)),
            "q25": float(np.percentile(two_deltas, 25)),
            "q75": float(np.percentile(two_deltas, 75)),
        },
        "spectral_exponent": {
            "quantity": "beta from S(omega) ~ omega^(-beta)",
            "n_heads": len(betas),
            "median": float(np.median(betas)),
            "mean": float(np.mean(betas)),
            "std": float(np.std(betas)),
        },
        "random_control": {
            "mean_beta": float(np.mean(rand_betas)) if len(rand_betas) > 0 else None,
            "std_beta": float(np.std(rand_betas)) if len(rand_betas) > 0 else None,
        },
        "per_head": per_head,
    }

    out_path = os.path.join(OUTPUT_DIR, "spectral_analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
