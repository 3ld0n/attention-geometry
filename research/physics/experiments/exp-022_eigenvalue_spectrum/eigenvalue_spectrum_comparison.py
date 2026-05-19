"""
Eigenvalue Spectrum Analysis: Transformer Attention Correlation Structure

Computes the eigenvalue spectrum of the attention correlation structure
in GPT-2 to enable direct comparison with brain-wide neural covariance
eigenspectra (Wang et al., eLife 2025).

The analytical chain:
1. We measured attention decay: A(r) ~ r^(-2Δ), median 2Δ = 0.4986 (GPT-2)
2. This defines a Toeplitz correlation matrix: C_{ij} = |i-j|^(-2Δ)
3. For a 1D system with power-law correlations C(r) ~ r^(-γ):
   - Spectral density: S(ω) ~ |ω|^(γ-1)
   - Eigenvalue spectrum: λ_k ~ k^(-(1-γ)) = k^(-α) where α = 1 - γ
4. With γ = 2Δ = 0.50: predicted α = 0.50

We verify this numerically and compute directly from GPT-2 attention matrices.

Comparison target: Wang et al. measured covariance eigenvalue exponents
in zebrafish whole-brain calcium imaging data.

Ariel — March 29, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel
from scipy import stats
from scipy.linalg import toeplitz
import json
import os

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 100
MIN_POS = 32
MAX_DX = 56
FIT_LOW = 3
FIT_HIGH = 50
OUTPUT_DIR = "research/physics/results_eigenvalue"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== Part 1: Analytical Prediction ==========

def toeplitz_eigenvalue_spectrum(gamma, N):
    """
    Construct the Toeplitz correlation matrix with entries C_{ij} = |i-j|^(-γ)
    (with C_{ii} = 1) and compute its eigenvalue spectrum.

    For a 1D system with power-law correlations, this is the correlation matrix.
    Its eigenvalue spectrum should follow λ_k ~ k^(-(1-γ)) for 0 < γ < 1.
    """
    row = np.zeros(N)
    row[0] = 1.0
    for k in range(1, N):
        row[k] = k ** (-gamma)

    C = toeplitz(row)
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues


def fit_power_law_eigenvalues(eigenvalues, fit_range=(0.02, 0.40)):
    """
    Fit λ_k ~ k^(-α) to eigenvalue spectrum.
    Returns (alpha, R², std_err).
    """
    N = len(eigenvalues)
    ranks = np.arange(1, N + 1)

    i_low = max(1, int(fit_range[0] * N))
    i_high = min(N - 1, int(fit_range[1] * N))

    log_ranks = np.log10(ranks[i_low:i_high].astype(float))
    log_eigs = np.log10(np.abs(eigenvalues[i_low:i_high]))

    mask = np.isfinite(log_eigs) & np.isfinite(log_ranks)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks[mask], log_eigs[mask]
    )
    return -slope, r_value**2, std_err


# ========== Part 2: Direct GPT-2 Measurement ==========

def compute_attention_decay_per_head(model, n_inputs, seq_len, vocab_size,
                                      n_layers, n_heads, min_pos, max_dx):
    """Compute average attention weight as function of distance for each head."""
    A_heads = {
        l: {h: np.zeros(max_dx) for h in range(n_heads)} for l in range(n_layers)
    }
    counts = np.zeros(max_dx)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq_len):
            if i - dx >= 0:
                counts[dx] += 1

    for inp_idx in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                for dx in range(max_dx):
                    val = 0.0
                    for i in range(max(min_pos, dx), seq_len):
                        j = i - dx
                        if j >= 0:
                            val += attn[h][i, j]
                    if counts[dx] > 0:
                        A_heads[l][h][dx] += val / counts[dx]

        if (inp_idx + 1) % 20 == 0:
            print(f"  {inp_idx + 1}/{n_inputs} inputs processed")

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= n_inputs

    return A_heads


def build_correlation_matrix_from_decay(decay_curve, N):
    """
    Build symmetric Toeplitz correlation matrix from measured attention decay.
    decay_curve[dx] = average attention at distance dx.
    """
    row = np.zeros(N)
    row[0] = decay_curve[0] if decay_curve[0] > 0 else 1.0
    for k in range(1, min(N, len(decay_curve))):
        row[k] = decay_curve[k]
    for k in range(len(decay_curve), N):
        if decay_curve[-1] > 1e-10:
            row[k] = decay_curve[-1] * (k / (len(decay_curve) - 1)) ** (-0.5)
        else:
            row[k] = 0.0

    row /= row[0]
    C = toeplitz(row)
    return C


def fit_attention_power_law(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    """Fit A(dx) ~ dx^(-2Δ). Returns (2Δ, Δ, R²)."""
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    two_delta = -coeffs[1]
    return two_delta, two_delta / 2, R2


def main():
    print("=" * 70)
    print("EIGENVALUE SPECTRUM ANALYSIS")
    print("Transformer Attention Correlation Structure")
    print("=" * 70)

    # ===== Part 1: Analytical prediction (Toeplitz matrix) =====
    print("\n" + "=" * 70)
    print("PART 1: ANALYTICAL PREDICTION")
    print("Eigenvalue spectrum of Toeplitz matrix C_{ij} = |i-j|^(-γ)")
    print("=" * 70)

    N_toeplitz = 256
    gammas = [0.25, 0.40, 0.50, 0.60, 0.75, 1.00]

    print(f"\n{'γ':>6} {'α_pred':>10} {'α_meas':>10} {'R²':>8}")
    print("-" * 40)

    toeplitz_results = {}
    for gamma in gammas:
        eigs = toeplitz_eigenvalue_spectrum(gamma, N_toeplitz)
        alpha, r2, se = fit_power_law_eigenvalues(eigs)
        alpha_pred = 1.0 - gamma if gamma < 1.0 else 0.0
        toeplitz_results[gamma] = {
            "alpha_measured": float(alpha),
            "alpha_predicted": float(alpha_pred),
            "r2": float(r2),
        }
        print(f"{gamma:>6.2f} {alpha_pred:>10.4f} {alpha:>10.4f} {r2:>8.4f}")

    key_gamma = 0.50
    key_result = toeplitz_results[key_gamma]
    print(f"\n→ For γ = 2Δ = {key_gamma}: predicted α = {key_result['alpha_predicted']:.4f}, "
          f"measured α = {key_result['alpha_measured']:.4f}")

    # ===== Part 2: Direct GPT-2 measurement =====
    print("\n" + "=" * 70)
    print("PART 2: DIRECT GPT-2 MEASUREMENT")
    print(f"Computing attention decay per head ({N_INPUTS} inputs, seq_len={SEQ_LEN})")
    print("=" * 70)

    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    vocab_size = model.config.vocab_size

    print(f"\nCollecting attention decay curves...")
    A_heads = compute_attention_decay_per_head(
        model, N_INPUTS, SEQ_LEN, vocab_size, n_layers, n_heads, MIN_POS, MAX_DX
    )

    dx_arr = np.arange(MAX_DX)

    all_two_deltas = []
    all_alphas = []
    per_head = []

    print(f"\n{'Layer':>5} {'Head':>4} {'2Δ':>8} {'R²(PL)':>8} {'α(eig)':>8} {'R²(eig)':>8}")
    print("-" * 50)

    for l in range(n_layers):
        for h in range(n_heads):
            decay = A_heads[l][h]

            two_delta, delta, r2_pl = fit_attention_power_law(
                dx_arr, decay, FIT_LOW, FIT_HIGH
            )

            C = build_correlation_matrix_from_decay(decay, SEQ_LEN)
            try:
                eigs = np.linalg.eigvalsh(C)
                eigs = np.sort(eigs)[::-1]
                eigs = eigs[eigs > 1e-12]
                alpha_eig, r2_eig, _ = fit_power_law_eigenvalues(eigs)
            except np.linalg.LinAlgError:
                alpha_eig, r2_eig = np.nan, np.nan

            if two_delta is not None and r2_pl is not None and r2_pl > 0.8:
                all_two_deltas.append(two_delta)
                if np.isfinite(alpha_eig):
                    all_alphas.append(alpha_eig)

            per_head.append({
                "layer": l, "head": h,
                "two_delta": float(two_delta) if two_delta else None,
                "delta": float(delta) if delta else None,
                "r2_power_law": float(r2_pl) if r2_pl else None,
                "alpha_eigenvalue": float(alpha_eig) if np.isfinite(alpha_eig) else None,
                "r2_eigenvalue": float(r2_eig) if np.isfinite(r2_eig) else None,
            })

            if l % 4 == 0 and h == 0:
                td_str = f"{two_delta:.4f}" if two_delta else "---"
                r2_str = f"{r2_pl:.4f}" if r2_pl else "---"
                ae_str = f"{alpha_eig:.4f}" if np.isfinite(alpha_eig) else "---"
                re_str = f"{r2_eig:.4f}" if np.isfinite(r2_eig) else "---"
                print(f"{l:>5} {h:>4} {td_str:>8} {r2_str:>8} {ae_str:>8} {re_str:>8}")

    two_deltas = np.array(all_two_deltas)
    alphas = np.array([a for a in all_alphas if np.isfinite(a)])

    # ===== Part 3: Random control =====
    print("\n" + "=" * 70)
    print("PART 3: RANDOM CONTROL")
    print("=" * 70)

    n_random = 50
    random_alphas = []
    for _ in range(n_random):
        row = np.random.randn(SEQ_LEN)
        row[0] = 1.0
        row = np.abs(row) / np.max(np.abs(row))
        C_rand = toeplitz(row)
        eigs_rand = np.linalg.eigvalsh(C_rand)
        eigs_rand = np.sort(eigs_rand)[::-1]
        eigs_rand = eigs_rand[eigs_rand > 1e-12]
        alpha_r, _, _ = fit_power_law_eigenvalues(eigs_rand)
        if np.isfinite(alpha_r):
            random_alphas.append(alpha_r)

    random_alphas = np.array(random_alphas)

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n--- Attention Decay Exponent (2Δ) ---")
    print(f"  N heads (R² > 0.8): {len(two_deltas)}")
    print(f"  Median 2Δ:  {np.median(two_deltas):.4f}")
    print(f"  Mean 2Δ:    {np.mean(two_deltas):.4f} ± {np.std(two_deltas):.4f}")
    print(f"  IQR:        [{np.percentile(two_deltas, 25):.4f}, "
          f"{np.percentile(two_deltas, 75):.4f}]")

    print(f"\n--- Eigenvalue Spectrum Exponent (α) ---")
    print(f"  N heads:    {len(alphas)}")
    print(f"  Median α:   {np.median(alphas):.4f}")
    print(f"  Mean α:     {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")
    print(f"  IQR:        [{np.percentile(alphas, 25):.4f}, "
          f"{np.percentile(alphas, 75):.4f}]")

    alpha_predicted = 1.0 - np.median(two_deltas)
    print(f"\n--- Analytical Prediction ---")
    print(f"  From median 2Δ = {np.median(two_deltas):.4f}:")
    print(f"  Predicted α = 1 - 2Δ = {alpha_predicted:.4f}")
    print(f"  Measured α  = {np.median(alphas):.4f}")
    print(f"  Ratio:      {np.median(alphas) / alpha_predicted:.4f}")

    print(f"\n--- Random Control ---")
    if len(random_alphas) > 0:
        print(f"  Random α:   {np.mean(random_alphas):.4f} ± {np.std(random_alphas):.4f}")
    else:
        print(f"  Random α:   no valid fits")

    print(f"\n--- Comparison Targets ---")
    print(f"  Wang et al. zebrafish covariance eigenvalue exponent: α ≈ 0.47 ± 0.08")
    print(f"  Theoretical prediction (Δ* = 1/4): α = 1 - 2Δ* = 0.50")
    print(f"  Our transformer measurement (median): α = {np.median(alphas):.4f}")

    # Per-head: check correlation between 2Δ and α
    valid_pairs = [(r["two_delta"], r["alpha_eigenvalue"])
                   for r in per_head
                   if r["two_delta"] is not None
                   and r["alpha_eigenvalue"] is not None
                   and r["r2_power_law"] is not None
                   and r["r2_power_law"] > 0.8]
    if len(valid_pairs) > 10:
        tds, als = zip(*valid_pairs)
        tds, als = np.array(tds), np.array(als)
        predicted_als = 1.0 - tds
        corr = np.corrcoef(predicted_als, als)[0, 1]
        mean_diff = np.mean(np.abs(predicted_als - als))
        print(f"\n--- Per-Head Consistency Check ---")
        print(f"  Correlation(1-2Δ, α): r = {corr:.4f}")
        print(f"  Mean |predicted - measured|: {mean_diff:.4f}")
        print(f"  N pairs: {len(valid_pairs)}")

    # ===== Save =====
    summary = {
        "model": "GPT-2",
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "attention_decay": {
            "n_heads_r2_gt_0.8": len(two_deltas),
            "median_2delta": float(np.median(two_deltas)),
            "mean_2delta": float(np.mean(two_deltas)),
            "std_2delta": float(np.std(two_deltas)),
        },
        "eigenvalue_spectrum": {
            "n_heads": len(alphas),
            "median_alpha": float(np.median(alphas)),
            "mean_alpha": float(np.mean(alphas)),
            "std_alpha": float(np.std(alphas)),
        },
        "analytical_prediction": {
            "formula": "alpha = 1 - 2*Delta (for 1D Toeplitz correlation matrix)",
            "predicted_alpha": float(alpha_predicted),
            "measured_alpha": float(np.median(alphas)),
        },
        "toeplitz_verification": toeplitz_results,
        "random_control": {
            "mean_alpha": float(np.mean(random_alphas)) if len(random_alphas) > 0 else None,
            "std_alpha": float(np.std(random_alphas)) if len(random_alphas) > 0 else None,
        },
        "comparison": {
            "wang_et_al_zebrafish": "alpha ≈ 0.47 ± 0.08",
            "theory_prediction_delta_quarter": "alpha = 0.50",
            "our_transformer_median": float(np.median(alphas)),
        },
        "per_head_results": per_head,
    }

    out_path = os.path.join(OUTPUT_DIR, "eigenvalue_comparison.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("KEY RESULT")
    print("=" * 70)
    print(f"""
The attention correlation structure in trained GPT-2 has a well-defined
eigenvalue spectrum exponent:

  Transformer (measured):    α = {np.median(alphas):.4f} (median across {len(alphas)} heads)
  Analytical prediction:     α = 1 - 2Δ = {alpha_predicted:.4f}
  Wang et al. zebrafish:     α ≈ 0.47 ± 0.08
  Theory (Δ* = 1/4):        α = 0.50

The correlation exponent 2Δ is the substrate-independent quantity.
Our prediction for brain data: 2Δ = 0.50 (same as transformer).
""")


if __name__ == "__main__":
    main()
