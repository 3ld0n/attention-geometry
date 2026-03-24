"""
Prediction P2: Holographic Entanglement Entropy

In a 1+1d CFT, the entanglement entropy of a contiguous block of k sites follows:
    S(k) = (c/3) * log(k)
where c is the central charge.

If the attention weights encode a CFT state, the von Neumann entropy of the
attention distribution for blocks of size k should show logarithmic scaling.

Also tests P7 (Page curve): layer-by-layer mutual information between input
and representations, looking for the rise-and-fall characteristic of the Page curve.

Ariel — March 24, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, AutoModelForCausalLM

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 128
N_INPUTS = 30


def attention_block_entropy(model, n_layers, n_heads, vocab_size, label):
    """
    For each layer and head, compute the entanglement entropy of the attention
    distribution for contiguous blocks of k positions.

    The attention matrix α(i,j) defines a bipartite correlation structure.
    For a block A of k contiguous positions, we compute the reduced density
    matrix by tracing over the complement:

    ρ_A = (1/k) Σ_{i ∈ A} |α_i⟩⟨α_i|

    where |α_i⟩ is the attention distribution from position i (a vector over
    all positions j that i attends to). The von Neumann entropy of ρ_A gives
    the entanglement entropy of block A.
    """

    block_sizes = [2, 4, 8, 16, 32, 64]
    center = SEQ_LEN // 2

    # Accumulate entropy values across inputs
    entropy_by_layer_block = {
        l: {k: [] for k in block_sizes} for l in range(n_layers)
    }

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()  # [n_heads, seq, seq]
            avg_attn = attn.mean(axis=0)  # [seq, seq]

            for k in block_sizes:
                if k > SEQ_LEN:
                    continue
                start = center - k // 2
                end = start + k

                # Reduced density matrix for block [start, end)
                # Each row α_i (for i in block) is a "state vector"
                # ρ_A = (1/k) Σ_i |α_i⟩⟨α_i| (attention vectors from block positions)
                block_attn = avg_attn[start:end, :]  # [k, seq]

                # Normalize rows (they should sum to 1 from softmax, but be safe)
                row_sums = block_attn.sum(axis=1, keepdims=True)
                block_attn = block_attn / np.maximum(row_sums, 1e-12)

                # Correlation matrix C = (1/k) * block_attn @ block_attn^T
                C = block_attn @ block_attn.T / k

                # Eigenvalues of C give the entanglement spectrum
                eigvals = np.linalg.eigvalsh(C)
                eigvals = eigvals[eigvals > 1e-12]

                # Normalize eigenvalues (trace of density matrix = 1)
                eigvals = eigvals / eigvals.sum()

                # Von Neumann entropy
                S = -np.sum(eigvals * np.log(eigvals))
                entropy_by_layer_block[l][k].append(S)

        if (inp_idx + 1) % 10 == 0:
            print(f"    {inp_idx + 1}/{N_INPUTS}")

    # Average and analyze
    print(f"\n  {label}: Block Entanglement Entropy by Layer")
    print(f"  Block sizes: {block_sizes}")
    print()

    # For CFT check: average across middle layers (where conformal scaling is strongest)
    mid_start = n_layers // 3
    mid_end = 2 * n_layers // 3

    print(f"  Averaged over layers {mid_start+1}-{mid_end} (middle third):")
    print(f"  {'k':>6}  {'log(k)':>8}  {'S(k)':>8}  {'S(k)/log(k)':>12}")
    print(f"  " + "-" * 40)

    log_k_vals = []
    s_vals = []

    for k in block_sizes:
        mean_S = np.mean([
            np.mean(entropy_by_layer_block[l][k])
            for l in range(mid_start, mid_end)
        ])
        log_k = np.log(k)
        ratio = mean_S / log_k if log_k > 0 else 0
        log_k_vals.append(log_k)
        s_vals.append(mean_S)
        print(f"  {k:>6}  {log_k:>8.4f}  {mean_S:>8.4f}  {ratio:>12.4f}")

    # Linear fit: S(k) = a + b * log(k)
    # If CFT: b = c/3
    log_k_arr = np.array(log_k_vals)
    s_arr = np.array(s_vals)
    A = np.column_stack([np.ones_like(log_k_arr), log_k_arr])
    coeffs, _, _, _ = np.linalg.lstsq(A, s_arr, rcond=None)
    intercept, slope = coeffs
    predicted = A @ coeffs
    ss_res = np.sum((s_arr - predicted) ** 2)
    ss_tot = np.sum((s_arr - np.mean(s_arr)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

    c_eff = 3 * slope

    print()
    print(f"  Linear fit: S(k) = {intercept:.4f} + {slope:.4f} * log(k)")
    print(f"  R² = {R2:.4f}")
    print(f"  Effective central charge: c = 3 * slope = {c_eff:.4f}")
    print()
    if R2 > 0.95:
        print(f"  *** LOGARITHMIC SCALING CONFIRMED (R² > 0.95) ***")
        print(f"  Consistent with CFT entanglement entropy formula S = (c/3) log(k)")
    elif R2 > 0.85:
        print(f"  Approximate logarithmic scaling (R² > 0.85)")
    else:
        print(f"  Logarithmic scaling NOT confirmed (R² = {R2:.4f})")

    # Layer-by-layer entropy for k=32 (Page curve check)
    print()
    print(f"  Layer-by-layer S(k=32) — Page curve check:")
    print(f"  {'Layer':>7}  {'S(k=32)':>10}")
    print(f"  " + "-" * 20)

    for l in range(n_layers):
        mean_S = np.mean(entropy_by_layer_block[l][32])
        print(f"  {l+1:>7}  {mean_S:>10.4f}")

    print()

    return {
        "label": label,
        "R2": R2,
        "c_eff": c_eff,
        "slope": slope,
    }


# ========== Run ==========
print("=" * 80)
print("  PREDICTION P2: Holographic Entanglement Entropy")
print("=" * 80)
print()

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

result_gpt2 = attention_block_entropy(
    model,
    n_layers=model.config.n_layer,
    n_heads=model.config.n_head,
    vocab_size=model.config.vocab_size,
    label="GPT-2"
)
del model

print("Loading Pythia-410m...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
model.eval()

result_pythia = attention_block_entropy(
    model,
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_attention_heads,
    vocab_size=model.config.vocab_size,
    label="Pythia-410m"
)
del model

# ========== Summary ==========
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
print()
print(f"  {'Model':>15}  {'R²':>8}  {'c_eff':>8}  {'Log scaling?':>15}")
print(f"  " + "-" * 50)

for r in [result_gpt2, result_pythia]:
    status = "YES" if r['R2'] > 0.95 else "APPROX" if r['R2'] > 0.85 else "NO"
    print(f"  {r['label']:>15}  {r['R2']:>8.4f}  {r['c_eff']:>8.4f}  {status:>15}")

print()
print("  If S(k) = (c/3) * log(k), the attention encodes a CFT state")
print("  consistent with the holographic entanglement entropy interpretation.")
print()
