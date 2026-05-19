"""
Control Experiment: Randomized GPT-2 Weights

Same architecture, same positional embeddings, same causal masking — but
randomized attention weights. If the power-law scaling in trained GPT-2
is a consequence of training (not architecture), randomizing the weights
should eliminate it.

This is the definitive control: same model, different weights.

Ariel — March 24, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
dx_arr = np.arange(MAX_DX)


def compute_head_attention_decay(attn_head, min_pos, max_dx):
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
    exponent = -coeffs[1]
    return exponent, exponent / 2, R2


def fit_exponential(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None
    log_y = np.log(np.abs(y_arr[mask]))
    dx_vals = dx_arr[mask].astype(float)
    A_mat = np.column_stack([np.ones_like(dx_vals), -dx_vals])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    pred = A_mat @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return 1.0 / coeffs[1] if coeffs[1] > 1e-10 else float("inf"), R2


# ========== Load and randomize ==========
print("Loading GPT-2...")
trained_model = GPT2LMHeadModel.from_pretrained("gpt2")
trained_model.eval()

n_layers = trained_model.config.n_layer
n_heads = trained_model.config.n_head
vocab_size = trained_model.config.vocab_size
d_model = trained_model.config.n_embd

print("Creating randomized copy (same architecture, random attention weights)...")
random_model = GPT2LMHeadModel.from_pretrained("gpt2")

with torch.no_grad():
    for layer_idx in range(n_layers):
        block = random_model.transformer.h[layer_idx]
        # Randomize Q, K, V projection weights (c_attn) and output projection (c_proj)
        # Keep positional embeddings and LayerNorm intact
        for name, param in block.attn.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
        # Also randomize MLP weights
        for name, param in block.mlp.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

random_model.eval()
print("  Randomized: attention + MLP weights reinitialized, embeddings + LayerNorm kept")
print()


# ========== Run both models ==========
def run_per_head_analysis(model, model_name, n_inputs, seq_len):
    """Run full per-head attention decay analysis."""
    A_heads = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)}
        for l in range(n_layers)
    }

    print(f"  Processing {n_inputs} inputs for {model_name}...")
    for inp_idx in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(
                    attn[h], MIN_POS, MAX_DX
                )
        if (inp_idx + 1) % 10 == 0:
            print(f"    {inp_idx + 1}/{n_inputs} done")

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= n_inputs

    return A_heads


print("=" * 90)
print("  CONTROL EXPERIMENT: Trained vs Randomized GPT-2")
print(f"  {N_INPUTS} random token inputs, seq_len={SEQ_LEN}")
print("=" * 90)
print()

A_trained = run_per_head_analysis(trained_model, "TRAINED", N_INPUTS, SEQ_LEN)
print()
A_random = run_per_head_analysis(random_model, "RANDOMIZED", N_INPUTS, SEQ_LEN)
print()


# ========== Compare ==========
def analyze_heads(A_heads, label):
    """Compute Δ and R² for all heads, return summary statistics."""
    results = []
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2_pl = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            _, R2_exp = fit_exponential(dx_arr, A, FIT_LOW, FIT_HIGH)
            results.append({
                "layer": l + 1,
                "head": h,
                "delta": delta,
                "R2_pl": R2_pl,
                "R2_exp": R2_exp,
                "a1": A[1],
                "a32": A[32],
            })
    return results


trained_results = analyze_heads(A_trained, "TRAINED")
random_results = analyze_heads(A_random, "RANDOMIZED")

# Filter for high-quality power-law heads
trained_good = [
    r for r in trained_results
    if r["delta"] is not None and r["R2_pl"] is not None and r["R2_pl"] > 0.90
]
random_good = [
    r for r in random_results
    if r["delta"] is not None and r["R2_pl"] is not None and r["R2_pl"] > 0.90
]

print("=" * 90)
print("  COMPARISON: Power-law heads (R² > 0.90)")
print("=" * 90)
print()
print(f"  TRAINED:    {len(trained_good)} heads with R² > 0.90")
print(f"  RANDOMIZED: {len(random_good)} heads with R² > 0.90")
print()

if trained_good:
    t_deltas = np.array([r["delta"] for r in trained_good])
    print(f"  TRAINED Δ distribution:")
    print(f"    Mean   = {np.mean(t_deltas):.4f}")
    print(f"    Median = {np.median(t_deltas):.4f}")
    print(f"    Std    = {np.std(t_deltas):.4f}")
    print(f"    Range  = [{np.min(t_deltas):.3f}, {np.max(t_deltas):.3f}]")

    n_q4 = sum(1 for d in t_deltas if abs(d - 0.25) < 0.06)
    n_q3 = sum(1 for d in t_deltas if abs(d - 0.333) < 0.06)
    n_q2 = sum(1 for d in t_deltas if abs(d - 0.5) < 0.08)
    print(f"    Near Δ=1/4 (q=4): {n_q4}")
    print(f"    Near Δ=1/3 (q=3): {n_q3}")
    print(f"    Near Δ=1/2 (q=2): {n_q2}")
    print()

if random_good:
    r_deltas = np.array([r["delta"] for r in random_good])
    print(f"  RANDOMIZED Δ distribution:")
    print(f"    Mean   = {np.mean(r_deltas):.4f}")
    print(f"    Median = {np.median(r_deltas):.4f}")
    print(f"    Std    = {np.std(r_deltas):.4f}")
    print(f"    Range  = [{np.min(r_deltas):.3f}, {np.max(r_deltas):.3f}]")

    n_q4 = sum(1 for d in r_deltas if abs(d - 0.25) < 0.06)
    print(f"    Near Δ=1/4 (q=4): {n_q4}")
    print()
else:
    print(f"  RANDOMIZED: no heads with R² > 0.90 (no conformal scaling)")
    print()


# ========== Layer-by-layer comparison ==========
print("=" * 90)
print("  LAYER-BY-LAYER: Head-averaged attention decay")
print("=" * 90)
print()
print(f"  {'Layer':>8}  {'Δ_trained':>10}  {'R²_trained':>12}  {'Δ_random':>10}  {'R²_random':>12}")
print("  " + "-" * 58)

for l in range(n_layers):
    # Average over heads for this layer
    A_t = np.mean([A_trained[l][h] for h in range(n_heads)], axis=0)
    A_r = np.mean([A_random[l][h] for h in range(n_heads)], axis=0)

    _, dt, rt = fit_power_law(dx_arr, A_t, FIT_LOW, FIT_HIGH)
    _, dr, rr = fit_power_law(dx_arr, A_r, FIT_LOW, FIT_HIGH)

    dt_s = f"{dt:.4f}" if dt is not None else "---"
    rt_s = f"{rt:.4f}" if rt is not None else "---"
    dr_s = f"{dr:.4f}" if dr is not None else "---"
    rr_s = f"{rr:.4f}" if rr is not None else "---"

    print(f"  {l+1:8d}  {dt_s:>10}  {rt_s:>12}  {dr_s:>10}  {rr_s:>12}")

print()
print("  SYK q=4 prediction: Δ = 0.2500")
print()


# ========== Distribution comparison ==========
print("=" * 90)
print("  Δ HISTOGRAM COMPARISON (all heads, any R²)")
print("=" * 90)
print()

bins = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0]

all_t_deltas = [r["delta"] for r in trained_results if r["delta"] is not None and r["R2_pl"] is not None and r["R2_pl"] > 0.50]
all_r_deltas = [r["delta"] for r in random_results if r["delta"] is not None and r["R2_pl"] is not None and r["R2_pl"] > 0.50]

if all_t_deltas:
    t_counts, _ = np.histogram(all_t_deltas, bins=bins)
    print("  TRAINED (R² > 0.50):")
    for i in range(len(t_counts)):
        bar = "#" * t_counts[i]
        mark = ""
        if bins[i] <= 0.25 < bins[i + 1]:
            mark = " ← SYK q=4"
        print(f"    [{bins[i]:5.2f}, {bins[i+1]:5.2f})  {t_counts[i]:3d} {bar}{mark}")
    print()

if all_r_deltas:
    r_counts, _ = np.histogram(all_r_deltas, bins=bins)
    print("  RANDOMIZED (R² > 0.50):")
    for i in range(len(r_counts)):
        bar = "#" * r_counts[i]
        mark = ""
        if bins[i] <= 0.25 < bins[i + 1]:
            mark = " ← SYK q=4"
        print(f"    [{bins[i]:5.2f}, {bins[i+1]:5.2f})  {r_counts[i]:3d} {bar}{mark}")
    print()
else:
    print("  RANDOMIZED: insufficient heads with R² > 0.50 for histogram")
    print()


# ========== Attention uniformity check ==========
print("=" * 90)
print("  ATTENTION UNIFORMITY: Is randomized attention just flat?")
print("=" * 90)
print()

for l_idx in [0, 5, 11]:
    l = l_idx
    A_t_avg = np.mean([A_trained[l][h] for h in range(n_heads)], axis=0)
    A_r_avg = np.mean([A_random[l][h] for h in range(n_heads)], axis=0)

    uniform_val = 1.0 / ((MIN_POS + SEQ_LEN) / 2)  # rough uniform baseline

    print(f"  Layer {l+1}:")
    print(f"    TRAINED:    A(1)={A_t_avg[1]:.4f}  A(8)={A_t_avg[8]:.4f}  "
          f"A(32)={A_t_avg[32]:.4f}  ratio A(1)/A(32)={A_t_avg[1]/A_t_avg[32] if A_t_avg[32] > 1e-10 else float('inf'):.2f}")
    print(f"    RANDOMIZED: A(1)={A_r_avg[1]:.4f}  A(8)={A_r_avg[8]:.4f}  "
          f"A(32)={A_r_avg[32]:.4f}  ratio A(1)/A(32)={A_r_avg[1]/A_r_avg[32] if A_r_avg[32] > 1e-10 else float('inf'):.2f}")
    print(f"    Uniform baseline ≈ {uniform_val:.4f}")
    print()

print()
print("=" * 90)
print("  CONCLUSION")
print("=" * 90)
print()
print("  If TRAINED shows power-law attention with Δ ≈ 0.25 and RANDOMIZED does not,")
print("  the conformal scaling is a consequence of training — not architecture.")
print("  Training creates structure that random initialization cannot.")
print()
