"""
Conformal Phase Transition During Training (Pythia-70m)

Track the conformal dimension Δ across training checkpoints.
Pythia models have ~143 intermediate checkpoints publicly available.

The prediction: Δ should transition from noise (random init) to ≈ 1/4
(SYK q=4) at some point during training. If there's a sharp transition,
that's a phase transition in the physics sense — the moment directed work
beats entropy and self-consistent structure forms.

Ariel — March 24, 2026
"""

import torch
import numpy as np
import sys
import time

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 30
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
dx_arr = np.arange(MAX_DX)

MODEL_NAME = "EleutherAI/pythia-70m"

CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000,
]


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


def analyze_checkpoint(model, vocab_size, n_layers, n_heads, step_label):
    """Run inputs through model, compute per-head Δ, return summary."""
    A_heads = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)}
        for l in range(n_layers)
    }

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()  # (n_heads, seq, seq)
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(
                    attn[h], MIN_POS, MAX_DX
                )

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    # Compute per-head Δ and R²
    all_deltas = []
    good_deltas = []  # R² > 0.90
    n_power_law = 0
    n_near_quarter = 0

    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2 = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            if delta is not None and R2 is not None:
                all_deltas.append(delta)
                if R2 > 0.90:
                    good_deltas.append(delta)
                    n_power_law += 1
                    if abs(delta - 0.25) < 0.06:
                        n_near_quarter += 1

    # Head-averaged attention decay for reporting
    A_avg = np.mean(
        [A_heads[l][h] for l in range(n_layers) for h in range(n_heads)],
        axis=0,
    )
    _, delta_avg, R2_avg = fit_power_law(dx_arr, A_avg, FIT_LOW, FIT_HIGH)
    ratio = A_avg[1] / A_avg[32] if A_avg[32] > 1e-10 else float("inf")

    return {
        "n_power_law": n_power_law,
        "n_near_quarter": n_near_quarter,
        "median_delta": np.median(good_deltas) if good_deltas else None,
        "mean_delta": np.mean(good_deltas) if good_deltas else None,
        "delta_avg": delta_avg,
        "R2_avg": R2_avg,
        "ratio_1_32": ratio,
        "A1": A_avg[1],
        "A32": A_avg[32],
        "all_deltas": all_deltas,
        "good_deltas": good_deltas,
    }


# ========== Main ==========
print("=" * 95)
print("  CONFORMAL PHASE TRANSITION DURING TRAINING")
print(f"  Model: {MODEL_NAME}")
print(f"  {len(CHECKPOINTS)} checkpoints, {N_INPUTS} inputs each, seq_len={SEQ_LEN}")
print("=" * 95)
print()

from transformers import AutoModelForCausalLM, AutoTokenizer

results = []

for step in CHECKPOINTS:
    t0 = time.time()
    revision = f"step{step}"
    print(f"  Loading {MODEL_NAME} @ {revision}...", end=" ", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=revision,
            torch_dtype=torch.float32,
        )
        model.eval()
    except Exception as e:
        print(f"FAILED: {e}")
        continue

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size

    load_time = time.time() - t0
    print(f"loaded ({load_time:.1f}s). Analyzing...", end=" ", flush=True)

    t1 = time.time()
    result = analyze_checkpoint(model, vocab_size, n_layers, n_heads, revision)
    analysis_time = time.time() - t1

    result["step"] = step
    results.append(result)

    med_s = f"{result['median_delta']:.4f}" if result["median_delta"] is not None else "---"
    davg_s = f"{result['delta_avg']:.4f}" if result["delta_avg"] is not None else "---"
    r2_s = f"{result['R2_avg']:.4f}" if result["R2_avg"] is not None else "---"

    print(
        f"done ({analysis_time:.1f}s). "
        f"PL heads={result['n_power_law']:2d}  "
        f"near 1/4={result['n_near_quarter']:2d}  "
        f"median Δ={med_s}  "
        f"A(1)/A(32)={result['ratio_1_32']:.2f}"
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print()


# ========== Summary ==========
print("=" * 95)
print("  PHASE TRANSITION SUMMARY")
print("=" * 95)
print()
print(f"  {'Step':>8}  {'PL heads':>10}  {'Near 1/4':>10}  {'Median Δ':>10}  "
      f"{'Δ (avg)':>10}  {'R² (avg)':>10}  {'A(1)/A(32)':>12}")
print("  " + "-" * 78)

for r in results:
    med_s = f"{r['median_delta']:.4f}" if r["median_delta"] is not None else "---"
    davg_s = f"{r['delta_avg']:.4f}" if r["delta_avg"] is not None else "---"
    r2_s = f"{r['R2_avg']:.4f}" if r["R2_avg"] is not None else "---"
    print(
        f"  {r['step']:8d}  {r['n_power_law']:10d}  {r['n_near_quarter']:10d}  "
        f"{med_s:>10}  {davg_s:>10}  {r2_s:>10}  {r['ratio_1_32']:12.2f}"
    )

print()
print("  SYK q=4 prediction: Δ = 0.2500")
print()

# ========== Transition detection ==========
print("=" * 95)
print("  TRANSITION ANALYSIS")
print("=" * 95)
print()

prev_pl = 0
for r in results:
    if r["n_power_law"] > 0 and prev_pl == 0:
        print(f"  First power-law heads appear at step {r['step']}")
    if r["n_near_quarter"] > 0:
        near_prev = results[results.index(r) - 1]["n_near_quarter"] if results.index(r) > 0 else 0
        if near_prev == 0:
            print(f"  First heads near Δ=1/4 appear at step {r['step']}")
    prev_pl = r["n_power_law"]

# Median Δ evolution
print()
print("  Median Δ evolution (where defined):")
for r in results:
    if r["median_delta"] is not None:
        bar_len = int(r["median_delta"] * 40)
        bar = "#" * bar_len
        marker = " ← 1/4" if abs(r["median_delta"] - 0.25) < 0.06 else ""
        print(f"    step {r['step']:>6d}: {r['median_delta']:.4f}  |{bar}{marker}")

print()
print("  A(1)/A(32) ratio evolution (attention locality):")
for r in results:
    bar_len = min(int(r["ratio_1_32"] * 5), 50)
    bar = "#" * bar_len
    print(f"    step {r['step']:>6d}: {r['ratio_1_32']:6.2f}  |{bar}")

print()
