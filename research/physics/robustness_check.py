"""
Robustness Check: Fit Range and R² Threshold Sensitivity

Tests whether the core finding (median Δ ≈ 0.25) is stable across:
1. Different fit ranges: [2,55], [3,50], [4,45], [5,40], [3,35], [5,50]
2. Different R² thresholds: 0.80, 0.85, 0.90, 0.95

If Δ moves significantly with analysis choices, the result is fragile.
If it stays near 0.25 regardless, the claim is robust.

Ariel — March 24, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

n_layers = model.config.n_layer
n_heads = model.config.n_head
vocab_size = model.config.vocab_size

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 60
MIN_POS = 32


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


def fit_power_law(dx_arr, y_arr, cutoff_low, cutoff_high):
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


# Compute attention decay curves once
dx_arr = np.arange(MAX_DX)
A_heads = {
    l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
}

print(f"\nProcessing {N_INPUTS} inputs (seq_len={SEQ_LEN})...")
for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()
        for h in range(n_heads):
            A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)
    if (inp_idx + 1) % 10 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

for l in range(n_layers):
    for h in range(n_heads):
        A_heads[l][h] /= N_INPUTS

print("\nAttention data collected. Running robustness analysis...\n")


# ========== Robustness Test 1: Fit Range Sensitivity ==========
fit_ranges = [
    (2, 55), (3, 50), (4, 45), (5, 40), (3, 35), (5, 50),
    (3, 30), (2, 40), (4, 50), (5, 55),
]

R2_THRESHOLD = 0.90

print("=" * 90)
print("  ROBUSTNESS TEST 1: Fit Range Sensitivity (R² > 0.90)")
print("=" * 90)
print()
print(f"  {'Fit Range':>15}  {'PL Heads':>10}  {'Near 1/4':>10}  {'Median Δ':>10}  "
      f"{'Mean Δ':>10}  {'Std Δ':>10}")
print("  " + "-" * 70)

for low, high in fit_ranges:
    good_deltas = []
    n_near_quarter = 0
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2 = fit_power_law(dx_arr, A, low, high)
            if delta is not None and R2 is not None and R2 > R2_THRESHOLD:
                good_deltas.append(delta)
                if abs(delta - 0.25) < 0.06:
                    n_near_quarter += 1

    if good_deltas:
        d = np.array(good_deltas)
        print(f"  [{low:2d}, {high:2d}]         {len(d):10d}  {n_near_quarter:10d}  "
              f"{np.median(d):10.4f}  {np.mean(d):10.4f}  {np.std(d):10.4f}")
    else:
        print(f"  [{low:2d}, {high:2d}]                0           0       ---       ---       ---")

print()


# ========== Robustness Test 2: R² Threshold Sensitivity ==========
R2_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.97]
FIT_LOW, FIT_HIGH = 3, 50

print("=" * 90)
print(f"  ROBUSTNESS TEST 2: R² Threshold Sensitivity (fit range [{FIT_LOW}, {FIT_HIGH}])")
print("=" * 90)
print()
print(f"  {'R² Threshold':>15}  {'PL Heads':>10}  {'Near 1/4':>10}  {'Median Δ':>10}  "
      f"{'Mean Δ':>10}  {'Std Δ':>10}")
print("  " + "-" * 70)

for thresh in R2_THRESHOLDS:
    good_deltas = []
    n_near_quarter = 0
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2 = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            if delta is not None and R2 is not None and R2 > thresh:
                good_deltas.append(delta)
                if abs(delta - 0.25) < 0.06:
                    n_near_quarter += 1

    if good_deltas:
        d = np.array(good_deltas)
        print(f"  R² > {thresh:.2f}        {len(d):10d}  {n_near_quarter:10d}  "
              f"{np.median(d):10.4f}  {np.mean(d):10.4f}  {np.std(d):10.4f}")
    else:
        print(f"  R² > {thresh:.2f}               0           0       ---       ---       ---")

print()


# ========== Summary ==========
print("=" * 90)
print("  ROBUSTNESS SUMMARY")
print("=" * 90)
print()
print("  If median Δ stays near 0.25 across all fit ranges and R² thresholds,")
print("  the finding is robust to analysis choices.")
print()
print("  SYK q=4 prediction: Δ = 0.2500")
print()
