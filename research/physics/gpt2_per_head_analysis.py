"""
Per-Head Attention Scaling Analysis (GPT-2)

Follow-up to gpt2_conformal_test.py. The head-averaged attention weights show
power-law decay with Δ in the range 0.12-0.42. But individual heads may show
different scaling — some might converge to Δ = 1/4 while others show different
universality classes.

If specific heads show Δ = 1/4 precisely (SYK q=4), while others show Δ = 1/3
(q=3) or Δ = 1/2 (q=2), that would indicate different operator dimensions in
the conformal tower.

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

n_layers = model.config.n_layer   # 12
n_heads = model.config.n_head     # 12
vocab_size = model.config.vocab_size

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50

dx_arr = np.arange(MAX_DX)


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
    A = np.column_stack([np.ones_like(dx_vals), -dx_vals])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return 1.0 / coeffs[1] if coeffs[1] > 1e-10 else float("inf"), R2


# Accumulators: per-head attention decay for each layer
# Shape: A_heads[layer][head] = running sum of A(Δx)
A_heads = {
    l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
}

print(f"\nProcessing {N_INPUTS} inputs (seq_len={SEQ_LEN})...")

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

    if (inp_idx + 1) % 10 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

# Average
for l in range(n_layers):
    for h in range(n_heads):
        A_heads[l][h] /= N_INPUTS

# ========== Analysis ==========
print()
print("=" * 90)
print("  PER-HEAD ATTENTION SCALING: Δ and R² for Each Head at Each Layer")
print("=" * 90)
print()

# Collect all results for summary
all_results = []

for l in range(n_layers):
    print(f"  Layer {l+1}:")
    print(f"    {'Head':>6}  {'A(1)':>8}  {'A(8)':>8}  {'A(32)':>8}  "
          f"{'Δ':>8}  {'R²(PL)':>8}  {'R²(Exp)':>8}  {'Better':>10}  {'Type':>12}")
    print("    " + "-" * 86)

    for h in range(n_heads):
        A = A_heads[l][h]
        _, delta, R2_pl = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
        _, R2_exp = fit_exponential(dx_arr, A, FIT_LOW, FIT_HIGH)

        a1 = A[1] if A[1] > 0 else 0
        a8 = A[8] if A[8] > 0 else 0
        a32 = A[32] if A[32] > 0 else 0

        if delta is None:
            delta_s = "---"
            r2pl_s = "---"
        else:
            delta_s = f"{delta:.4f}"
            r2pl_s = f"{R2_pl:.4f}"

        r2exp_s = f"{R2_exp:.4f}" if R2_exp is not None else "---"

        if R2_pl is not None and R2_exp is not None:
            better = "PL" if R2_pl > R2_exp else "Exp"
        else:
            better = "---"

        # Classify head type based on attention pattern
        if a1 > 0.1:
            htype = "LOCAL"
        elif a32 > 0.01 and abs(a1 - a32) < 0.005:
            htype = "UNIFORM"
        elif R2_pl is not None and R2_pl > 0.9 and delta is not None:
            if abs(delta - 0.25) < 0.08:
                htype = "SYK q=4"
            elif abs(delta - 0.333) < 0.08:
                htype = "SYK q=3"
            elif abs(delta - 0.5) < 0.1:
                htype = "SYK q=2"
            else:
                htype = f"CONFORMAL"
        else:
            htype = "MIXED"

        print(f"    {h:6d}  {a1:8.4f}  {a8:8.4f}  {a32:8.4f}  "
              f"{delta_s:>8}  {r2pl_s:>8}  {r2exp_s:>8}  {better:>10}  {htype:>12}")

        all_results.append({
            "layer": l + 1,
            "head": h,
            "delta": delta,
            "R2_pl": R2_pl,
            "R2_exp": R2_exp,
            "a1": a1,
            "a32": a32,
            "type": htype,
        })

    print()


# ========== Summary: heads near SYK prediction ==========
print("=" * 90)
print("  HEADS WITH Δ NEAR SYK PREDICTIONS (R² > 0.90)")
print("=" * 90)
print()

syk_targets = [
    (0.25, "q=4 (Δ=1/4)", 0.06),
    (0.333, "q=3 (Δ=1/3)", 0.06),
    (0.5, "q=2 (Δ=1/2)", 0.08),
]

for target, label, tolerance in syk_targets:
    matches = [
        r
        for r in all_results
        if r["delta"] is not None
        and r["R2_pl"] is not None
        and abs(r["delta"] - target) < tolerance
        and r["R2_pl"] > 0.90
    ]
    print(f"  {label}: {len(matches)} heads")
    for m in matches:
        print(
            f"    Layer {m['layer']:2d} Head {m['head']:2d}: "
            f"Δ={m['delta']:.4f}  R²={m['R2_pl']:.4f}"
        )
    print()


# ========== Distribution of Δ ==========
print("=" * 90)
print("  DISTRIBUTION OF Δ (power-law heads with R² > 0.90)")
print("=" * 90)
print()

good_deltas = [
    r["delta"]
    for r in all_results
    if r["delta"] is not None and r["R2_pl"] is not None and r["R2_pl"] > 0.90
]

if good_deltas:
    deltas = np.array(good_deltas)
    print(f"  N = {len(deltas)} heads with R² > 0.90")
    print(f"  Mean Δ = {np.mean(deltas):.4f}")
    print(f"  Median Δ = {np.median(deltas):.4f}")
    print(f"  Std Δ = {np.std(deltas):.4f}")
    print(f"  Min Δ = {np.min(deltas):.4f}")
    print(f"  Max Δ = {np.max(deltas):.4f}")
    print()

    # Histogram
    bins = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0]
    counts, _ = np.histogram(deltas, bins=bins)
    print("  Histogram:")
    for i in range(len(counts)):
        bar = "#" * counts[i]
        label = f"  [{bins[i]:.2f}, {bins[i+1]:.2f})"
        syk_mark = ""
        if bins[i] <= 0.25 < bins[i + 1]:
            syk_mark = " ← SYK q=4"
        if bins[i] <= 0.333 < bins[i + 1]:
            syk_mark = " ← SYK q=3"
        if bins[i] <= 0.5 < bins[i + 1]:
            syk_mark = " ← SYK q=2"
        print(f"  {label:20s} {counts[i]:3d} {bar}{syk_mark}")
    print()


# ========== Layer-averaged Δ ==========
print("=" * 90)
print("  LAYER-AVERAGED Δ (R² > 0.90 heads only)")
print("=" * 90)
print()

for l in range(1, n_layers + 1):
    layer_deltas = [
        r["delta"]
        for r in all_results
        if r["layer"] == l
        and r["delta"] is not None
        and r["R2_pl"] is not None
        and r["R2_pl"] > 0.90
    ]
    if layer_deltas:
        d = np.array(layer_deltas)
        print(
            f"  Layer {l:2d}: N={len(d):2d}  "
            f"Mean Δ = {np.mean(d):.4f}  "
            f"Std = {np.std(d):.4f}  "
            f"Range = [{np.min(d):.3f}, {np.max(d):.3f}]"
        )
    else:
        print(f"  Layer {l:2d}: no heads with R² > 0.90")

print()
print("  SYK q=4 prediction: Δ = 0.2500")
print("  SYK q=3 prediction: Δ = 0.3333")
print("  SYK q=2 prediction: Δ = 0.5000")
print()
