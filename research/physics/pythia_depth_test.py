"""
Depth Test: Pythia 70m vs 160m vs 410m (fully trained only)

Tests whether deeper models stabilize Δ closer to 1/4.
GPT-2 (12 layers) holds Δ at 0.25. Pythia-70m (6 layers) doesn't stabilize as tightly.
If 160m (12 layers) and 410m (24 layers) tighten the Δ distribution, depth matters.

Ariel — March 24, 2026
"""

import torch
import numpy as np
import time
from transformers import AutoModelForCausalLM

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 30
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


MODELS = [
    ("EleutherAI/pythia-70m", "Pythia-70m", 6, 8),
    ("EleutherAI/pythia-160m", "Pythia-160m", 12, 12),
    ("EleutherAI/pythia-410m", "Pythia-410m", 24, 16),
]

print("=" * 95)
print("  DEPTH TEST: Does deeper Pythia stabilize Δ at 1/4?")
print("=" * 95)
print()

all_model_results = []

for model_name, label, expected_layers, expected_heads in MODELS:
    print(f"Loading {label} ({model_name})...", end=" ", flush=True)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
    except Exception as e:
        print(f"FAILED: {e}")
        continue

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size
    total_heads = n_layers * n_heads
    load_time = time.time() - t0
    print(f"loaded ({load_time:.1f}s). {n_layers} layers, {n_heads} heads, {total_heads} total.")

    A_heads = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
    }

    print(f"  Processing {N_INPUTS} inputs...", end=" ", flush=True)
    t1 = time.time()
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    analysis_time = time.time() - t1

    good_deltas = []
    n_near_quarter = 0
    layer_deltas = {}
    for l in range(n_layers):
        layer_deltas[l] = []
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2 = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            if delta is not None and R2 is not None and R2 > 0.90:
                good_deltas.append(delta)
                layer_deltas[l].append(delta)
                if abs(delta - 0.25) < 0.06:
                    n_near_quarter += 1

    result = {
        "label": label,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads": total_heads,
        "pl_heads": len(good_deltas),
        "near_quarter": n_near_quarter,
        "median_delta": np.median(good_deltas) if good_deltas else None,
        "mean_delta": np.mean(good_deltas) if good_deltas else None,
        "std_delta": np.std(good_deltas) if good_deltas else None,
        "layer_deltas": layer_deltas,
    }
    all_model_results.append(result)

    med_s = f"{result['median_delta']:.4f}" if result['median_delta'] is not None else "---"
    print(f"done ({analysis_time:.1f}s). PL heads: {result['pl_heads']}/{total_heads}, "
          f"near 1/4: {n_near_quarter}, median Δ: {med_s}")

    # Deep layer analysis
    deep_start = max(0, n_layers - 3)
    deep_deltas = []
    for l in range(deep_start, n_layers):
        deep_deltas.extend(layer_deltas.get(l, []))
    if deep_deltas:
        dd = np.array(deep_deltas)
        print(f"  Deep layers ({deep_start+1}-{n_layers}): N={len(dd)}, "
              f"mean Δ={np.mean(dd):.4f}, std={np.std(dd):.4f}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()


# ========== Comparison ==========
print("=" * 95)
print("  DEPTH COMPARISON SUMMARY")
print("=" * 95)
print()
print(f"  {'Model':>15}  {'Layers':>8}  {'Total Heads':>13}  {'PL Heads':>10}  "
      f"{'Near 1/4':>10}  {'Median Δ':>10}  {'Std Δ':>8}")
print("  " + "-" * 78)

for r in all_model_results:
    med_s = f"{r['median_delta']:.4f}" if r['median_delta'] is not None else "---"
    std_s = f"{r['std_delta']:.4f}" if r['std_delta'] is not None else "---"
    print(f"  {r['label']:>15}  {r['n_layers']:8d}  {r['total_heads']:13d}  "
          f"{r['pl_heads']:10d}  {r['near_quarter']:10d}  {med_s:>10}  {std_s:>8}")

print()
print("  GPT-2 reference: 12 layers, 144 heads, 44 PL heads, median Δ = 0.2493")
print("  SYK q=4 prediction: Δ = 0.2500")
print()
print("  If median Δ tightens toward 0.25 with increasing depth,")
print("  depth stabilizes the conformal fixed point.")
print()
