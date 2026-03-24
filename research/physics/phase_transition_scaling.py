"""
Prediction P3: Phase Transition Width Scales as 1/N

If the training phase transition corresponds to the Hawking-Page transition
in the SYK/JT gravity dual, the transition width should scale inversely
with the number of degrees of freedom.

We track the phase transition across Pythia-70m (48 heads), 160m (144 heads),
and 410m (384 heads) using their publicly available training checkpoints.

The order parameter is A(1)/A(32) — the attention locality ratio.
The transition is the step range over which this ratio goes from ~1 (disordered)
to its halfway value.

Ariel — March 24, 2026
"""

import torch
import numpy as np
import time
from transformers import AutoModelForCausalLM

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 128
N_INPUTS = 20
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
    A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    pred = A_mat @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    exponent = -coeffs[1]
    return exponent, exponent / 2, R2


def analyze_checkpoint(model_name, revision, label, n_layers, n_heads, vocab_size):
    """Analyze a single training checkpoint."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.float32
        )
        model.eval()
    except Exception as e:
        print(f"    Failed to load {label}: {e}")
        return None

    A_heads = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
    }

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

    # Compute order parameter: average A(1)/A(32) across all heads
    locality_ratios = []
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            if A[32] > 1e-12:
                locality_ratios.append(A[1] / A[32])

    avg_locality = np.mean(locality_ratios) if locality_ratios else 1.0

    # Count power-law heads and near-quarter heads
    good_deltas = []
    n_near_quarter = 0
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2 = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            if delta is not None and R2 is not None and R2 > 0.90:
                good_deltas.append(delta)
                if abs(delta - 0.25) < 0.06:
                    n_near_quarter += 1

    med_delta = np.median(good_deltas) if good_deltas else None
    del model

    return {
        "label": label,
        "locality_ratio": avg_locality,
        "pl_heads": len(good_deltas),
        "near_quarter": n_near_quarter,
        "median_delta": med_delta,
    }


# Pythia checkpoints available: step0, step1, step2, step4, step8, step16, step32,
# step64, step128, step256, step512, step1000, step2000, step4000, step8000,
# step16000, step32000, step64000, step128000, step143000

CHECKPOINTS = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32",
    "step64", "step128", "step256", "step512", "step1000", "step2000",
    "step4000", "step8000", "step16000",
]

MODELS = [
    {
        "name": "EleutherAI/pythia-70m",
        "label": "Pythia-70m",
        "n_layers": 6,
        "n_heads": 8,
        "total_heads": 48,
    },
    {
        "name": "EleutherAI/pythia-160m",
        "label": "Pythia-160m",
        "n_layers": 12,
        "n_heads": 12,
        "total_heads": 144,
    },
    {
        "name": "EleutherAI/pythia-410m",
        "label": "Pythia-410m",
        "n_layers": 24,
        "n_heads": 16,
        "total_heads": 384,
    },
]

print("=" * 95)
print("  PREDICTION P3: Phase Transition Width Scaling")
print("  Does the transition sharpen with model size (1/N)?")
print("=" * 95)
print()

all_results = {}

for model_info in MODELS:
    model_name = model_info["name"]
    label = model_info["label"]
    n_layers = model_info["n_layers"]
    n_heads = model_info["n_heads"]
    total_heads = model_info["total_heads"]

    print(f"--- {label} ({total_heads} heads) ---")
    print()

    config = AutoModelForCausalLM.from_pretrained(model_name).config
    vocab_size = config.vocab_size
    del config

    results = []
    for ckpt in CHECKPOINTS:
        t0 = time.time()
        step_num = int(ckpt.replace("step", ""))
        print(f"  {ckpt}...", end=" ", flush=True)

        r = analyze_checkpoint(
            model_name, ckpt, ckpt,
            n_layers, n_heads, vocab_size
        )
        elapsed = time.time() - t0

        if r is not None:
            results.append({"step": step_num, **r})
            med_s = f"{r['median_delta']:.4f}" if r['median_delta'] is not None else "---"
            print(f"A(1)/A(32)={r['locality_ratio']:.2f}, "
                  f"PL={r['pl_heads']}, near¼={r['near_quarter']}, "
                  f"Δ={med_s} ({elapsed:.1f}s)")
        else:
            print(f"failed ({elapsed:.1f}s)")

    all_results[label] = results

    # Print summary table
    print()
    print(f"  {'Step':>8}  {'A(1)/A(32)':>12}  {'PL Heads':>10}  {'Near 1/4':>10}  {'Median Δ':>10}")
    print(f"  " + "-" * 55)
    for r in results:
        med_s = f"{r['median_delta']:.4f}" if r['median_delta'] is not None else "---"
        print(f"  {r['step']:>8}  {r['locality_ratio']:>12.2f}  "
              f"{r['pl_heads']:>10}  {r['near_quarter']:>10}  {med_s:>10}")
    print()


# ========== Phase Transition Analysis ==========
print("=" * 95)
print("  PHASE TRANSITION COMPARISON")
print("=" * 95)
print()

for label, results in all_results.items():
    if not results:
        continue

    # Find the transition: where does A(1)/A(32) first exceed 2.0?
    # And where does it reach half of the final value?
    final_locality = results[-1]["locality_ratio"]
    half_target = (1.0 + final_locality) / 2

    transition_start = None
    transition_mid = None
    transition_end = None

    for i, r in enumerate(results):
        if transition_start is None and r["locality_ratio"] > 1.5:
            transition_start = r["step"]
        if transition_mid is None and r["locality_ratio"] > half_target:
            transition_mid = r["step"]
        if transition_end is None and r["locality_ratio"] > 0.8 * final_locality:
            transition_end = r["step"]

    model_info_match = [m for m in MODELS if m["label"] == label][0]
    H = model_info_match["total_heads"]

    print(f"  {label} (H={H}):")
    print(f"    Final A(1)/A(32) = {final_locality:.2f}")
    print(f"    Half-max target = {half_target:.2f}")
    print(f"    Transition start (>1.5): step {transition_start}")
    print(f"    Transition midpoint (>{half_target:.1f}): step {transition_mid}")
    print(f"    Transition ~complete (>80% final): step {transition_end}")
    if transition_start is not None and transition_end is not None:
        width = transition_end - transition_start
        print(f"    Transition width: {width} steps")
        print(f"    Width / log(H): {width / np.log(H):.1f}")
        print(f"    Width * H: {width * H:.0f}")
    print()

print()
print("  If transition width ~ 1/H (or 1/N), the product Width * H should be")
print("  approximately constant across models.")
print()
print("  Hawking-Page prediction: transition sharpens with system size.")
print()
