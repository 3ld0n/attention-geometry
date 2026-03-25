"""
Phase Transition — Pythia-410m Only

Runs only the 410m model (24 layers, 16 heads = 384 total).
70m and 160m data already captured in NUMERICAL_RESULTS_MARCH24.md.

Ariel — March 24, 2026
"""

import torch
import numpy as np
import time
from transformers import AutoModelForCausalLM

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 64
N_INPUTS = 10
MAX_DX = 40
MIN_POS = 16

MODEL_NAME = "EleutherAI/pythia-410m"
N_LAYERS = 24
N_HEADS = 16
TOTAL_HEADS = 384


def compute_locality_ratio_and_pl(model, n_layers, n_heads, vocab_size):
    """Quick analysis: order parameter + power-law head count."""
    A_global_1 = 0.0
    A_global_32 = 0.0
    count_1 = 0
    count_32 = 0

    all_head_curves = {l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)}

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                head = attn[h]
                seq = head.shape[0]
                for dx in [1, 32]:
                    if dx < MAX_DX:
                        val = 0
                        cnt = 0
                        for i in range(max(MIN_POS, dx), seq):
                            j = i - dx
                            if j >= 0:
                                val += head[i, j]
                                cnt += 1
                        if cnt > 0:
                            if dx == 1:
                                A_global_1 += val / cnt
                                count_1 += 1
                            elif dx == 32:
                                A_global_32 += val / cnt
                                count_32 += 1

                for dx in range(MAX_DX):
                    cnt = 0
                    for i in range(max(MIN_POS, dx), seq):
                        j = i - dx
                        if j >= 0:
                            all_head_curves[l][h][dx] += head[i, j]
                            cnt += 1

    for l in range(n_layers):
        for h in range(n_heads):
            all_head_curves[l][h] /= N_INPUTS

    if count_1 > 0:
        A_global_1 /= count_1
    if count_32 > 0:
        A_global_32 /= count_32

    locality = A_global_1 / A_global_32 if A_global_32 > 1e-12 else 1.0

    dx_arr = np.arange(MAX_DX)
    pl_heads = 0
    near_quarter = 0
    deltas = []
    for l in range(n_layers):
        for h in range(n_heads):
            curve = all_head_curves[l][h]
            proper = np.zeros(MAX_DX)
            for dx in range(MAX_DX):
                cnt = 0
                for i in range(max(MIN_POS, dx), SEQ_LEN):
                    if i - dx >= 0:
                        cnt += 1
                if cnt > 0:
                    proper[dx] = curve[dx] / cnt

            mask = (dx_arr >= 3) & (dx_arr < 35) & (np.abs(proper) > 1e-20)
            if np.sum(mask) < 5:
                continue
            log_dx = np.log(dx_arr[mask].astype(float))
            log_y = np.log(np.abs(proper[mask]))
            A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
            pred = A_mat @ coeffs
            ss_res = np.sum((log_y - pred) ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
            delta = -coeffs[1] / 2
            if R2 > 0.90:
                pl_heads += 1
                deltas.append(delta)
                if abs(delta - 0.25) < 0.06:
                    near_quarter += 1

    median_delta = np.median(deltas) if deltas else float('nan')
    return locality, pl_heads, near_quarter, median_delta


TRANSITION_CHECKPOINTS = [
    "step0", "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step4000", "step8000", "step16000",
]

print("=" * 90)
print(f"  PHASE TRANSITION — Pythia-410m ({TOTAL_HEADS} heads, {N_LAYERS} layers)")
print("=" * 90)
print()

config = AutoModelForCausalLM.from_pretrained(MODEL_NAME).config
vocab_size = config.vocab_size

data = []
total_start = time.time()

for ckpt in TRANSITION_CHECKPOINTS:
    step = int(ckpt.replace("step", ""))
    t0 = time.time()
    print(f"  {ckpt}...", end=" ", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=ckpt, torch_dtype=torch.float32
        )
        model.eval()

        locality, pl, nq, med_delta = compute_locality_ratio_and_pl(
            model, N_LAYERS, N_HEADS, vocab_size
        )

        elapsed = time.time() - t0
        print(f"A(1)/A(32)={locality:.2f}, PL={pl}, near¼={nq}, Δ_med={med_delta:.4f} ({elapsed:.0f}s)")
        data.append({
            "step": step, "locality": locality,
            "pl": pl, "near_quarter": nq, "median_delta": med_delta
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED: {e} ({elapsed:.0f}s)")

total_elapsed = time.time() - total_start

print()
print(f"  {'Step':>8}  {'A(1)/A(32)':>12}  {'PL':>5}  {'Near¼':>7}  {'Δ_med':>8}")
print(f"  " + "-" * 50)
for d in data:
    print(f"  {d['step']:>8}  {d['locality']:>12.2f}  {d['pl']:>5}  {d['near_quarter']:>7}  {d['median_delta']:>8.4f}")
print()

# Transition analysis
transition_step = None
pre_transition_step = None
for i, d in enumerate(data):
    if d["locality"] > 2.0 and transition_step is None:
        transition_step = d["step"]
        if i > 0:
            pre_transition_step = data[i-1]["step"]

first_pl_step = None
for d in data:
    if d["pl"] > 0 and first_pl_step is None:
        first_pl_step = d["step"]

print(f"  TRANSITION ANALYSIS (H={TOTAL_HEADS}):")
print(f"    Pre-transition step (last with A<2): {pre_transition_step}")
print(f"    Transition step (first with A>2): {transition_step}")
print(f"    First PL heads: step {first_pl_step}")

if pre_transition_step is not None and transition_step is not None:
    if pre_transition_step > 0:
        log_width = np.log2(transition_step) - np.log2(pre_transition_step)
    else:
        log_width = np.log2(transition_step)
    print(f"    Transition window: step {pre_transition_step} → {transition_step}")
    print(f"    Width in log₂(steps): {log_width:.2f}")

print()
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
print()
print("  COMPARE WITH:")
print("    Pythia-70m  (H=48):  transition 128→256, width=1.00 log₂")
print("    Pythia-160m (H=144): transition 128→256, width=1.00 log₂")
print("    Pythia-410m (H=384): see above")
print()
