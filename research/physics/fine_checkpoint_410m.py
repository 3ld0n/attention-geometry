"""
Fine Checkpoint Sampling — Pythia-410m

Samples the 16k-143k range to track the Δ trajectory.
Does the system show a prethermal plateau near Δ ≈ 0.50
before converging to Δ ≈ 0.25?

Ariel — March 25, 2026
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


def compute_metrics(model, n_layers, n_heads, vocab_size):
    """Order parameter + power-law head analysis."""
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
    near_half = sum(1 for d in deltas if abs(d - 0.50) < 0.06) if deltas else 0
    return locality, pl_heads, near_quarter, near_half, median_delta


# Focus on the 16k-143k range where Δ should be flowing from ~0.55 toward ~0.28
CHECKPOINTS = [
    "step16000", "step32000", "step64000", "step100000", "step143000",
]

print("=" * 90)
print(f"  FINE CHECKPOINT SAMPLING — Pythia-410m (steps 16k-143k)")
print(f"  Question: does Δ plateau near 0.50 or flow smoothly toward 0.25?")
print("=" * 90)
print()

config = AutoModelForCausalLM.from_pretrained(MODEL_NAME).config
vocab_size = config.vocab_size

data = []
total_start = time.time()

for ckpt in CHECKPOINTS:
    step = int(ckpt.replace("step", ""))
    t0 = time.time()
    print(f"  {ckpt}...", end=" ", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=ckpt, torch_dtype=torch.float32
        )
        model.eval()

        locality, pl, nq, nh, med_delta = compute_metrics(
            model, N_LAYERS, N_HEADS, vocab_size
        )

        elapsed = time.time() - t0
        print(f"A={locality:.2f}, PL={pl}, near¼={nq}, near½={nh}, Δ_med={med_delta:.4f} ({elapsed:.0f}s)")
        data.append({
            "step": step, "locality": locality,
            "pl": pl, "near_quarter": nq, "near_half": nh,
            "median_delta": med_delta
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED: {e} ({elapsed:.0f}s)")

total_elapsed = time.time() - total_start

print()
print(f"  {'Step':>8}  {'A(1)/A(32)':>12}  {'PL':>5}  {'Near¼':>7}  {'Near½':>7}  {'Δ_med':>8}")
print(f"  " + "-" * 55)
for d in data:
    print(f"  {d['step']:>8}  {d['locality']:>12.2f}  {d['pl']:>5}  {d['near_quarter']:>7}  {d['near_half']:>7}  {d['median_delta']:>8.4f}")

print()
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
print()
print("  COMBINED TRAJECTORY (including earlier data):")
print(f"  {'Step':>8}  {'Δ_med':>8}  {'note':>20}")
print(f"  " + "-" * 40)
earlier = [
    (0, 0.0642, "disordered"),
    (256, 0.2103, "first PL head"),
    (512, 0.2015, "pre-transition"),
    (1000, 0.5542, "transition"),
    (2000, 0.5804, "plateau?"),
    (4000, 0.5354, "plateau?"),
    (8000, 0.4707, "plateau?"),
    (16000, 0.5528, "plateau?"),
]
for step, delta, note in earlier:
    print(f"  {step:>8}  {delta:>8.4f}  {note:>20}")
for d in data:
    print(f"  {d['step']:>8}  {d['median_delta']:>8.4f}  {'NEW':>20}")
print()
print("  KEY QUESTION: Does Δ stay near 0.50 (prethermal q=2 plateau)")
print("  or flow smoothly toward 0.25 (ordinary RG flow)?")
print()
