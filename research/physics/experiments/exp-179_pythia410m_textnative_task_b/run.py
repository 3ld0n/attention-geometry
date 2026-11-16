"""
exp-179 — Pythia-410m text-native Δ-window heads: positional retrieval (Task B)

Pre-registration: attention-geometry 2246e55 (committed and pushed before this script).
Tests whether the competitor structure found in exp-178 (ablating text-native heads
IMPROVES content retrieval) reflects a positive positional-retrieval function:
do these heads carry Task B (list-lookup at stated position) function?

Instrument: individual W_K ablation on 20-item Task B battery (task_b_items.json).
Population: 16 text-native Δ-window heads in Pythia-410m from exp-118.

Declared prior: H_positional (≥10/16 degrade Task B); RoPE→positional, AbsPE→content.

Ariel — 2026-11-15, ~7:00 PM MST, solo physics room.
"""

from __future__ import annotations
import copy
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "2246e55"

# 16 text-native Δ-window heads from exp-118 (layer, head) for Pythia-410m
TEXT_NATIVE: List[Tuple[int, int]] = [
    (0, 3), (1, 8), (2, 8),
    (3, 2), (3, 7), (3, 14),
    (4, 12),
    (7, 11),
    (8, 15),
    (9, 15),
    (12, 3), (12, 13),
    (14, 15),
    (19, 15),
    (21, 11),
    (22, 1),
]

# Competitor heads from exp-178 (ablation improved Task C)
# Heads where ΔP_abl > 0 on Task C (exp-178 results)
COMPETITOR_HEADS: List[Tuple[int, int]] = [
    (1, 8), (2, 8), (3, 2), (3, 7), (3, 14), (7, 11), (8, 15), (12, 3), (21, 11),
]

D_MODEL = 1024
D_HEAD  = 64
N_HEADS = 16
N_LAYERS = 24

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Task B items ───────────────────────────────────────────────────────────────

TASK_B_PATH = Path(__file__).parent / "task_b_items.json"
with open(TASK_B_PATH) as f:
    task_b_data = json.load(f)
TASK_B_ITEMS = task_b_data["items"]
assert len(TASK_B_ITEMS) == 20, f"Expected 20 items, got {len(TASK_B_ITEMS)}"

# ── Model and tokenizer ───────────────────────────────────────────────────────

print("Loading Pythia-410m...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model_orig = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-410m",
    torch_dtype=torch.float32,
    attn_implementation="eager",
)
model_orig.to(DEVICE)
model_orig.eval()
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── Token verification ─────────────────────────────────────────────────────────

print("Verifying all Task B targets are single tokens...")
for item in TASK_B_ITEMS:
    target = item["target_token"]
    ids = tokenizer.encode(target, add_special_tokens=False)
    assert len(ids) == 1, (
        f"Item {item['id']}: target '{target}' has {len(ids)} tokens: {ids}. "
        "Pre-registration verified single-token; tokenizer version mismatch?"
    )
print("  All 20 targets verified single-token.")

# ── W_K helpers ───────────────────────────────────────────────────────────────

def get_wk(model, layer, head):
    """Extract W_K for (layer, head) from query_key_value weight matrix."""
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.detach().cpu().float().numpy()
    # QKV layout: [Q block | K block | V block], each D_MODEL rows
    # K head h starts at row D_MODEL + h*D_HEAD
    offset = D_MODEL + head * D_HEAD
    return W[offset: offset + D_HEAD, :].copy()  # (d_head, d_model)


def set_wk_zero(model, layer, head):
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.data
    offset = D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[offset: offset + D_HEAD, :] = 0.0


def set_wk_random(model, layer, head, sigma, rng):
    """Set W_K to random Gaussian with given std (sham control)."""
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.data
    offset = D_MODEL + head * D_HEAD
    rand = torch.tensor(
        rng.normal(0, sigma, (D_HEAD, D_MODEL)), dtype=W.dtype, device=W.device
    )
    with torch.no_grad():
        W[offset: offset + D_HEAD, :] = rand


def restore_wk(model, layer, head, W_K_saved):
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.data
    offset = D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[offset: offset + D_HEAD, :] = torch.tensor(W_K_saved, dtype=W.dtype, device=W.device)


def wk_norm(wk_array):
    return float(np.linalg.norm(wk_array))


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(item):
    """Build Task B prompt: numbered list + positional retrieval cue."""
    cat = item["category"]
    lst = item["list_items"]
    N = item["probe_position"]
    lines = [f"Here is a list of {cat}:"]
    for i, w in enumerate(lst, 1):
        lines.append(f"{i}. {w}")
    lines.append(f"\nWhat is item number {N} in the list? Item number {N} is")
    return "\n".join(lines)


# ── Forward pass helper ───────────────────────────────────────────────────────

def token_logprob(model, prompt, target_token):
    """Return log-prob of target_token (single token, includes leading space) following prompt."""
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    assert len(target_ids) == 1, f"target '{target_token}' tokenized to {len(target_ids)} tokens"
    target_id = target_ids[0]

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]  # logits at last position
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(log_probs[target_id].cpu())


# ── Run experiment ─────────────────────────────────────────────────────────────

print(f"\nRunning exp-179 (pre-reg: {PREREG_COMMIT})")
print(f"Population: {len(TEXT_NATIVE)} text-native Δ-window heads in Pythia-410m")
print(f"Battery: {len(TASK_B_ITEMS)} Task B items (positional retrieval)")
print()

rng = np.random.default_rng(179)

# Pre-build all prompts and cache baseline log-probs
print("Computing baselines...")
prompts = [build_prompt(item) for item in TASK_B_ITEMS]
targets = [item["target_token"] for item in TASK_B_ITEMS]
baselines = [token_logprob(model_orig, p, t) for p, t in zip(prompts, targets)]
print(f"  Mean baseline log-prob: {np.mean(baselines):.3f}")

results_per_head = []

for layer, head in TEXT_NATIVE:
    head_id = f"L{layer}H{head}"
    print(f"\nHead {head_id}:")

    # Save W_K
    W_K_saved = get_wk(model_orig, layer, head)
    sigma_K = wk_norm(W_K_saved) / (D_HEAD * D_MODEL) ** 0.5

    # --- Ablation ---
    set_wk_zero(model_orig, layer, head)
    abl_lps = [token_logprob(model_orig, p, t) for p, t in zip(prompts, targets)]
    restore_wk(model_orig, layer, head, W_K_saved)

    # --- Sham ---
    set_wk_random(model_orig, layer, head, sigma_K, rng)
    sham_lps = [token_logprob(model_orig, p, t) for p, t in zip(prompts, targets)]
    restore_wk(model_orig, layer, head, W_K_saved)

    # Compute deltas
    delta_abl  = [a - b for a, b in zip(abl_lps,  baselines)]
    delta_sham = [s - b for s, b in zip(sham_lps, baselines)]

    mean_abl  = float(np.mean(delta_abl))
    mean_sham = float(np.mean(delta_sham))
    items_degraded = sum(1 for d in delta_abl if d < 0)
    items_improved = sum(1 for d in delta_abl if d > 0)

    is_competitor_from_178 = (layer, head) in COMPETITOR_HEADS
    carries_positional = mean_abl < -0.05 and items_degraded >= 12

    print(f"  ΔP_B abl={mean_abl:+.3f}  sham={mean_sham:+.3f}  "
          f"degraded={items_degraded}/20  improved={items_improved}/20  "
          f"competitor_178={is_competitor_from_178}  carries_pos={carries_positional}")

    results_per_head.append({
        "layer": layer,
        "head": head,
        "head_id": head_id,
        "mean_delta_abl": round(mean_abl, 4),
        "mean_delta_sham": round(mean_sham, 4),
        "items_degraded": items_degraded,
        "items_improved": items_improved,
        "per_item_delta_abl": [round(d, 4) for d in delta_abl],
        "per_item_delta_sham": [round(d, 4) for d in delta_sham],
        "is_competitor_from_exp178": is_competitor_from_178,
        "carries_positional_function": carries_positional,
    })

# ── Compute summary statistics ─────────────────────────────────────────────────

mean_deltas = [r["mean_delta_abl"] for r in results_per_head]
n_positional = sum(1 for r in results_per_head if r["carries_positional_function"])
cv = float(np.std(mean_deltas) / abs(np.mean(mean_deltas))) if np.mean(mean_deltas) != 0 else float("inf")

competitor_heads_set = {(r["layer"], r["head"]) for r in results_per_head if r["is_competitor_from_exp178"]}
n_competitor_positional = sum(
    1 for r in results_per_head
    if r["is_competitor_from_exp178"] and r["carries_positional_function"]
)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Heads with positional function (H_positional): {n_positional}/16")
print(f"  Threshold: ΔP_B < -0.05 AND items_degraded ≥ 12/20")
print(f"H_positional {'CONFIRMED' if n_positional >= 10 else 'FALSIFIED'} (threshold: ≥10/16)")
print(f"CV of mean ΔP_B: {cv:.3f} ({'CONFIRMED' if cv >= 0.5 else 'FALSIFIED'} H_graded; threshold: ≥0.5)")
print(f"Competitor heads from exp-178 carrying positional function: {n_competitor_positional}/{len(competitor_heads_set)}")
print(f"H_competitor_null: {'NOT FIRED' if n_competitor_positional < 5 else 'FIRED'} (fires if ≥5 competitor heads carry positional)")

print(f"\nPer-head ΔP_B (ablation):")
for r in sorted(results_per_head, key=lambda x: x["mean_delta_abl"]):
    comp = "C" if r["is_competitor_from_exp178"] else " "
    pos  = "P" if r["carries_positional_function"] else " "
    print(f"  {comp}{pos} {r['head_id']:8s}  ΔP_abl={r['mean_delta_abl']:+.3f}  "
          f"degraded={r['items_degraded']}/20")

# ── Save results ───────────────────────────────────────────────────────────────

results = {
    "experiment": "exp-179",
    "prereg_commit": PREREG_COMMIT,
    "date": "2026-11-15",
    "model": "EleutherAI/pythia-410m",
    "battery": "task_b_items.json",
    "n_items": len(TASK_B_ITEMS),
    "n_heads": len(TEXT_NATIVE),
    "baselines": {
        "mean_log_prob": round(float(np.mean(baselines)), 4),
        "per_item": [round(b, 4) for b in baselines],
    },
    "hypotheses": {
        "H_positional": {
            "verdict": "confirmed" if n_positional >= 10 else "falsified",
            "threshold": ">=10/16 heads with ΔP_B < -0.05 AND items_degraded >= 12/20",
            "result": f"{n_positional}/16"
        },
        "H_graded": {
            "verdict": "confirmed" if cv >= 0.5 else "falsified",
            "threshold": "CV >= 0.5",
            "result": round(cv, 4)
        },
        "H_competitor_null": {
            "verdict": "not_fired" if n_competitor_positional < 5 else "fired",
            "threshold": "fires if >=5 competitor heads carry positional function",
            "result": f"{n_competitor_positional}/{len(competitor_heads_set)}"
        },
    },
    "per_head_results": results_per_head,
    "mean_delta_abl_all_heads": round(float(np.mean(mean_deltas)), 4),
    "cv_mean_delta": round(cv, 4),
    "n_positional_heads": n_positional,
    "n_competitor_positional": n_competitor_positional,
    "n_competitor_total": len(competitor_heads_set),
}

out_path = Path(__file__).parent / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
