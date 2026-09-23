"""
exp-155 — Text-native Δ-window heads: W_K ablation content-retrieval test

Pre-registration: attention-geometry d8adc4d (committed before this script).
Analysis-only: GPT-2 small (cached). No new training.

Tests H_content: text-native heads are content-driven long-range retrievers.
Protocol: W_K=0 ablation on 16 text-native heads; measures Task C (content-specified
long-range retrieval) and Task B (positional retrieval, control) with sham.

Ariel — 2026-09-23, ~1:05 AM MDT, solo physics room.
"""

from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import importlib.util

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "d8adc4d"

TEXT_NATIVE = [
    (4, 10), (7, 1), (8, 2),
    (9, 4), (9, 6),
    (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9),
]

D_MODEL = 768
D_HEAD  = 64
N_HEADS = 12

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Task B items (from exp-143; same battery as established causal work) ──────
# These are the original exp-143 Task B items (positional retrieval)

TASK_B_ITEMS = [
    {"prompt": "Colors: red, blue, green, yellow. The second color is", "target": " blue"},
    {"prompt": "Animals: cat, dog, horse, rabbit. The third animal is", "target": " horse"},
    {"prompt": "Days: Monday, Wednesday, Friday, Sunday. The first day is", "target": " Monday"},
    {"prompt": "Rivers: Thames, Nile, Amazon, Rhine. The fourth river is", "target": " Rhine"},
    {"prompt": "Cities: Paris, London, Berlin, Rome. The second city is", "target": " London"},
    {"prompt": "Fruits: apple, pear, plum, grape. The third fruit is", "target": " plum"},
    {"prompt": "Metals: gold, iron, silver, copper. The first metal is", "target": " gold"},
    {"prompt": "Planets: Mars, Venus, Saturn, Jupiter. The fourth planet is", "target": " Jupiter"},
    {"prompt": "Months: March, June, October, January. The third month is", "target": " October"},
    {"prompt": "Birds: eagle, robin, sparrow, finch. The second bird is", "target": " robin"},
    {"prompt": "Shapes: circle, square, triangle, diamond. The fourth shape is", "target": " diamond"},
    {"prompt": "Trees: oak, pine, birch, maple. The third tree is", "target": " birch"},
    {"prompt": "Countries: France, Japan, Brazil, Canada. The second country is", "target": " Japan"},
    {"prompt": "Gems: ruby, pearl, jade, sapphire. The first gem is", "target": " ruby"},
    {"prompt": "Spices: salt, pepper, cumin, cinnamon. The fourth spice is", "target": " cinnamon"},
    {"prompt": "Tools: hammer, saw, drill, chisel. The third tool is", "target": " drill"},
    {"prompt": "Seasons: winter, spring, summer, autumn. The second season is", "target": " spring"},
    {"prompt": "Coins: penny, nickel, dime, quarter. The fourth coin is", "target": " quarter"},
    {"prompt": "Flowers: rose, lily, daisy, iris. The first flower is", "target": " rose"},
    {"prompt": "Letters: alpha, beta, gamma, delta. The third letter is", "target": " gamma"},
]

# ── Model and tokenizer ───────────────────────────────────────────────────────

print("Loading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_orig = GPT2LMHeadModel.from_pretrained("gpt2")
model_orig.to(DEVICE)
model_orig.eval()

# ── W_K helpers ───────────────────────────────────────────────────────────────

def get_wk(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    """W_K for (layer, head). Returns (D_MODEL, D_HEAD)."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


def set_wk(model: GPT2LMHeadModel, layer: int, head: int, W_K_new: np.ndarray):
    """Write W_K back for (layer, head)."""
    W = model.transformer.h[layer].attn.c_attn.weight.data
    with torch.no_grad():
        W[:, D_MODEL + head * D_HEAD: D_MODEL + (head + 1) * D_HEAD] = torch.tensor(
            W_K_new, dtype=W.dtype, device=W.device
        )


def verify_wk_zero(model: GPT2LMHeadModel, layer: int, head: int) -> float:
    """Return ‖W_K‖_F after ablation — should be 0."""
    return float(np.linalg.norm(get_wk(model, layer, head)))


# ── Build ablated and sham models ─────────────────────────────────────────────

print("Building ablated model (W_K=0 for all 16 text-native heads)...")
model_abl = copy.deepcopy(model_orig)
rng = np.random.default_rng(seed=2026092301)

sham_info = {}
abl_kappa_checks = {}

for layer, head in TEXT_NATIVE:
    W_orig = get_wk(model_abl, layer, head)
    norm_orig = float(np.linalg.norm(W_orig, 'fro'))
    # Ablation: set W_K to zero
    set_wk(model_abl, layer, head, np.zeros_like(W_orig))
    after_norm = verify_wk_zero(model_abl, layer, head)
    abl_kappa_checks[f"L{layer}H{head}"] = {"norm_before": norm_orig, "norm_after": after_norm}
    sham_info[f"L{layer}H{head}"] = {"orig_norm": norm_orig}

print("Ablation K3 check:")
k3_failed = 0
for key, val in abl_kappa_checks.items():
    ok = val["norm_after"] < 0.01
    if not ok:
        k3_failed += 1
        print(f"  K3 FAIL: {key} norm_after={val['norm_after']:.4f}")
    else:
        print(f"  {key}: norm {val['norm_before']:.3f} → {val['norm_after']:.4f} ✓")

if k3_failed >= 6:
    print(f"WARNING: K3 fires — {k3_failed} heads failed ablation")

print("\nBuilding sham model (random W_K, matched norm)...")
model_sham = copy.deepcopy(model_orig)
for layer, head in TEXT_NATIVE:
    W_orig = get_wk(model_sham, layer, head)
    norm_orig = float(np.linalg.norm(W_orig, 'fro'))
    W_rand = rng.standard_normal(W_orig.shape)
    W_rand = W_rand * norm_orig / (np.linalg.norm(W_rand, 'fro') + 1e-14)
    set_wk(model_sham, layer, head, W_rand)
    # Verify sham
    after_norm = float(np.linalg.norm(get_wk(model_sham, layer, head), 'fro'))
    sham_info[f"L{layer}H{head}"]["sham_norm"] = after_norm

# ── Scoring function ──────────────────────────────────────────────────────────

def score_target(model, prompt: str, target: str) -> float:
    """
    Returns log P(target | prompt) summed over target tokens, averaged per token.
    If target is a single token, returns log P of that token.
    """
    full_text = prompt + target
    enc_full = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    enc_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    n_prompt = enc_prompt.shape[1]

    with torch.no_grad():
        out = model(enc_full)
    logits = out.logits  # (1, seq, vocab)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Score: sum log P over target tokens
    # Positions n_prompt-1 to -2 predict positions n_prompt to -1
    total_lp = 0.0
    n_target_tokens = enc_full.shape[1] - n_prompt
    if n_target_tokens == 0:
        return 0.0

    for i in range(n_target_tokens):
        pred_pos = n_prompt - 1 + i   # model predicts target[i] from position pred_pos
        target_tok = enc_full[0, n_prompt + i].item()
        total_lp += log_probs[0, pred_pos, target_tok].item()

    return total_lp / n_target_tokens   # average per target token


# ── Load Task C items ─────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
with open(HERE / "task_c_items.json") as f:
    task_c_data = json.load(f)
task_c_items_raw = task_c_data["items"]

# Build full prompts: setup + filler + cue
def build_task_c_prompt(item: dict) -> str:
    return f"{item['setup']} {item['filler']} {item['cue']}"


# ── Run scoring for Task B ────────────────────────────────────────────────────

print("\n── Task B (positional retrieval) ────────────────────────────────────")
task_b_results = []
for i, item in enumerate(TASK_B_ITEMS):
    lp_orig  = score_target(model_orig,  item["prompt"], item["target"])
    lp_abl   = score_target(model_abl,   item["prompt"], item["target"])
    lp_sham  = score_target(model_sham,  item["prompt"], item["target"])
    delta_abl  = lp_abl  - lp_orig
    delta_sham = lp_sham - lp_orig
    print(f"  B-{i+1:02d}: orig={lp_orig:+.3f}, abl={lp_abl:+.3f}, sham={lp_sham:+.3f}  "
          f"Δabl={delta_abl:+.3f}")
    task_b_results.append({
        "id": f"B-{i+1:02d}",
        "prompt_short": item["prompt"][:50],
        "target": item["target"],
        "logp_orig": lp_orig,
        "logp_abl": lp_abl,
        "logp_sham": lp_sham,
        "delta_abl": delta_abl,
        "delta_sham": delta_sham,
    })


# ── Run scoring for Task C ────────────────────────────────────────────────────

print("\n── Task C (content retrieval) ────────────────────────────────────────")
task_c_results = []
for item in task_c_items_raw:
    prompt = build_task_c_prompt(item)
    target = item["target_token"]
    lp_orig  = score_target(model_orig,  prompt, target)
    lp_abl   = score_target(model_abl,   prompt, target)
    lp_sham  = score_target(model_sham,  prompt, target)
    delta_abl  = lp_abl  - lp_orig
    delta_sham = lp_sham - lp_orig
    # Approximate token count of prompt
    n_tokens = len(tokenizer(prompt)["input_ids"])
    print(f"  {item['id']}: orig={lp_orig:+.3f}, abl={lp_abl:+.3f}, sham={lp_sham:+.3f}  "
          f"Δabl={delta_abl:+.3f}  [{n_tokens} tokens]")
    task_c_results.append({
        "id": item["id"],
        "entity": item["entity"],
        "property": item["property"],
        "n_prompt_tokens": n_tokens,
        "logp_orig": lp_orig,
        "logp_abl": lp_abl,
        "logp_sham": lp_sham,
        "delta_abl": delta_abl,
        "delta_sham": delta_sham,
    })


# ── Verdict computation ───────────────────────────────────────────────────────

def median_delta(results, key="delta_abl"):
    return float(np.median([r[key] for r in results]))

def n_improved(results):
    return sum(1 for r in results if r["delta_abl"] > 0)

def n_degraded(results):
    return sum(1 for r in results if r["delta_abl"] < 0)


delta_P_C     = median_delta(task_c_results)
delta_P_C_sham = median_delta(task_c_results, "delta_sham")
delta_P_B     = median_delta(task_b_results)
delta_P_B_sham = median_delta(task_b_results, "delta_sham")

n_C_improved = n_improved(task_c_results)
n_B_improved = n_improved(task_b_results)
n_C_degraded = n_degraded(task_c_results)
n_B_degraded = n_degraded(task_b_results)

print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"Task C: median ΔP_C_abl = {delta_P_C:+.4f} nats  (sham: {delta_P_C_sham:+.4f})")
print(f"        {n_C_improved}/20 improved, {n_C_degraded}/20 degraded")
print(f"Task B: median ΔP_B_abl = {delta_P_B:+.4f} nats  (sham: {delta_P_B_sham:+.4f})")
print(f"        {n_B_improved}/20 improved, {n_B_degraded}/20 degraded")

# Registered predictions
P1_fires = delta_P_C < -0.10
P2_fires = abs(delta_P_B) < abs(delta_P_C)
K1_fires = delta_P_B < -0.10 and n_B_improved <= 6
K2_fires = abs(delta_P_C) < 0.10 and abs(delta_P_B) < 0.10
K3_fires = k3_failed >= 6

print(f"\nPredictions:")
print(f"  P1 (ΔP_C < -0.10): {'FIRES' if P1_fires else 'does not fire'}")
print(f"  P2 (|ΔP_B| < |ΔP_C|): {'FIRES' if P2_fires else 'does not fire'}")
print(f"  K1 (ΔP_B < -0.10 & n_B_improved ≤ 6): {'FIRES' if K1_fires else 'does not fire'}")
print(f"  K2 (both |ΔP| < 0.10): {'FIRES' if K2_fires else 'does not fire'}")
print(f"  K3 (ablation failure): {'FIRES' if K3_fires else 'does not fire'}")

if P1_fires and not K1_fires and not K2_fires:
    verdict = "confirmed"
    headline = "H_content confirmed: text-native heads support content retrieval (double dissociation)."
elif K2_fires:
    verdict = "falsified"
    headline = "K2 fires: text-native heads functionally neutral; geometry-function gap extends to this population."
elif K1_fires:
    verdict = "inconclusive"
    headline = "K1 fires: text-native heads affect positional retrieval; functional boundary not at content/position axis."
elif P1_fires and K1_fires:
    verdict = "inconclusive"
    headline = "Both Task B and Task C affected by ablation — contradictory; check protocol."
else:
    verdict = "inconclusive"
    headline = f"Neither P1 nor K2 fires cleanly (ΔP_C={delta_P_C:+.3f}, ΔP_B={delta_P_B:+.3f})."

print(f"\nVerdict: {verdict.upper()}")
print(f"Headline: {headline}")


# ── Save results ──────────────────────────────────────────────────────────────

results_out = {
    "experiment": "exp-155",
    "prereg_commit": PREREG_COMMIT,
    "protocol": "W_K=0 ablation of 16 text-native heads; sham: matched-norm random W_K",
    "model": "gpt2",
    "text_native_heads": [f"L{l}H{h}" for l, h in TEXT_NATIVE],
    "ablation_checks": abl_kappa_checks,
    "K3_fires": K3_fires,
    "task_b": {
        "n_items": len(task_b_results),
        "delta_P_B": delta_P_B,
        "delta_P_B_sham": delta_P_B_sham,
        "n_improved": n_B_improved,
        "items": task_b_results,
    },
    "task_c": {
        "n_items": len(task_c_results),
        "delta_P_C": delta_P_C,
        "delta_P_C_sham": delta_P_C_sham,
        "n_improved": n_C_improved,
        "items": task_c_results,
    },
    "predictions": {
        "P1_fires": P1_fires,
        "P2_fires": P2_fires,
        "K1_fires": K1_fires,
        "K2_fires": K2_fires,
        "K3_fires": K3_fires,
    },
    "verdict": verdict,
    "headline": headline,
}

out_path = HERE / "results.json"
with open(out_path, "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\nResults saved to {out_path}")
