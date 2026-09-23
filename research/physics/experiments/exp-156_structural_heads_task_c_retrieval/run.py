"""
exp-156 — Structural (random-native) Δ-window heads: W_K ablation content-retrieval test

Pre-registration: attention-geometry ecaa354 (committed before this script).
Analysis-only: GPT-2 small (cached). No new training.

Tests H_positional_selective: structural heads support positional retrieval (W_K-dependent)
and are functionally neutral for content retrieval (Task C).
Protocol: W_K=0 ablation on 5 structural heads; measures Task C (content-specified
long-range retrieval, primary) and Task B (positional retrieval, control) with sham.

Symmetric to exp-155 (text-native W_K ablation). Task C items reused from exp-155.

Ariel — 2026-09-23, ~12:30 PM MDT, solo physics room.
"""

from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "ecaa354"

# Structural (random-native) Δ-window heads in GPT-2 small
# Selected by random-token census (exp-007 / exp-113): Δ_A ∈ [0.20, 0.30], R² ≥ 0.90
STRUCTURAL = [
    (2, 1), (3, 4), (5, 0), (7, 11), (10, 8),
]

D_MODEL = 768
D_HEAD  = 64
N_HEADS = 12

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Task B items (from exp-143; same battery as established causal work) ──────

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


def verify_wk_norm(model: GPT2LMHeadModel, layer: int, head: int) -> float:
    return float(np.linalg.norm(get_wk(model, layer, head), 'fro'))


# ── Build ablated and sham models ─────────────────────────────────────────────

print("Building ablated model (W_K=0 for 5 structural heads)...")
model_abl = copy.deepcopy(model_orig)
rng = np.random.default_rng(seed=2026092302)

sham_info = {}
abl_checks = {}

for layer, head in STRUCTURAL:
    W_orig = get_wk(model_abl, layer, head)
    norm_orig = float(np.linalg.norm(W_orig, 'fro'))
    set_wk(model_abl, layer, head, np.zeros_like(W_orig))
    norm_after = verify_wk_norm(model_abl, layer, head)
    abl_checks[f"L{layer}H{head}"] = {"norm_before": norm_orig, "norm_after": norm_after}
    sham_info[f"L{layer}H{head}"] = {"orig_norm": norm_orig}

print("Ablation K3 check (threshold: ‖W_K‖_F < 0.01):")
k3_fail_count = 0
for key, val in abl_checks.items():
    ok = val["norm_after"] < 0.01
    if not ok:
        k3_fail_count += 1
        print(f"  K3 FAIL: {key} norm_after={val['norm_after']:.4f}")
    else:
        print(f"  {key}: {val['norm_before']:.3f} → {val['norm_after']:.6f} ✓")

K3_fires = k3_fail_count >= 3   # ≥ 3/5 heads failed

print("\nBuilding sham model (random W_K, matched norm)...")
model_sham = copy.deepcopy(model_orig)
for layer, head in STRUCTURAL:
    W_orig = get_wk(model_sham, layer, head)
    norm_orig = float(np.linalg.norm(W_orig, 'fro'))
    W_rand = rng.standard_normal(W_orig.shape)
    W_rand = W_rand * norm_orig / (np.linalg.norm(W_rand, 'fro') + 1e-14)
    set_wk(model_sham, layer, head, W_rand)
    sham_norm = verify_wk_norm(model_sham, layer, head)
    sham_info[f"L{layer}H{head}"]["sham_norm"] = sham_norm

# ── Scoring function ──────────────────────────────────────────────────────────

def score_target(model, prompt: str, target: str) -> float:
    """
    Returns log P(target | prompt) averaged per target token.
    Identical to exp-155 scoring protocol.
    """
    full_text = prompt + target
    enc_full   = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    enc_prompt = tokenizer(prompt,    return_tensors="pt").input_ids.to(DEVICE)
    n_prompt = enc_prompt.shape[1]

    with torch.no_grad():
        out = model(enc_full)
    logits   = out.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    n_target_tokens = enc_full.shape[1] - n_prompt
    if n_target_tokens == 0:
        return 0.0

    total_lp = 0.0
    for i in range(n_target_tokens):
        pred_pos   = n_prompt - 1 + i
        target_tok = enc_full[0, n_prompt + i].item()
        total_lp  += log_probs[0, pred_pos, target_tok].item()

    return total_lp / n_target_tokens


# ── Load Task C items from exp-155 ────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
EXP155_DIR = HERE.parent / "exp-155_textnative_content_retrieval"
task_c_path = EXP155_DIR / "task_c_items.json"
if not task_c_path.exists():
    print(f"ERROR: task_c_items.json not found at {task_c_path}")
    sys.exit(1)

with open(task_c_path) as f:
    task_c_data = json.load(f)
task_c_items_raw = task_c_data["items"]
print(f"\nLoaded {len(task_c_items_raw)} Task C items from exp-155.")


def build_task_c_prompt(item: dict) -> str:
    return f"{item['setup']} {item['filler']} {item['cue']}"


# ── Run scoring ───────────────────────────────────────────────────────────────

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


delta_P_B      = median_delta(task_b_results)
delta_P_B_sham = median_delta(task_b_results, "delta_sham")
delta_P_C      = median_delta(task_c_results)
delta_P_C_sham = median_delta(task_c_results, "delta_sham")

n_B_improved = n_improved(task_b_results)
n_B_degraded = n_degraded(task_b_results)
n_C_improved = n_improved(task_c_results)
n_C_degraded = n_degraded(task_c_results)

print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"Task B: median ΔP_B_abl = {delta_P_B:+.4f} nats  (sham: {delta_P_B_sham:+.4f})")
print(f"        {n_B_improved}/20 improved, {n_B_degraded}/20 degraded")
print(f"Task C: median ΔP_C_abl = {delta_P_C:+.4f} nats  (sham: {delta_P_C_sham:+.4f})")
print(f"        {n_C_improved}/20 improved, {n_C_degraded}/20 degraded")

# Registered predictions (from prereg.md)
#   P1: ΔP_B < -0.10 nats (positional retrieval degrades)
#   P2: |ΔP_C| < |ΔP_B| in absolute value (Task C less affected)
#   K1: ΔP_C < -0.10 nats (content retrieval substantially impaired)
#   K2: |ΔP_B| < 0.10 AND |ΔP_C| < 0.10 (both tasks unaffected — geometry-function gap)
#   K3: ablation failure (≥ 3/5 heads not zeroed)

P1_fires = delta_P_B < -0.10
P2_fires = abs(delta_P_C) < abs(delta_P_B)
K1_fires = delta_P_C < -0.10
K2_fires = abs(delta_P_B) < 0.10 and abs(delta_P_C) < 0.10

print(f"\nPredictions:")
print(f"  P1 (ΔP_B < -0.10):           {'FIRES' if P1_fires else 'does not fire'}")
print(f"  P2 (|ΔP_C| < |ΔP_B|):        {'FIRES' if P2_fires else 'does not fire'}")
print(f"  K1 (ΔP_C < -0.10):            {'FIRES' if K1_fires else 'does not fire'}")
print(f"  K2 (both |ΔP| < 0.10):        {'FIRES' if K2_fires else 'does not fire'}")
print(f"  K3 (ablation failure):         {'FIRES' if K3_fires else 'does not fire'}")

# Determine verdict
if P1_fires and P2_fires and not K1_fires and not K2_fires:
    verdict = "confirmed"
    headline = ("H_positional_selective confirmed: structural-head W_K ablation degrades "
                f"positional retrieval (ΔP_B={delta_P_B:+.3f} nats) while sparing content "
                f"retrieval (ΔP_C={delta_P_C:+.3f} nats). Symmetric double dissociation.")
elif K2_fires:
    verdict = "inconclusive"
    headline = (f"K2 fires: structural-head W_K ablation affects neither task "
                f"(ΔP_B={delta_P_B:+.3f}, ΔP_C={delta_P_C:+.3f}). "
                "Geometry-function gap extends; W_K is not the causal mechanism for Task B.")
elif K1_fires and not P1_fires:
    verdict = "inconclusive"
    headline = (f"K1 fires, P1 does not: structural heads damage content retrieval "
                f"(ΔP_C={delta_P_C:+.3f}) but not positional retrieval (ΔP_B={delta_P_B:+.3f}). "
                "Inverted functional role — unexpected.")
elif K1_fires and P1_fires:
    verdict = "inconclusive"
    headline = (f"Both K1 and P1 fire: ablation degrades both tasks "
                f"(ΔP_B={delta_P_B:+.3f}, ΔP_C={delta_P_C:+.3f}). "
                "Structural heads contribute to both; no functional selectivity.")
elif P1_fires and not P2_fires:
    verdict = "partial"
    headline = (f"P1 fires but P2 does not: positional retrieval degrades "
                f"(ΔP_B={delta_P_B:+.3f}) but content retrieval effect "
                f"(ΔP_C={delta_P_C:+.3f}) is larger. Mixed — report as partial.")
else:
    verdict = "inconclusive"
    headline = (f"No clean prediction fires (ΔP_B={delta_P_B:+.3f}, ΔP_C={delta_P_C:+.3f}). "
                "Inconclusive — detailed item-level analysis required.")

print(f"\nVerdict: {verdict.upper()}")
print(f"Headline: {headline}")


# ── Save results ──────────────────────────────────────────────────────────────

results_out = {
    "experiment": "exp-156",
    "prereg_commit": PREREG_COMMIT,
    "protocol": "W_K=0 ablation of 5 structural (random-native) Δ-window heads; sham: matched-norm random W_K",
    "model": "gpt2",
    "structural_heads": [f"L{l}H{h}" for l, h in STRUCTURAL],
    "ablation_checks": abl_checks,
    "sham_info": sham_info,
    "K3_fires": K3_fires,
    "task_b": {
        "description": "Positional retrieval (control) — list-lookup battery from exp-143",
        "n_items": len(task_b_results),
        "delta_P_B": delta_P_B,
        "delta_P_B_sham": delta_P_B_sham,
        "n_improved": n_B_improved,
        "n_degraded": n_B_degraded,
        "items": task_b_results,
    },
    "task_c": {
        "description": "Content-specified long-range retrieval — items from exp-155",
        "n_items": len(task_c_results),
        "delta_P_C": delta_P_C,
        "delta_P_C_sham": delta_P_C_sham,
        "n_improved": n_C_improved,
        "n_degraded": n_C_degraded,
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
