"""
exp-157 — Structural (random-native) Δ-window heads: W_V ablation Task B test

Pre-registration: attention-geometry b6adea5 (committed before this script).
Analysis-only: GPT-2 small (cached). No new training.

Tests whether the structural heads' value pathway carries positional retrieval function.
exp-156 showed W_K ablation ≈ sham (both +0.485/+0.490 nats) — interference finding.
This experiment tests W_V = 0: does silencing the head entirely degrade Task B?

Two hypotheses:
  H_value_pathway: W_V ablation degrades Task B (ΔP_B < -0.10 nats)
  H_attention_shape: W_V ablation improves or is neutral (matching W_K=0 result)

Ariel — 2026-09-23, ~4:25 PM MDT, solo physics room.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "b6adea5"

# Structural (random-native) Δ-window heads in GPT-2 small
STRUCTURAL = [
    (2, 1), (3, 4), (5, 0), (7, 11), (10, 8),
]

D_MODEL = 768
D_HEAD  = 64
N_HEADS = 12

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SHAM_SEED = 2026092357  # distinct from all prior experiments

# ── Task B items (from exp-143; same battery used in all causal experiments) ──

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

# ── Model loading ─────────────────────────────────────────────────────────────

print("Loading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_orig = GPT2LMHeadModel.from_pretrained("gpt2")
model_orig.to(DEVICE)
model_orig.eval()

# ── W_V helpers ───────────────────────────────────────────────────────────────

def get_wv(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    """W_V for (layer, head). Returns (D_MODEL, D_HEAD) numpy array."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = 2 * D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


def get_bv(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    """Bias_V for (layer, head). Returns (D_HEAD,) numpy array."""
    b = model.transformer.h[layer].attn.c_attn.bias.detach().cpu().double().numpy()
    offset = 2 * D_MODEL + head * D_HEAD
    return b[offset: offset + D_HEAD]


def set_wv(model: GPT2LMHeadModel, layer: int, head: int,
           W_V_new: np.ndarray, b_V_new: np.ndarray | None = None):
    """Write W_V (and optionally bias_V) back for (layer, head)."""
    W = model.transformer.h[layer].attn.c_attn.weight.data
    b = model.transformer.h[layer].attn.c_attn.bias.data
    offset = 2 * D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[:, offset: offset + D_HEAD] = torch.tensor(
            W_V_new, dtype=W.dtype, device=W.device
        )
        if b_V_new is not None:
            b[offset: offset + D_HEAD] = torch.tensor(
                b_V_new, dtype=b.dtype, device=b.device
            )


def verify_wv_norm(model: GPT2LMHeadModel, layer: int, head: int) -> float:
    return float(np.linalg.norm(get_wv(model, layer, head), 'fro'))


# ── Build ablated and sham models ─────────────────────────────────────────────

print("\nBuilding ablated model (W_V=0, bias_V=0 for 5 structural heads)...")
model_abl = copy.deepcopy(model_orig)
rng = np.random.default_rng(seed=SHAM_SEED)

abl_checks = {}
sham_info   = {}

for layer, head in STRUCTURAL:
    W_orig = get_wv(model_abl, layer, head)
    norm_orig = float(np.linalg.norm(W_orig, 'fro'))
    set_wv(model_abl, layer, head,
           np.zeros_like(W_orig), np.zeros(D_HEAD))
    norm_after = verify_wv_norm(model_abl, layer, head)
    abl_checks[f"L{layer}H{head}"] = {
        "norm_before": norm_orig,
        "norm_after":  norm_after,
    }
    sham_info[f"L{layer}H{head}"] = {"orig_norm": norm_orig}

print("Ablation K2 check (threshold: ‖W_V‖_F < 0.01):")
k2_fail_count = 0
for key, val in abl_checks.items():
    ok = val["norm_after"] < 0.01
    if not ok:
        k2_fail_count += 1
        print(f"  K2 FAIL: {key} norm_after={val['norm_after']:.4f}")
    else:
        print(f"  {key}: {val['norm_before']:.3f} → {val['norm_after']:.6f} ✓")

K2_fires = k2_fail_count >= 3   # ≥ 3/5 heads failed

print("\nBuilding sham model (random W_V matched norm, bias_V=0 for 5 structural heads)...")
model_sham = copy.deepcopy(model_orig)
for layer, head in STRUCTURAL:
    W_orig = get_wv(model_sham, layer, head)
    norm_orig = float(np.linalg.norm(W_orig, 'fro'))
    W_rand = rng.standard_normal(W_orig.shape)
    W_rand = W_rand * norm_orig / (np.linalg.norm(W_rand, 'fro') + 1e-14)
    set_wv(model_sham, layer, head, W_rand, np.zeros(D_HEAD))
    sham_norm = verify_wv_norm(model_sham, layer, head)
    sham_info[f"L{layer}H{head}"]["sham_norm"] = sham_norm
    print(f"  L{layer}H{head}: orig={norm_orig:.3f} → sham={sham_norm:.3f}")

# ── Verify: forward pass shows head writes zero (ablated model) ───────────────

print("\nVerifying head value outputs are zero in ablated model...")
_test_input = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)

hooks = []
value_outputs = {}

def make_hook(layer, head):
    def hook(module, input, output):
        # output of c_attn: (batch, seq, 3*n_embd)
        # value portion: output[..., 2*D_MODEL : 3*D_MODEL]
        v_all = output[0, :, 2 * D_MODEL : 3 * D_MODEL].detach().cpu()
        v_head = v_all[:, head * D_HEAD : (head + 1) * D_HEAD]
        key = f"L{layer}H{head}"
        value_outputs[key] = float(v_head.abs().max().item())
    return hook

for layer, head in STRUCTURAL:
    h = model_abl.transformer.h[layer].attn.c_attn.register_forward_hook(
        make_hook(layer, head)
    )
    hooks.append(h)

with torch.no_grad():
    model_abl(_test_input)

for h in hooks:
    h.remove()

all_zero = True
for key, max_val in value_outputs.items():
    ok = max_val < 1e-6
    if not ok:
        all_zero = False
    print(f"  {key}: max |v| = {max_val:.2e}  {'✓ zeroed' if ok else '✗ NOT ZERO'}")

if not all_zero:
    print("WARNING: some head value outputs are not zero — ablation may be incomplete")

# ── Scoring function ──────────────────────────────────────────────────────────

def score_target(model, prompt: str, target: str) -> float:
    """
    Returns log P(target | prompt) averaged per target token.
    Identical to exp-155/156 scoring protocol.
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

with open(task_c_path) as f:
    task_c_data = json.load(f)
task_c_items_raw = task_c_data["items"]
print(f"\nLoaded {len(task_c_items_raw)} Task C items from exp-155.")


def build_task_c_prompt(item: dict) -> str:
    return f"{item['setup']} {item['filler']} {item['cue']}"


# ── Run scoring ───────────────────────────────────────────────────────────────

print("\n── Task B (positional retrieval; primary) ───────────────────────────")
task_b_results = []
for i, item in enumerate(TASK_B_ITEMS):
    lp_orig  = score_target(model_orig,  item["prompt"], item["target"])
    lp_abl   = score_target(model_abl,   item["prompt"], item["target"])
    lp_sham  = score_target(model_sham,  item["prompt"], item["target"])
    delta_abl  = lp_abl  - lp_orig
    delta_sham = lp_sham - lp_orig
    print(f"  B-{i+1:02d}: orig={lp_orig:+.3f}  abl={lp_abl:+.3f}  sham={lp_sham:+.3f}  "
          f"Δabl={delta_abl:+.3f}  Δsham={delta_sham:+.3f}")
    task_b_results.append({
        "id": f"B-{i+1:02d}",
        "prompt_short": item["prompt"][:55],
        "target": item["target"],
        "logp_orig":  lp_orig,
        "logp_abl":   lp_abl,
        "logp_sham":  lp_sham,
        "delta_abl":  delta_abl,
        "delta_sham": delta_sham,
    })

print("\n── Task C (content retrieval; secondary) ────────────────────────────")
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
    print(f"  {item['id']}: orig={lp_orig:+.3f}  abl={lp_abl:+.3f}  sham={lp_sham:+.3f}  "
          f"Δabl={delta_abl:+.3f}  [{n_tokens} tok]")
    task_c_results.append({
        "id": item["id"],
        "entity": item["entity"],
        "property": item["property"],
        "n_prompt_tokens": n_tokens,
        "logp_orig":  lp_orig,
        "logp_abl":   lp_abl,
        "logp_sham":  lp_sham,
        "delta_abl":  delta_abl,
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

# Reference: exp-156 W_K ablation result
WK_ABL_DELTA_P_B = +0.485   # from exp-156

print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"Task B: median ΔP_B_abl  = {delta_P_B:+.4f} nats  (sham: {delta_P_B_sham:+.4f})")
print(f"        exp-156 W_K=0 was: +0.485 nats (reference)")
print(f"        {n_B_improved}/20 improved, {n_B_degraded}/20 degraded")
print(f"Task C: median ΔP_C_abl  = {delta_P_C:+.4f} nats  (sham: {delta_P_C_sham:+.4f})")
print(f"        exp-156 W_K=0 was: -0.031 nats (reference)")
print(f"        {n_C_improved}/20 improved, {n_C_degraded}/20 degraded")

# Registered predictions (from prereg.md)
#   P1: ΔP_B < -0.10 nats AND substantially worse than W_K ablation (H_value_pathway)
#   P2: ΔP_B > 0 or |ΔP_B| ≤ 0.10 AND close to W_K=0 result (H_attention_shape)
#   P3: |ΔP_C| < 0.10 nats (Task C neutral)
#   K1: ΔP_C < -0.10 nats
#   K2: ablation failure (≥ 3/5 heads not zeroed)
#   K3: sham substantially outperforms ablation in degradation direction

P1_fires = delta_P_B < -0.10
P2_fires = delta_P_B > 0.0 or abs(delta_P_B) <= 0.10
P3_fires = abs(delta_P_C) < 0.10
K1_fires = delta_P_C < -0.10
# K2 already computed above
K3_fires = (delta_P_B_sham > delta_P_B + 0.20)

# Also: is W_V ablation close to W_K ablation?
wv_close_to_wk = abs(delta_P_B - WK_ABL_DELTA_P_B) < 0.15

print(f"\nPredictions:")
print(f"  P1 (ΔP_B < -0.10, H_value_pathway):    {'FIRES' if P1_fires else 'does not fire'}")
print(f"  P2 (ΔP_B ≥ 0 or neutral, H_attention): {'FIRES' if P2_fires else 'does not fire'}")
print(f"  P3 (|ΔP_C| < 0.10, Task C neutral):    {'FIRES' if P3_fires else 'does not fire'}")
print(f"  K1 (ΔP_C < -0.10, C substantially degraded): {'FIRES' if K1_fires else 'does not fire'}")
print(f"  K2 (ablation failure):                  {'FIRES' if K2_fires else 'does not fire'}")
print(f"  K3 (sham > ablation + 0.20):            {'FIRES' if K3_fires else 'does not fire'}")
print(f"  W_V ≈ W_K=0 (|ΔP_B - 0.485| < 0.15):  {'YES' if wv_close_to_wk else 'NO'}")

# Determine verdict
if P1_fires and not K1_fires and not K2_fires:
    verdict = "confirmed"
    headline = (
        f"H_value_pathway confirmed: W_V ablation of structural heads degrades Task B "
        f"(ΔP_B={delta_P_B:+.3f} nats, {n_B_improved}/20 improved) while W_K=0 improved it "
        f"(+0.485 nats). Value payload carries positional retrieval function."
    )
elif not P1_fires and not K1_fires and not K2_fires and wv_close_to_wk:
    verdict = "confirmed"
    headline = (
        f"H_attention_shape confirmed: W_V ablation matches W_K ablation "
        f"(ΔP_B={delta_P_B:+.3f} vs exp-156 +0.485 nats). Value write was also interfering. "
        f"Gain-of-function (exp-141) operates through attention distribution, not value payload."
    )
elif not P1_fires and not K1_fires and not K2_fires:
    verdict = "inconclusive"
    headline = (
        f"Intermediate result: W_V ablation (ΔP_B={delta_P_B:+.3f} nats) lies between "
        f"original and W_K=0 (+0.485 nats). Routing interferes more than value write. "
        f"Neither H cleanly confirmed."
    )
elif K2_fires:
    verdict = "aborted"
    headline = "K2: ablation failure — head value outputs not zeroed."
elif K1_fires and P1_fires:
    verdict = "inconclusive"
    headline = (
        f"Both tasks degrade (ΔP_B={delta_P_B:+.3f}, ΔP_C={delta_P_C:+.3f}). "
        f"Structural value writes contribute to both — unexpected; inconsistent with exp-156."
    )
else:
    verdict = "inconclusive"
    headline = (
        f"No clean prediction fires (ΔP_B={delta_P_B:+.3f}, ΔP_C={delta_P_C:+.3f}). "
        f"Detailed item-level analysis required."
    )

print(f"\nVerdict: {verdict.upper()}")
print(f"Headline: {headline}")


# ── Save results ──────────────────────────────────────────────────────────────

results_out = {
    "experiment":  "exp-157",
    "prereg_commit": PREREG_COMMIT,
    "protocol": ("W_V=0 ablation (+ bias_V=0) of 5 structural (random-native) Δ-window heads; "
                 "sham: matched-norm random W_V, bias_V=0"),
    "model": "gpt2",
    "structural_heads": [f"L{l}H{h}" for l, h in STRUCTURAL],
    "ablation_checks": abl_checks,
    "sham_info": sham_info,
    "K2_fires": K2_fires,
    "value_outputs_zeroed": all_zero,
    "reference_wk_ablation_delta_P_B": WK_ABL_DELTA_P_B,
    "task_b": {
        "description": "Positional retrieval — list-lookup battery from exp-143",
        "n_items": len(task_b_results),
        "delta_P_B":      delta_P_B,
        "delta_P_B_sham": delta_P_B_sham,
        "n_improved": n_B_improved,
        "n_degraded": n_B_degraded,
        "items": task_b_results,
    },
    "task_c": {
        "description": "Content-specified long-range retrieval — items from exp-155",
        "n_items": len(task_c_results),
        "delta_P_C":      delta_P_C,
        "delta_P_C_sham": delta_P_C_sham,
        "n_improved": n_C_improved,
        "n_degraded": n_C_degraded,
        "items": task_c_results,
    },
    "predictions": {
        "P1_fires": P1_fires,
        "P2_fires": P2_fires,
        "P3_fires": P3_fires,
        "K1_fires": K1_fires,
        "K2_fires": K2_fires,
        "K3_fires": K3_fires,
        "wv_close_to_wk_ablation": wv_close_to_wk,
    },
    "verdict": verdict,
    "headline": headline,
}

out_path = HERE / "results.json"
with open(out_path, "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\nResults saved to {out_path}")
