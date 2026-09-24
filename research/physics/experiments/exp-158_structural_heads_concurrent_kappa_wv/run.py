"""
exp-158 — Structural (random-native) Δ-window heads: concurrent κ̃ amplification + W_V=0

Pre-registration: attention-geometry 09fd989 (committed and pushed before this script).
Analysis-only: GPT-2 small (cached). No new training.

Tests whether the gain-of-function from exp-141 (κ̃ amplification → +0.270 nats Task B)
survives value silencing. H_interference_only predicts ΔP_B ≈ exp-157's W_V=0 result
(+0.412 nats), because amplified attention routing × zero value = zero head output.

If confirmed: the gain-of-function mechanism from exp-141 is entirely mediated through
the value write (what the head writes when attention is concentrated), not an independent
routing mechanism.

Ariel — 2026-09-23, ~8:25 PM MDT, solo physics room.
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

PREREG_COMMIT = "09fd989"

# Structural (random-native) Δ-window heads in GPT-2 small
STRUCTURAL = [
    (2, 1), (3, 4), (5, 0), (7, 11), (10, 8),
]

D_MODEL = 768
D_HEAD  = 64
N_HEADS = 12

# κ̃ amplification parameters (same as exp-141)
GAMMA         = 2.0
N_INPUTS      = 50    # census inputs for positional field
SEQ_LEN       = 512   # same as exp-141/exp-112
SEED_CENSUS   = 42    # same as exp-112/exp-141

SHAM_SEED_BASE = 2026092358  # distinct from all prior experiments

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Reference results from prior experiments
WK_ABL_DELTA_P_B  = +0.485   # exp-156: W_K=0
WV_ABL_DELTA_P_B  = +0.412   # exp-157: W_V=0
KAPPA_AMP_DELTA_P_B = +0.270  # exp-141: κ̃ amplification alone

# ── Task B items (from exp-143/157; primary battery) ─────────────────────────

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
print(f"Device: {DEVICE}")

# ── κ̃ positional field utilities (from exp-141) ─────────────────────────────

def compute_positional_field_ln1(model: GPT2LMHeadModel,
                                  layer: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Positional field δ at ln_1(h) output for attention block `layer`.
    Same protocol as exp-141: hook ln_1 output, not raw residual stream.
    """
    model.eval()
    tok_ids = rng.integers(0, model.config.vocab_size, size=(N_INPUTS, SEQ_LEN))
    tokens  = torch.tensor(tok_ids, dtype=torch.long, device=DEVICE)
    captured = []
    def hook_fn(mod, inp, out):
        captured.append(out.detach().cpu().float().numpy())
    ln1_module = model.transformer.h[layer].ln_1
    handle = ln1_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(tokens)
    handle.remove()
    acts  = np.concatenate(captured, axis=0)  # (N_INPUTS, SEQ_LEN, D_MODEL)
    xbar  = acts.mean(axis=0)                  # (SEQ_LEN, D_MODEL)
    m     = xbar.mean(axis=0, keepdims=True)
    return xbar - m                            # centered positional field


def compute_kappa(W_K: np.ndarray, delta: np.ndarray) -> float:
    """κ̃(W_K) = isotropic-normalised positional capture (exp-137 formula)."""
    reads = delta @ W_K  # (SEQ_LEN, D_HEAD)
    cap = float((np.linalg.norm(reads, axis=1) ** 2).sum() /
                (np.linalg.norm(delta, axis=1) ** 2).sum())
    norm_factor = float((W_K ** 2).sum()) / D_MODEL
    return cap / norm_factor if norm_factor > 1e-14 else 0.0


def get_wk(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    return W[:, D_MODEL + head * D_HEAD : D_MODEL + (head + 1) * D_HEAD]


def set_wk(model: GPT2LMHeadModel, layer: int, head: int, W_K_new: np.ndarray):
    W = model.transformer.h[layer].attn.c_attn.weight.data
    with torch.no_grad():
        W[:, D_MODEL + head * D_HEAD : D_MODEL + (head + 1) * D_HEAD] = torch.tensor(
            W_K_new, dtype=W.dtype, device=W.device
        )


def get_wv(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = 2 * D_MODEL + head * D_HEAD
    return W[:, offset : offset + D_HEAD]


def set_wv(model: GPT2LMHeadModel, layer: int, head: int,
           W_V_new: np.ndarray, b_V_new: np.ndarray | None = None):
    W = model.transformer.h[layer].attn.c_attn.weight.data
    b = model.transformer.h[layer].attn.c_attn.bias.data
    offset = 2 * D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[:, offset : offset + D_HEAD] = torch.tensor(
            W_V_new, dtype=W.dtype, device=W.device
        )
        if b_V_new is not None:
            b[offset : offset + D_HEAD] = torch.tensor(
                b_V_new, dtype=b.dtype, device=b.device
            )

# ── Build combined model: κ̃ amplification + W_V=0 ────────────────────────────

print("\nBuilding combined model (κ̃ amp W_K + W_V=0 for 5 structural heads)...")
rng_census = np.random.default_rng(seed=SEED_CENSUS)
model_combined = copy.deepcopy(model_orig)
model_sham     = copy.deepcopy(model_orig)

kappa_report  = {}
wv_abl_checks = {}
wv_sham_info  = {}

for i, (layer, head) in enumerate(STRUCTURAL):
    # --- Positional field for this layer ---
    delta = compute_positional_field_ln1(model_orig, layer, rng_census)

    # Top-4 PC directions of positional field
    _, _, Vt = np.linalg.svd(delta, full_matrices=False)
    P_k = Vt[:4]  # (4, D_MODEL)

    # --- W_K: κ̃ amplification (same as exp-141) ---
    W_K = get_wk(model_orig, layer, head)
    W_K_proj   = P_k.T @ (P_k @ W_K)   # positional projection
    W_K_amp    = W_K + GAMMA * W_K_proj  # amplified

    kappa_before    = compute_kappa(W_K,     delta)
    kappa_after_amp = compute_kappa(W_K_amp, delta)

    set_wk(model_combined, layer, head, W_K_amp)

    # Sham W_K: orthogonal complement, matched Frobenius norm
    sham_rng  = np.random.default_rng(SHAM_SEED_BASE + i)
    rand_vecs = sham_rng.standard_normal((4, D_MODEL))
    for pk_row in P_k:
        rand_vecs -= (rand_vecs @ pk_row)[:, None] * pk_row
    P_perp, _ = np.linalg.qr(rand_vecs.T)
    P_perp = P_perp.T
    W_K_perp_proj = P_perp.T @ (P_perp @ W_K)
    delta_amp_norm  = np.linalg.norm(GAMMA * W_K_proj, "fro")
    delta_perp_norm = np.linalg.norm(W_K_perp_proj, "fro")
    gamma_sham = delta_amp_norm / delta_perp_norm if delta_perp_norm > 1e-14 else 0.0
    W_K_sham = W_K + gamma_sham * W_K_perp_proj
    kappa_after_sham = compute_kappa(W_K_sham, delta)

    set_wk(model_sham, layer, head, W_K_sham)

    kappa_report[f"L{layer}H{head}"] = {
        "kappa_before":     float(kappa_before),
        "kappa_after_amp":  float(kappa_after_amp),
        "kappa_after_sham": float(kappa_after_sham),
        "gamma":            GAMMA,
        "gamma_sham":       float(gamma_sham),
        "amp_ratio":        float(kappa_after_amp / kappa_before) if kappa_before > 1e-14 else 0.0,
    }
    print(f"  L{layer}H{head}: κ̃ {kappa_before:.3f} → amp {kappa_after_amp:.3f} "
          f"(sham {kappa_after_sham:.3f}, ratio {kappa_after_amp / kappa_before:.2f}×)")

    # --- W_V = 0 + bias_V = 0 (applied to BOTH combined and sham) ---
    W_V_orig = get_wv(model_orig, layer, head)
    norm_orig = float(np.linalg.norm(W_V_orig, "fro"))

    set_wv(model_combined, layer, head, np.zeros_like(W_V_orig), np.zeros(D_HEAD))
    set_wv(model_sham,     layer, head, np.zeros_like(W_V_orig), np.zeros(D_HEAD))

    norm_after = float(np.linalg.norm(
        get_wv(model_combined, layer, head), "fro"
    ))
    wv_abl_checks[f"L{layer}H{head}"] = {
        "norm_before": norm_orig,
        "norm_after":  norm_after,
    }
    wv_sham_info[f"L{layer}H{head}"] = {"orig_norm": norm_orig}

print("\nW_V zeroing K1 check (threshold: ‖W_V‖_F < 0.01):")
k1_fail_count = 0
for key, val in wv_abl_checks.items():
    ok = val["norm_after"] < 0.01
    if not ok:
        k1_fail_count += 1
    print(f"  {key}: {val['norm_before']:.3f} → {val['norm_after']:.6f} "
          f"{'✓' if ok else '✗ K1-FAIL'}")
K1_fires = k1_fail_count >= 3

print("\nκ̃ amplification K2 check (threshold: ratio < 1.5):")
k2_fail_count = 0
for key, val in kappa_report.items():
    ok = val["amp_ratio"] >= 1.5
    if not ok:
        k2_fail_count += 1
    print(f"  {key}: ratio = {val['amp_ratio']:.2f}× {'✓' if ok else '✗ K2-FAIL'}")
K2_fires = k2_fail_count >= 1  # any failure is informative

# ── Verify zero head value outputs ───────────────────────────────────────────

print("\nVerifying head value outputs are zero in combined model...")
_test_input = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
hooks = []
value_outputs = {}

def make_hook(layer, head):
    def hook(module, input, output):
        v_all  = output[0, :, 2 * D_MODEL : 3 * D_MODEL].detach().cpu()
        v_head = v_all[:, head * D_HEAD : (head + 1) * D_HEAD]
        value_outputs[f"L{layer}H{head}"] = float(v_head.abs().max().item())
    return hook

for layer, head in STRUCTURAL:
    h = model_combined.transformer.h[layer].attn.c_attn.register_forward_hook(
        make_hook(layer, head)
    )
    hooks.append(h)

with torch.no_grad():
    model_combined(_test_input)
for h in hooks:
    h.remove()

all_zero = True
for key, max_val in value_outputs.items():
    ok = max_val < 1e-6
    if not ok:
        all_zero = False
    print(f"  {key}: max |v| = {max_val:.2e}  {'✓ zeroed' if ok else '✗ NOT ZERO'}")

# ── Scoring function ──────────────────────────────────────────────────────────

def score_target(model, prompt: str, target: str) -> float:
    """log P(target | prompt) averaged per target token. Same as exp-157."""
    full_text  = prompt + target
    enc_full   = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    enc_prompt = tokenizer(prompt,    return_tensors="pt").input_ids.to(DEVICE)
    n_prompt   = enc_prompt.shape[1]
    with torch.no_grad():
        out = model(enc_full)
    log_probs = torch.nn.functional.log_softmax(out.logits, dim=-1)
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
with open(EXP155_DIR / "task_c_items.json") as f:
    task_c_items_raw = json.load(f)["items"]
print(f"\nLoaded {len(task_c_items_raw)} Task C items from exp-155.")

def build_task_c_prompt(item: dict) -> str:
    return f"{item['setup']} {item['filler']} {item['cue']}"

# ── Run scoring ───────────────────────────────────────────────────────────────

print("\n── Task B (positional retrieval; primary) ───────────────────────────")
task_b_results = []
for i, item in enumerate(TASK_B_ITEMS):
    lp_orig  = score_target(model_orig,     item["prompt"], item["target"])
    lp_comb  = score_target(model_combined, item["prompt"], item["target"])
    lp_sham  = score_target(model_sham,     item["prompt"], item["target"])
    d_comb = lp_comb - lp_orig
    d_sham = lp_sham - lp_orig
    print(f"  B-{i+1:02d}: orig={lp_orig:+.3f}  comb={lp_comb:+.3f}  sham={lp_sham:+.3f}  "
          f"Δcomb={d_comb:+.3f}  Δsham={d_sham:+.3f}")
    task_b_results.append({
        "id": f"B-{i+1:02d}",
        "prompt_short": item["prompt"][:55],
        "target": item["target"],
        "logp_orig": lp_orig,
        "logp_comb": lp_comb,
        "logp_sham": lp_sham,
        "delta_comb": d_comb,
        "delta_sham": d_sham,
    })

print("\n── Task C (content retrieval; secondary) ────────────────────────────")
task_c_results = []
for item in task_c_items_raw:
    prompt = build_task_c_prompt(item)
    target = item["target_token"]
    lp_orig = score_target(model_orig,     prompt, target)
    lp_comb = score_target(model_combined, prompt, target)
    lp_sham = score_target(model_sham,     prompt, target)
    d_comb = lp_comb - lp_orig
    d_sham = lp_sham - lp_orig
    n_tok  = len(tokenizer(prompt)["input_ids"])
    print(f"  {item['id']}: orig={lp_orig:+.3f}  comb={lp_comb:+.3f}  sham={lp_sham:+.3f}  "
          f"Δcomb={d_comb:+.3f}  [{n_tok} tok]")
    task_c_results.append({
        "id": item["id"],
        "entity": item["entity"],
        "property": item["property"],
        "n_prompt_tokens": n_tok,
        "logp_orig": lp_orig,
        "logp_comb": lp_comb,
        "logp_sham": lp_sham,
        "delta_comb": d_comb,
        "delta_sham": d_sham,
    })

# ── Verdict computation ───────────────────────────────────────────────────────

def median_delta(results, key):
    return float(np.median([r[key] for r in results]))

def n_count(results, key, direction="positive"):
    if direction == "positive":
        return sum(1 for r in results if r[key] > 0)
    return sum(1 for r in results if r[key] < 0)

delta_P_B      = median_delta(task_b_results, "delta_comb")
delta_P_B_sham = median_delta(task_b_results, "delta_sham")
delta_P_C      = median_delta(task_c_results, "delta_comb")
delta_P_C_sham = median_delta(task_c_results, "delta_sham")

n_B_improved = n_count(task_b_results, "delta_comb", "positive")
n_B_degraded = n_count(task_b_results, "delta_comb", "negative")
n_C_improved = n_count(task_c_results, "delta_comb", "positive")
n_C_degraded = n_count(task_c_results, "delta_comb", "negative")

# Registered predictions
# H_interference_only: |ΔP_B - WV_ABL_DELTA_P_B| ≤ 0.15
# H_additional_mechanism: ΔP_B > WV_ABL_DELTA_P_B + 0.15
# H_null: |ΔP_B| < 0.10
# H_degradation: ΔP_B < -0.10
# K1: ≥ 3/5 heads W_V not zeroed
# K2: any head kappa_ratio < 1.5
# K3: sham > comb + 0.20

H_interference_only   = abs(delta_P_B - WV_ABL_DELTA_P_B) <= 0.15
H_additional_mechanism = delta_P_B > WV_ABL_DELTA_P_B + 0.15
H_null                = abs(delta_P_B) < 0.10
H_degradation         = delta_P_B < -0.10
K3_fires              = delta_P_B_sham > delta_P_B + 0.20

# Summary
print(f"\n── Summary ──────────────────────────────────────────────────────────")
print(f"Task B: ΔP_B_comb     = {delta_P_B:+.4f} nats  (sham: {delta_P_B_sham:+.4f})")
print(f"        exp-157 W_V=0 = +0.412 nats  (reference)")
print(f"        exp-141 κ̃_amp = +0.270 nats  (reference; different battery)")
print(f"        {n_B_improved}/20 improved, {n_B_degraded}/20 degraded")
print(f"Task C: ΔP_C_comb     = {delta_P_C:+.4f} nats  (sham: {delta_P_C_sham:+.4f})")
print(f"        {n_C_improved}/20 improved, {n_C_degraded}/20 degraded")
print(f"\nHypotheses:")
print(f"  H_interference_only  (|ΔP_B − 0.412| ≤ 0.15): {'YES' if H_interference_only else 'NO'}")
print(f"  H_additional_mechanism (ΔP_B > 0.562):          {'YES' if H_additional_mechanism else 'NO'}")
print(f"  H_null (|ΔP_B| < 0.10):                         {'YES' if H_null else 'NO'}")
print(f"  H_degradation (ΔP_B < -0.10):                   {'YES' if H_degradation else 'NO'}")
print(f"Kill conditions:")
print(f"  K1 (W_V not zeroed, ≥3/5): {'FIRES' if K1_fires else 'does not fire'}")
print(f"  K2 (κ̃ ratio < 1.5, any):   {'FIRES' if K2_fires else 'does not fire'}")
print(f"  K3 (sham > comb + 0.20):    {'FIRES' if K3_fires else 'does not fire'}")

# Verdict
if K1_fires:
    verdict = "aborted"
    headline = "K1: W_V not zeroed — manipulation failure."
elif H_interference_only and not H_degradation and not K2_fires:
    verdict = "confirmed"
    headline = (
        f"H_interference_only confirmed: κ̃ amplification adds nothing beyond W_V=0 "
        f"(ΔP_B_comb={delta_P_B:+.3f} vs exp-157 W_V=0=+0.412 nats, |diff|={abs(delta_P_B - WV_ABL_DELTA_P_B):.3f}). "
        f"Gain-of-function from exp-141 is mediated entirely through the value write, "
        f"not routing-independent mechanism."
    )
elif H_additional_mechanism and not K1_fires and not K2_fires:
    verdict = "confirmed"
    headline = (
        f"H_additional_mechanism: κ̃ amplification improves Task B beyond W_V=0 baseline "
        f"(ΔP_B_comb={delta_P_B:+.3f} vs exp-157 +0.412 nats). Independent mechanism exists."
    )
elif H_null and not K1_fires:
    verdict = "inconclusive"
    headline = (
        f"H_null: combined manipulation neutral (ΔP_B={delta_P_B:+.3f}). "
        f"Neither prediction cleanly confirmed."
    )
elif H_degradation and not K1_fires:
    verdict = "inconclusive"
    headline = (
        f"H_degradation: combined manipulation degrades Task B (ΔP_B={delta_P_B:+.3f}). "
        f"Unexpected interaction. Investigate interaction between κ̃ and V=0."
    )
else:
    verdict = "inconclusive"
    headline = (
        f"No registered hypothesis cleanly confirmed (ΔP_B={delta_P_B:+.3f}). "
        f"Item-level analysis required."
    )

print(f"\nVerdict: {verdict.upper()}")
print(f"Headline: {headline}")

# ── Save results ──────────────────────────────────────────────────────────────

results_out = {
    "experiment": "exp-158",
    "prereg_commit": PREREG_COMMIT,
    "protocol": (
        "Concurrent κ̃ amplification (γ=+2.0, ln_1 positional field, top-4 PC) + "
        "W_V=0 + bias_V=0, 5 structural (random-native) Δ-window heads; "
        "sham: orthogonal W_K (matched norm) + W_V=0"
    ),
    "model": "gpt2",
    "gamma": GAMMA,
    "structural_heads": [f"L{l}H{h}" for l, h in STRUCTURAL],
    "kappa_report": kappa_report,
    "wv_ablation_checks": wv_abl_checks,
    "value_outputs_zeroed": all_zero,
    "K1_fires": K1_fires,
    "K2_fires": K2_fires,
    "K3_fires": K3_fires,
    "references": {
        "exp_141_kappa_amp_delta_P_B": KAPPA_AMP_DELTA_P_B,
        "exp_156_wk_abl_delta_P_B":   WK_ABL_DELTA_P_B,
        "exp_157_wv_abl_delta_P_B":   WV_ABL_DELTA_P_B,
    },
    "task_b": {
        "description": "Positional retrieval — 20-item battery from exp-143/157",
        "n_items": len(task_b_results),
        "delta_P_B":      delta_P_B,
        "delta_P_B_sham": delta_P_B_sham,
        "n_improved": n_B_improved,
        "n_degraded": n_B_degraded,
        "items": task_b_results,
    },
    "task_c": {
        "description": "Content retrieval — 20-item battery from exp-155",
        "n_items": len(task_c_results),
        "delta_P_C":      delta_P_C,
        "delta_P_C_sham": delta_P_C_sham,
        "n_improved": n_C_improved,
        "n_degraded": n_C_degraded,
        "items": task_c_results,
    },
    "hypotheses": {
        "H_interference_only":    H_interference_only,
        "H_additional_mechanism": H_additional_mechanism,
        "H_null":                 H_null,
        "H_degradation":          H_degradation,
    },
    "verdict": verdict,
    "headline": headline,
}

out_path = HERE / "results.json"
with open(out_path, "w") as f:
    json.dump(results_out, f, indent=2)
print(f"\nResults saved to {out_path}")
