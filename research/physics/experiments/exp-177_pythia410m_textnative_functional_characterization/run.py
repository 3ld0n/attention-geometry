"""
exp-177 — Pythia-410m text-native Δ-window heads: individual functional characterization

Pre-registration: attention-geometry 9ebf6ef (committed and pushed before this script).
Cross-family replication of exp-159 (GPT-2 small, 2026-09-24).

Three instruments:
  1. Individual-head Task C contribution (per-head W_K ablation)
  2. Attention distribution on Task C items (attention-on-target analysis)
  3. Recency / lost-in-the-middle probe (position sensitivity in baseline model)

Population: 16 text-native Δ-window heads in Pythia-410m from exp-118.
Multi-token target protocol: sum of token log-probs for all target tokens.

Ariel — 2026-11-13, ~12:30 AM MST, solo physics room.
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

PREREG_COMMIT = "9ebf6ef"

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

D_MODEL = 1024
D_HEAD  = 64
N_HEADS = 16
N_LAYERS = 24

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Task C items (from exp-155) ───────────────────────────────────────────────

TASK_C_PATH = Path(__file__).parent.parent / "exp-155_textnative_content_retrieval" / "task_c_items.json"
with open(TASK_C_PATH) as f:
    task_c_data = json.load(f)
TASK_C_ITEMS = task_c_data["items"]
assert len(TASK_C_ITEMS) == 20, f"Expected 20 items, got {len(TASK_C_ITEMS)}"

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

# ── W_K helpers ───────────────────────────────────────────────────────────────
# In GPT-NeoX / Pythia:
#   attention.query_key_value.weight has shape [3*d_model, d_model]
#   Layout: rows [0:d_model] = Q, [d_model:2*d_model] = K, [2*d_model:3*d_model] = V
#   For head h: W_K is rows [d_model + h*d_head : d_model + (h+1)*d_head, :]

def _qkv_weight(model, layer):
    return model.gpt_neox.layers[layer].attention.query_key_value.weight


def get_wk(model, layer, head):
    W = _qkv_weight(model, layer).detach().cpu().double().numpy()
    start = D_MODEL + head * D_HEAD
    return W[start: start + D_HEAD, :].copy()


def set_wk_zero(model, layer, head):
    W = _qkv_weight(model, layer).data
    start = D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[start: start + D_HEAD, :] = 0.0


def set_wk(model, layer, head, W_K_new):
    """Set W_K for head to a new value (shape [d_head, d_model])."""
    W = _qkv_weight(model, layer).data
    start = D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[start: start + D_HEAD, :] = torch.tensor(
            W_K_new, dtype=W.dtype, device=W.device
        )


def wk_frobenius(model, layer, head):
    return float(np.linalg.norm(get_wk(model, layer, head)))


# ── Log-prob helpers ──────────────────────────────────────────────────────────

def build_prompt(item):
    return item["setup"] + " " + item["filler"] + " " + item["cue"]


def target_logprob(model, prompt: str, target_token: str) -> float:
    """Return log-prob of target_token (may be multi-token) following prompt.

    For multi-token targets, returns the sum of per-token log-probs
    (the log-probability of the full target sequence given prompt).
    """
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    if len(target_ids) == 1:
        # Single-token: standard protocol matching exp-159
        with torch.no_grad():
            out = model(prompt_ids)
        logits = out.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        return float(log_probs[target_ids[0]].cpu())
    else:
        # Multi-token: sum log P(tok_i | context + tok_1..tok_{i-1})
        total_lp = 0.0
        current_ids = prompt_ids
        for tok_id in target_ids:
            with torch.no_grad():
                out = model(current_ids)
            logits = out.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            total_lp += float(log_probs[tok_id].cpu())
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[tok_id]], device=DEVICE)
            ], dim=-1)
        return total_lp


def is_single_token(target_token: str) -> bool:
    ids = tokenizer.encode(target_token, add_special_tokens=False)
    return len(ids) == 1


# ── Build prompts ─────────────────────────────────────────────────────────────

PROMPTS = [build_prompt(it) for it in TASK_C_ITEMS]
TARGETS = [it["target_token"] for it in TASK_C_ITEMS]
SINGLE_TOKEN_MASK = [is_single_token(t) for t in TARGETS]
n_single = sum(SINGLE_TOKEN_MASK)
print(f"Task C: {len(TASK_C_ITEMS)} items, {n_single} single-token, "
      f"{len(TASK_C_ITEMS)-n_single} multi-token (sum-log-prob protocol)")

# ── Baseline ─────────────────────────────────────────────────────────────────

print("\nComputing baseline log-probs...")
t0 = time.time()
baseline_lps = []
for prompt, tgt in zip(PROMPTS, TARGETS):
    baseline_lps.append(target_logprob(model_orig, prompt, tgt))
baseline_mean = np.mean(baseline_lps)
print(f"  Baseline mean log-prob: {baseline_mean:.3f}  ({time.time()-t0:.1f}s)")

# ── Instrument 1 — Individual-head W_K ablation ───────────────────────────────

print("\n--- Instrument 1: Individual-head W_K ablation ---")
results_inst1 = []

for idx, (layer, head) in enumerate(TEXT_NATIVE):
    print(f"  [{idx+1:02d}/16] L{layer}H{head}:", end=" ", flush=True)

    # Save original W_K
    orig_wk = get_wk(model_orig, layer, head)
    orig_norm = float(np.linalg.norm(orig_wk))

    # --- Ablation (W_K = 0) ---
    model_abl = copy.deepcopy(model_orig)
    set_wk_zero(model_abl, layer, head)
    model_abl.to(DEVICE)
    model_abl.eval()

    abl_lps = []
    for prompt, tgt in zip(PROMPTS, TARGETS):
        abl_lps.append(target_logprob(model_abl, prompt, tgt))
    del model_abl
    if DEVICE == "mps":
        torch.mps.empty_cache()

    delta_abl = np.mean(abl_lps) - baseline_mean
    n_improved_abl = sum(a > b for a, b in zip(abl_lps, baseline_lps))
    n_degraded_abl = sum(a < b - 0.05 for a, b in zip(abl_lps, baseline_lps))

    # --- Sham (matched-norm random W_K) ---
    rng = np.random.default_rng(seed=42 + layer * 100 + head)
    sham_wk = rng.standard_normal(orig_wk.shape)
    sham_wk = sham_wk * (orig_norm / (np.linalg.norm(sham_wk) + 1e-12))

    model_sham = copy.deepcopy(model_orig)
    set_wk(model_sham, layer, head, sham_wk)
    model_sham.to(DEVICE)
    model_sham.eval()

    sham_lps = []
    for prompt, tgt in zip(PROMPTS, TARGETS):
        sham_lps.append(target_logprob(model_sham, prompt, tgt))
    del model_sham
    if DEVICE == "mps":
        torch.mps.empty_cache()

    delta_sham = np.mean(sham_lps) - baseline_mean
    n_improved_sham = sum(s > b for s, b in zip(sham_lps, baseline_lps))

    result = {
        "layer": layer,
        "head": head,
        "orig_wk_norm": orig_norm,
        "delta_abl": float(delta_abl),
        "delta_sham": float(delta_sham),
        "n_degraded_abl": n_degraded_abl,
        "n_improved_abl": n_improved_abl,
        "n_improved_sham": n_improved_sham,
        "item_abl_lps": [float(x) for x in abl_lps],
        "item_sham_lps": [float(x) for x in sham_lps],
    }
    results_inst1.append(result)
    print(f"ΔP_abl={delta_abl:+.3f} ΔP_sham={delta_sham:+.3f} degraded={n_degraded_abl}/20")

# CV of |ΔP_C_i|
delta_abls = np.array([r["delta_abl"] for r in results_inst1])
abs_deltas = np.abs(delta_abls)
cv_abls = float(np.std(abs_deltas) / np.mean(abs_deltas)) if np.mean(abs_deltas) > 0 else 0.0
n_degraded_H_positive = sum(r["n_degraded_abl"] >= 10 for r in results_inst1)  # per-head: ≥10/20 items
n_heads_degrade_task_c = sum(r["delta_abl"] < -0.05 for r in results_inst1)

print(f"\n  CV of |ΔP_C_i| = {cv_abls:.3f}  (H_graded threshold: 0.5)")
print(f"  Heads with mean ΔP_C < -0.05: {n_heads_degrade_task_c}/16  (H_positive threshold: 10)")

# ── Instrument 2 — Attention-on-target analysis ───────────────────────────────

print("\n--- Instrument 2: Attention-on-target analysis ---")

# Run model with output_attentions
def get_all_attention_weights(model, prompt):
    """Return attention weights for all layers and heads. Shape: [n_layers][n_heads, seq_len, seq_len]"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    # out.attentions is a tuple of [n_layers] tensors, each [1, n_heads, seq_len, seq_len]
    return out.attentions, input_ids[0]


def find_target_token_pos(input_ids, target_token):
    """Find the position of the first token of target_token in input_ids."""
    tgt_ids = tokenizer.encode(target_token, add_special_tokens=False)
    tgt_first = tgt_ids[0]
    ids_list = input_ids.tolist()
    # Find the last occurrence (since the target typically appears at the end of setup)
    for pos in range(len(ids_list) - 1, -1, -1):
        if ids_list[pos] == tgt_first:
            return pos
    return -1  # not found


attn_on_target = []
for idx, (layer, head) in enumerate(TEXT_NATIVE):
    head_attns = []
    for item_idx, (item, prompt, tgt) in enumerate(zip(TASK_C_ITEMS, PROMPTS, TARGETS)):
        attns, input_ids = get_all_attention_weights(model_orig, prompt)
        # attns[layer] has shape [1, n_heads, seq_len, seq_len]
        # We want the attention from the last query position to the target token position
        attn_matrix = attns[layer][0, head].cpu().float().numpy()  # [seq_len, seq_len]
        last_query_pos = attn_matrix.shape[0] - 1
        target_pos = find_target_token_pos(input_ids, tgt)
        if target_pos >= 0:
            attn_on_tgt = float(attn_matrix[last_query_pos, target_pos])
        else:
            attn_on_tgt = 0.0
        head_attns.append(attn_on_tgt)

    mean_attn = float(np.mean(head_attns))
    attn_on_target.append({
        "layer": layer,
        "head": head,
        "mean_attn_on_target": mean_attn,
        "per_item_attn": [float(x) for x in head_attns],
    })

    # Print progress every 4 heads
    if (idx + 1) % 4 == 0:
        print(f"  [{idx+1:02d}/16] completed")

# Spearman ρ between |ΔP_C_i| and mean attention-on-target
abs_deltas_inst1 = np.abs([r["delta_abl"] for r in results_inst1])
mean_attns = np.array([r["mean_attn_on_target"] for r in attn_on_target])
spearman_rho, spearman_p = stats.spearmanr(abs_deltas_inst1, mean_attns)
print(f"\n  Spearman ρ(|ΔP_C_i|, attn_on_target) = {spearman_rho:.3f}  p={spearman_p:.3f}")
print(f"  H_content_selective threshold: ρ ≥ 0.40")

# ── Instrument 3 — Recency / LiM probe ───────────────────────────────────────

print("\n--- Instrument 3: Recency / LiM probe ---")

# Select 12 items that can have their setup redistributed
# Strategy: items where "setup" is a short sentence (≤ 15 words) and
# filler is ≥ 20 words, so we can rearrange the context
recency_items = []
for item in TASK_C_ITEMS:
    setup_words = len(item["setup"].split())
    filler_words = len(item["filler"].split())
    if setup_words <= 15 and filler_words >= 15:
        recency_items.append(item)

# If we don't have 12, take the ones with the most filler words
if len(recency_items) < 12:
    recency_items = sorted(TASK_C_ITEMS,
                           key=lambda x: len(x["filler"].split()),
                           reverse=True)[:12]

print(f"  Using {len(recency_items)} items for recency probe")

def make_begin_context(item):
    """Setup at beginning: setup + filler + cue"""
    return item["setup"] + " " + item["filler"] + " " + item["cue"]


def make_end_context(item):
    """Setup at end: filler + setup + cue"""
    return item["filler"] + " " + item["setup"] + " " + item["cue"]


def make_middle_context(item):
    """Setup in middle: half_filler + setup + half_filler + cue"""
    filler_words = item["filler"].split()
    mid = len(filler_words) // 2
    first_half = " ".join(filler_words[:mid])
    second_half = " ".join(filler_words[mid:])
    return first_half + " " + item["setup"] + " " + second_half + " " + item["cue"]


lps_begin, lps_mid, lps_end = [], [], []
for item in recency_items:
    tgt = item["target_token"]
    lps_begin.append(target_logprob(model_orig, make_begin_context(item), tgt))
    lps_mid.append(target_logprob(model_orig, make_middle_context(item), tgt))
    lps_end.append(target_logprob(model_orig, make_end_context(item), tgt))

mean_begin = float(np.mean(lps_begin))
mean_mid   = float(np.mean(lps_mid))
mean_end   = float(np.mean(lps_end))

print(f"\n  Recency probe results (n={len(recency_items)} items):")
print(f"    Begin: {mean_begin:.3f}")
print(f"    Mid:   {mean_mid:.3f}")
print(f"    End:   {mean_end:.3f}")
print(f"    End - Begin: {mean_end - mean_begin:+.3f}  (H_recency threshold: ≥ +0.10)")
monotone = mean_end > mean_mid > mean_begin
print(f"    Monotone end > mid > begin: {monotone}")

# ── Hypothesis verdicts ───────────────────────────────────────────────────────

print("\n--- Hypothesis verdicts ---")

H_graded = cv_abls >= 0.5
H_positive = n_heads_degrade_task_c >= 10
H_content_selective = spearman_rho >= 0.40
H_recency = (mean_end - mean_begin) >= 0.10

K1_fires = not H_graded
K2_fires = not H_positive
K3_fires = not H_content_selective
K4_fires = not H_recency

print(f"H_graded     (CV ≥ 0.5):             {'CONFIRMED' if H_graded else 'DEAD'}  (CV={cv_abls:.3f})")
print(f"H_positive   (≥10/16 heads degrade): {'CONFIRMED' if H_positive else 'DEAD'}  ({n_heads_degrade_task_c}/16)")
print(f"H_content_selective (ρ ≥ 0.40):      {'CONFIRMED' if H_content_selective else 'DEAD'}  (ρ={spearman_rho:.3f})")
print(f"H_recency    (end-begin ≥ 0.10):     {'CONFIRMED' if H_recency else 'DEAD'}  ({mean_end - mean_begin:+.3f})")

n_confirmed = sum([H_graded, H_positive, H_content_selective, H_recency])
if n_confirmed == 4:
    h_rep_verdict = "CONFIRMED"
elif n_confirmed >= 2:
    h_rep_verdict = "PARTIAL"
else:
    h_rep_verdict = "FALSIFIED"

print(f"\nH_rep ({n_confirmed}/4 sub-hypotheses confirmed): {h_rep_verdict}")

# Single-token sub-analysis
st_indices = [i for i, m in enumerate(SINGLE_TOKEN_MASK) if m]
print(f"\nSingle-token sub-analysis ({n_single} items):")
for r in results_inst1:
    head_id = (r["layer"], r["head"])
    st_deltas = [r["item_abl_lps"][i] - baseline_lps[i] for i in st_indices]
    r["delta_abl_single_token"] = float(np.mean(st_deltas)) if st_deltas else None

st_means = [r["delta_abl_single_token"] for r in results_inst1 if r["delta_abl_single_token"] is not None]
if st_means:
    print(f"  Mean ΔP_C (single-token only): {np.mean(st_means):.3f}")

# ── Save results ──────────────────────────────────────────────────────────────

results = {
    "exp": "exp-177",
    "prereg_commit": PREREG_COMMIT,
    "model": "EleutherAI/pythia-410m",
    "n_text_native_heads": len(TEXT_NATIVE),
    "text_native_heads": TEXT_NATIVE,
    "n_task_c_items": len(TASK_C_ITEMS),
    "n_single_token_items": n_single,
    "n_multi_token_items": len(TASK_C_ITEMS) - n_single,
    "baseline_mean_lp": float(baseline_mean),
    "baseline_per_item": [float(x) for x in baseline_lps],
    "instrument_1": {
        "cv_abs_delta_abl": cv_abls,
        "n_heads_degrade_task_c": n_heads_degrade_task_c,
        "head_results": results_inst1,
        "H_graded": H_graded,
        "H_positive": H_positive,
        "K1_fires": K1_fires,
        "K2_fires": K2_fires,
    },
    "instrument_2": {
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "attn_on_target_per_head": attn_on_target,
        "H_content_selective": H_content_selective,
        "K3_fires": K3_fires,
    },
    "instrument_3": {
        "n_items": len(recency_items),
        "mean_lp_begin": mean_begin,
        "mean_lp_mid": mean_mid,
        "mean_lp_end": mean_end,
        "end_minus_begin": float(mean_end - mean_begin),
        "monotone": monotone,
        "H_recency": H_recency,
        "K4_fires": K4_fires,
        "per_item_begin": [float(x) for x in lps_begin],
        "per_item_mid": [float(x) for x in lps_mid],
        "per_item_end": [float(x) for x in lps_end],
    },
    "H_rep": h_rep_verdict,
    "n_confirmed": n_confirmed,
}

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

OUT = Path(__file__).parent / "results.json"
OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, cls=_NumpyEncoder)
print(f"\nResults saved to {OUT}")
print("Done.")
