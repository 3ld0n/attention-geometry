"""
exp-178 — Pythia-410m text-native Δ-window heads: native-tokenizer Task C battery

Pre-registration: attention-geometry acd81b0 (committed and pushed before this script).
Architecture-conditionality discrimination test: does exp-177's null replicate when
all 20 Task C targets are single tokens in the GPT-NeoX tokenizer?

Three instruments (same as exp-177 and exp-159):
  1. Individual-head Task C contribution (per-head W_K ablation)
  2. Attention distribution on Task C items (attention-on-target analysis)
  3. Recency / lost-in-the-middle probe (position sensitivity in baseline model)

Population: 16 text-native Δ-window heads in Pythia-410m from exp-118.
Battery: task_c_native_items.json (20 items, all single-token in GPT-NeoX tokenizer).

Declared prior: H_arch — the architecture-conditionality null replicates.
Based on exp-177's 6-item single-token sub-analysis (mean ΔP_C = +0.014).

Ariel — 2026-11-14, ~12:40 AM MST, solo physics room.
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

PREREG_COMMIT = "acd81b0"

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

# ── Task C native items ───────────────────────────────────────────────────────

TASK_C_PATH = Path(__file__).parent / "task_c_native_items.json"
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

def get_wk(model, layer, head):
    """Extract W_K for (layer, head) from query_key_value weight matrix."""
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.detach().cpu().float().numpy()
    # QKV layout: rows = [Q_0..Q_{n-1}, K_0..K_{n-1}, V_0..V_{n-1}] in d_model slices
    # Each head slice is d_head rows starting at D_MODEL + head*D_HEAD (K block)
    offset = D_MODEL + head * D_HEAD
    return W[offset: offset + D_HEAD, :].copy()  # shape (d_head, d_model)


def set_wk_zero(model, layer, head):
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.data
    offset = D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[offset: offset + D_HEAD, :] = 0.0


def set_wk(model, layer, head, W_K_new):
    W = model.gpt_neox.layers[layer].attention.query_key_value.weight.data
    offset = D_MODEL + head * D_HEAD
    with torch.no_grad():
        W[offset: offset + D_HEAD, :] = torch.tensor(W_K_new, dtype=W.dtype, device=W.device)


def verify_wk_norm(model, layer, head):
    return float(np.linalg.norm(get_wk(model, layer, head)))


# ── Forward pass helpers ──────────────────────────────────────────────────────

def build_prompt(item):
    return item["setup"] + " " + item["filler"] + " " + item["cue"]


def token_logprob(model, prompt, target_token):
    """Return log-prob of target_token (single token) following prompt."""
    target_id = tokenizer.encode(target_token, add_special_tokens=False)
    assert len(target_id) == 1, f"Expected single token for '{target_token}', got {target_id}"
    target_id = target_id[0]

    tok_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(tok_ids)
    logits = out.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(log_probs[target_id].cpu())


def run_task_c(model, items=TASK_C_ITEMS):
    """Run Task C battery. Returns list of log-probs."""
    return [token_logprob(model, build_prompt(it), it["target_token"]) for it in items]


def delta_p(ablation_lps, baseline_lps):
    return float(np.mean(np.array(ablation_lps) - np.array(baseline_lps)))


def n_improved(ablation_lps, baseline_lps):
    return int(np.sum(np.array(ablation_lps) > np.array(baseline_lps)))


# ── Baseline ─────────────────────────────────────────────────────────────────

print("\n=== Baseline ===")
t0 = time.time()
baseline_lps = run_task_c(model_orig)
print(f"Baseline mean log-prob: {np.mean(baseline_lps):.4f} | elapsed: {time.time()-t0:.1f}s")
print(f"Baseline per-item: {[round(x,3) for x in baseline_lps]}")

# ── Instrument 1: Individual-head Task C contribution ────────────────────────

print("\n=== Instrument 1: Individual-head W_K ablation ===")
instrument1_results = []

for layer, head in TEXT_NATIVE:
    label = f"L{layer}H{head}"

    # Save original W_K
    orig_wk = get_wk(model_orig, layer, head)
    orig_norm = float(np.linalg.norm(orig_wk))

    # === ABLATION ===
    set_wk_zero(model_orig, layer, head)
    abl_norm = verify_wk_norm(model_orig, layer, head)
    assert abl_norm < 1e-6, f"{label}: ablation failed, norm={abl_norm}"
    abl_lps = run_task_c(model_orig)
    dp_abl = delta_p(abl_lps, baseline_lps)
    n_imp_abl = n_improved(abl_lps, baseline_lps)

    # Restore
    set_wk(model_orig, layer, head, orig_wk)
    check_norm = verify_wk_norm(model_orig, layer, head)
    assert abs(check_norm - orig_norm) < 1e-3, f"{label}: restore failed"

    # === SHAM ===
    rng = np.random.default_rng(seed=layer * 100 + head)
    sham_wk = rng.standard_normal(orig_wk.shape) * (orig_norm / np.sqrt(orig_wk.size))
    sham_wk = sham_wk.astype(np.float32)
    set_wk(model_orig, layer, head, sham_wk)
    sham_lps = run_task_c(model_orig)
    dp_sham = delta_p(sham_lps, baseline_lps)
    n_imp_sham = n_improved(sham_lps, baseline_lps)

    # Restore again
    set_wk(model_orig, layer, head, orig_wk)

    result = {
        "layer": layer, "head": head, "label": label,
        "dp_abl": round(dp_abl, 4),
        "dp_sham": round(dp_sham, 4),
        "n_improved_abl": n_imp_abl,
        "n_improved_sham": n_imp_sham,
        "orig_wk_norm": round(orig_norm, 4),
    }
    instrument1_results.append(result)
    print(f"  {label}: ΔP_abl={dp_abl:+.4f} ({n_imp_abl}/20 improved) | ΔP_sham={dp_sham:+.4f} ({n_imp_sham}/20)")

# ── Instrument 1 analysis ─────────────────────────────────────────────────────

print("\n--- Instrument 1 Summary ---")
dp_abs = np.array([abs(r["dp_abl"]) for r in instrument1_results])
cv = float(np.std(dp_abs) / np.mean(dp_abs)) if np.mean(dp_abs) > 1e-8 else 0.0
mean_dp = float(np.mean([r["dp_abl"] for r in instrument1_results]))
n_degrading = sum(1 for r in instrument1_results if r["dp_abl"] < -0.05)
print(f"CV of |ΔP_C_i|: {cv:.4f} (threshold: 0.5) → H_graded {'CONFIRMED' if cv >= 0.5 else 'DEAD (K1)'}")
print(f"Heads with ΔP_C < -0.05: {n_degrading}/16 → H_positive {'CONFIRMED' if n_degrading >= 10 else f'DEAD (K2 fires) — {n_degrading} < 10'}")
print(f"Mean ΔP_C across all heads: {mean_dp:+.4f}")

sorted_by_dp = sorted(instrument1_results, key=lambda r: r["dp_abl"])
print("\nTop degraders (most negative ΔP_C):")
for r in sorted_by_dp[:5]:
    print(f"  {r['label']}: ΔP_abl={r['dp_abl']:+.4f} ({r['n_improved_abl']}/20 items)")

# ── Instrument 2: Attention-on-target ────────────────────────────────────────

print("\n=== Instrument 2: Attention-on-target ===")

instrument2_results = []

for item in TASK_C_ITEMS:
    prompt = build_prompt(item)
    target_token = item["target_token"]
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    # Tokenize and find position of first target-token-like token in prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    n_tokens = input_ids.shape[1]

    # Find position of target-related content in context
    # Strategy: find the token(s) that appear in the setup sentence (the property token)
    # We look for target_id in the input_ids
    setup_encoded = tokenizer.encode(item["setup"], add_special_tokens=False)
    # Find target_id in setup_encoded to get approximate position
    target_positions = [i for i, t in enumerate(input_ids[0].tolist()) if t == target_id]

    with torch.no_grad():
        out = model_orig(input_ids, output_attentions=True)

    # out.attentions: tuple of (batch, n_heads, seq, seq) per layer
    # For each head in TEXT_NATIVE, get attention at the last query position (the cue)
    # onto positions where the target property appears in the setup
    head_attn_on_target = {}
    for layer, head in TEXT_NATIVE:
        attn = out.attentions[layer][0, head, -1, :]  # shape (seq,)
        attn_np = attn.cpu().float().numpy()
        if target_positions:
            mean_attn = float(np.mean([attn_np[p] for p in target_positions if p < len(attn_np)]))
        else:
            mean_attn = 0.0
        head_attn_on_target[(layer, head)] = mean_attn

    instrument2_results.append({
        "item_id": item["id"],
        "target_token": target_token,
        "n_target_positions": len(target_positions),
        "head_attn": {f"L{l}H{h}": round(v, 6) for (l, h), v in head_attn_on_target.items()},
    })

# Compute Spearman correlation between |ΔP_C_i| and mean attention-on-target
print("Computing Spearman ρ(|ΔP_C_i|, mean attention-on-target)...")
abs_contributions = np.array([abs(r["dp_abl"]) for r in instrument1_results])
mean_attns = []
for layer, head in TEXT_NATIVE:
    # mean attention across all items for this head
    per_item = [item_r["head_attn"][f"L{layer}H{head}"] for item_r in instrument2_results]
    mean_attns.append(float(np.mean(per_item)))

rho, pval = stats.spearmanr(abs_contributions, mean_attns)
print(f"Spearman ρ = {rho:.4f} (p = {pval:.4f})")
h_content = rho >= 0.40
print(f"H_content_selective: {'CONFIRMED' if h_content else 'DEAD (K3)'} (threshold: ρ ≥ 0.40)")

# ── Instrument 3: Position probe ─────────────────────────────────────────────

print("\n=== Instrument 3: Position probe ===")

def split_filler(filler_text):
    sentences = filler_text.split('. ')
    mid = len(sentences) // 2
    first_half = '. '.join(sentences[:mid]) + '.'
    second_half = '. '.join(sentences[mid:])
    return first_half.strip(), second_half.strip()


instrument3_results = []

for item in TASK_C_ITEMS:
    setup = item["setup"]
    filler = item["filler"]
    cue = item["cue"]
    target_token = item["target_token"]

    filler_first, filler_second = split_filler(filler)

    prompts = {
        "begin":  setup + " " + filler + " " + cue,
        "end":    filler + " " + setup + " " + cue,
        "middle": filler_first + " " + setup + " " + filler_second + " " + cue,
    }

    lps = {cond: token_logprob(model_orig, prompt, target_token)
           for cond, prompt in prompts.items()}

    instrument3_results.append({
        "item_id": item["id"],
        "entity": item["entity"],
        "property": item["property"],
        "target_token": target_token,
        "lp_begin": round(lps["begin"], 4),
        "lp_middle": round(lps["middle"], 4),
        "lp_end": round(lps["end"], 4),
    })

# ── Instrument 3 analysis ─────────────────────────────────────────────────────

print("\n--- Instrument 3 Summary ---")

lp_begin  = np.array([r["lp_begin"]  for r in instrument3_results])
lp_middle = np.array([r["lp_middle"] for r in instrument3_results])
lp_end    = np.array([r["lp_end"]    for r in instrument3_results])

mean_begin  = float(np.mean(lp_begin))
mean_middle = float(np.mean(lp_middle))
mean_end    = float(np.mean(lp_end))

print(f"Mean log-prob by target position:")
print(f"  Begin (setup first):  {mean_begin:.4f}")
print(f"  Middle (setup mid):   {mean_middle:.4f}")
print(f"  End (setup last):     {mean_end:.4f}")

end_minus_begin = mean_end - mean_begin
h_recency = (mean_end > mean_begin and abs(end_minus_begin) >= 0.10)
h_lim = (mean_middle < mean_begin and mean_middle < mean_end)

print(f"\nEnd − Begin = {end_minus_begin:+.4f} nats")
print(f"H_recency: {'CONFIRMED' if h_recency else 'DEAD (K4)'}")
print(f"U-shape (H_lim pattern): {'PRESENT' if h_lim else 'absent'}")

# ── Results summary ───────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Instrument 1:")
print(f"  H_graded  (CV ≥ 0.5): {'CONFIRMED' if cv >= 0.5 else 'DEAD (K1)'} — CV = {cv:.4f}")
print(f"  H_positive (≥10 degrade): {'CONFIRMED' if n_degrading >= 10 else 'DEAD (K2)'} — {n_degrading}/16")
print(f"Instrument 2:")
print(f"  H_content_selective (ρ ≥ 0.40): {'CONFIRMED' if h_content else 'DEAD (K3)'} — ρ = {rho:.4f}")
print(f"Instrument 3:")
print(f"  H_recency: {'CONFIRMED' if h_recency else 'DEAD (K4)'} — end-begin = {end_minus_begin:+.4f}")
print(f"  U-shape: {'YES (H_lim fires)' if h_lim else 'no'}")

n_confirmed = sum([cv >= 0.5, n_degrading >= 10, h_content, h_recency])
print(f"\nSub-hypotheses confirmed: {n_confirmed}/4")
if n_confirmed == 4:
    print("H_rep: CONFIRMED")
    print("Architecture-conditionality was a task-design ARTIFACT.")
elif n_confirmed >= 2:
    print("H_rep: PARTIAL")
elif n_confirmed <= 1:
    print(f"H_rep: FALSIFIED — declared prior (H_arch) SUPPORTED")

print(f"\nDeclared prior H_arch: {'SUPPORTED' if n_degrading < 10 else 'REFUTED'}")

# ── Save results ──────────────────────────────────────────────────────────────

results = {
    "experiment": "exp-178",
    "date": "2026-11-14",
    "model": "EleutherAI/pythia-410m",
    "prereg_commit": PREREG_COMMIT,
    "battery": "task_c_native_items.json (20 single-token targets, GPT-NeoX tokenizer)",
    "instrument1": {
        "cv_abs_contributions": round(cv, 4),
        "h_graded": cv >= 0.5,
        "n_degrading_heads": n_degrading,
        "h_positive": n_degrading >= 10,
        "mean_dp_all_heads": round(mean_dp, 4),
        "per_head": instrument1_results,
    },
    "instrument2": {
        "spearman_rho": round(float(rho), 4),
        "spearman_pval": round(float(pval), 4),
        "h_content_selective": h_content,
        "mean_attn_per_head": {
            f"L{layer}H{head}": round(float(np.mean([r["head_attn"][f"L{layer}H{head}"] for r in instrument2_results])), 6)
            for layer, head in TEXT_NATIVE
        },
    },
    "instrument3": {
        "mean_lp_begin": round(mean_begin, 4),
        "mean_lp_middle": round(mean_middle, 4),
        "mean_lp_end": round(mean_end, 4),
        "end_minus_begin": round(end_minus_begin, 4),
        "h_recency": h_recency,
        "h_lim_pattern": h_lim,
        "per_item": instrument3_results,
    },
    "summary": {
        "n_subhypotheses_confirmed": n_confirmed,
        "h_rep": "confirmed" if n_confirmed == 4 else ("partial" if n_confirmed >= 2 else "falsified"),
        "declared_prior_h_arch": "supported" if n_degrading < 10 else "refuted",
    },
}

out_path = Path(__file__).parent / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
