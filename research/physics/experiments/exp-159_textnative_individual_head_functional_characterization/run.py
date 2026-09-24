"""
exp-159 — Text-native Δ-window heads: individual functional characterization

Pre-registration: attention-geometry c751a2b (committed before this script).
Three instruments:
  1. Individual-head Task C contribution (per-head W_K ablation)
  2. Attention distribution on Task C items (attention-on-target analysis)
  3. Lost-in-the-middle probe (position sensitivity in baseline model)

Ariel — 2026-09-24, ~12:30 AM MDT, solo physics room.
"""

from __future__ import annotations
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "c751a2b"

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
print(f"Device: {DEVICE}")

# ── Task C items (from exp-155) ───────────────────────────────────────────────

TASK_C_PATH = Path(__file__).parent.parent / "exp-155_textnative_content_retrieval" / "task_c_items.json"
with open(TASK_C_PATH) as f:
    task_c_data = json.load(f)
TASK_C_ITEMS = task_c_data["items"]
assert len(TASK_C_ITEMS) == 20, f"Expected 20 items, got {len(TASK_C_ITEMS)}"

# ── Model and tokenizer ───────────────────────────────────────────────────────

print("Loading GPT-2 small (eager attention for output_attentions support)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_orig = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model_orig.to(DEVICE)
model_orig.eval()

# ── W_K helpers ───────────────────────────────────────────────────────────────

def get_wk(model, layer, head):
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD].copy()


def set_wk_zero(model, layer, head):
    W = model.transformer.h[layer].attn.c_attn.weight.data
    with torch.no_grad():
        W[:, D_MODEL + head * D_HEAD: D_MODEL + (head + 1) * D_HEAD] = 0.0


def set_wk(model, layer, head, W_K_new):
    W = model.transformer.h[layer].attn.c_attn.weight.data
    with torch.no_grad():
        W[:, D_MODEL + head * D_HEAD: D_MODEL + (head + 1) * D_HEAD] = torch.tensor(
            W_K_new, dtype=W.dtype, device=W.device
        )


def verify_wk_norm(model, layer, head):
    return float(np.linalg.norm(get_wk(model, layer, head)))


# ── Forward pass helpers ──────────────────────────────────────────────────────

def build_prompt(item):
    return item["setup"] + " " + item["filler"] + " " + item["cue"]


def token_logprob(model, prompt, target_token):
    """Return log-prob of target_token following prompt."""
    tok_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    target_id = tokenizer.encode(target_token)[0]
    with torch.no_grad():
        out = model(tok_ids)
    logits = out.logits[0, -1, :]  # last position
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
print(f"Baseline Task C mean log-prob: {np.mean(baseline_lps):.4f} | elapsed: {time.time()-t0:.1f}s")

# ── Instrument 1: Individual-head Task C contribution ────────────────────────

print("\n=== Instrument 1: Individual-head W_K ablation ===")
instrument1_results = []

for layer, head in TEXT_NATIVE:
    label = f"L{layer}H{head}"

    # Save original W_K
    orig_wk = get_wk(model_orig, layer, head)

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

    # === SHAM: matched-norm random W_K ===
    rng = np.random.default_rng(seed=(layer * 100 + head + 159))
    orig_norm = float(np.linalg.norm(orig_wk))
    rand_wk = rng.standard_normal(orig_wk.shape)
    rand_wk = rand_wk * orig_norm / float(np.linalg.norm(rand_wk))
    set_wk(model_orig, layer, head, rand_wk)
    sham_lps = run_task_c(model_orig)
    dp_sham = delta_p(sham_lps, baseline_lps)
    n_imp_sham = n_improved(sham_lps, baseline_lps)

    # Restore
    set_wk(model_orig, layer, head, orig_wk)
    assert verify_wk_norm(model_orig, layer, head) > 0.1  # restored

    result = {
        "head": label,
        "layer": layer,
        "head_idx": head,
        "dp_c_ablation": round(dp_abl, 4),
        "dp_c_sham": round(dp_sham, 4),
        "n_improved_ablation": n_imp_abl,
        "n_improved_sham": n_imp_sham,
        "item_lps_ablation": [round(x, 4) for x in abl_lps],
        "item_lps_sham": [round(x, 4) for x in sham_lps],
    }
    instrument1_results.append(result)
    print(f"  {label}: ΔP_C(abl)={dp_abl:+.3f} ({n_imp_abl}/20 improved) | ΔP_C(sham)={dp_sham:+.3f} | restored norm={check_norm:.3f}")


# ── Instrument 1 analysis ─────────────────────────────────────────────────────

abs_contribs = [abs(r["dp_c_ablation"]) for r in instrument1_results]
mean_abs = float(np.mean(abs_contribs))
std_abs = float(np.std(abs_contribs))
cv = std_abs / mean_abs if mean_abs > 0 else 0.0
n_degrading = sum(1 for r in instrument1_results if r["dp_c_ablation"] < -0.05)

print(f"\n--- Instrument 1 Summary ---")
print(f"CV of |ΔP_C|: {cv:.3f} (threshold 0.5; H_graded fires if CV ≥ 0.5)")
print(f"H_graded: {'CONFIRMED' if cv >= 0.5 else 'DEAD'}")
print(f"Heads with ΔP_C < -0.05: {n_degrading}/16 (threshold 10; H_positive fires if ≥ 10)")
print(f"H_positive: {'CONFIRMED' if n_degrading >= 10 else 'DEAD'}")
print(f"K1 (CV < 0.5): {'fires' if cv < 0.5 else 'does not fire'}")
print(f"K2 (fewer than 10/16 degrade): {'fires' if n_degrading < 10 else 'does not fire'}")

# Ranked
ranked = sorted(instrument1_results, key=lambda r: r["dp_c_ablation"])
print("\nRanked by ΔP_C (ablation, most degrading first):")
for r in ranked:
    marker = "  <<" if r["dp_c_ablation"] < -0.05 else ""
    print(f"  {r['head']}: ΔP_C={r['dp_c_ablation']:+.3f}{marker}")


# ── Instrument 2: Attention distribution on Task C items ─────────────────────

print("\n=== Instrument 2: Attention distribution on Task C items ===")

# Use model's built-in output_attentions=True.
# out.attentions is a tuple of (n_layers,) tensors, each shape (1, n_heads, seq_len, seq_len).

instrument2_results = []

for item_idx, item in enumerate(TASK_C_ITEMS):
    prompt = build_prompt(item)
    tok_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    seq_len = tok_ids.shape[1]

    # Find target token position in the tokenized full prompt
    # The setup sentence appears at the start of the prompt
    full_ids = tokenizer.encode(prompt)
    setup_ids = tokenizer.encode(item["setup"])
    target_token = item["target_token"]
    target_id = tokenizer.encode(target_token)[0]

    # Find target token in the setup portion (first len(setup_ids) tokens)
    target_positions = []
    for pos, tid in enumerate(setup_ids):
        if tid == target_id:
            target_positions.append(pos)

    # Use the first occurrence in setup
    target_pos = target_positions[0] if target_positions else None

    # Forward pass — use output_attentions=True to get all layer attention tensors
    with torch.no_grad():
        out = model_orig(tok_ids, output_attentions=True)

    # out.attentions: tuple of tensors, one per layer, shape (1, n_heads, seq_len, seq_len)
    all_attentions = out.attentions  # length = n_layers = 12

    # For each text-native head, extract attention weight from last position to target_pos
    head_attn_on_target = {}
    for layer, head in TEXT_NATIVE:
        if all_attentions is None or layer >= len(all_attentions):
            head_attn_on_target[f"L{layer}H{head}"] = None
            continue
        attn = all_attentions[layer]  # (1, n_heads, seq_len, seq_len)
        if attn is None or attn.shape[2] != seq_len:
            head_attn_on_target[f"L{layer}H{head}"] = None
            continue
        if target_pos is not None and target_pos < seq_len:
            attn_weight = float(attn[0, head, -1, target_pos].cpu())
        else:
            attn_weight = None
        head_attn_on_target[f"L{layer}H{head}"] = attn_weight

    item_result = {
        "item_id": item["id"],
        "entity": item["entity"],
        "property": item["property"],
        "target_token": target_token,
        "target_pos_in_setup": target_pos,
        "seq_len": seq_len,
        "attn_on_target": head_attn_on_target,
    }
    instrument2_results.append(item_result)

    if item_idx < 3 or item_idx % 5 == 0:
        attn_vals = [v for v in head_attn_on_target.values() if v is not None]
        mean_v = np.mean(attn_vals) if attn_vals else float('nan')
        print(f"  Item {item['id']} ({item['entity']}/{item['property']}): "
              f"target_pos={target_pos}, seq_len={seq_len}, "
              f"mean attn-on-target={mean_v:.4f}")


# ── Instrument 2 analysis ─────────────────────────────────────────────────────

print("\n--- Instrument 2 Summary ---")

# Mean attention-on-target per head
head_mean_attn = {}
for layer, head in TEXT_NATIVE:
    label = f"L{layer}H{head}"
    vals = [r["attn_on_target"].get(label) for r in instrument2_results]
    valid = [v for v in vals if v is not None]
    head_mean_attn[label] = float(np.mean(valid)) if valid else None

print("Mean attention-on-target per head (baseline, last-position → target-prop-position):")
for label, v in sorted(head_mean_attn.items(), key=lambda x: -(x[1] or 0)):
    print(f"  {label}: {v:.4f}" if v is not None else f"  {label}: N/A")

# Spearman correlation between individual head contribution (|ΔP_C|) and mean attn-on-target
ind1_dict = {r["head"]: abs(r["dp_c_ablation"]) for r in instrument1_results}
attn_vals_for_corr = []
contrib_vals_for_corr = []
for label in ind1_dict:
    a = head_mean_attn.get(label)
    c = ind1_dict.get(label)
    if a is not None and c is not None:
        attn_vals_for_corr.append(a)
        contrib_vals_for_corr.append(c)

from scipy.stats import spearmanr
if len(attn_vals_for_corr) >= 5:
    rho, pval = spearmanr(contrib_vals_for_corr, attn_vals_for_corr)
    print(f"\nSpearman ρ (|ΔP_C_i| vs mean attn-on-target): {rho:.3f} (p={pval:.3f})")
    print(f"H_content_selective: {'CONFIRMED' if rho >= 0.40 else 'DEAD'} (threshold 0.40)")
    print(f"K3 (ρ < 0.40): {'fires' if rho < 0.40 else 'does not fire'}")
else:
    rho, pval = None, None
    print("Insufficient data for Spearman correlation.")


# ── Instrument 3: Lost-in-the-middle probe ───────────────────────────────────

print("\n=== Instrument 3: Lost-in-the-middle probe ===")

# Construct variants of Task C items where the target property appears at
# beginning, middle, or end of the combined context window.
# Strategy: rearrange the filler text around the setup sentence.

# For each item we create three conditions:
#   - "begin": setup sentence first, then filler, then cue
#     (this is the original layout — target info near start of context)
#   - "end": filler first, then setup sentence, then cue
#     (target info near end of context)
#   - "middle": first half of filler, then setup sentence, then second half of filler, then cue
#     (target info in middle of context)

def split_filler(filler_text):
    """Split filler at a sentence boundary near the midpoint."""
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
        "begin": setup + " " + filler + " " + cue,    # original layout
        "end":   filler + " " + setup + " " + cue,    # setup near end
        "middle": filler_first + " " + setup + " " + filler_second + " " + cue,
    }

    lps = {}
    for condition, prompt in prompts.items():
        lps[condition] = token_logprob(model_orig, prompt, target_token)

    item_result = {
        "item_id": item["id"],
        "entity": item["entity"],
        "property": item["property"],
        "target_token": target_token,
        "lp_begin": round(lps["begin"], 4),
        "lp_middle": round(lps["middle"], 4),
        "lp_end": round(lps["end"], 4),
    }
    instrument3_results.append(item_result)


# ── Instrument 3 analysis ─────────────────────────────────────────────────────

print("\n--- Instrument 3 Summary ---")

lp_begin = np.array([r["lp_begin"] for r in instrument3_results])
lp_middle = np.array([r["lp_middle"] for r in instrument3_results])
lp_end = np.array([r["lp_end"] for r in instrument3_results])

mean_begin = float(np.mean(lp_begin))
mean_middle = float(np.mean(lp_middle))
mean_end = float(np.mean(lp_end))

print(f"Mean log-prob by target position:")
print(f"  Begin (setup first):  {mean_begin:.4f}")
print(f"  Middle (setup mid):   {mean_middle:.4f}")
print(f"  End (setup last):     {mean_end:.4f}")

diff_middle_vs_begin = mean_middle - mean_begin
diff_middle_vs_end = mean_middle - mean_end
lim_detected = (mean_middle < mean_begin and mean_middle < mean_end
                and abs(diff_middle_vs_begin) >= 0.15 and abs(diff_middle_vs_end) >= 0.15)

print(f"\nMiddle vs Begin: {diff_middle_vs_begin:+.4f} nats")
print(f"Middle vs End:   {diff_middle_vs_end:+.4f} nats")
print(f"H_lim (lost-in-middle): {'CONFIRMED' if lim_detected else 'DEAD'}")
print(f"K4 (no position sensitivity): {'fires' if not lim_detected else 'does not fire'}")


# ── Results JSON ──────────────────────────────────────────────────────────────

# Final verdicts
k1_fires = cv < 0.5
k2_fires = n_degrading < 10
k3_fires = (rho is not None and rho < 0.40)
k4_fires = not lim_detected

h_graded = not k1_fires
h_positive = not k2_fires
h_content_selective = (rho is not None and rho >= 0.40)
h_lim = lim_detected

# Overall verdict
if h_graded and h_positive and h_content_selective:
    verdict = "confirmed"
elif not h_graded and not h_positive:
    verdict = "falsified"
else:
    verdict = "partial"

rho_str = f"{rho:.3f}" if rho is not None else "N/A"
headline = (
    f"Individual Task C contributions: CV={cv:.2f} ({'graded' if h_graded else 'uniform'}), "
    f"{n_degrading}/16 degrading heads, "
    f"Spearman rho(contrib, attn-on-target)={rho_str}, "
    f"LiM: {'detected' if lim_detected else 'not detected'}."
)

results = {
    "experiment": "exp-159",
    "prereg_commit": PREREG_COMMIT,
    "model": "gpt2",
    "text_native_heads": [f"L{l}H{h}" for l, h in TEXT_NATIVE],
    "baseline_task_c_mean_lp": round(float(np.mean(baseline_lps)), 4),
    "baseline_task_c_item_lps": [round(x, 4) for x in baseline_lps],
    "instrument1": {
        "description": "Individual-head Task C contribution (W_K ablation)",
        "head_results": instrument1_results,
        "summary": {
            "abs_contributions": [round(x, 4) for x in abs_contribs],
            "mean_abs_contribution": round(mean_abs, 4),
            "std_abs_contribution": round(std_abs, 4),
            "cv": round(cv, 4),
            "n_degrading_05": n_degrading,
            "h_graded": h_graded,
            "h_positive": h_positive,
            "k1_fires": k1_fires,
            "k2_fires": k2_fires,
        },
        "ranked": [r["head"] for r in sorted(instrument1_results, key=lambda r: r["dp_c_ablation"])],
    },
    "instrument2": {
        "description": "Attention distribution on Task C items (attention-on-target analysis)",
        "item_results": instrument2_results,
        "head_mean_attn_on_target": {k: (round(v, 6) if v is not None else None) for k, v in head_mean_attn.items()},
        "spearman_rho_contrib_vs_attn": round(rho, 4) if rho is not None else None,
        "spearman_pval": round(pval, 4) if pval is not None else None,
        "h_content_selective": h_content_selective,
        "k3_fires": k3_fires,
    },
    "instrument3": {
        "description": "Lost-in-the-middle probe (target position sensitivity, baseline model)",
        "item_results": instrument3_results,
        "summary": {
            "mean_lp_begin": round(mean_begin, 4),
            "mean_lp_middle": round(mean_middle, 4),
            "mean_lp_end": round(mean_end, 4),
            "diff_middle_vs_begin": round(float(diff_middle_vs_begin), 4),
            "diff_middle_vs_end": round(float(diff_middle_vs_end), 4),
            "h_lim_confirmed": h_lim,
            "k4_fires": k4_fires,
        },
    },
    "verdict": verdict,
    "headline": headline,
}

def _convert(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    if isinstance(obj, (bool, type(None))):
        return obj
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj

out_path = Path(__file__).parent / "results.json"
with open(out_path, "w") as f:
    json.dump(_convert(results), f, indent=2)

print(f"\n=== Final Verdict: {verdict.upper()} ===")
print(f"Headline: {headline}")
print(f"Results written to {out_path}")
