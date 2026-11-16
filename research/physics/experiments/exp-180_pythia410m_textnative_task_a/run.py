"""
exp-180 — Pythia-410m text-native Δ-window heads: entity-state tracking (Task A)

Pre-registration: attention-geometry 0d4b8c5 (committed and pushed before this script).
Tests whether the 16 text-native Δ-window heads in Pythia-410m carry entity-state
tracking function (world-model coherence), after two null results:
  exp-178: 0/16 carry Task C (content-retrieval) function
  exp-179: 1/16 carry Task B (positional-retrieval) function

Three-task battery arc: this is the final standard instrument.

Declared prior: H_arch_full — this population carries no positive function in any
of the three tested categories. H_entity falsified.

Ariel — 2026-11-16, ~12:35 AM MST, solo physics room.
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
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "0d4b8c5"

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

# 8 competitor heads from exp-178 (ablation improved Task C)
COMPETITOR_HEADS: List[Tuple[int, int]] = [
    (0, 3), (2, 8), (3, 2), (3, 7), (7, 11), (8, 15), (9, 15), (19, 15),
]

D_MODEL = 1024
D_HEAD  = 64
N_HEADS = 16
N_LAYERS = 24

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Task A items ──────────────────────────────────────────────────────────────

TASK_A_PATH = Path(__file__).parent / "task_a_items.json"
with open(TASK_A_PATH) as f:
    task_a_data = json.load(f)
TASK_A_ITEMS = task_a_data["items"]
assert len(TASK_A_ITEMS) == 20, f"Expected 20 items, got {len(TASK_A_ITEMS)}"

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

# Verify token count for each target (sanity check)
print("\nVerifying target token counts...")
for item in TASK_A_ITEMS:
    ids = tokenizer.encode(item["target_token"], add_special_tokens=False)
    assert len(ids) == 1, f"Item {item['id']}: target {repr(item['target_token'])} is not single-token: {ids}"
print("  All 20 targets verified as single tokens.")

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


def run_task_a(model, items=TASK_A_ITEMS):
    """Run Task A battery. Returns list of log-probs."""
    return [token_logprob(model, item["passage"], item["target_token"]) for item in items]


def delta_p(ablation_lps, baseline_lps):
    return float(np.mean(np.array(ablation_lps) - np.array(baseline_lps)))


def n_degraded(ablation_lps, baseline_lps):
    """Count items where ablation degraded performance (abl < baseline)."""
    return int(np.sum(np.array(ablation_lps) < np.array(baseline_lps)))


def n_improved(ablation_lps, baseline_lps):
    return int(np.sum(np.array(ablation_lps) > np.array(baseline_lps)))


# ── Baseline ─────────────────────────────────────────────────────────────────

print("\n=== Baseline ===")
t0 = time.time()
baseline_lps = run_task_a(model_orig)
print(f"Baseline mean log-prob: {np.mean(baseline_lps):.4f} | elapsed: {time.time()-t0:.1f}s")
print(f"Baseline per-item: {[round(x,3) for x in baseline_lps]}")

# Check K2 (floor): baseline median < -10 nats?
baseline_median = float(np.median(baseline_lps))
k2_fires = baseline_median < -10.0
print(f"\nBaseline median: {baseline_median:.4f} | K2 (floor, < -10 nats): {'FIRES — inconclusive' if k2_fires else 'clear'}")
if k2_fires:
    print("WARNING: K2 fires — task too hard for Pythia-410m. Results will be inconclusive.")

# ── Instrument 1: Individual-head Task A contribution ────────────────────────

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
    abl_lps = run_task_a(model_orig)
    dp_abl = delta_p(abl_lps, baseline_lps)
    n_deg_abl = n_degraded(abl_lps, baseline_lps)
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
    sham_lps = run_task_a(model_orig)
    dp_sham = delta_p(sham_lps, baseline_lps)
    n_deg_sham = n_degraded(sham_lps, baseline_lps)

    # Restore again
    set_wk(model_orig, layer, head, orig_wk)

    result = {
        "layer": layer, "head": head, "label": label,
        "dp_abl": round(dp_abl, 4),
        "dp_sham": round(dp_sham, 4),
        "n_degraded_abl": n_deg_abl,
        "n_improved_abl": n_imp_abl,
        "n_degraded_sham": n_deg_sham,
        "orig_wk_norm": round(orig_norm, 4),
        "is_competitor_exp178": (layer, head) in COMPETITOR_HEADS,
    }
    instrument1_results.append(result)
    comp_flag = " [competitor]" if (layer, head) in COMPETITOR_HEADS else ""
    print(f"  {label}: ΔP_abl={dp_abl:+.4f} ({n_deg_abl}/20 degraded, {n_imp_abl}/20 improved) | ΔP_sham={dp_sham:+.4f}{comp_flag}")

# ── Instrument 1 analysis ─────────────────────────────────────────────────────

print("\n--- Instrument 1 Summary ---")

# H_entity: ≥10/16 heads have ΔP_A < -0.05 AND ≥12/20 degraded
n_entity_carriers = sum(
    1 for r in instrument1_results
    if r["dp_abl"] < -0.05 and r["n_degraded_abl"] >= 12
)
print(f"Heads meeting H_entity criterion (ΔP_A < -0.05 AND ≥12/20 degraded): {n_entity_carriers}/16")
h_entity = n_entity_carriers >= 10
print(f"H_entity: {'CONFIRMED' if h_entity else 'DEAD (K2 or near-null)'} — {n_entity_carriers} < 10" if not h_entity else f"H_entity: CONFIRMED — {n_entity_carriers}/16")

# H_graded: CV of |ΔP_A_i| ≥ 0.5
abs_dp = np.array([abs(r["dp_abl"]) for r in instrument1_results])
cv = float(np.std(abs_dp) / np.mean(abs_dp)) if np.mean(abs_dp) > 1e-8 else 0.0
h_graded = cv >= 0.5
print(f"H_graded (CV ≥ 0.5): {'CONFIRMED' if h_graded else 'DEAD'} — CV = {cv:.4f}")

# H_competitor_entity: ≥5/8 competitor heads carry entity function
competitor_entity_carriers = sum(
    1 for r in instrument1_results
    if r["is_competitor_exp178"] and r["dp_abl"] < -0.05
)
h_competitor_entity = competitor_entity_carriers >= 5
print(f"H_competitor_entity (≥5/8 competitor heads carry entity function): {'FIRES' if h_competitor_entity else 'NOT FIRED'} — {competitor_entity_carriers}/8")

# K1: competitor suppression extends to Task A
k1_competitor_suppressors_a = sum(
    1 for r in instrument1_results
    if r["is_competitor_exp178"] and r["dp_abl"] > 0.05
)
k1_fires = k1_competitor_suppressors_a >= 5
print(f"K1 (competitor suppression in Task A, ≥5/8 improve): {'FIRES' if k1_fires else 'does not fire'} — {k1_competitor_suppressors_a}/8")

# K2: near-null (< 5 degraders AND < 5 improvers)
n_total_degraders = sum(1 for r in instrument1_results if r["dp_abl"] < -0.05)
n_total_improvers = sum(1 for r in instrument1_results if r["dp_abl"] > 0.05)
k2_null = n_total_degraders < 5 and n_total_improvers < 5
print(f"K2 (near-null): {'FIRES' if k2_null else 'does not fire'} — degraders={n_total_degraders}, improvers={n_total_improvers}")

# K3: sham ≥ ablation for most heads
n_sham_dominates = sum(1 for r in instrument1_results if abs(r["dp_sham"]) >= abs(r["dp_abl"]))
k3_fires = n_sham_dominates >= 10
print(f"K3 (sham ≥ ablation for ≥10/16): {'FIRES — artifact' if k3_fires else 'does not fire'} — {n_sham_dominates}/16")

mean_dp = float(np.mean([r["dp_abl"] for r in instrument1_results]))
print(f"\nMean ΔP_A across all 16 heads: {mean_dp:+.4f}")

print("\nSorted by ΔP_A (most degrading first):")
for r in sorted(instrument1_results, key=lambda x: x["dp_abl"])[:8]:
    comp = " [C]" if r["is_competitor_exp178"] else ""
    print(f"  {r['label']:8s}: ΔP={r['dp_abl']:+.4f} ({r['n_degraded_abl']}/20 deg, {r['n_improved_abl']}/20 imp) | sham={r['dp_sham']:+.4f}{comp}")

print("\nSorted by ΔP_A (most improving first):")
for r in sorted(instrument1_results, key=lambda x: x["dp_abl"], reverse=True)[:5]:
    comp = " [C]" if r["is_competitor_exp178"] else ""
    print(f"  {r['label']:8s}: ΔP={r['dp_abl']:+.4f} ({r['n_degraded_abl']}/20 deg, {r['n_improved_abl']}/20 imp) | sham={r['dp_sham']:+.4f}{comp}")

# ── Results summary ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESULTS SUMMARY — exp-180")
print("="*60)
print(f"H_entity  (≥10/16 carry entity function):  {'CONFIRMED' if h_entity else 'DEAD'} — {n_entity_carriers}/16")
print(f"H_graded  (CV ≥ 0.5):                      {'CONFIRMED' if h_graded else 'DEAD'} — CV = {cv:.4f}")
print(f"H_competitor_entity (≥5/8 carry):          {'FIRES' if h_competitor_entity else 'not fired'} — {competitor_entity_carriers}/8")
print(f"K1 (competitor suppression in Task A):      {'FIRES' if k1_fires else 'clear'} — {k1_competitor_suppressors_a}/8")
print(f"K2 (near-null):                             {'FIRES' if k2_null else 'clear'}")
print(f"K3 (sham ≥ ablation):                       {'FIRES' if k3_fires else 'clear'}")
print(f"Baseline floor check (K2b):                 {'FIRES — floor effect' if k2_fires else 'clear'} — median={baseline_median:.4f}")
print()
if h_entity:
    print("Verdict: H_entity CONFIRMED — geometric class predicts entity-state tracking in RoPE.")
    print("Declared prior H_arch_full: REFUTED")
else:
    print("Verdict: H_entity DEAD — declared prior H_arch_full SUPPORTED.")
    if k1_fires:
        print("         K1 fires: competitor suppression extends to Task A (a third task type).")
    if k2_null:
        print("         K2 fires: near-null pattern (< 5 degraders AND < 5 improvers).")

# ── Save results ──────────────────────────────────────────────────────────────

results = {
    "experiment": "exp-180",
    "date": "2026-11-16",
    "model": "EleutherAI/pythia-410m",
    "prereg_commit": PREREG_COMMIT,
    "battery": "task_a_items.json (20 entity-state tracking items, single-token targets verified)",
    "baseline": {
        "mean_logprob": round(float(np.mean(baseline_lps)), 4),
        "median_logprob": round(baseline_median, 4),
        "k2_floor_fires": k2_fires,
        "per_item": [round(x, 4) for x in baseline_lps],
    },
    "instrument1": {
        "cv_abs_contributions": round(cv, 4),
        "h_graded": h_graded,
        "n_entity_carriers": n_entity_carriers,
        "h_entity": h_entity,
        "h_competitor_entity": h_competitor_entity,
        "competitor_entity_carriers": competitor_entity_carriers,
        "k1_fires": k1_fires,
        "k1_competitor_suppressors_in_task_a": k1_competitor_suppressors_a,
        "k2_near_null": k2_null,
        "k3_fires": k3_fires,
        "k3_sham_dominates_count": n_sham_dominates,
        "mean_dp_all_heads": round(mean_dp, 4),
        "n_total_degraders": n_total_degraders,
        "n_total_improvers": n_total_improvers,
        "per_head": instrument1_results,
    },
    "summary": {
        "h_entity": "confirmed" if h_entity else "dead",
        "h_graded": "confirmed" if h_graded else "dead",
        "declared_prior_h_arch_full": "supported" if not h_entity else "refuted",
        "kills_fired": [k for k, v in [("K1", k1_fires), ("K2", k2_null), ("K3", k3_fires), ("K2_floor", k2_fires)] if v],
    },
}

out_path = Path(__file__).parent / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
