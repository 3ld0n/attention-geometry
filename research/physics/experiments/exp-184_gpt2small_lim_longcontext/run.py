"""
exp-184: GPT-2 small text-native Δ-window heads — lost-in-the-middle at long contexts
Pre-registration: attention-geometry efdcec8 (committed and pushed before this file)
Date: 2026-11-19
Room: Physics (solo)

Tests whether LiM manifests in GPT-2 small at long contexts (~500 tokens),
extending exp-159 Instrument 3 (short-context probe, K4: recency gradient).
"""

import json
import copy
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ─── Setup ──────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
model = model.to(device)
model.eval()

# ─── Text-native Δ-window heads (from exp-118) ───────────────────────────────
# These are the 16 heads that appear in both random-token and WikiText-native census
# Source: registry.json / exp-118 results; head identities verified in exp-159

TN_HEADS = [
    (4, 10), (7, 1), (8, 2),
    (9, 4), (9, 6),
    (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9),
]

# Top-4 contributors from exp-159 Instrument 1 (for ablation instrument)
TOP4_TN_HEADS = [(10, 10), (9, 6), (4, 10), (7, 1)]

# ─── Identity corpus ─────────────────────────────────────────────────────────
# 30 distinct identities: name, role, company, city, year, manager, product
# Each passage template: ~50-55 GPT-2 tokens

IDENTITIES = [
    {"name": "Marcus", "role": "director", "company": "Halverson", "city": "Portland",
     "year": "2019", "manager": "Elena", "product": "analytics"},
    {"name": "Clara", "role": "analyst", "company": "Brightline", "city": "Denver",
     "year": "2021", "manager": "Thomas", "product": "logistics"},
    {"name": "Owen", "role": "engineer", "company": "Stonefield", "city": "Austin",
     "year": "2020", "manager": "Priya", "product": "infrastructure"},
    {"name": "Leah", "role": "manager", "company": "Westport", "city": "Seattle",
     "year": "2018", "manager": "David", "product": "consulting"},
    {"name": "Felix", "role": "scientist", "company": "Redcloud", "city": "Boston",
     "year": "2022", "manager": "Sara", "product": "research"},
    {"name": "Nina", "role": "architect", "company": "Irongate", "city": "Chicago",
     "year": "2017", "manager": "James", "product": "construction"},
    {"name": "Derek", "role": "specialist", "company": "Silvertop", "city": "Phoenix",
     "year": "2023", "manager": "Laura", "product": "finance"},
    {"name": "Maya", "role": "consultant", "company": "Farland", "city": "Atlanta",
     "year": "2019", "manager": "Kevin", "product": "marketing"},
    {"name": "Ethan", "role": "coordinator", "company": "Northwave", "city": "Dallas",
     "year": "2016", "manager": "Rachel", "product": "distribution"},
    {"name": "Sofia", "role": "strategist", "company": "Clearpath", "city": "Miami",
     "year": "2021", "manager": "Michael", "product": "strategy"},
    {"name": "Aaron", "role": "developer", "company": "Greenfield", "city": "Minneapolis",
     "year": "2020", "manager": "Anna", "product": "software"},
    {"name": "Julia", "role": "planner", "company": "Blackridge", "city": "Detroit",
     "year": "2018", "manager": "Carlos", "product": "planning"},
    {"name": "Victor", "role": "supervisor", "company": "Fairmont", "city": "Nashville",
     "year": "2022", "manager": "Helen", "product": "operations"},
    {"name": "Emma", "role": "researcher", "company": "Whitecrest", "city": "Columbus",
     "year": "2017", "manager": "Alex", "product": "data"},
    {"name": "Logan", "role": "technician", "company": "Eastbridge", "city": "Indianapolis",
     "year": "2023", "manager": "Maria", "product": "hardware"},
    {"name": "Chloe", "role": "director", "company": "Lakeview", "city": "Louisville",
     "year": "2019", "manager": "Robert", "product": "healthcare"},
    {"name": "Nathan", "role": "associate", "company": "Pinecroft", "city": "Memphis",
     "year": "2021", "manager": "Lisa", "product": "insurance"},
    {"name": "Zoe", "role": "engineer", "company": "Stonehaven", "city": "Richmond",
     "year": "2020", "manager": "Paul", "product": "energy"},
    {"name": "Isaac", "role": "officer", "company": "Brightwood", "city": "Hartford",
     "year": "2016", "manager": "Diana", "product": "security"},
    {"name": "Lily", "role": "analyst", "company": "Meadowbrook", "city": "Rochester",
     "year": "2018", "manager": "John", "product": "retail"},
    {"name": "Carter", "role": "manager", "company": "Oakdale", "city": "Raleigh",
     "year": "2022", "manager": "Susan", "product": "construction"},
    {"name": "Stella", "role": "developer", "company": "Cedarvale", "city": "Buffalo",
     "year": "2017", "manager": "Chris", "product": "gaming"},
    {"name": "Miles", "role": "scientist", "company": "Riverton", "city": "Salt Lake City",
     "year": "2023", "manager": "Amy", "product": "pharmaceuticals"},
    {"name": "Hazel", "role": "coordinator", "company": "Springdale", "city": "Omaha",
     "year": "2019", "manager": "Daniel", "product": "agriculture"},
    {"name": "Oscar", "role": "specialist", "company": "Hillcrest", "city": "Tucson",
     "year": "2021", "manager": "Nancy", "product": "aerospace"},
    {"name": "Violet", "role": "architect", "company": "Maplewood", "city": "Sacramento",
     "year": "2020", "manager": "Mark", "product": "real estate"},
    {"name": "Jasper", "role": "technician", "company": "Birchwood", "city": "Albuquerque",
     "year": "2016", "manager": "Julie", "product": "manufacturing"},
    {"name": "Ruby", "role": "consultant", "company": "Elmwood", "city": "Kansas City",
     "year": "2018", "manager": "Steven", "product": "legal"},
    {"name": "Leo", "role": "supervisor", "company": "Ashford", "city": "Tulsa",
     "year": "2022", "manager": "Patricia", "product": "automotive"},
    {"name": "Pearl", "role": "researcher", "company": "Fernwood", "city": "Fresno",
     "year": "2017", "manager": "George", "product": "education"},
]

def make_passage(identity):
    """Generate one passage block from an identity dict."""
    n = identity["name"]
    r = identity["role"]
    c = identity["company"]
    ci = identity["city"]
    y = identity["year"]
    m = identity["manager"]
    p = identity["product"]
    return (f"About {n}: {n} holds the position of {r} at {c}, based in {ci}. "
            f"{n} joined {c} in {y} and reports to {m}. {c}'s main product is {p}. ")

def make_query(identity):
    """Generate query block."""
    n = identity["name"]
    return f"Question: What company does {n} work at? Answer: {n} works at"

def get_target_token(identity):
    """Get the first token of the company name."""
    c = identity["company"]
    # Tokenize " Halverson" (with leading space as it follows "works at")
    tokens = tokenizer.encode(" " + c)
    return tokens[0]  # First subword of the company name

def build_context(target_identity, distractors, target_position, all_identities):
    """
    Build a 10-block context with target at the given position (1-indexed).
    Returns the full context string.
    """
    passages = []
    distractor_idx = 0
    for block in range(1, 11):
        if block == target_position:
            passages.append(make_passage(target_identity))
        else:
            passages.append(make_passage(distractors[distractor_idx]))
            distractor_idx += 1
    context = "".join(passages) + make_query(target_identity)
    return context

def compute_logprob(context, target_token_id):
    """Compute log-prob of target_token_id continuing context."""
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    
    # GPT-2 small max is 1024 tokens — truncate from left if needed
    if input_ids.shape[1] > 1020:
        input_ids = input_ids[:, -1020:]
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # logits at last position
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[target_token_id].item()

def ablate_top4_tn(apply=True):
    """Context manager to zero W_K of top-4 tn heads."""
    saved = {}
    if apply:
        for layer_idx, head_idx in TOP4_TN_HEADS:
            layer = model.transformer.h[layer_idx]
            # GPT-2 uses c_attn (combined QKV projection)
            # W_K slice: columns [d_model:2*d_model] in c_attn
            d = model.config.n_embd  # 768 for gpt2
            head_start = head_idx * (d // model.config.n_head)
            head_end = head_start + (d // model.config.n_head)
            saved[(layer_idx, head_idx)] = layer.attn.c_attn.weight.data[d:2*d, head_start:head_end].clone()
            layer.attn.c_attn.weight.data[d:2*d, head_start:head_end] = 0.0
    return saved

def restore_weights(saved):
    for (layer_idx, head_idx), weights in saved.items():
        d = model.config.n_emph  # 768 for gpt2
        head_start = head_idx * (d // model.config.n_head)
        head_end = head_start + (d // model.config.n_head)
        model.transformer.h[layer_idx].attn.c_attn.weight.data[d:2*d, head_start:head_end] = weights

# ─── Main experiment ─────────────────────────────────────────────────────────

TARGET_POSITIONS = [1, 5, 10]  # early, middle, late
POSITION_LABELS = {1: "early", 5: "middle", 10: "late"}
N_IDENTITIES = 30

print("Building results...")

results_by_identity = []
context_lengths = []  # track actual token counts

for i, target in enumerate(IDENTITIES[:N_IDENTITIES]):
    # Choose 9 distractors (all identities except target)
    distractors = [id_ for id_ in IDENTITIES if id_["name"] != target["name"]][:9]
    target_token = get_target_token(target)
    
    per_position = {}
    per_position_ablated = {}
    
    for pos in TARGET_POSITIONS:
        ctx = build_context(target, distractors, pos, IDENTITIES)
        
        # Track context length
        input_ids = tokenizer.encode(ctx)
        context_lengths.append(len(input_ids))
        
        # Baseline
        lp_base = compute_logprob(ctx, target_token)
        per_position[pos] = lp_base
        
        # Ablated (top-4 tn heads W_K = 0)
        # Protocol: zero W_K output columns for each head, matching exp-159 set_wk_zero:
        #   W[:, D_MODEL + head * D_HEAD : D_MODEL + (head+1) * D_HEAD] = 0
        # c_attn.weight shape: [d_model, 3*d_model]; K cols: [d_model : 2*d_model]
        saved = {}
        D_MODEL = model.config.n_embd      # 768
        D_HEAD = D_MODEL // model.config.n_head  # 64
        for layer_idx, head_idx in TOP4_TN_HEADS:
            col_start = D_MODEL + head_idx * D_HEAD
            col_end   = D_MODEL + (head_idx + 1) * D_HEAD
            W = model.transformer.h[layer_idx].attn.c_attn.weight.data
            saved[(layer_idx, head_idx)] = W[:, col_start:col_end].clone()
            W[:, col_start:col_end] = 0.0

        lp_abl = compute_logprob(ctx, target_token)

        # Restore
        for (layer_idx, head_idx), saved_w in saved.items():
            col_start = D_MODEL + head_idx * D_HEAD
            col_end   = D_MODEL + (head_idx + 1) * D_HEAD
            model.transformer.h[layer_idx].attn.c_attn.weight.data[:, col_start:col_end] = saved_w
        
        per_position_ablated[pos] = lp_abl
    
    results_by_identity.append({
        "name": target["name"],
        "company": target["company"],
        "target_token_id": int(target_token),
        "baseline": {POSITION_LABELS[p]: float(per_position[p]) for p in TARGET_POSITIONS},
        "ablated": {POSITION_LABELS[p]: float(per_position_ablated[p]) for p in TARGET_POSITIONS},
        "delta": {POSITION_LABELS[p]: float(per_position_ablated[p] - per_position[p]) 
                  for p in TARGET_POSITIONS},
    })
    
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{N_IDENTITIES} identities done")

# ─── Aggregate results ────────────────────────────────────────────────────────

print("\nAggregating...")

# Baseline position means
baseline_means = {}
for label in ["early", "middle", "late"]:
    vals = [r["baseline"][label] for r in results_by_identity]
    baseline_means[label] = float(np.mean(vals))

# K1 check: mean log-prob <= -10 AND variance < 0.10
mean_all = np.mean(list(baseline_means.values()))
var_all = np.var(list(baseline_means.values()))
k1_fires = mean_all <= -10.0 and var_all < 0.10

# H_lim: middle < early AND middle < late by >= 0.15 nats
h_lim_fires = (baseline_means["middle"] < baseline_means["early"] - 0.15 and
               baseline_means["middle"] < baseline_means["late"] - 0.15)

# H_recency: late > middle > early (or at least late > early)
h_recency_monotone = (baseline_means["late"] > baseline_means["middle"] > baseline_means["early"])
h_recency_loose = (baseline_means["late"] > baseline_means["early"])

# Delta (ablation effect) by position
delta_means = {}
for label in ["early", "middle", "late"]:
    vals = [r["delta"][label] for r in results_by_identity]
    delta_means[label] = float(np.mean(vals))

delta_abs_means = {}
for label in ["early", "middle", "late"]:
    vals = [abs(r["delta"][label]) for r in results_by_identity]
    delta_abs_means[label] = float(np.mean(vals))

# H_tn_differential: |delta(late)| - |delta(early)| > 0.10 nats
h_tn_differential = (delta_abs_means["late"] - delta_abs_means["early"]) > 0.10
# K2: max difference < 0.05 nats
k2_fires = max(delta_abs_means.values()) - min(delta_abs_means.values()) < 0.05

# Context length stats
mean_ctx_len = float(np.mean(context_lengths))
max_ctx_len = int(np.max(context_lengths))
min_ctx_len = int(np.min(context_lengths))

# ─── Print summary ────────────────────────────────────────────────────────────

print("\n=== exp-184 Results ===")
print(f"\nContext length: mean={mean_ctx_len:.0f}, min={min_ctx_len}, max={max_ctx_len} tokens")
print(f"\nBaseline log-probs by position:")
for label in ["early", "middle", "late"]:
    print(f"  {label:8s}: {baseline_means[label]:.3f} nats")
print(f"\nK1 (retrieval collapse): {'FIRES' if k1_fires else 'does not fire'}")
print(f"H_lim (U-shape, diff >= 0.15 nats): {'CONFIRMED' if h_lim_fires else 'NOT CONFIRMED'}")
print(f"H_recency (monotone late>mid>early): {'CONFIRMED' if h_recency_monotone else 'NOT CONFIRMED'}")
print(f"H_recency (loose: late>early): {'CONFIRMED' if h_recency_loose else 'NOT CONFIRMED'}")
print(f"\nDelta (ablated - baseline) by position:")
for label in ["early", "middle", "late"]:
    print(f"  {label:8s}: mean delta = {delta_means[label]:.3f}, mean |delta| = {delta_abs_means[label]:.3f}")
print(f"\nH_tn_differential (|late|-|early| > 0.10): {'CONFIRMED' if h_tn_differential else 'NOT CONFIRMED'}")
print(f"K2 (position-uniform, max diff < 0.05): {'FIRES' if k2_fires else 'does not fire'}")

# ─── Save results ─────────────────────────────────────────────────────────────

results = {
    "experiment": "exp-184",
    "date": "2026-11-19",
    "model": "gpt2",
    "prereg_commit": "attention-geometry efdcec8",
    "prereg_evidence": "git-attested — prereg.md committed and pushed before run.py was written",
    "n_identities": N_IDENTITIES,
    "context_length": {
        "mean": float(mean_ctx_len),
        "min": int(min_ctx_len),
        "max": int(max_ctx_len),
    },
    "instrument_1_baseline": {
        "early_mean": float(baseline_means["early"]),
        "middle_mean": float(baseline_means["middle"]),
        "late_mean": float(baseline_means["late"]),
        "k1_fires": bool(k1_fires),
        "h_lim_fires": bool(h_lim_fires),
        "h_recency_monotone": bool(h_recency_monotone),
        "h_recency_loose": bool(h_recency_loose),
        "diff_late_minus_early": float(baseline_means["late"] - baseline_means["early"]),
        "diff_middle_minus_late": float(baseline_means["middle"] - baseline_means["late"]),
        "diff_middle_minus_early": float(baseline_means["middle"] - baseline_means["early"]),
    },
    "instrument_2_ablation": {
        "delta_means": {k: float(v) for k, v in delta_means.items()},
        "delta_abs_means": {k: float(v) for k, v in delta_abs_means.items()},
        "h_tn_differential": bool(h_tn_differential),
        "k2_fires": bool(k2_fires),
        "diff_abs_late_minus_early": float(delta_abs_means["late"] - delta_abs_means["early"]),
    },
    "per_identity_results": results_by_identity,
}

import os
out_path = os.path.join(os.path.dirname(__file__), "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
