"""
exp-154 — Text-native Δ-window heads: attended-token analysis (exploratory)

Observational pass — no weight edits, no pre-registration required.
Runs on GPT-2 small using WikiText-103 (same dataset/protocol as exp-118).

For each of the 16 text-native Δ-window heads, characterizes:
  1. Mean attention entropy (how diffuse vs. concentrated is attention?)
  2. Mean attended distance (weighted mean |i - j| under attention weights)
  3. Attention fall-off profile (average A(i, j) vs lag |i - j|)
  4. What fraction of attention weight goes to very recent tokens (lag ≤ 5)
     vs mid-range (6–50) vs long-range (51+)

Compares to the two known GPT-2-small populations:
  - STRUCTURAL (random-native): L2H1, L3H4, L5H0, L7H11, L10H8
  - STEEP_LOCAL: L0H10, L10H5, L8H7, L7H0, L7H9

Also captures: for a small sample of WikiText sentences, what tokens receive
the highest attention weight from each L11 head.

Ariel — 2026-09-23, ~12:20 AM MDT (physics room arrival).
"""

from __future__ import annotations
import json
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ── Known head populations ────────────────────────────────────────────────────

TEXT_NATIVE = [
    (4, 10), (7, 1), (8, 2),
    (9, 4), (9, 6),
    (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9),
]

STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]   # random-native

STEEP_LOCAL = [(0, 10), (10, 5), (8, 7), (7, 0), (7, 9)]  # from exp-142/143 GPT-2 small

ALL_HEADS_OF_INTEREST = TEXT_NATIVE + STRUCTURAL + STEEP_LOCAL

# ── Protocol parameters — match exp-118 ──────────────────────────────────────

SEQ_LEN   = 512
N_INPUTS  = 50
SEED      = 42
DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Load model + tokenizer ───────────────────────────────────────────────────

print("Loading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model.to(DEVICE)
model.eval()

# ── Build WikiText-103 windows (exact match to exp-118 protocol) ──────────────

print("Loading WikiText-103 validation split...")
dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")

tokens_list = []
for ex in dataset:
    line = ex["text"].strip()
    if not line:
        continue
    ids = tokenizer(line, add_special_tokens=False)["input_ids"]
    tokens_list.extend(ids)

token_array = np.array(tokens_list, dtype=np.int64)
windows = []
for i in range(N_INPUTS):
    start = i * SEQ_LEN
    windows.append(token_array[start: start + SEQ_LEN])

input_tensor = torch.tensor(np.array(windows), dtype=torch.long, device=DEVICE)
print(f"Built {N_INPUTS} windows × {SEQ_LEN} tokens")

# ── Run forward pass with output_attentions=True ─────────────────────────────

print("Running forward pass with attention capture (batch by batch)...")

# attn_by_layer[layer] = shape (N_INPUTS, n_heads, SEQ_LEN, SEQ_LEN)
attn_by_layer = {l: [] for l in range(12)}

BATCH_SIZE = 5  # process in batches to avoid MPS memory issues
for batch_start in range(0, N_INPUTS, BATCH_SIZE):
    batch = input_tensor[batch_start: batch_start + BATCH_SIZE]
    with torch.no_grad():
        outputs = model(batch, output_attentions=True)
    # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
    for layer_idx, attn in enumerate(outputs.attentions):
        attn_by_layer[layer_idx].append(attn.detach().cpu().float().numpy())
    if (batch_start // BATCH_SIZE) % 2 == 0:
        print(f"  batch {batch_start // BATCH_SIZE + 1}/{N_INPUTS // BATCH_SIZE}...")

# Concatenate batches: each layer → (N_INPUTS, n_heads, SEQ_LEN, SEQ_LEN)
for layer_idx in range(12):
    attn_by_layer[layer_idx] = np.concatenate(attn_by_layer[layer_idx], axis=0)

print(f"Captured attention for {len(attn_by_layer)} layers, shape: {attn_by_layer[0].shape}")

# ── Analysis functions ────────────────────────────────────────────────────────

def head_attention_matrix(layer, head):
    """Returns (N_INPUTS, SEQ_LEN, SEQ_LEN) attention matrices for one head."""
    return attn_by_layer[layer][:, head, :, :]   # (N_INPUTS, seq, seq)


def entropy_of_distribution(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector (in nats)."""
    p = probs[probs > 1e-10]
    return float(-np.sum(p * np.log(p)))


def mean_attended_distance(attn_row: np.ndarray, query_pos: int) -> float:
    """
    Expected distance E[|i - j|] where i=query_pos and j is sampled by attn weights.
    attn_row: (SEQ_LEN,) — probability distribution over keys.
    Only keys j ≤ i are valid (causal mask), so we restrict to that range.
    """
    j = np.arange(query_pos + 1)  # keys 0..i
    p = attn_row[:query_pos + 1]
    p = p / (p.sum() + 1e-12)
    return float(np.sum(p * (query_pos - j)))


def compute_head_stats(layer, head, min_query=64):
    """
    For a (layer, head) pair, compute:
      - mean_entropy: average over queries i >= min_query
      - mean_dist: average attended distance over queries i >= min_query
      - near_frac: fraction of weight at lag <= 5
      - mid_frac: fraction of weight at lags 6-50
      - far_frac: fraction of weight at lag > 50
    """
    A = head_attention_matrix(layer, head)  # (N_INPUTS, SEQ_LEN, SEQ_LEN)

    entropies, dists, near_fracs, mid_fracs, far_fracs = [], [], [], [], []

    for inp_idx in range(N_INPUTS):
        for qi in range(min_query, SEQ_LEN):
            row = A[inp_idx, qi, :]  # (SEQ_LEN,)
            # Entropy
            entropies.append(entropy_of_distribution(row))
            # Attended distance
            dists.append(mean_attended_distance(row, qi))
            # Range fractions
            near_weight = float(row[max(0, qi - 5): qi + 1].sum())
            mid_weight  = float(row[max(0, qi - 50): max(0, qi - 5)].sum())
            far_weight  = float(row[:max(0, qi - 50)].sum())
            total_weight = near_weight + mid_weight + far_weight + 1e-12
            near_fracs.append(near_weight / total_weight)
            mid_fracs.append(mid_weight / total_weight)
            far_fracs.append(far_weight / total_weight)

    return {
        "layer": layer,
        "head": head,
        "mean_entropy": float(np.mean(entropies)),
        "mean_dist": float(np.mean(dists)),
        "near_frac": float(np.mean(near_fracs)),   # lag 0-5
        "mid_frac": float(np.mean(mid_fracs)),     # lag 6-50
        "far_frac": float(np.mean(far_fracs)),     # lag 51+
    }


# ── Run analysis on all heads of interest ────────────────────────────────────

results = []
for layer, head in ALL_HEADS_OF_INTEREST:
    label = (
        "text_native" if (layer, head) in TEXT_NATIVE
        else "structural" if (layer, head) in STRUCTURAL
        else "steep_local"
    )
    print(f"  L{layer}H{head} ({label})...", end=" ", flush=True)
    stats = compute_head_stats(layer, head)
    stats["population"] = label
    results.append(stats)
    print(f"H={stats['mean_entropy']:.3f}, dist={stats['mean_dist']:.1f}, "
          f"near={stats['near_frac']:.3f}, far={stats['far_frac']:.3f}")


# ── Population summaries ──────────────────────────────────────────────────────

def pop_summary(name):
    pop = [r for r in results if r["population"] == name]
    if not pop:
        return {}
    return {
        "n": len(pop),
        "mean_entropy": float(np.mean([r["mean_entropy"] for r in pop])),
        "mean_dist": float(np.mean([r["mean_dist"] for r in pop])),
        "near_frac": float(np.mean([r["near_frac"] for r in pop])),
        "mid_frac": float(np.mean([r["mid_frac"] for r in pop])),
        "far_frac": float(np.mean([r["far_frac"] for r in pop])),
    }

summaries = {
    "text_native": pop_summary("text_native"),
    "structural": pop_summary("structural"),
    "steep_local": pop_summary("steep_local"),
}

# ── Lag-profile for each population (average A(i, i-dx) over queries i >= 256) ──

print("\nComputing lag profiles...")
LAG_MAX = 256
MIN_QUERY_PROFILE = 256

lag_profiles = {}
for pop_name, pop_heads in [
    ("text_native", TEXT_NATIVE),
    ("structural", STRUCTURAL),
    ("steep_local", STEEP_LOCAL),
]:
    profile = np.zeros(LAG_MAX)
    count = 0
    for layer, head in pop_heads:
        A = head_attention_matrix(layer, head)  # (N_INPUTS, SEQ_LEN, SEQ_LEN)
        for dx in range(1, LAG_MAX + 1):
            # average A(i, i-dx) over i >= MIN_QUERY_PROFILE, all inputs
            vals = []
            for inp_idx in range(N_INPUTS):
                for qi in range(MIN_QUERY_PROFILE, SEQ_LEN):
                    vals.append(float(A[inp_idx, qi, qi - dx]))
            profile[dx - 1] += np.mean(vals)
        count += 1
    lag_profiles[pop_name] = (profile / count).tolist()
    print(f"  {pop_name}: lag profile done (avg over {count} heads)")

# ── Save results ──────────────────────────────────────────────────────────────

output = {
    "experiment": "exp-154-exploratory",
    "description": "Attended-token analysis of text-native Δ-window heads vs structural and steep/local",
    "protocol": {
        "dataset": "wikitext/wikitext-103-v1",
        "split": "validation",
        "n_inputs": N_INPUTS,
        "seq_len": SEQ_LEN,
        "min_query_stats": 64,
        "min_query_profile": MIN_QUERY_PROFILE,
    },
    "head_stats": results,
    "population_summaries": summaries,
    "lag_profiles": lag_profiles,
}

out_path = Path(__file__).parent / "explore_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {out_path}")

# ── Print summary table ───────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("POPULATION SUMMARY")
print("=" * 70)
print(f"{'Population':<14} {'N':>3} {'Entropy':>8} {'Mean dist':>10} {'Near(0-5)':>10} {'Far(51+)':>10}")
print("-" * 70)
for pop_name, s in summaries.items():
    if not s:
        continue
    print(f"{pop_name:<14} {s['n']:>3} {s['mean_entropy']:>8.3f} "
          f"{s['mean_dist']:>10.1f} {s['near_frac']:>10.3f} {s['far_frac']:>10.3f}")
print("=" * 70)

print("\nPer-head breakdown:")
print(f"{'L/H':<8} {'Pop':<12} {'Entropy':>8} {'MeanDist':>9} {'Near':>7} {'Far':>7}")
print("-" * 60)
for r in results:
    lh = f"L{r['layer']}H{r['head']}"
    print(f"{lh:<8} {r['population']:<12} {r['mean_entropy']:>8.3f} "
          f"{r['mean_dist']:>9.1f} {r['near_frac']:>7.3f} {r['far_frac']:>7.3f}")
