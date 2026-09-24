"""
exp-160 — L4H10 entity-vs-property attention analysis

Pre-registration: attention-geometry 066fa20 (committed before this script).

Hypothesis: L4H10 (early-layer text-native outlier) attends to the entity name
rather than the property token, providing an entity-anchoring signal for deeper
retrieval heads.

Ariel — 2026-09-24, afternoon MDT, solo physics room.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

PREREG_COMMIT = "066fa20"

TEXT_NATIVE = [
    (4, 10), (7, 1), (8, 2),
    (9, 4), (9, 6),
    (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9),
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load Task C battery ───────────────────────────────────────────────────────

TASK_C_PATH = (
    Path(__file__).parent.parent
    / "exp-155_textnative_content_retrieval"
    / "task_c_items.json"
)
with open(TASK_C_PATH) as f:
    task_c_data = json.load(f)
ITEMS = task_c_data["items"]
assert len(ITEMS) == 20

# ── Model ─────────────────────────────────────────────────────────────────────

print("Loading GPT-2 small (eager attention for output_attentions support)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model.to(DEVICE)
model.eval()

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_token_positions(token_ids: list[int], target_text: str, zone_text: str, zone_offset_chars: int, prompt: str) -> list[int]:
    """
    Find token positions (indices into token_ids) corresponding to target_text
    within zone_text (which starts at zone_offset_chars in prompt).

    Strategy:
    - Locate zone in the prompt by character offset
    - Decode each token prefix to find where zone_offset_chars falls, then
      search forward for target_text occurrences within the zone.
    - Returns list of token indices (0-indexed into full token sequence).
    """
    # Build cumulative character offsets for each token
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    cum_chars = []
    pos = 0
    for ts in token_strs:
        cum_chars.append(pos)
        pos += len(ts)
    # Total chars
    cum_chars.append(pos)

    zone_end_chars = zone_offset_chars + len(zone_text)

    # Find positions of target_text within the zone
    positions = []
    search_start = 0
    zone_lower = zone_text.lower()
    target_lower = target_text.lower().strip()
    while True:
        idx = zone_lower.find(target_lower, search_start)
        if idx == -1:
            break
        # Absolute char position in prompt
        abs_start = zone_offset_chars + idx
        abs_end = abs_start + len(target_lower)
        # Find all tokens that overlap [abs_start, abs_end)
        for ti, (cs, ce) in enumerate(zip(cum_chars[:-1], cum_chars[1:])):
            if cs < abs_end and ce > abs_start:
                positions.append(ti)
        search_start = idx + 1

    return positions


def run_item(item: dict) -> dict:
    """
    For one Task C item, extract attention weights from the last position to:
    - entity tokens in the cue zone
    - entity tokens in the setup zone
    - property token in the setup zone
    Returns per-head attention sums for each zone.
    """
    setup = item["setup"]
    filler = item["filler"]
    cue = item["cue"]
    entity = item["entity"]
    property_str = item["property"]
    target_token = item["target_token"]  # e.g. " crimson"

    # Build full prompt and record zone character offsets
    prompt = setup + " " + filler + " " + cue
    setup_offset = 0
    setup_len = len(setup)
    # filler starts at setup_len + 1 (the space)
    cue_offset = setup_len + 1 + len(filler) + 1
    cue_len = len(cue)

    # Tokenize
    tok_ids = tokenizer.encode(prompt)
    n_tokens = len(tok_ids)

    # Find entity token positions in cue zone
    entity_cue_positions = find_token_positions(
        tok_ids, entity, cue, cue_offset, prompt
    )
    # Find entity token positions in setup zone
    entity_setup_positions = find_token_positions(
        tok_ids, entity, setup, setup_offset, prompt
    )
    # Find property token positions in setup zone
    # The property token in the target is " crimson" (space-prefixed); search for
    # "crimson" without leading space in setup text
    property_setup_positions = find_token_positions(
        tok_ids, property_str.strip(), setup, setup_offset, prompt
    )

    # Run forward pass with output_attentions=True
    input_ids = torch.tensor([tok_ids], device=DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    # outputs.attentions: tuple of n_layers tensors, each [1, n_heads, seq, seq]
    # We want attention FROM the last token TO each position
    attn_weights = outputs.attentions  # 12 layers

    result = {
        "item_id": item["id"],
        "entity": entity,
        "property": property_str,
        "seq_len": n_tokens,
        "entity_cue_positions": entity_cue_positions,
        "entity_setup_positions": entity_setup_positions,
        "property_setup_positions": property_setup_positions,
        "n_entity_cue_tokens": len(entity_cue_positions),
        "n_entity_setup_tokens": len(entity_setup_positions),
        "n_property_tokens": len(property_setup_positions),
        "per_head": {},
    }

    last_pos = n_tokens - 1  # attention FROM this position

    for (layer, head) in TEXT_NATIVE:
        # attn_weights[layer] shape: [1, n_heads, seq, seq]
        # attn from last_pos to position j: attn_weights[layer][0, head, last_pos, j]
        attn_row = attn_weights[layer][0, head, last_pos, :].cpu().float().numpy()

        # Sum attention on entity-cue tokens
        attn_entity_cue = float(sum(attn_row[p] for p in entity_cue_positions)) if entity_cue_positions else 0.0
        # Sum attention on entity-setup tokens
        attn_entity_setup = float(sum(attn_row[p] for p in entity_setup_positions)) if entity_setup_positions else 0.0
        # Sum attention on property tokens
        attn_property = float(sum(attn_row[p] for p in property_setup_positions)) if property_setup_positions else 0.0

        head_key = f"L{layer}H{head}"
        result["per_head"][head_key] = {
            "attn_entity_cue": attn_entity_cue,
            "attn_entity_setup": attn_entity_setup,
            "attn_property": attn_property,
        }

    return result


# ── Main run ──────────────────────────────────────────────────────────────────

print("Running entity-vs-property attention analysis on 20 Task C items...")
item_results = []
for i, item in enumerate(ITEMS):
    r = run_item(item)
    item_results.append(r)
    print(
        f"  {item['id']:5s}  entity={item['entity']:12s}  property={item['property']:12s}  "
        f"cue_positions={r['entity_cue_positions']}  "
        f"L4H10: entity_cue={r['per_head']['L4H10']['attn_entity_cue']:.4f}  "
        f"entity_setup={r['per_head']['L4H10']['attn_entity_setup']:.4f}  "
        f"property={r['per_head']['L4H10']['attn_property']:.4f}"
    )

# ── Aggregate per-head means ──────────────────────────────────────────────────

print("\nAggregating per-head means...")
head_keys = [f"L{l}H{h}" for (l, h) in TEXT_NATIVE]

head_means = {}
for hk in head_keys:
    entity_cue_vals = [r["per_head"][hk]["attn_entity_cue"] for r in item_results]
    entity_setup_vals = [r["per_head"][hk]["attn_entity_setup"] for r in item_results]
    property_vals = [r["per_head"][hk]["attn_property"] for r in item_results]
    head_means[hk] = {
        "mean_attn_entity_cue": float(np.mean(entity_cue_vals)),
        "mean_attn_entity_setup": float(np.mean(entity_setup_vals)),
        "mean_attn_property": float(np.mean(property_vals)),
        "ratio_entity_cue_over_property": float(
            np.mean(entity_cue_vals) / (np.mean(property_vals) + 1e-6)
        ),
    }

# ── Hypothesis tests ──────────────────────────────────────────────────────────

l4h10 = head_means["L4H10"]
l4h10_ratio = l4h10["ratio_entity_cue_over_property"]

# H_entity_anchor: L4H10 mean_attn_entity_cue > mean_attn_property, ratio >= 2.0
h_entity_anchor = (
    l4h10["mean_attn_entity_cue"] > l4h10["mean_attn_property"]
    and l4h10_ratio >= 2.0
)
k1_fires = not h_entity_anchor

# H_entity_specificity: L4H10 ratio in top quartile (rank <= 4 of 16)
all_ratios = [(hk, head_means[hk]["ratio_entity_cue_over_property"]) for hk in head_keys]
all_ratios_sorted = sorted(all_ratios, key=lambda x: x[1], reverse=True)
l4h10_rank = next(i+1 for i, (hk, _) in enumerate(all_ratios_sorted) if hk == "L4H10")
h_entity_specificity = l4h10_rank <= 4
k2_fires = not h_entity_specificity

# H_entity_setup (exploratory): L4H10 attn_entity_setup >= attn_property
h_entity_setup = l4h10["mean_attn_entity_setup"] >= l4h10["mean_attn_property"]

# ── Report ────────────────────────────────────────────────────────────────────

print("\n=== RESULTS ===")
print(f"\nL4H10:")
print(f"  mean attn on entity-cue tokens:   {l4h10['mean_attn_entity_cue']:.5f}")
print(f"  mean attn on entity-setup tokens: {l4h10['mean_attn_entity_setup']:.5f}")
print(f"  mean attn on property tokens:     {l4h10['mean_attn_property']:.5f}")
print(f"  entity_cue / property ratio:      {l4h10_ratio:.2f}")

print(f"\nH_entity_anchor: {'CONFIRMED' if h_entity_anchor else 'DEAD'}")
print(f"  (ratio={l4h10_ratio:.2f} {'≥' if l4h10_ratio >= 2.0 else '<'} 2.0; "
      f"entity_cue {'>' if l4h10['mean_attn_entity_cue'] > l4h10['mean_attn_property'] else '≤'} property)")
print(f"  K1 {'fires' if k1_fires else 'does not fire'}")

print(f"\nH_entity_specificity: {'CONFIRMED' if h_entity_specificity else 'DEAD'}")
print(f"  L4H10 rank in ratio distribution: {l4h10_rank}/16 (top quartile = rank ≤ 4)")
print(f"  K2 {'fires' if k2_fires else 'does not fire'}")

print(f"\nH_entity_setup (exploratory): {'CONFIRMED' if h_entity_setup else 'NOT CONFIRMED'}")
print(f"  L4H10 entity-setup={l4h10['mean_attn_entity_setup']:.5f} vs property={l4h10['mean_attn_property']:.5f}")

print("\nAll-head entity/property ratios (ranked):")
for rank, (hk, ratio) in enumerate(all_ratios_sorted, 1):
    marker = " ← L4H10" if hk == "L4H10" else ""
    print(f"  {rank:2d}. {hk:8s}  entity_cue/property = {ratio:6.2f}{marker}")

print("\nAll-head summary:")
print(f"  {'Head':8s}  {'entity_cue':11s}  {'entity_setup':13s}  {'property':10s}  ratio")
for hk in head_keys:
    m = head_means[hk]
    marker = " ←" if hk == "L4H10" else ""
    print(
        f"  {hk:8s}  {m['mean_attn_entity_cue']:.5f}     {m['mean_attn_entity_setup']:.5f}       "
        f"{m['mean_attn_property']:.5f}    {m['ratio_entity_cue_over_property']:.2f}{marker}"
    )

# ── Save results ──────────────────────────────────────────────────────────────

out = {
    "experiment": "exp-160",
    "prereg_commit": PREREG_COMMIT,
    "model": "gpt2",
    "text_native_heads": [f"L{l}H{h}" for (l, h) in TEXT_NATIVE],
    "item_results": item_results,
    "head_means": head_means,
    "entity_cue_property_ratios_ranked": [
        {"head": hk, "ratio": ratio, "rank": i+1}
        for i, (hk, ratio) in enumerate(all_ratios_sorted)
    ],
    "l4h10_rank_in_ratio_distribution": l4h10_rank,
    "hypothesis_results": {
        "H_entity_anchor": {
            "result": "confirmed" if h_entity_anchor else "dead",
            "l4h10_mean_attn_entity_cue": l4h10["mean_attn_entity_cue"],
            "l4h10_mean_attn_property": l4h10["mean_attn_property"],
            "ratio": l4h10_ratio,
            "k1_fires": k1_fires,
        },
        "H_entity_specificity": {
            "result": "confirmed" if h_entity_specificity else "dead",
            "l4h10_rank": l4h10_rank,
            "k2_fires": k2_fires,
        },
        "H_entity_setup_exploratory": {
            "result": "confirmed" if h_entity_setup else "not_confirmed",
            "l4h10_mean_attn_entity_setup": l4h10["mean_attn_entity_setup"],
            "l4h10_mean_attn_property": l4h10["mean_attn_property"],
        },
    },
    "verdict": "pending",
    "headline": "pending",
}

OUT_PATH = Path(__file__).parent / "results.json"
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults written to {OUT_PATH}")
