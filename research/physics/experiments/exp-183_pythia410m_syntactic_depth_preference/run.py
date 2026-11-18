"""
exp-183: Pythia-410m text-native Δ-window heads — syntactic depth preference
via spaCy dependency parse

Pre-registration: attention-geometry 73b74c9
Question: Geometry — instrument refinement/exploration (Direction block pause applies;
logged as such)

Declared prior: H_flat — moderate confidence (~60%). By analogy with exp-182's K3
fire (no semantic targeting), expect no syntactic depth selectivity. The absolute-
key-position mechanism (exp-138) suggests the tn character is about sequence
position, not linguistic depth.
"""

import json
import math
import numpy as np
import torch
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime, timezone

# ─── constants ──────────────────────────────────────────────────────────────

MODEL_ID = "EleutherAI/pythia-410m"
N_LAYERS = 24
N_HEADS = 16
SEQ_LEN = 128
N_PASSAGES = 100
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# From exp-118: 16 text-native Δ-window heads [layer, head]
TEXTNATIVE_HEADS = [
    [0,3],[1,8],[2,8],[3,2],[3,7],[3,14],[4,12],[7,11],
    [8,15],[9,15],[12,3],[12,13],[14,15],[19,15],[21,11],[22,1]
]

# Depth bins
SHALLOW_MAX = 1   # depth 0-1
MEDIUM_MAX = 3    # depth 2-3
# DEEP = depth 4+


# ─── helpers ────────────────────────────────────────────────────────────────

def get_token_depth(token):
    """Compute spaCy dependency parse depth: edges from root (root depth = 0)."""
    depth = 0
    current = token
    while current.head.i != current.i:
        current = current.head
        depth += 1
        if depth > 50:  # safety cap for malformed parses
            break
    return depth


def depth_bin(depth):
    """Map integer depth to bin name."""
    if depth <= SHALLOW_MAX:
        return "SHALLOW"
    elif depth <= MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "DEEP"


def select_control_heads(textnative_heads, heads_list):
    """Select 16 control heads: one per text-native head's layer, highest |delta-0.25|
    outside [0.20, 0.30], excluding text-native heads."""
    tn_set = set(map(tuple, textnative_heads))
    delta_map = {(h['layer'], h['head']): h['delta'] for h in heads_list}
    controls = []
    for layer, head in textnative_heads:
        candidates = []
        for h in range(N_HEADS):
            if (layer, h) not in tn_set:
                d = delta_map.get((layer, h), None)
                if d is not None and not (0.20 <= d <= 0.30):
                    candidates.append((h, d))
        if candidates:
            best = max(candidates, key=lambda x: abs(x[1] - 0.25))
            controls.append([layer, best[0]])
        else:
            for h in range(N_HEADS):
                if (layer, h) not in tn_set:
                    controls.append([layer, h])
                    break
    return controls


def align_tokens_to_depth(token_ids, tokenizer, nlp):
    """
    Decode each GPT-NeoX token and align to spaCy dependency depth.

    Returns: list of dicts per token position, each with:
        - 'depth': int, spaCy dependency depth (0 = root)
        - 'depth_bin': 'SHALLOW', 'MEDIUM', or 'DEEP'
        - 'aligned': bool
    """
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    doc = nlp(full_text)
    spacy_tokens = list(doc)

    # Pre-compute spaCy token depths
    spacy_depths = [get_token_depth(t) for t in spacy_tokens]

    result = []
    for i, (char_start, char_end) in enumerate(offsets):
        if i >= len(token_ids):
            break

        # Find spaCy token with maximum character overlap
        best_spacy_idx = None
        best_overlap = 0
        for s_idx, stok in enumerate(spacy_tokens):
            s_start = stok.idx
            s_end = stok.idx + len(stok.text)
            overlap = max(0, min(char_end, s_end) - max(char_start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spacy_idx = s_idx

        if best_spacy_idx is not None and best_overlap > 0:
            d = spacy_depths[best_spacy_idx]
            result.append({
                'depth': d,
                'depth_bin': depth_bin(d),
                'aligned': True,
            })
        else:
            # Unaligned: assign MEDIUM as default (least extreme)
            result.append({'depth': 2, 'depth_bin': 'MEDIUM', 'aligned': False})

    # Pad to SEQ_LEN
    while len(result) < SEQ_LEN:
        result.append({'depth': 2, 'depth_bin': 'MEDIUM', 'aligned': False})

    return result[:SEQ_LEN]


def compute_depth_preference_ratios(attn_passages, depth_bin_passages, n_actual):
    """
    Compute preference ratios for SHALLOW, MEDIUM, DEEP per head.

    attn_passages: (n_passages, seq_len, seq_len) — causal attention weights
    depth_bin_passages: list of n_passages × seq_len lists of dicts

    Returns: {'SHALLOW': float, 'MEDIUM': float, 'DEEP': float}
    """
    eps = 1e-6
    total_attn = {'SHALLOW': 0.0, 'MEDIUM': 0.0, 'DEEP': 0.0}
    total_freq = {'SHALLOW': 0.0, 'MEDIUM': 0.0, 'DEEP': 0.0}
    total_query_positions = 0

    for p in range(n_actual):
        a = attn_passages[p]  # (seq_len, seq_len)
        bins = [t['depth_bin'] for t in depth_bin_passages[p]]

        freq = {}
        for b in ('SHALLOW', 'MEDIUM', 'DEEP'):
            freq[b] = bins.count(b) / SEQ_LEN

        for i in range(1, SEQ_LEN):
            row = a[i, :i+1]
            if row.sum() < 1e-8:
                continue
            total_query_positions += 1
            for b in ('SHALLOW', 'MEDIUM', 'DEEP'):
                bin_attn = sum(row[j] for j in range(i+1) if bins[j] == b)
                total_attn[b] += bin_attn
                total_freq[b] += freq[b]

    pref = {}
    for b in ('SHALLOW', 'MEDIUM', 'DEEP'):
        mean_attn = total_attn[b] / max(total_query_positions, 1)
        mean_freq = total_freq[b] / max(total_query_positions, 1)
        pref[b] = mean_attn / (mean_freq + eps)

    return pref


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_var = ((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2)
    pooled_std = math.sqrt(pooled_var)
    return abs(float(np.mean(a)) - float(np.mean(b))) / (pooled_std + 1e-10)


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading Pythia-410m...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        output_attentions=True,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )
    model.eval()

    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading WikiText-103 test split...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    all_text = " ".join(line for line in dataset["text"] if line.strip())
    encoded = tokenizer(all_text, return_tensors="pt", add_special_tokens=False)
    input_ids_full = encoded["input_ids"][0]
    print(f"Total test tokens: {len(input_ids_full)}")

    # Same non-overlapping passages as exp-181/182
    passages_ids = []
    for i in range(N_PASSAGES):
        start = i * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(input_ids_full):
            break
        passages_ids.append(input_ids_full[start:end].tolist())
    n_actual = len(passages_ids)
    print(f"Using {n_actual} passages of {SEQ_LEN} tokens")

    # ── spaCy depth classification for each passage ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running spaCy dependency depth classification...")
    depth_bin_passages = []
    shallow_counts = []
    medium_counts = []
    deep_counts = []
    alignment_failures_total = 0
    total_tokens_total = 0

    for p_idx, ids in enumerate(passages_ids):
        if p_idx % 20 == 0:
            print(f"  spaCy passage {p_idx}/{n_actual}...")
        depth_info = align_tokens_to_depth(ids, tokenizer, nlp)
        failures = sum(1 for t in depth_info if not t['aligned'])
        alignment_failures_total += failures
        total_tokens_total += SEQ_LEN
        shallow_count = sum(1 for t in depth_info if t['depth_bin'] == 'SHALLOW')
        medium_count  = sum(1 for t in depth_info if t['depth_bin'] == 'MEDIUM')
        deep_count    = sum(1 for t in depth_info if t['depth_bin'] == 'DEEP')
        shallow_counts.append(shallow_count)
        medium_counts.append(medium_count)
        deep_counts.append(deep_count)
        depth_bin_passages.append(depth_info)

    shallow_frac = sum(shallow_counts) / (n_actual * SEQ_LEN)
    medium_frac  = sum(medium_counts)  / (n_actual * SEQ_LEN)
    deep_frac    = sum(deep_counts)    / (n_actual * SEQ_LEN)
    alignment_failure_rate = alignment_failures_total / total_tokens_total
    print(f"Depth bin fractions: shallow={shallow_frac:.4f}, medium={medium_frac:.4f}, deep={deep_frac:.4f}")
    print(f"Alignment failure rate: {alignment_failure_rate:.4f}")

    # Kill conditions
    k1_fires = shallow_frac < 0.05
    k2_fires = deep_frac < 0.05

    # ── load control heads from exp-118 ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading exp-118 for control head selection...")
    exp118_path = "research/physics/experiments/exp-118_wikitext_census_cross_family/results.json"
    with open(exp118_path) as f:
        exp118_results = json.load(f)

    pythia_410m_entry = None
    for entry in exp118_results:
        if "pythia-410m" in entry.get("model", "").lower():
            pythia_410m_entry = entry
            break
    if pythia_410m_entry is None:
        raise ValueError("Pythia-410m entry not found in exp-118 results")

    heads_list = pythia_410m_entry.get("heads", [])
    control_heads = select_control_heads(TEXTNATIVE_HEADS, heads_list)
    print(f"Control heads: {control_heads}")

    # All heads to process
    all_heads_to_process = set(map(tuple, TEXTNATIVE_HEADS)) | set(map(tuple, control_heads))

    # ── forward passes ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running forward passes...")
    attn_accum = {(l, h): [] for l, h in all_heads_to_process}

    with torch.no_grad():
        for p_idx, ids in enumerate(passages_ids):
            if p_idx % 10 == 0:
                print(f"  Forward pass {p_idx}/{n_actual}...")
            ids_tensor = torch.tensor(ids).unsqueeze(0).to(DEVICE)
            outputs = model(ids_tensor, output_attentions=True)
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                layer_np = layer_attn[0].cpu().numpy()  # (n_heads, seq_len, seq_len)
                for head_idx in range(N_HEADS):
                    if (layer_idx, head_idx) in attn_accum:
                        attn_accum[(layer_idx, head_idx)].append(layer_np[head_idx])

    print(f"[{datetime.now(timezone.utc).isoformat()}] Computing depth preference ratios...")

    # ── compute preference ratios per head ──
    pref_results = {}
    for l, h in all_heads_to_process:
        attn_passages_arr = np.stack(attn_accum[(l, h)])  # (n_passages, seq_len, seq_len)
        pref = compute_depth_preference_ratios(attn_passages_arr, depth_bin_passages, n_actual)
        pref_results[(l, h)] = pref

    # ── extract tn and ctrl preference ratios ──
    tn_shallow = [pref_results[(l,h)]['SHALLOW'] for l,h in TEXTNATIVE_HEADS]
    tn_medium  = [pref_results[(l,h)]['MEDIUM']  for l,h in TEXTNATIVE_HEADS]
    tn_deep    = [pref_results[(l,h)]['DEEP']    for l,h in TEXTNATIVE_HEADS]

    ctrl_shallow = [pref_results[(tuple(lh))]['SHALLOW'] for lh in control_heads]
    ctrl_medium  = [pref_results[(tuple(lh))]['MEDIUM']  for lh in control_heads]
    ctrl_deep    = [pref_results[(tuple(lh))]['DEEP']    for lh in control_heads]

    # ── hypothesis testing ──
    # H_deep: ≥10/16 tn heads deep_pref > 1.10
    h_deep_count = sum(1 for x in tn_deep if x > 1.10)
    h_deep_fires = h_deep_count >= 10

    # H_shallow: ≥10/16 tn heads shallow_pref > 1.10
    h_shallow_count = sum(1 for x in tn_shallow if x > 1.10)
    h_shallow_fires = h_shallow_count >= 10

    # H_flat: ≥10/16 tn heads have all |bin_pref - 1.0| < 0.15
    h_flat_count = sum(
        1 for s, m, d in zip(tn_shallow, tn_medium, tn_deep)
        if abs(s - 1.0) < 0.15 and abs(m - 1.0) < 0.15 and abs(d - 1.0) < 0.15
    )
    h_flat_fires = h_flat_count >= 10

    # H_contrast: d > 0.3 for any bin
    d_shallow = cohens_d(tn_shallow, ctrl_shallow)
    d_medium  = cohens_d(tn_medium,  ctrl_medium)
    d_deep    = cohens_d(tn_deep,    ctrl_deep)
    h_contrast_fires = d_shallow > 0.3 or d_medium > 0.3 or d_deep > 0.3

    # K3: no population difference
    k3_fires = d_shallow < 0.3 and d_medium < 0.3 and d_deep < 0.3

    # ── build results ──
    results = {
        "exp": "exp-183",
        "date": "2026-11-18",
        "model": MODEL_ID,
        "prereg_commit": "73b74c9",
        "prereg_evidence": "git-attested — registration pushed before run.py was written",
        "n_passages": n_actual,
        "seq_len": SEQ_LEN,
        "textnative_heads": TEXTNATIVE_HEADS,
        "control_heads": control_heads,
        "depth_stats": {
            "shallow_frac": float(shallow_frac),
            "medium_frac":  float(medium_frac),
            "deep_frac":    float(deep_frac),
            "alignment_failure_rate": float(alignment_failure_rate),
        },
        "textnative_preference": {
            "shallow": tn_shallow,
            "medium":  tn_medium,
            "deep":    tn_deep,
        },
        "control_preference": {
            "shallow": ctrl_shallow,
            "medium":  ctrl_medium,
            "deep":    ctrl_deep,
        },
        "summary_stats": {
            "tn_shallow_mean": float(np.mean(tn_shallow)),
            "tn_shallow_std":  float(np.std(tn_shallow)),
            "tn_medium_mean":  float(np.mean(tn_medium)),
            "tn_medium_std":   float(np.std(tn_medium)),
            "tn_deep_mean":    float(np.mean(tn_deep)),
            "tn_deep_std":     float(np.std(tn_deep)),
            "ctrl_shallow_mean": float(np.mean(ctrl_shallow)),
            "ctrl_medium_mean":  float(np.mean(ctrl_medium)),
            "ctrl_deep_mean":    float(np.mean(ctrl_deep)),
            "d_shallow": float(d_shallow),
            "d_medium":  float(d_medium),
            "d_deep":    float(d_deep),
        },
        "hypotheses": {
            "H_deep": {
                "fires": bool(h_deep_fires),
                "count": int(h_deep_count),
                "threshold": "≥10/16 tn heads deep_pref > 1.10",
                "verdict": "CONFIRMED" if h_deep_fires else "NOT CONFIRMED",
            },
            "H_shallow": {
                "fires": bool(h_shallow_fires),
                "count": int(h_shallow_count),
                "threshold": "≥10/16 tn heads shallow_pref > 1.10",
                "verdict": "CONFIRMED" if h_shallow_fires else "NOT CONFIRMED",
            },
            "H_flat": {
                "fires": bool(h_flat_fires),
                "count": int(h_flat_count),
                "threshold": "≥10/16 tn heads all |depth_pref - 1.0| < 0.15",
                "verdict": "CONFIRMED" if h_flat_fires else "NOT CONFIRMED",
            },
            "H_contrast": {
                "fires": bool(h_contrast_fires),
                "d_shallow": float(d_shallow),
                "d_medium":  float(d_medium),
                "d_deep":    float(d_deep),
                "threshold": "d > 0.3 for any depth bin",
                "verdict": "CONFIRMED" if h_contrast_fires else "NOT CONFIRMED",
            },
        },
        "kill_conditions": {
            "K1_shallow_undersampling": {
                "fires": bool(k1_fires),
                "shallow_frac": float(shallow_frac),
                "threshold": 0.05,
            },
            "K2_deep_undersampling": {
                "fires": bool(k2_fires),
                "deep_frac": float(deep_frac),
                "threshold": 0.05,
            },
            "K3_no_population_difference": {
                "fires": bool(k3_fires),
                "d_shallow": float(d_shallow),
                "d_medium":  float(d_medium),
                "d_deep":    float(d_deep),
            },
        },
        "per_head_preference": {
            f"{l}_{h}": {
                "shallow_pref": float(pref_results[(l,h)]['SHALLOW']),
                "medium_pref":  float(pref_results[(l,h)]['MEDIUM']),
                "deep_pref":    float(pref_results[(l,h)]['DEEP']),
                "is_textnative": [l,h] in TEXTNATIVE_HEADS,
            }
            for l, h in all_heads_to_process
        }
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    outfile = "research/physics/experiments/exp-183_pythia410m_syntactic_depth_preference/results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Results written to {outfile}")

    # ── summary ──
    print("\n=== exp-183 RESULTS SUMMARY ===")
    print(f"\nDepth bin fractions: shallow={shallow_frac:.4f}, medium={medium_frac:.4f}, deep={deep_frac:.4f}")
    print(f"Alignment failure rate: {alignment_failure_rate:.4f}")
    print(f"\nK1 (shallow_frac <0.05): {'FIRES' if k1_fires else 'does not fire'}")
    print(f"K2 (deep_frac <0.05):    {'FIRES' if k2_fires else 'does not fire'}")
    print(f"K3 (d<0.3 all bins):     {'FIRES' if k3_fires else 'does not fire'}")
    print(f"\nH_deep    (≥10/16 deep_pref>1.10):   {'CONFIRMED' if h_deep_fires else 'NOT CONFIRMED'} ({h_deep_count}/16)")
    print(f"H_shallow (≥10/16 shallow_pref>1.10): {'CONFIRMED' if h_shallow_fires else 'NOT CONFIRMED'} ({h_shallow_count}/16)")
    print(f"H_flat    (≥10/16 all|pref-1|<0.15):  {'CONFIRMED' if h_flat_fires else 'NOT CONFIRMED'} ({h_flat_count}/16)")
    print(f"H_contrast (d>0.3 any bin):            {'CONFIRMED' if h_contrast_fires else 'NOT CONFIRMED'} "
          f"(d_shallow={d_shallow:.3f}, d_medium={d_medium:.3f}, d_deep={d_deep:.3f})")
    print(f"\nTn shallow_pref: mean={np.mean(tn_shallow):.4f} ± {np.std(tn_shallow):.4f}")
    print(f"Tn medium_pref:  mean={np.mean(tn_medium):.4f} ± {np.std(tn_medium):.4f}")
    print(f"Tn deep_pref:    mean={np.mean(tn_deep):.4f} ± {np.std(tn_deep):.4f}")
    print(f"Ctrl shallow_pref: mean={np.mean(ctrl_shallow):.4f}")
    print(f"Ctrl medium_pref:  mean={np.mean(ctrl_medium):.4f}")
    print(f"Ctrl deep_pref:    mean={np.mean(ctrl_deep):.4f}")
    print(f"\nPer-head shallow_pref (tn): {[f'{x:.3f}' for x in tn_shallow]}")
    print(f"Per-head medium_pref  (tn): {[f'{x:.3f}' for x in tn_medium]}")
    print(f"Per-head deep_pref    (tn): {[f'{x:.3f}' for x in tn_deep]}")


if __name__ == "__main__":
    main()
