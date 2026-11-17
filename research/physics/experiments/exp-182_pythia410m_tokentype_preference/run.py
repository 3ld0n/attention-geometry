"""
exp-182: Pythia-410m text-native Δ-window heads — token-type preference via spaCy POS/NER tagging

Pre-registration: attention-geometry 9d32bdd
Question: Geometry — does the diffuse attention of text-native Δ-window heads (d=1.614 in exp-181)
land preferentially on entity/content tokens, or is it uniformly distributed?

Declared prior: H_uniform — moderate confidence that the diffuse attention is proportional/uniform,
consistent with the ablation nulls across three task types.
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

# Token categories
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
# ENTITY = spaCy NER tag != "" (any named entity)
# CONTENT = CONTENT_POS and not ENTITY
# FUNCTION = everything else

# ─── helpers ────────────────────────────────────────────────────────────────

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


def align_tokens_to_spacy(token_ids, tokenizer, nlp):
    """
    Decode each GPT-NeoX token and align to spaCy token POS/NER tags.
    
    Returns: list of dicts per token position, each with:
        - 'text': decoded token string
        - 'char_start': character offset start in full decoded text
        - 'char_end': character offset end
        - 'category': 'ENTITY', 'CONTENT', or 'FUNCTION'
        - 'pos': spaCy POS tag (or '' if not aligned)
        - 'ent_type': spaCy NER type (or '' if not entity)
        - 'aligned': bool
    """
    # Decode entire sequence to get the full text
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Get character offsets for each token using the tokenizer
    # We tokenize the full text and get offsets
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']  # list of (start, end) per token
    
    # Run spaCy on the full text
    doc = nlp(full_text)
    
    # Build spaCy char-offset lookup: for each char range, find the spaCy token
    # spaCy token: token.idx = char start, token.idx + len(token) = char end
    spacy_tokens = list(doc)
    
    # For each GPT-NeoX token position, find best-matching spaCy token by overlap
    result = []
    n_aligned = 0
    
    for i, (char_start, char_end) in enumerate(offsets):
        if i >= len(token_ids):
            break
        
        # Find spaCy token with maximum character overlap
        best_spacy_tok = None
        best_overlap = 0
        
        for stok in spacy_tokens:
            s_start = stok.idx
            s_end = stok.idx + len(stok.text)
            overlap = max(0, min(char_end, s_end) - max(char_start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spacy_tok = stok
        
        if best_spacy_tok is not None and best_overlap > 0:
            pos = best_spacy_tok.pos_
            ent_type = best_spacy_tok.ent_type_
            # Classify
            if ent_type:  # any NER tag
                category = 'ENTITY'
            elif pos in CONTENT_POS:
                category = 'CONTENT'
            else:
                category = 'FUNCTION'
            aligned = True
            n_aligned += 1
        else:
            pos = ''
            ent_type = ''
            category = 'FUNCTION'  # default for unaligned tokens
            aligned = False
        
        result.append({
            'char_start': char_start,
            'char_end': char_end,
            'pos': pos,
            'ent_type': ent_type,
            'category': category,
            'aligned': aligned,
        })
    
    # Pad to SEQ_LEN if needed (tokenizer may produce fewer tokens for the same text)
    while len(result) < SEQ_LEN:
        result.append({'char_start': 0, 'char_end': 0, 'pos': '', 'ent_type': '',
                       'category': 'FUNCTION', 'aligned': False})
    
    return result[:SEQ_LEN], n_aligned


def compute_preference_ratios(attn_passages, token_type_passages, n_actual):
    """
    Compute preference ratios for ENTITY, CONTENT, FUNCTION per head.
    
    attn_passages: (n_passages, seq_len, seq_len) — causal attention weights
    token_type_passages: list of n_passages × seq_len category strings
    
    For each category c:
        preference_ratio[c] = mean over all (passage, query_pos_i, key_pos_j in c) 
                              of attn[i, j] / (freq[c] + eps)
    where freq[c] = n_tokens_in_c / seq_len (per passage, variable).
    
    Returns: {'ENTITY': float, 'CONTENT': float, 'FUNCTION': float}
    """
    eps = 1e-6
    total_attn = {'ENTITY': 0.0, 'CONTENT': 0.0, 'FUNCTION': 0.0}
    total_freq = {'ENTITY': 0.0, 'CONTENT': 0.0, 'FUNCTION': 0.0}
    total_query_positions = 0
    
    for p in range(n_actual):
        a = attn_passages[p]  # (seq_len, seq_len)
        cats = [t['category'] for t in token_type_passages[p]]  # seq_len categories
        
        # Frequency of each category in this passage
        freq = {}
        for c in ('ENTITY', 'CONTENT', 'FUNCTION'):
            freq[c] = cats.count(c) / SEQ_LEN
        
        # For each query position i, compute mean attention on each category
        for i in range(1, SEQ_LEN):
            row = a[i, :i+1]  # attending to positions 0..i
            if row.sum() < 1e-8:
                continue
            total_query_positions += 1
            for c in ('ENTITY', 'CONTENT', 'FUNCTION'):
                # sum of attention weights on positions j where cats[j] == c
                cat_attn = sum(row[j] for j in range(i+1) if cats[j] == c)
                total_attn[c] += cat_attn
                total_freq[c] += freq[c]
    
    # preference_ratio = mean_attn[c] / mean_freq[c]
    pref = {}
    for c in ('ENTITY', 'CONTENT', 'FUNCTION'):
        mean_attn = total_attn[c] / max(total_query_positions, 1)
        mean_freq = total_freq[c] / max(total_query_positions, 1)
        pref[c] = mean_attn / (mean_freq + eps)
    
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
    input_ids_full = encoded["input_ids"][0]  # (total_tokens,)
    print(f"Total test tokens: {len(input_ids_full)}")
    
    # Same non-overlapping passages as exp-181 (first N_PASSAGES × SEQ_LEN tokens)
    passages_ids = []
    for i in range(N_PASSAGES):
        start = i * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(input_ids_full):
            break
        passages_ids.append(input_ids_full[start:end].tolist())
    n_actual = len(passages_ids)
    print(f"Using {n_actual} passages of {SEQ_LEN} tokens")
    
    # ── spaCy token-type classification for each passage ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running spaCy token-type classification...")
    token_type_passages = []  # list of n_actual × seq_len lists of dicts
    alignment_failures_total = 0
    total_tokens_total = 0
    entity_counts = []
    content_counts = []
    
    for p_idx, ids in enumerate(passages_ids):
        if p_idx % 20 == 0:
            print(f"  spaCy passage {p_idx}/{n_actual}...")
        token_types, n_aligned = align_tokens_to_spacy(ids, tokenizer, nlp)
        alignment_failures = sum(1 for t in token_types if not t['aligned'])
        alignment_failures_total += alignment_failures
        total_tokens_total += SEQ_LEN
        entity_count = sum(1 for t in token_types if t['category'] == 'ENTITY')
        content_count = sum(1 for t in token_types if t['category'] == 'CONTENT')
        entity_counts.append(entity_count)
        content_counts.append(content_count)
        token_type_passages.append(token_types)
    
    entity_frac = sum(entity_counts) / (n_actual * SEQ_LEN)
    content_frac = sum(content_counts) / (n_actual * SEQ_LEN)
    alignment_failure_rate = alignment_failures_total / total_tokens_total
    print(f"Entity fraction: {entity_frac:.4f} (K1 threshold: 0.03)")
    print(f"Content fraction: {content_frac:.4f}")
    print(f"Alignment failure rate: {alignment_failure_rate:.4f} (K2 threshold: 0.20)")
    
    # K1: entity undersampling
    k1_fires = entity_frac < 0.03
    # K2: alignment failure
    k2_alignment_fires = alignment_failure_rate > 0.20
    
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
    
    # All heads to process (tn + ctrl, deduplicated by position)
    all_heads_to_process = set(map(tuple, TEXTNATIVE_HEADS)) | set(map(tuple, control_heads))
    
    # ── forward passes ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running forward passes...")
    # attn_accum[(layer, head)] = list of (seq_len, seq_len) numpy arrays
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
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Computing preference ratios...")
    
    # ── compute preference ratios per head ──
    pref_results = {}
    for l, h in all_heads_to_process:
        attn_passages_arr = np.stack(attn_accum[(l, h)])  # (n_passages, seq_len, seq_len)
        pref = compute_preference_ratios(attn_passages_arr, token_type_passages, n_actual)
        pref_results[(l, h)] = pref
    
    # ── extract tn and ctrl preference ratios ──
    tn_entity_pref  = [pref_results[(l,h)]['ENTITY']   for l,h in TEXTNATIVE_HEADS]
    tn_content_pref = [pref_results[(l,h)]['CONTENT']  for l,h in TEXTNATIVE_HEADS]
    tn_func_pref    = [pref_results[(l,h)]['FUNCTION'] for l,h in TEXTNATIVE_HEADS]
    
    ctrl_entity_pref  = [pref_results[(l,h)]['ENTITY']   for l,h in control_heads]
    ctrl_content_pref = [pref_results[(l,h)]['CONTENT']  for l,h in control_heads]
    ctrl_func_pref    = [pref_results[(l,h)]['FUNCTION'] for l,h in control_heads]
    
    # ── hypothesis testing ──
    # H_entity_prefer: ≥10/16 tn heads entity_preference_ratio > 1.10
    h_entity_count = sum(1 for x in tn_entity_pref if x > 1.10)
    h_entity_fires = h_entity_count >= 10
    
    # H_content_prefer: ≥10/16 tn heads content_preference_ratio > 1.10
    h_content_count = sum(1 for x in tn_content_pref if x > 1.10)
    h_content_fires = h_content_count >= 10
    
    # H_uniform: ≥10/16 tn heads have both |entity_pref - 1.0| < 0.10 AND |content_pref - 1.0| < 0.10
    h_uniform_count = sum(
        1 for ep, cp in zip(tn_entity_pref, tn_content_pref)
        if abs(ep - 1.0) < 0.10 and abs(cp - 1.0) < 0.10
    )
    h_uniform_fires = h_uniform_count >= 10
    
    # H_population_contrast: d(tn, ctrl) > 0.3 for entity OR content
    d_entity = cohens_d(tn_entity_pref, ctrl_entity_pref)
    d_content = cohens_d(tn_content_pref, ctrl_content_pref)
    h_pop_contrast_fires = d_entity > 0.3 or d_content > 0.3
    
    # ── K3 kill: no population difference in preference ratios ──
    k3_fires = d_entity < 0.3 and d_content < 0.3
    
    # ── build results ──
    results = {
        "exp": "exp-182",
        "date": "2026-11-17",
        "model": MODEL_ID,
        "prereg_commit": "9d32bdd",
        "prereg_evidence": "git-attested — registration pushed before run.py was written",
        "n_passages": n_actual,
        "seq_len": SEQ_LEN,
        "textnative_heads": TEXTNATIVE_HEADS,
        "control_heads": control_heads,
        "token_type_stats": {
            "entity_frac": float(entity_frac),
            "content_frac": float(content_frac),
            "function_frac": float(1.0 - entity_frac - content_frac),
            "alignment_failure_rate": float(alignment_failure_rate),
        },
        "textnative_preference": {
            "entity":  tn_entity_pref,
            "content": tn_content_pref,
            "function": tn_func_pref,
        },
        "control_preference": {
            "entity":  ctrl_entity_pref,
            "content": ctrl_content_pref,
            "function": ctrl_func_pref,
        },
        "summary_stats": {
            "tn_entity_pref_mean":  float(np.mean(tn_entity_pref)),
            "tn_entity_pref_std":   float(np.std(tn_entity_pref)),
            "tn_content_pref_mean": float(np.mean(tn_content_pref)),
            "tn_content_pref_std":  float(np.std(tn_content_pref)),
            "ctrl_entity_pref_mean":  float(np.mean(ctrl_entity_pref)),
            "ctrl_content_pref_mean": float(np.mean(ctrl_content_pref)),
            "d_entity":  float(d_entity),
            "d_content": float(d_content),
        },
        "hypotheses": {
            "H_entity_prefer": {
                "fires": bool(h_entity_fires),
                "count": int(h_entity_count),
                "threshold": "≥10/16 tn heads entity_pref > 1.10",
                "verdict": "CONFIRMED" if h_entity_fires else "NOT CONFIRMED",
            },
            "H_content_prefer": {
                "fires": bool(h_content_fires),
                "count": int(h_content_count),
                "threshold": "≥10/16 tn heads content_pref > 1.10",
                "verdict": "CONFIRMED" if h_content_fires else "NOT CONFIRMED",
            },
            "H_uniform": {
                "fires": bool(h_uniform_fires),
                "count": int(h_uniform_count),
                "threshold": "≥10/16 tn heads |entity_pref-1.0|<0.10 AND |content_pref-1.0|<0.10",
                "verdict": "CONFIRMED" if h_uniform_fires else "NOT CONFIRMED",
            },
            "H_population_contrast": {
                "fires": bool(h_pop_contrast_fires),
                "d_entity": float(d_entity),
                "d_content": float(d_content),
                "threshold": "d_entity > 0.3 OR d_content > 0.3",
                "verdict": "CONFIRMED" if h_pop_contrast_fires else "NOT CONFIRMED",
            },
        },
        "kill_conditions": {
            "K1_entity_undersampling": {
                "fires": bool(k1_fires),
                "entity_frac": float(entity_frac),
                "threshold": 0.03,
            },
            "K2_alignment_failure": {
                "fires": bool(k2_alignment_fires),
                "failure_rate": float(alignment_failure_rate),
                "threshold": 0.20,
            },
            "K3_no_population_difference": {
                "fires": bool(k3_fires),
                "d_entity": float(d_entity),
                "d_content": float(d_content),
            },
        },
        "per_head_preference": {
            f"{l}_{h}": {
                "entity_pref": float(pref_results[(l,h)]['ENTITY']),
                "content_pref": float(pref_results[(l,h)]['CONTENT']),
                "function_pref": float(pref_results[(l,h)]['FUNCTION']),
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
    
    outfile = "research/physics/experiments/exp-182_pythia410m_tokentype_preference/results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Results written to {outfile}")
    
    # ── summary ──
    print("\n=== exp-182 RESULTS SUMMARY ===")
    print(f"\nToken-type composition: entity={entity_frac:.3f}, content={content_frac:.3f}, "
          f"function={1-entity_frac-content_frac:.3f}")
    print(f"Alignment failure rate: {alignment_failure_rate:.4f}")
    print(f"\nK1 (entity undersampling <0.03): {'FIRES' if k1_fires else 'does not fire'}")
    print(f"K2 (alignment failure >0.20):   {'FIRES' if k2_alignment_fires else 'does not fire'}")
    print(f"K3 (no population difference):   {'FIRES' if k3_fires else 'does not fire'}")
    print(f"\nH_entity_prefer  (≥10/16 entity_pref>1.10):  "
          f"{'CONFIRMED' if h_entity_fires else 'NOT CONFIRMED'} ({h_entity_count}/16)")
    print(f"H_content_prefer (≥10/16 content_pref>1.10): "
          f"{'CONFIRMED' if h_content_fires else 'NOT CONFIRMED'} ({h_content_count}/16)")
    print(f"H_uniform        (≥10/16 both |pref-1|<0.10): "
          f"{'CONFIRMED' if h_uniform_fires else 'NOT CONFIRMED'} ({h_uniform_count}/16)")
    print(f"H_population_contrast (d>0.3 for entity/content): "
          f"{'CONFIRMED' if h_pop_contrast_fires else 'NOT CONFIRMED'} "
          f"(d_entity={d_entity:.3f}, d_content={d_content:.3f})")
    print(f"\nTn entity_pref:  mean={np.mean(tn_entity_pref):.4f} ± {np.std(tn_entity_pref):.4f}")
    print(f"Ctrl entity_pref: mean={np.mean(ctrl_entity_pref):.4f} ± {np.std(ctrl_entity_pref):.4f}")
    print(f"Tn content_pref:  mean={np.mean(tn_content_pref):.4f} ± {np.std(tn_content_pref):.4f}")
    print(f"Ctrl content_pref: mean={np.mean(ctrl_content_pref):.4f} ± {np.std(ctrl_content_pref):.4f}")
    print(f"\nPer-head entity_pref (tn):  {[f'{x:.3f}' for x in tn_entity_pref]}")
    print(f"Per-head content_pref (tn): {[f'{x:.3f}' for x in tn_content_pref]}")


if __name__ == "__main__":
    main()
