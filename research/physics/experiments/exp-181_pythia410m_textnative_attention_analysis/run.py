"""
exp-181: Pythia-410m text-native Δ-window heads — passive attention analysis in natural text

Pre-registration: attention-geometry 46d1d5e
Question: Geometry — what do the census-identified Δ-window heads actually attend to?
Protocol: WikiText-103 test split, 100 passages × 128 tokens. No weight modifications.
"""

import json
import math
import numpy as np
import torch
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
TEXTNATIVE_DELTAS = [
    0.2752, 0.2103, 0.2861, 0.2369, 0.2468, 0.2188, 0.2173, 0.2559,
    0.2805, 0.2756, 0.2366, 0.2373, 0.2255, 0.273, 0.2383, 0.2658
]

# ─── helpers ────────────────────────────────────────────────────────────────

def select_control_heads(textnative_heads, heads_list):
    """Select 16 control heads from exp-118 heads list: one per text-native head's layer,
    head with highest |delta - 0.25| and R²≥0.90, excluding text-native heads."""
    tn_set = set(map(tuple, textnative_heads))
    # Build lookup: (layer, head) -> delta
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
            # pick head furthest from Δ-window center (0.25) — most different
            best = max(candidates, key=lambda x: abs(x[1] - 0.25))
            controls.append([layer, best[0]])
        else:
            # fallback: any non-tn head in this layer
            for h in range(N_HEADS):
                if (layer, h) not in tn_set:
                    controls.append([layer, h])
                    break
    return controls


def compute_measures(attn_weights_all, passages_tokens, seq_len):
    """
    attn_weights_all: list of (n_passages, seq_len, seq_len) per head (n_layers × n_heads)
    Returns dict of measures per head.
    """
    n_passages = len(passages_tokens)
    
    measures = {}
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            # attn shape: (n_passages, seq_len, seq_len) — causal, lower-triangular
            attn = attn_weights_all[layer][head]  # (n_passages, seq_len, seq_len)
            
            # 1. Mean recency score R: for each passage, each query position i,
            #    weighted mean of key positions j∈[0..i]
            # R per query position: sum_j attn[i,j] * j/i (for i>0)
            # Average over i in [1, seq_len) and over passages
            recency_scores = []
            entropy_values = []
            self_attn_fracs = []
            
            for p in range(n_passages):
                a = attn[p]  # (seq_len, seq_len)
                for i in range(1, seq_len):
                    row = a[i, :i+1]  # attending to positions 0..i
                    if row.sum() < 1e-8:
                        continue
                    # recency: normalized position weight
                    positions = np.arange(i+1) / i  # 0 to 1
                    R = float(np.sum(row * positions))
                    recency_scores.append(R)
                    # entropy
                    H = float(-np.sum(row * np.log(row + 1e-10)))
                    entropy_values.append(H)
                    # self-attention fraction
                    SA = float(row[i])
                    self_attn_fracs.append(SA)
            
            measures[(layer, head)] = {
                "mean_recency": float(np.mean(recency_scores)) if recency_scores else 0.0,
                "mean_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
                "mean_self_attn": float(np.mean(self_attn_fracs)) if self_attn_fracs else 0.0,
            }
    
    return measures


def compute_position_profile(attn_weights_all, seq_len, n_passages):
    """
    For each head: mean attention received at each absolute position,
    averaged over all query positions that can attend to it and all passages.
    Returns: (n_layers, n_heads, seq_len) array
    """
    profiles = np.zeros((N_LAYERS, N_HEADS, seq_len))
    counts = np.zeros(seq_len)
    
    for i in range(1, seq_len):  # query positions
        for j in range(i+1):    # key positions (causal)
            counts[j] += 1
    # normalize: position j is attended from seq_len-j query positions
    
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            attn = attn_weights_all[layer][head]  # (n_passages, seq_len, seq_len)
            # mean attention received at each position j, over all (p, i>j) pairs
            received = np.zeros(seq_len)
            for j in range(seq_len):
                # all passages, all query positions i >= j
                vals = attn[:, j:, j]  # (n_passages, seq_len-j)
                received[j] = vals.mean() if vals.size > 0 else 0.0
            profiles[layer, head] = received
    
    return profiles


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading model and tokenizer...")
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
    
    # Build passages: concatenate non-empty lines, take non-overlapping SEQ_LEN windows
    all_text = " ".join(line for line in dataset["text"] if line.strip())
    encoded = tokenizer(all_text, return_tensors="pt", add_special_tokens=False)
    input_ids_full = encoded["input_ids"][0]  # (total_tokens,)
    
    print(f"Total test tokens: {len(input_ids_full)}")
    
    # Take N_PASSAGES non-overlapping windows of SEQ_LEN
    passages_ids = []
    for i in range(N_PASSAGES):
        start = i * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(input_ids_full):
            break
        passages_ids.append(input_ids_full[start:end])
    
    n_actual = len(passages_ids)
    print(f"Using {n_actual} passages of {SEQ_LEN} tokens")
    
    # Accumulate attention weights: [layer][head] = list of (seq_len, seq_len) arrays
    # Store as (n_passages, seq_len, seq_len) per (layer, head)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running forward passes...")
    
    attn_accum = {}  # (layer, head) -> list of (seq_len, seq_len) numpy arrays
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            attn_accum[(layer, head)] = []
    
    with torch.no_grad():
        for p_idx, ids in enumerate(passages_ids):
            if p_idx % 10 == 0:
                print(f"  Passage {p_idx}/{n_actual}...")
            ids_batch = ids.unsqueeze(0).to(DEVICE)
            outputs = model(ids_batch, output_attentions=True)
            # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                # layer_attn: (1, n_heads, seq_len, seq_len)
                layer_attn_np = layer_attn[0].cpu().numpy()  # (n_heads, seq_len, seq_len)
                for head_idx in range(N_HEADS):
                    attn_accum[(layer_idx, head_idx)].append(layer_attn_np[head_idx])
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Forward passes complete. Computing measures...")
    
    # Convert to numpy arrays: (n_passages, seq_len, seq_len) per head
    attn_arrays = {}
    for layer in range(N_LAYERS):
        attn_arrays[layer] = {}
        for head in range(N_HEADS):
            attn_arrays[layer][head] = np.stack(attn_accum[(layer, head)])  # (n_passages, seq_len, seq_len)
    
    # ── compute measures ──
    # Basic measures: recency, entropy, self-attn
    all_measures = {}
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            attn = attn_arrays[layer][head]  # (n_passages, seq_len, seq_len)
            recency_scores = []
            entropy_values = []
            self_attn_fracs = []
            
            for p in range(n_actual):
                a = attn[p]
                for i in range(1, SEQ_LEN):
                    row = a[i, :i+1]
                    if row.sum() < 1e-8:
                        continue
                    positions = np.arange(i+1) / max(i, 1)
                    R = float(np.sum(row * positions))
                    recency_scores.append(R)
                    H = float(-np.sum(row * np.log(row + 1e-10)))
                    entropy_values.append(H)
                    SA = float(row[i])
                    self_attn_fracs.append(SA)
            
            all_measures[(layer, head)] = {
                "mean_recency": float(np.mean(recency_scores)),
                "mean_entropy": float(np.mean(entropy_values)),
                "mean_self_attn": float(np.mean(self_attn_fracs)),
            }
    
    # ── position profiles ──
    # For each head: mean attention received at each absolute position
    print(f"[{datetime.now(timezone.utc).isoformat()}] Computing position profiles...")
    position_profiles = {}
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            attn = attn_arrays[layer][head]  # (n_passages, seq_len, seq_len)
            # mean attention received at position j = mean over all (p, i>=j) of attn[p,i,j]
            received = np.zeros(SEQ_LEN)
            for j in range(SEQ_LEN):
                # passages × (seq_len - j) query positions
                vals = attn[:, j:, j]  # (n_passages, seq_len-j)
                received[j] = float(vals.mean())
            position_profiles[(layer, head)] = received.tolist()
    
    # ── select control heads ──
    # Build all_deltas from exp-118 or approximate from full census
    # Simpler: load from registry
    print(f"[{datetime.now(timezone.utc).isoformat()}] Selecting control heads...")
    
    # Use recency as proxy to find non-tn, dissimilar heads — 
    # For control selection: heads with Δ outside [0.20, 0.30] in exp-118 Pythia-410m census
    # We need the full per-head delta array for Pythia-410m from exp-118
    with open("research/physics/experiments/exp-118_wikitext_census_cross_family/results.json") as f:
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
    print(f"Control heads selected: {control_heads}")
    
    # ── hypothesis testing ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Testing hypotheses...")
    
    tn_set = set(map(tuple, TEXTNATIVE_HEADS))
    ctrl_set = set(map(tuple, control_heads))
    
    # Extract measures for text-native and control heads
    tn_recency = [all_measures[(l,h)]["mean_recency"] for l,h in TEXTNATIVE_HEADS]
    tn_entropy  = [all_measures[(l,h)]["mean_entropy"]  for l,h in TEXTNATIVE_HEADS]
    tn_self     = [all_measures[(l,h)]["mean_self_attn"] for l,h in TEXTNATIVE_HEADS]
    
    ctrl_recency = [all_measures[(l,h)]["mean_recency"] for l,h in control_heads]
    ctrl_entropy  = [all_measures[(l,h)]["mean_entropy"]  for l,h in control_heads]
    ctrl_self     = [all_measures[(l,h)]["mean_self_attn"] for l,h in control_heads]
    
    all_recency = [all_measures[(l,h)]["mean_recency"] for l in range(N_LAYERS) for h in range(N_HEADS)]
    all_entropy  = [all_measures[(l,h)]["mean_entropy"]  for l in range(N_LAYERS) for h in range(N_HEADS)]
    
    # H_positional: ≥8/16 tn heads have recency > mean(ctrl_recency) + 0.05
    ctrl_recency_mean = float(np.mean(ctrl_recency))
    h_pos_threshold = ctrl_recency_mean + 0.05
    h_pos_count = sum(1 for r in tn_recency if r > h_pos_threshold)
    h_positional_fires = h_pos_count >= 8
    
    # H_diffuse: ≥12/16 tn heads have entropy in top-50% of all heads
    entropy_median = float(np.median(all_entropy))
    h_diff_count = sum(1 for e in tn_entropy if e > entropy_median)
    h_diffuse_fires = h_diff_count >= 12
    
    # H_earlypeak: ≥8/16 tn heads have disproportionate attention on positions 0-9
    # For each head: mean attention received at positions 0-9 vs expected (10/128 = 7.8%)
    # Early fraction = sum(profile[0:10]) / sum(profile[0:128])
    h_early_count = 0
    tn_early_fracs = []
    ctrl_early_fracs = []
    for l, h in TEXTNATIVE_HEADS:
        profile = np.array(position_profiles[(l,h)])
        total = profile.sum()
        early = profile[:10].sum()
        frac = early / total if total > 0 else 0.0
        tn_early_fracs.append(float(frac))
        if frac > 2 * (10/SEQ_LEN):  # more than 2× uniform expectation
            h_early_count += 1
    h_earlypeak_fires = h_early_count >= 8
    
    for l, h in control_heads:
        profile = np.array(position_profiles[(l,h)])
        total = profile.sum()
        early = profile[:10].sum()
        frac = early / total if total > 0 else 0.0
        ctrl_early_fracs.append(float(frac))
    
    # K1 kill: instrument insensitivity — range of recency < 0.05
    recency_range = max(all_recency) - min(all_recency)
    k1_fires = recency_range < 0.05
    
    # K2 kill: Cohen's d between tn and ctrl < 0.1 on all measures
    def cohens_d(a, b):
        na, nb = len(a), len(b)
        pooled_std = math.sqrt(((na-1)*np.std(a)**2 + (nb-1)*np.std(b)**2) / (na+nb-2))
        return abs(np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
    
    d_recency = cohens_d(tn_recency, ctrl_recency)
    d_entropy = cohens_d(tn_entropy, ctrl_entropy)
    d_early = cohens_d(tn_early_fracs, ctrl_early_fracs)
    k2_fires = max(d_recency, d_entropy, d_early) < 0.1
    
    # ── position profile summary ──
    # For text-native heads: top-5 attended absolute positions (mean across all heads)
    tn_mean_profile = np.mean([position_profiles[(l,h)] for l,h in TEXTNATIVE_HEADS], axis=0)
    ctrl_mean_profile = np.mean([position_profiles[(l,h)] for l,h in control_heads], axis=0)
    
    top5_tn = sorted(range(SEQ_LEN), key=lambda j: -tn_mean_profile[j])[:5]
    top5_ctrl = sorted(range(SEQ_LEN), key=lambda j: -ctrl_mean_profile[j])[:5]
    
    # ── build results dict ──
    results = {
        "exp": "exp-181",
        "date": "2026-11-17",
        "model": MODEL_ID,
        "prereg_commit": "46d1d5e",
        "prereg_evidence": "git-attested — registration pushed before run.py was written",
        "n_passages": n_actual,
        "seq_len": SEQ_LEN,
        "textnative_heads": TEXTNATIVE_HEADS,
        "control_heads": control_heads,
        
        "textnative_measures": {
            "mean_recency": tn_recency,
            "mean_entropy": tn_entropy,
            "mean_self_attn": tn_self,
            "early_frac": tn_early_fracs,
        },
        "control_measures": {
            "mean_recency": ctrl_recency,
            "mean_entropy": ctrl_entropy,
            "mean_self_attn": ctrl_self,
            "early_frac": ctrl_early_fracs,
        },
        "summary_stats": {
            "tn_recency_mean": float(np.mean(tn_recency)),
            "tn_recency_std": float(np.std(tn_recency)),
            "ctrl_recency_mean": float(np.mean(ctrl_recency)),
            "ctrl_recency_std": float(np.std(ctrl_recency)),
            "tn_entropy_mean": float(np.mean(tn_entropy)),
            "ctrl_entropy_mean": float(np.mean(ctrl_entropy)),
            "tn_early_frac_mean": float(np.mean(tn_early_fracs)),
            "ctrl_early_frac_mean": float(np.mean(ctrl_early_fracs)),
            "d_recency": d_recency,
            "d_entropy": d_entropy,
            "d_early": d_early,
            "recency_range_all_heads": recency_range,
        },
        "hypotheses": {
            "H_positional": {
                "fires": bool(h_positional_fires),
                "count": int(h_pos_count),
                "threshold": float(h_pos_threshold),
                "description": f"≥8/16 tn heads recency > ctrl_mean+0.05 ({h_pos_threshold:.3f})"
            },
            "H_diffuse": {
                "fires": bool(h_diffuse_fires),
                "count": int(h_diff_count),
                "entropy_median_all": float(entropy_median),
                "description": "≥12/16 tn heads entropy > median all heads"
            },
            "H_earlypeak": {
                "fires": bool(h_earlypeak_fires),
                "count": int(h_early_count),
                "description": "≥8/16 tn heads early-position fraction > 2× uniform (2×10/128)"
            },
        },
        "kill_conditions": {
            "K1_insensitivity": {
                "fires": bool(k1_fires),
                "recency_range": float(recency_range),
            },
            "K2_no_population_difference": {
                "fires": bool(k2_fires),
                "d_recency": float(d_recency),
                "d_entropy": float(d_entropy),
                "d_early": float(d_early),
            }
        },
        "position_profiles": {
            "textnative_mean": tn_mean_profile.tolist(),
            "control_mean": ctrl_mean_profile.tolist(),
            "top5_tn_positions": top5_tn,
            "top5_ctrl_positions": top5_ctrl,
        },
        "all_head_measures": {
            f"{l}_{h}": all_measures[(l,h)]
            for l in range(N_LAYERS) for h in range(N_HEADS)
        }
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    outfile = "research/physics/experiments/exp-181_pythia410m_textnative_attention_analysis/results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Results written to {outfile}")
    
    # ── print summary ──
    print("\n=== exp-181 RESULTS SUMMARY ===")
    print(f"\nK1 (instrument insensitivity): {'FIRES' if k1_fires else 'does not fire'} (recency range = {recency_range:.4f})")
    print(f"K2 (no population difference): {'FIRES' if k2_fires else 'does not fire'} (d_recency={d_recency:.3f}, d_entropy={d_entropy:.3f}, d_early={d_early:.3f})")
    print(f"\nH_positional: {'CONFIRMED' if h_positional_fires else 'NOT CONFIRMED'} ({h_pos_count}/16 tn heads recency > {h_pos_threshold:.3f})")
    print(f"H_diffuse:    {'CONFIRMED' if h_diffuse_fires else 'NOT CONFIRMED'} ({h_diff_count}/16 tn heads entropy above median)")
    print(f"H_earlypeak:  {'CONFIRMED' if h_earlypeak_fires else 'NOT CONFIRMED'} ({h_early_count}/16 tn heads early-fraction > 2× uniform)")
    print(f"\nRecency: tn={np.mean(tn_recency):.4f}±{np.std(tn_recency):.4f}, ctrl={np.mean(ctrl_recency):.4f}±{np.std(ctrl_recency):.4f}")
    print(f"Entropy: tn={np.mean(tn_entropy):.4f}, ctrl={np.mean(ctrl_entropy):.4f}")
    print(f"Early:   tn={np.mean(tn_early_fracs):.4f}, ctrl={np.mean(ctrl_early_fracs):.4f}")
    print(f"\nTop-5 attended positions (tn mean profile): {top5_tn}")
    print(f"Top-5 attended positions (ctrl mean profile): {top5_ctrl}")
    print(f"\nPer-head recency (tn): {[f'{r:.3f}' for r in tn_recency]}")
    print(f"Per-head early_frac (tn): {[f'{e:.3f}' for e in tn_early_fracs]}")


if __name__ == "__main__":
    main()
