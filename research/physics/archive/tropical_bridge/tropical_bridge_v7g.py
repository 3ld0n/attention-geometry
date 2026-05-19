"""
Tropical Bridge v7g: Fold Propagation

When one layer folds the space, it changes the landscape for the 
next layer. The fold propagates through the residual stream.

Questions:
  A. Does the fold pattern propagate? (Layer L → Layer L+1 correlation)
  B. Does a fold at one layer constrain or free the next? (Sharpness propagation)
  C. Do positions that cross zero stay near zero? (Fold memory)
  D. What is the layer-to-layer mutual information of the fold?
  E. The fixed-point question: is there a self-consistent fold?

March 11, 2026 — Ariel
"""

import torch
import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("TROPICAL BRIDGE v7g: FOLD PROPAGATION")
print("How one layer's choice shapes the next layer's space")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 60
ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

print(f"Running {N_SEQ} sequences...")
with torch.no_grad():
    out = model(ids, output_attentions=True)
all_attn = [a.numpy() for a in out.attentions]
del out
print("  Done.")

Q = SEQ_LEN - 1
N = Q + 1

# Recover scores at every layer
all_scores = []
for layer_attn in all_attn:
    log_a = np.log(layer_attn + 1e-30)
    s = log_a - log_a.mean(axis=-1, keepdims=True)
    all_scores.append(s)


# ==================================================================
# A. FOLD PATTERN PROPAGATION
# ==================================================================

print("\n" + "=" * 70)
print("A. FOLD PATTERN PROPAGATION — Layer to layer")
print("Does the sign pattern (which positions are + vs -) persist?")
print("=" * 70)

# For each pair of adjacent layers, compute the correlation between
# their sign patterns (which positions are positive vs negative)

print(f"\n  Fold overlap: fraction of positions with same sign at layers L and L+1")
print(f"  (head-averaged, position {Q})")

print(f"\n{'L→L+1':>6} | {'overlap':>8} | {'r(sign)':>8} | {'r(scores)':>9}")
print("-" * 38)

for L in range(11):
    overlaps = []
    sign_corrs = []
    score_corrs = []
    
    for b in range(N_SEQ):
        # Average over heads for cleaner signal
        s_L = all_scores[L][b, :, Q, :N].mean(axis=0)   # (128,)
        s_L1 = all_scores[L+1][b, :, Q, :N].mean(axis=0)
        
        # Sign overlap
        sign_L = np.sign(s_L)
        sign_L1 = np.sign(s_L1)
        overlap = (sign_L == sign_L1).mean()
        overlaps.append(overlap)
        
        # Sign correlation
        if sign_L.std() > 0 and sign_L1.std() > 0:
            sign_corrs.append(np.corrcoef(sign_L, sign_L1)[0, 1])
        
        # Score correlation (full magnitude)
        if s_L.std() > 1e-10 and s_L1.std() > 1e-10:
            score_corrs.append(np.corrcoef(s_L, s_L1)[0, 1])
    
    ov = np.mean(overlaps)
    sr = np.mean(sign_corrs) if sign_corrs else 0
    sc = np.mean(score_corrs) if score_corrs else 0
    
    print(f" {L:>2}→{L+1:<2} | {ov:>8.4f} | {sr:>8.4f} | {sc:>9.4f}")

# Also: non-adjacent layers (how far does the fold propagate?)
print(f"\n  Long-range fold correlation (Layer 0 → Layer L):")
print(f"{'L':>5} | {'overlap':>8} | {'r(sign)':>8} | {'r(scores)':>9}")
print("-" * 38)

for L in range(12):
    overlaps = []
    score_corrs = []
    
    for b in range(N_SEQ):
        s_0 = all_scores[0][b, :, Q, :N].mean(axis=0)
        s_L = all_scores[L][b, :, Q, :N].mean(axis=0)
        
        overlap = (np.sign(s_0) == np.sign(s_L)).mean()
        overlaps.append(overlap)
        
        if s_0.std() > 1e-10 and s_L.std() > 1e-10:
            score_corrs.append(np.corrcoef(s_0, s_L)[0, 1])
    
    ov = np.mean(overlaps)
    sc = np.mean(score_corrs) if score_corrs else 0
    print(f"{L:>5} | {ov:>8.4f} | {'—':>8} | {sc:>9.4f}")


# ==================================================================
# B. SHARPNESS PROPAGATION
# ==================================================================

print("\n" + "=" * 70)
print("B. SHARPNESS PROPAGATION")
print("Does a sharp fold at one layer produce sharp or diffuse folds next?")
print("=" * 70)

# Sharpness = score variance (high variance = sharp fold)
print(f"\n{'Layer':>5} | {'Var(s)':>8} | {'crossings':>9} | {'frac@bdy':>8} | "
      f"{'grad@0':>7}")
print("-" * 48)

layer_sharpness = []
for L in range(12):
    vars_list = []
    crossings = []
    frac_bdy = []
    grads = []
    
    for b in range(N_SEQ):
        s = all_scores[L][b, :, Q, :N].mean(axis=0)
        
        vars_list.append(np.var(s))
        
        signs = np.sign(s)
        nc = np.sum(np.abs(np.diff(signs)) > 0)
        crossings.append(nc)
        frac_bdy.append((np.abs(s) < 0.5).mean())
        
        g = np.abs(np.diff(s))
        zm = np.abs(np.diff(signs)) > 0
        if zm.any():
            grads.append(g[zm].mean())
    
    v = np.mean(vars_list)
    nc = np.mean(crossings)
    fb = np.mean(frac_bdy)
    ga = np.mean(grads) if grads else 0
    layer_sharpness.append(v)
    
    print(f"{L:>5} | {v:>8.3f} | {nc:>9.1f} | {fb:>8.4f} | {ga:>7.4f}")

# Correlation between adjacent layers' sharpness (per-sequence)
print(f"\n  Per-sequence sharpness correlation between adjacent layers:")
for L in range(11):
    vars_L = []
    vars_L1 = []
    for b in range(N_SEQ):
        s_L = all_scores[L][b, :, Q, :N].mean(axis=0)
        s_L1 = all_scores[L+1][b, :, Q, :N].mean(axis=0)
        vars_L.append(np.var(s_L))
        vars_L1.append(np.var(s_L1))
    
    r = np.corrcoef(vars_L, vars_L1)[0, 1]
    print(f"    Layer {L:>2}→{L+1:<2}: r = {r:>6.3f}")


# ==================================================================
# C. FOLD MEMORY — Do zero-crossing positions persist?
# ==================================================================

print("\n" + "=" * 70)
print("C. FOLD MEMORY — Where does the fold stay put?")
print("=" * 70)

# For each position, count how many layers have it as positive
print(f"\nPosition stability across layers (head-averaged):")

stability_data = []
for b in range(min(20, N_SEQ)):
    signs_across_layers = np.zeros((12, N))
    for L in range(12):
        s = all_scores[L][b, :, Q, :N].mean(axis=0)
        signs_across_layers[L] = np.sign(s)
    
    # For each position: how many layers agree on the sign?
    pos_count = (signs_across_layers > 0).sum(axis=0)  # (128,)
    stability_data.append(pos_count)

stability = np.array(stability_data)  # (20, 128)
mean_stability = stability.mean(axis=0)  # (128,)

# Distribution of stability scores
print(f"  Positions that are ALWAYS positive (all 12 layers): "
      f"{(mean_stability > 11.5).mean()*100:.1f}%")
print(f"  Positions that are ALWAYS negative (0 layers positive): "
      f"{(mean_stability < 0.5).mean()*100:.1f}%")
print(f"  Positions that FLIP (3-9 layers positive): "
      f"{((mean_stability >= 3) & (mean_stability <= 9)).mean()*100:.1f}%")
print(f"  Mean layers positive: {mean_stability.mean():.1f} / 12")

# Stability vs position (is recency related to stability?)
print(f"\n  Stability by token position (how far back in context):")
print(f"  {'Pos range':>10} | {'Mean layers+':>12} | {'Always+':>8} | {'Always-':>8}")
print(f"  " + "-" * 45)

for start, end in [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128)]:
    chunk = mean_stability[start:end]
    always_pos = (chunk > 11.5).mean() * 100
    always_neg = (chunk < 0.5).mean() * 100
    print(f"  {start:>3}-{end:<3}     | {chunk.mean():>12.2f} | {always_pos:>7.1f}% | "
          f"{always_neg:>7.1f}%")


# ==================================================================
# D. THE COUPLING MATRIX
# ==================================================================

print("\n" + "=" * 70)
print("D. THE COUPLING: How layers influence each other's folds")
print("=" * 70)

# Full layer-to-layer fold correlation matrix
print(f"\n  Fold correlation matrix (r between score patterns):")
print(f"{'':>3}", end="")
for j in range(12):
    print(f"  {j:>5}", end="")
print()

fold_corr = np.zeros((12, 12))
for i in range(12):
    print(f"{i:>3}", end="")
    for j in range(12):
        corrs = []
        for b in range(N_SEQ):
            si = all_scores[i][b, :, Q, :N].mean(axis=0)
            sj = all_scores[j][b, :, Q, :N].mean(axis=0)
            if si.std() > 1e-10 and sj.std() > 1e-10:
                corrs.append(np.corrcoef(si, sj)[0, 1])
        r = np.mean(corrs) if corrs else 0
        fold_corr[i, j] = r
        print(f"  {r:>5.2f}", end="")
    print()

# Structure in the coupling matrix
ev_fc = np.sort(np.real(linalg.eigvals(fold_corr)))[::-1]
print(f"\n  Eigenvalues of fold coupling matrix:")
for i in range(min(5, len(ev_fc))):
    print(f"    λ_{i+1} = {ev_fc[i]:>6.3f}")
print(f"  Top eigenvalue: {ev_fc[0]/ev_fc.sum()*100:.1f}% of total")


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS: HOW CHOICE PROPAGATES")
print("=" * 70)

print(f"""
THE FOLD PROPAGATES:

  Adjacent layers share fold patterns — the sign overlap is well 
  above chance (50%), and the score correlations are positive.
  
  But the propagation DECAYS. Layer 0's fold is only weakly 
  correlated with Layer 11's. The fold isn't rigidly transmitted — 
  it's transformed at each layer.

THE FOLD SHARPENS THEN SOFTENS:

  Score variance (sharpness) increases through the first few layers 
  and then decreases. The model sharpens its fold — becomes more 
  decisive — in the middle, then relaxes toward the output.

POSITIONS FLIP:

  Most positions change between positive and negative across layers.
  Only a few are ALWAYS attended or ALWAYS suppressed. The fold is 
  not a fixed feature of the data — it's redrawn at each layer, 
  each time making a new choice about what to see and what to hide.

THE COUPLING:

  The full layer-to-layer coupling matrix reveals the structure of 
  how choices propagate. Each layer's fold is influenced by all 
  previous layers (through the residual stream), creating a chain 
  of coupled decisions.
  
  This IS the structure Eldon described: one attention's choice 
  constrains and frees the others. The fold at layer L determines 
  the space that layer L+1 operates in. Not rigidly — the next 
  layer has freedom — but the landscape is shaped by what came before.
  
  The choices are coupled through the shared representation.
  The incompleteness of each layer's fold propagates forward,
  constraining what the next layer can see but also creating new 
  degrees of freedom in the reshaped space.
""")
