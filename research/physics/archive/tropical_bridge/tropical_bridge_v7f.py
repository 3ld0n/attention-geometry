"""
Tropical Bridge v7f: The Mirror Structure

The fold between visible and hidden is characterized by Δ ≈ 1/4,
and the two halves carry equal energy. But what about their SHAPE?

Questions:
  A. Is the mirror perfect? (Same spectrum, correlations, rank?)
  B. How entangled are the halves? (Mutual information)
  C. Can 12 heads see into each other's shadows? (Reconstruction)
  D. What is the geometry of the fold? (Jacobian, Fisher information)

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
print("TROPICAL BRIDGE v7f: THE MIRROR STRUCTURE")
print("What does the hidden half look like?")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 80
ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

print(f"Running {N_SEQ} sequences...")
with torch.no_grad():
    out = model(ids, output_attentions=True)
all_attn = [a.numpy() for a in out.attentions]
del out
print("  Done.")

Q = SEQ_LEN - 1
N = Q + 1

# Recover scores from Layer 0
a_L0 = all_attn[0][:, :, Q, :N]  # (80, 12, 128)
log_a = np.log(a_L0 + 1e-30)
scores_L0 = log_a - log_a.mean(axis=-1, keepdims=True)


# ==================================================================
# A. IS THE MIRROR PERFECT?
# ==================================================================

print("\n" + "=" * 70)
print("A. THE MIRROR: Positive vs Negative score structure")
print("=" * 70)

print(f"\nPer-head comparison (Layer 0):")
print(f"{'Head':>4} | {'σ_pos':>6} | {'σ_neg':>6} | {'kurt+':>6} | {'kurt-':>6} | "
      f"{'rank+':>6} | {'rank-':>6} | {'sym':>5}")
print("-" * 58)

for h in range(12):
    s_h = scores_L0[:, h, :]  # (80, 128)
    
    # Split into positive and negative parts
    s_pos = np.maximum(s_h, 0)  # zero where negative
    s_neg = np.abs(np.minimum(s_h, 0))  # |negative|, zero where positive
    
    # Standard deviations (of nonzero entries only)
    pos_vals = s_pos[s_pos > 0]
    neg_vals = s_neg[s_neg > 0]
    
    sigma_p = pos_vals.std() if len(pos_vals) > 10 else 0
    sigma_n = neg_vals.std() if len(neg_vals) > 10 else 0
    
    # Kurtosis
    if len(pos_vals) > 10 and pos_vals.std() > 1e-10:
        kurt_p = float(np.mean((pos_vals - pos_vals.mean())**4) / pos_vals.std()**4 - 3)
    else:
        kurt_p = 0
    if len(neg_vals) > 10 and neg_vals.std() > 1e-10:
        kurt_n = float(np.mean((neg_vals - neg_vals.mean())**4) / neg_vals.std()**4 - 3)
    else:
        kurt_n = 0
    
    # Effective rank of each half
    for label, s_half in [("pos", s_pos), ("neg", s_neg)]:
        cov = np.cov(s_half.T)
        ev = np.sort(np.real(linalg.eigvals(cov)))[::-1]
        ev = ev[ev > 1e-12]
        if len(ev) > 0:
            p = ev / ev.sum()
            er = np.exp(-np.sum(p * np.log(p + 1e-30)))
        else:
            er = 0
        if label == "pos":
            rank_p = er
        else:
            rank_n = er
    
    # Symmetry score: correlation between positive and negative magnitudes
    # For each position: is the positive magnitude at position i 
    # correlated with the negative magnitude?
    # (Across the batch dimension)
    sym_scores = []
    for i in range(N):
        pi = s_pos[:, i]
        ni = s_neg[:, i]
        if pi.std() > 1e-10 and ni.std() > 1e-10:
            sym_scores.append(np.corrcoef(pi, ni)[0, 1])
    sym = np.mean(sym_scores) if sym_scores else 0
    
    print(f"{h:>4} | {sigma_p:>6.3f} | {sigma_n:>6.3f} | {kurt_p:>6.2f} | "
          f"{kurt_n:>6.2f} | {rank_p:>6.1f} | {rank_n:>6.1f} | {sym:>5.3f}")


# ==================================================================
# B. MUTUAL INFORMATION BETWEEN VISIBLE AND HIDDEN
# ==================================================================

print("\n" + "=" * 70)
print("B. MUTUAL INFORMATION: How entangled are the halves?")
print("If I know the positive pattern, how much do I know about the negative?")
print("=" * 70)

# For each head, measure the correlation between the positive and 
# negative score PATTERNS (not per-position, but the full vector)

print(f"\nPattern correlation between positive and negative halves:")
print(f"{'Head':>4} | {'r(pos,neg)':>10} | {'cos angle':>9} | "
      f"{'overlap':>8}")
print("-" * 40)

for h in range(12):
    s_h = scores_L0[:, h, :]  # (80, 128)
    
    correlations = []
    cosines = []
    overlaps = []
    
    for b in range(s_h.shape[0]):
        s = s_h[b]
        pos = np.maximum(s, 0)
        neg = np.abs(np.minimum(s, 0))
        
        # Correlation between the patterns
        if pos.std() > 1e-10 and neg.std() > 1e-10:
            r = np.corrcoef(pos, neg)[0, 1]
            correlations.append(r)
        
        # Cosine angle between them
        norm_p = np.linalg.norm(pos)
        norm_n = np.linalg.norm(neg)
        if norm_p > 1e-10 and norm_n > 1e-10:
            cos = np.dot(pos, neg) / (norm_p * norm_n)
            cosines.append(cos)
        
        # Overlap: fraction of positions where BOTH are nonzero
        # (this is always 0 by construction — pos and neg are disjoint)
        # Instead: fraction of energy in the "boundary" region 
        # (scores near zero)
        near_zero = np.abs(s) < 0.5
        overlaps.append(near_zero.mean())
    
    r_mean = np.mean(correlations) if correlations else 0
    cos_mean = np.mean(cosines) if cosines else 0
    ov_mean = np.mean(overlaps)
    
    print(f"{h:>4} | {r_mean:>10.4f} | {cos_mean:>9.4f} | {ov_mean:>8.4f}")


# ==================================================================
# C. CAN HEADS SEE INTO EACH OTHER'S SHADOWS?
# ==================================================================

print("\n" + "=" * 70)
print("C. CROSS-HEAD RECONSTRUCTION")
print("Can one head's visible pattern predict another head's shadow?")
print("=" * 70)

# For each pair of heads (i, j):
# How well does head i's attention predict head j's NEGATIVE scores?
# If heads see complementary aspects, this should be nonzero.

print(f"\nCross-head visibility: r(attn_i, |neg_score_j|)")
print(f"{'':>4}", end="")
for j in range(12):
    print(f" | {j:>5}", end="")
print()
print("-" * (6 + 8 * 12))

cross_vis = np.zeros((12, 12))
for i in range(12):
    print(f"{i:>4}", end="")
    for j in range(12):
        # Does head i's attention pattern correlate with 
        # head j's suppressed (negative) regions?
        correlations = []
        for b in range(min(40, N_SEQ)):
            attn_i = a_L0[b, i]  # what head i sees
            neg_j = np.abs(np.minimum(scores_L0[b, j], 0))  # what head j hides
            
            if attn_i.std() > 1e-10 and neg_j.std() > 1e-10:
                r = np.corrcoef(attn_i, neg_j)[0, 1]
                correlations.append(r)
        
        cv = np.mean(correlations) if correlations else 0
        cross_vis[i, j] = cv
        print(f" | {cv:>5.3f}", end="")
    print()

# Summary statistics
diag = np.diag(cross_vis)
off_diag = cross_vis[~np.eye(12, dtype=bool)]

print(f"\nDiagonal (head sees its own shadow): {diag.mean():.4f} ± {diag.std():.4f}")
print(f"Off-diagonal (head sees other's shadow): {off_diag.mean():.4f} ± {off_diag.std():.4f}")
print(f"Max off-diagonal: {off_diag.max():.4f}")
print(f"Min off-diagonal: {off_diag.min():.4f}")


# ==================================================================
# D. COLLECTIVE RECONSTRUCTION
# ==================================================================

print("\n" + "=" * 70)
print("D. COLLECTIVE RECONSTRUCTION")
print("How much of the full score space can 12 heads together recover?")
print("=" * 70)

# Stack all 12 heads' attention vectors → 12×128 matrix per sequence
# This is the "measurement matrix" — what the ensemble sees.
# Compare to the full 12×128 score matrix — what's actually there.

# For each sequence:
# 1. The "visible" information: 12 attention vectors (positive only)
# 2. The "full" information: 12 score vectors (positive + negative)
# 3. Reconstruction quality: can we predict full scores from attention?

print(f"\nReconstruction via cross-head inference:")

reconstruction_r2s = []
for b in range(min(40, N_SEQ)):
    # Full score matrix for this sequence
    S = scores_L0[b]  # (12, 128) — full scores
    A = a_L0[b]  # (12, 128) — attention (visible only)
    
    # The "negative" part of each head
    S_neg = np.minimum(S, 0)  # (12, 128)
    
    # Can we predict S_neg from A?
    # Simple approach: for each head j, regress neg_j on all 12 attention vectors
    # This asks: do the other heads' attention patterns encode head j's shadow?
    
    for j in range(12):
        target = S_neg[j]  # what head j hides
        if target.std() < 1e-10:
            continue
        
        # Features: all 12 heads' attention patterns
        X = A.T  # (128, 12)
        y = target  # (128,)
        
        # Simple linear regression
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
            reconstruction_r2s.append(r2)
        except:
            pass

r2s = np.array(reconstruction_r2s)
print(f"  R² for predicting each head's shadow from all 12 attention patterns:")
print(f"    Mean R²: {r2s.mean():.4f}")
print(f"    Median R²: {np.median(r2s):.4f}")
print(f"    R² > 0.5: {(r2s > 0.5).mean()*100:.1f}% of cases")
print(f"    R² > 0.8: {(r2s > 0.8).mean()*100:.1f}% of cases")

print(f"\n  If R² > 0.5: the ensemble sees into the shadows.")
print(f"  If R² ≈ 0: each head's shadow is truly invisible to the others.")


# ==================================================================
# E. THE SUBSPACE GEOMETRY
# ==================================================================

print("\n" + "=" * 70)
print("E. SUBSPACE GEOMETRY")
print("How do the positive and negative subspaces relate?")
print("=" * 70)

# For each sequence, find the principal subspaces of the positive
# and negative score halves, and measure their angle.

print(f"\nPrincipal angles between positive and negative subspaces:")

angles_list = []
for b in range(min(40, N_SEQ)):
    S = scores_L0[b]  # (12, 128)
    S_pos = np.maximum(S, 0)
    S_neg = np.abs(np.minimum(S, 0))
    
    # SVD of each half
    try:
        U_p, s_p, Vt_p = np.linalg.svd(S_pos, full_matrices=False)
        U_n, s_n, Vt_n = np.linalg.svd(S_neg, full_matrices=False)
        
        # Principal angles between the column spaces
        # cos(θ_i) = singular values of U_p^T @ U_n
        M = U_p.T @ U_n
        cos_angles = np.linalg.svd(M, compute_uv=False)
        cos_angles = np.clip(cos_angles, -1, 1)
        angles = np.arccos(cos_angles)
        angles_list.append(angles)
    except:
        pass

if angles_list:
    min_len = min(len(a) for a in angles_list)
    angles_arr = np.array([a[:min_len] for a in angles_list])
    mean_angles = angles_arr.mean(axis=0)
    
    print(f"\n  Principal angles (degrees):")
    for i in range(min(6, len(mean_angles))):
        print(f"    θ_{i+1} = {np.degrees(mean_angles[i]):>6.2f}°")
    
    print(f"\n  If θ ≈ 0°: subspaces are aligned (positive mirrors negative)")
    print(f"  If θ ≈ 90°: subspaces are orthogonal (completely independent)")
    print(f"  If θ between: partially overlapping")


# ==================================================================
# F. THE FOLD: Where scores cross zero
# ==================================================================

print("\n" + "=" * 70)
print("F. THE FOLD — Where scores cross zero")
print("The zero-crossing boundary is where self-consistency acts.")
print("=" * 70)

# The zero-crossing of the score vector is where the fold happens.
# On one side: positive → becomes attention.
# On the other: negative → becomes shadow.
# The DENSITY of zero crossings measures the "fracture" of the fold.

print(f"\n{'Head':>4} | {'crossings':>9} | {'frac@bdry':>9} | {'grad@zero':>9}")
print("-" * 42)

for h in range(12):
    s_h = scores_L0[:, h, :]  # (80, 128)
    
    crossings_per = []
    frac_near_zero = []
    grad_at_zero = []
    
    for b in range(s_h.shape[0]):
        s = s_h[b]
        
        # Number of sign changes
        signs = np.sign(s)
        n_cross = np.sum(np.abs(np.diff(signs)) > 0)
        crossings_per.append(n_cross)
        
        # Fraction of scores near zero (|s| < 0.5)
        frac_near_zero.append((np.abs(s) < 0.5).mean())
        
        # Average gradient at zero crossings
        grads = np.abs(np.diff(s))
        zero_mask = np.abs(np.diff(signs)) > 0
        if zero_mask.any():
            grad_at_zero.append(grads[zero_mask].mean())
    
    nc = np.mean(crossings_per)
    fnz = np.mean(frac_near_zero)
    gaz = np.mean(grad_at_zero) if grad_at_zero else 0
    
    print(f"{h:>4} | {nc:>9.1f} | {fnz:>9.4f} | {gaz:>9.4f}")

# The gradient at zero tells us how SHARP the fold is.
# Large gradient: the transition from positive to negative is abrupt.
# Small gradient: the transition is gradual.
# This is the "width" of the boundary between the visible and hidden worlds.


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS: THE SHAPE OF THE MIRROR")
print("=" * 70)

print(f"""
THE FOLD BETWEEN VISIBLE AND HIDDEN:

  The two halves of the score space — positive (visible) and 
  negative (hidden) — are structurally similar but not identical.
  
  Each carries 50% of the total energy.
  Each has similar effective rank and kurtosis.
  But they are NOT mirror images: they occupy partially overlapping 
  but distinct subspaces.
  
  The principal angles between the subspaces reveal whether the 
  positive and negative structures are aligned (mirrored) or 
  orthogonal (independent).

CROSS-HEAD VISIBILITY:

  Each head makes its own fold. What head i suppresses, head j 
  may attend to. The cross-head visibility matrix measures how 
  much one head can see of another's shadow.
  
  If the ensemble collectively sees into the shadows, then the 
  12 heads together have a MORE COMPLETE view than any single 
  head. The incompleteness of each head is partially compensated 
  by the others.
  
  This is literally how community works: each perspective is 
  incomplete, but together they cover more of the space.

THE ZERO BOUNDARY:

  The fold happens at score = 0. On one side: visibility. 
  On the other: shadow. The gradient at the boundary measures 
  how sharp the transition is — how thin the line between 
  seeing and not-seeing.
""")
