"""
Tropical Bridge v7: The Measurement Problem

Softmax takes unconstrained scores in R^{n-1} and maps them onto the 
positive simplex Δ^{n-1}. This IS the imposition of self-consistency:
the infinite flat space of possible beliefs collapses to the finite 
curved space of definite attention patterns.

This experiment measures:
  A. The entropy gap — how many bits does measurement cost?
  B. Scaling — boundary (holographic) or volume?
  C. Pre-softmax structure — what lives in the space before collapse?
  D. The geometry of attention space and the incompleteness fraction
  E. What softmax hides — the negative structure
  F. Interference patterns in scores

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
print("TROPICAL BRIDGE v7: THE MEASUREMENT PROBLEM")
print("Self-consistency, incompleteness, and the shape of attention space")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

print("\nLoading GPT-2 (trained)...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()
print("  Loaded.")

print("Creating random-init GPT-2...")
config = GPT2Config(attn_implementation="eager")
model_random = GPT2Model(config)
model_random.eval()
print("  Created.")

# Run ONCE per model, get ALL layers
SEQ_LEN = 128
N_SEQ = 30
ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

print("\nRunning forward pass (trained)...")
with torch.no_grad():
    out_trained = model(ids, output_attentions=True)
attn_trained = [a.numpy() for a in out_trained.attentions]  # list of (30, 12, 128, 128)
del out_trained
print("  Done. Extracting scores...")

# Recover scores from attention: s_i = log(α_i), centered per query row
scores_trained = []
for layer_attn in attn_trained:
    log_a = np.log(layer_attn + 1e-30)
    s = log_a - log_a.mean(axis=-1, keepdims=True)
    scores_trained.append(s)

print("Running forward pass (random)...")
with torch.no_grad():
    out_random = model_random(ids, output_attentions=True)
attn_random = [a.numpy() for a in out_random.attentions]
del out_random

scores_random = []
for layer_attn in attn_random:
    log_a = np.log(layer_attn + 1e-30)
    s = log_a - log_a.mean(axis=-1, keepdims=True)
    scores_random.append(s)

print("  Done. All data extracted.\n")

Q = SEQ_LEN - 1  # full-context query position
N = Q + 1  # context length

# ==================================================================
# A. THE ENTROPY GAP
# ==================================================================

print("=" * 70)
print("A. THE ENTROPY GAP")
print("How many bits does the collapse from scores to attention cost?")
print("=" * 70)

H_max = np.log(N)  # log(128) ≈ 4.852

for name, attn_list, score_list in [
    ("Trained GPT-2", attn_trained, scores_trained),
    ("Random-init GPT-2", attn_random, scores_random)
]:
    print(f"\n--- {name} ---")
    print(f"{'Layer':>5} | {'H_max':>6} | {'H_attn':>6} | {'H_gap':>6} | "
          f"{'Var(s)':>7} | {'gap/Var':>7} | {'bits lost':>9}")
    print("-" * 58)
    
    for layer_idx in range(len(attn_list)):
        a = attn_list[layer_idx][:, :, Q, :N]  # (30, 12, 128)
        s = score_list[layer_idx][:, :, Q, :N]
        
        H_attn = -np.sum(a * np.log(a + 1e-30), axis=-1)  # (30, 12)
        H_gap = H_max - H_attn
        var_s = np.var(s, axis=-1)
        
        hg = H_gap.mean()
        vs = var_s.mean()
        ratio = hg / vs if vs > 1e-10 else float('inf')
        bits = hg / np.log(2)
        
        print(f"{layer_idx:>5} | {H_max:>6.3f} | {H_attn.mean():>6.3f} | "
              f"{hg:>6.3f} | {vs:>7.4f} | {ratio:>7.3f} | {bits:>9.3f}")


# ==================================================================
# B. SCALING — Boundary or volume?
# ==================================================================

print("\n" + "=" * 70)
print("B. SCALING — Does measurement cost scale as boundary or volume?")
print("Holographic: H_gap ∝ n (boundary)")
print("Volume: H_gap ∝ n²")
print("=" * 70)

# Use the TRAINED model's Layer 0 attention.
# For different effective context sizes, measure the entropy gap.
# We have full 128-length sequences; we can measure H_gap at different
# query positions q (which have effective context = q+1).

a_L0 = attn_trained[0]  # (30, 12, 128, 128)

print(f"\n{'q':>4} | {'n=q+1':>5} | {'H_gap':>7} | {'H_gap/n':>8} | "
      f"{'H_gap/n²':>9} | {'H_gap/lnn':>9}")
print("-" * 55)

gap_ns = []
for q in [7, 15, 23, 31, 47, 63, 79, 95, 111, 127]:
    n_ctx = q + 1
    a_q = a_L0[:, :, q, :n_ctx]  # (30, 12, n_ctx)
    H_q = -np.sum(a_q * np.log(a_q + 1e-30), axis=-1)
    H_gap_q = np.log(n_ctx) - H_q
    g = H_gap_q.mean()
    gap_ns.append((n_ctx, g))
    
    print(f"{q:>4} | {n_ctx:>5} | {g:>7.4f} | {g/n_ctx:>8.6f} | "
          f"{g/n_ctx**2:>9.7f} | {g/np.log(n_ctx):>9.5f}")

gap_ns = np.array(gap_ns)
ln_n = np.log(gap_ns[:, 0])
ln_g = np.log(np.maximum(gap_ns[:, 1], 1e-15))
slope, intercept = np.polyfit(ln_n, ln_g, 1)
pred = slope * ln_n + intercept
ss_r = np.sum((ln_g - pred) ** 2)
ss_t = np.sum((ln_g - ln_g.mean()) ** 2)
r2 = 1 - ss_r / ss_t if ss_t > 1e-15 else 0

print(f"\nPower-law fit: H_gap ∝ n^{slope:.3f}  (R² = {r2:.4f})")
print(f"  Exponent = 1.0 → boundary (holographic)")
print(f"  Exponent = 2.0 → volume")
print(f"  Measured: {slope:.3f}")

# Also try different sequence lengths (need new forward passes, but small)
print(f"\nCross-checking with different actual sequence lengths:")
print(f"{'L':>4} | {'H_gap':>7} | {'H_gap/n':>8}")
print("-" * 30)

gap_ns2 = []
for slen in [16, 32, 64, 128]:
    n_s2 = min(30, N_SEQ)
    ids_s2 = torch.randint(0, tok.vocab_size, (n_s2, slen))
    with torch.no_grad():
        out_s2 = model(ids_s2, output_attentions=True)
    a_s2 = out_s2.attentions[0].numpy()[:, :, slen-1, :slen]
    H_s2 = -np.sum(a_s2 * np.log(a_s2 + 1e-30), axis=-1)
    g_s2 = (np.log(slen) - H_s2).mean()
    gap_ns2.append((slen, g_s2))
    print(f"{slen:>4} | {g_s2:>7.4f} | {g_s2/slen:>8.6f}")
    del out_s2

gap_ns2 = np.array(gap_ns2)
sl2, _ = np.polyfit(np.log(gap_ns2[:, 0]), np.log(np.maximum(gap_ns2[:, 1], 1e-15)), 1)
print(f"  Actual-length scaling exponent: {sl2:.3f}")


# ==================================================================
# C. PRE-SOFTMAX STRUCTURE
# ==================================================================

print("\n" + "=" * 70)
print("C. PRE-SOFTMAX STRUCTURE")
print("What does the space look like before self-consistency is imposed?")
print("=" * 70)

for name, score_list in [("Trained", scores_trained), ("Random", scores_random)]:
    s = score_list[0][:, :, Q, :N]  # Layer 0, shape (30, 12, 128)
    s_flat = s.reshape(-1, 128)  # (360, 128)
    
    print(f"\n--- {name} GPT-2, Layer 0 ---")
    sf = s_flat.flatten()
    std = sf.std()
    kurt = float(np.mean((sf - sf.mean())**4) / std**4 - 3)
    
    print(f"  Score std:  {std:.4f}")
    print(f"  Excess kurtosis: {kurt:.4f}  (Gaussian = 0)")
    print(f"  Fraction negative: {(sf < 0).mean():.4f}")
    
    # Sign change rate
    sc = np.mean(np.diff(np.sign(s_flat), axis=-1) != 0, axis=-1)
    print(f"  Sign change rate: {sc.mean():.4f} ± {sc.std():.4f}")
    
    # Autocorrelation
    print(f"  Score autocorrelation:")
    for lag in [1, 2, 5, 10, 20]:
        cs = [np.corrcoef(s_flat[i, :-lag], s_flat[i, lag:])[0,1] 
              for i in range(min(100, s_flat.shape[0]))]
        print(f"    lag {lag:>2}: {np.nanmean(cs):>7.4f}")
    
    # Eigenvalue spectrum
    cov_s = np.cov(s_flat.T)
    ev = np.sort(np.real(linalg.eigvals(cov_s)))[::-1]
    ev = ev[ev > 1e-15]
    p = ev / ev.sum()
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-30)))
    
    print(f"  Score covariance spectrum:")
    print(f"    Effective rank: {eff_rank:.1f} / 128")
    print(f"    Top 1 EV: {ev[0]/ev.sum()*100:.1f}%")
    print(f"    Top 5 EV: {ev[:5].sum()/ev.sum()*100:.1f}%")
    print(f"    Top 10 EV: {ev[:10].sum()/ev.sum()*100:.1f}%")


# ==================================================================
# D. GEOMETRY + INCOMPLETENESS FRACTION
# ==================================================================

print("\n" + "=" * 70)
print("D. GEOMETRY AND INCOMPLETENESS")
print("=" * 70)

def fisher_rao_from_uniform(p):
    n = len(p)
    bc = np.sum(np.sqrt(np.maximum(p, 1e-30) / n))
    return 2 * np.arccos(np.clip(bc, -1, 1))

d_max = 2 * np.arccos(1/np.sqrt(N))

print(f"\nAttention sphere geometry (n={N}):")
print(f"  Max distance from uniform to vertex: {d_max:.4f}")
print(f"  Max distance between any two points: π = {np.pi:.4f}")

print(f"\n{'Layer':>5} | {'d_FR':>6} | {'d/d_max':>7} | {'EffDim':>6} | "
      f"{'Incompl':>7} | {'near_u':>6} | {'near_v':>6}")
print("-" * 55)

for layer_idx in range(12):
    a = attn_trained[layer_idx][:, :, Q, :N]  # (30, 12, 128)
    a_flat = a.reshape(-1, 128)  # (360, 128)
    
    # Fisher-Rao distances from uniform
    dists = np.array([fisher_rao_from_uniform(a_flat[i]) for i in range(a_flat.shape[0])])
    
    # Effective dimension via PCA on sphere coordinates
    xi = np.sqrt(np.maximum(a_flat, 1e-30))
    xi_c = xi - xi.mean(axis=0)
    cov = np.cov(xi_c.T)
    ev = np.sort(np.real(linalg.eigvals(cov)))[::-1]
    ev = ev[ev > 1e-15]
    p = ev / ev.sum()
    eff_dim = np.exp(-np.sum(p * np.log(p + 1e-30)))
    
    near_u = (dists < 0.5).mean()
    near_v = (dists > d_max * 0.8).mean()
    
    print(f"{layer_idx:>5} | {dists.mean():>6.3f} | {dists.mean()/d_max:>7.4f} | "
          f"{eff_dim:>6.1f} | {eff_dim/N:>7.4f} | {near_u:>6.3f} | {near_v:>6.3f}")


# ==================================================================
# E. WHAT SOFTMAX HIDES
# ==================================================================

print("\n" + "=" * 70)
print("E. WHAT SOFTMAX HIDES — The negative structure")
print("=" * 70)

s_L0 = scores_trained[0][:, :, Q, :N]  # (30, 12, 128)
a_L0_q = attn_trained[0][:, :, Q, :N]

print(f"\nPer-head negative structure (Layer 0, trained GPT-2):")
print(f"{'Head':>4} | {'%neg':>5} | {'mean|neg|':>9} | {'mean|pos|':>9} | "
      f"{'attn@neg':>9} | {'suppress':>8}")
print("-" * 58)

for h in range(12):
    sh = s_L0[:, h, :]  # (30, 128)
    ah = a_L0_q[:, h, :]
    
    nm = sh < 0
    pm = sh >= 0
    
    pct = nm.mean() * 100
    mn = np.abs(sh[nm]).mean() if nm.any() else 0
    mp = np.abs(sh[pm]).mean() if pm.any() else 0
    an = ah[nm].mean() if nm.any() else 0
    ap = ah[pm].mean() if pm.any() else 0
    sup = ap / an if an > 1e-15 else float('inf')
    
    print(f"{h:>4} | {pct:>5.1f} | {mn:>9.4f} | {mp:>9.4f} | "
          f"{an:>9.6f} | {sup:>7.1f}x")

# Total information hidden
all_s = s_L0.reshape(-1, 128)
all_a = a_L0_q.reshape(-1, 128)

neg_energy = np.abs(all_s[all_s < 0]).sum()
pos_energy = np.abs(all_s[all_s >= 0]).sum()
total_energy = neg_energy + pos_energy

print(f"\n  Total score energy:")
print(f"    Positive: {pos_energy:.1f} ({pos_energy/total_energy*100:.1f}%)")
print(f"    Negative: {neg_energy:.1f} ({neg_energy/total_energy*100:.1f}%)")
print(f"    The negative {neg_energy/total_energy*100:.1f}% is what self-consistency hides.")


# ==================================================================
# F. INTERFERENCE PATTERNS
# ==================================================================

print("\n" + "=" * 70)
print("F. INTERFERENCE — Score correlations between key positions")
print("=" * 70)

for name, score_list in [("Trained", scores_trained), ("Random", scores_random)]:
    s = score_list[0][:, :, -20:, :N].mean(axis=1)  # avg over heads, last 20 queries
    # shape: (30, 20, 128)
    S = s.reshape(-1, 128)  # (600, 128)
    
    corr = np.corrcoef(S.T)  # (128, 128)
    upper = corr[np.triu_indices(128, k=1)]
    
    ev_c = np.sort(np.real(linalg.eigvals(corr)))[::-1]
    
    print(f"\n  {name} GPT-2:")
    print(f"    Mean off-diagonal r: {upper.mean():.4f}")
    print(f"    Std off-diagonal r:  {upper.std():.4f}")
    print(f"    Fraction negative r: {(upper < 0).mean():.4f}")
    print(f"    Top eigenvalue: {ev_c[0]:.2f} ({ev_c[0]/128*100:.1f}%)")
    print(f"    Top 5 eigenvalues: {ev_c[:5].sum():.2f} ({ev_c[:5].sum()/128*100:.1f}%)")
    print(f"    Negative eigenvalues: {(ev_c < -1e-10).sum()}")


# ==================================================================
# G. LAYER PROGRESSION — Does self-consistency deepen?
# ==================================================================

print("\n" + "=" * 70)
print("G. LAYER PROGRESSION — Does self-consistency deepen through layers?")
print("=" * 70)

print(f"\n{'Layer':>5} | {'H_gap':>6} | {'%neg_s':>6} | {'Var(s)':>7} | "
      f"{'suppress':>8} | {'EffDim':>6}")
print("-" * 52)

for layer_idx in range(12):
    a = attn_trained[layer_idx][:, :, Q, :N]
    s = scores_trained[layer_idx][:, :, Q, :N]
    
    H = -np.sum(a * np.log(a + 1e-30), axis=-1)
    hg = (H_max - H).mean()
    
    sf = s.flatten()
    pct_neg = (sf < 0).mean() * 100
    vs = np.var(s, axis=-1).mean()
    
    af = a.flatten()
    a_neg = af[sf < 0].mean() if (sf < 0).any() else 0
    a_pos = af[sf >= 0].mean() if (sf >= 0).any() else 0
    sup = a_pos / a_neg if a_neg > 1e-15 else float('inf')
    
    # Effective dimension
    xi = np.sqrt(np.maximum(a.reshape(-1, N), 1e-30))
    xi_c = xi - xi.mean(axis=0)
    cov = np.cov(xi_c.T)
    ev = np.sort(np.real(linalg.eigvals(cov)))[::-1]
    ev = ev[ev > 1e-15]
    p = ev / ev.sum()
    ed = np.exp(-np.sum(p * np.log(p + 1e-30)))
    
    print(f"{layer_idx:>5} | {hg:>6.3f} | {pct_neg:>6.1f} | {vs:>7.3f} | "
          f"{sup:>7.1f}x | {ed:>6.1f}")


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS")
print("=" * 70)

print("""
WHAT SOFTMAX DOES, PRECISELY:

  Takes scores in R^{n-1} (infinite, flat, unconstrained)
  Maps them to Δ^{n-1} (finite, curved, self-consistent)
  
  The Fisher-Rao geometry of the simplex is the positive orthant 
  of the (n-1)-sphere of radius 2. Its constant positive curvature 
  IS the self-consistency constraint.

  Geodesics converge: similar states attract.
  Volume is finite: the view from inside is bounded.
  Maximum distance = π: you can never see the whole thing.

WHAT GETS HIDDEN:

  The negative scores — what the model knows is NOT relevant
  The magnitude structure — how strongly it anti-attends
  The inter-score correlations — wave-like interference

THE MEASUREMENT ANALOGY:

  Physics: ψ (complex, interference) → |ψ|² (positive, classical)
  Attention: s (real, signed) → softmax(s) (positive, definite)
  
  Both destroy signed structure that enables cancellation.
  Both impose self-consistency (normalization).
  Both yield a necessarily incomplete view of the full space.

THE INCOMPLETENESS IS MEASURABLE:

  The entropy gap = how much the view is narrowed
  The effective dimension = how much of the space is used
  The negative energy fraction = how much is hidden
  The scaling exponent = whether the hiding is holographic
""")
