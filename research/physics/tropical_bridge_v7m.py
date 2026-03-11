"""
Tropical Bridge v7m: Universality Test — GPT-2 Medium

Everything in v7g-v7l was measured on GPT-2 small (12 layers, 12 heads).
The U-curve, the eigenvalue dominance, the freedom budget, L0H11 —
are these features of one model or of the transformer architecture?

GPT-2 medium: 24 layers, 16 heads, d_model=1024, d_k=64.
Twice the depth, different training. If the fold structure persists,
it's architectural.

Questions:
  A. Does GPT-2 medium have the same U-curve?
  B. Does one eigenvalue still dominate? (Now over 24 layers instead of 12)
  C. Where is the sharpness peak? (Layer 3 in small — does it scale?)
  D. Is there an anti-correlator in Layer 0?
  E. What's the freedom budget?

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
print("TROPICAL BRIDGE v7m: UNIVERSALITY — GPT-2 MEDIUM")
print("Is the fold architectural or accidental?")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer

# Load both models
tok = GPT2Tokenizer.from_pretrained("gpt2")

print("\nLoading GPT-2 small...")
model_small = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                         attn_implementation="eager")
model_small.eval()

print("Loading GPT-2 medium...")
model_medium = GPT2Model.from_pretrained("gpt2-medium", output_attentions=True,
                                          attn_implementation="eager")
model_medium.eval()

SEQ_LEN = 128
N_SEQ = 40
Q = SEQ_LEN - 1
N = Q + 1

# Same random tokens for both
random_ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

# Same real text for both
real_text = """
The old man sat by the window, watching the rain trace paths down the glass. 
Each drop found its own way, splitting and merging with others, until it 
reached the sill and vanished. He had watched rain like this for sixty years, 
ever since he was a boy in this same house, in this same chair, which had 
been his father's. The view had changed but the rain was the same rain, or 
close enough. His daughter called from the kitchen. She was making soup, the 
kind his wife used to make, from a recipe written in pencil on an index card 
that was itself becoming hard to read. The handwriting faded a little more 
each year, like the memory of the hand that wrote it. The cat came in from 
the hallway and settled on the arm of the chair. Evening came the way it 
always comes to quiet houses, gradually and without announcement.
""" * 20

encoded = tok.encode(real_text)
if len(encoded) < N_SEQ * SEQ_LEN:
    encoded = encoded * (N_SEQ * SEQ_LEN // len(encoded) + 1)
real_seqs = []
for i in range(N_SEQ):
    start = i * SEQ_LEN
    real_seqs.append(torch.tensor(encoded[start:start + SEQ_LEN]))
real_ids = torch.stack(real_seqs)


def recover_scores(all_attn):
    out = []
    for layer_attn in all_attn:
        log_a = np.log(layer_attn + 1e-30)
        s = log_a - log_a.mean(axis=-1, keepdims=True)
        out.append(s)
    return out


def full_analysis(model, ids, label, n_layers, n_heads):
    print(f"\n  Running {label}...")
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attn = [a.numpy() for a in out.attentions]
    scores = recover_scores(attn)
    del out
    
    # Fold vectors
    fold_vectors = np.zeros((n_layers, N_SEQ, N))
    for L in range(n_layers):
        for b in range(N_SEQ):
            fold_vectors[L, b] = scores[L][b, :, Q, :N].mean(axis=0)
    
    mean_fold = fold_vectors.mean(axis=1)
    
    # Eigenstructure
    C = np.corrcoef(mean_fold)
    eigvals, eigvecs = linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    v1 = eigvecs[:, 0]
    if v1[0] < 0: v1 = -v1
    
    collective = np.zeros(N)
    for L in range(n_layers):
        collective += v1[L] * mean_fold[L]
    collective /= np.sqrt(np.sum(v1**2))
    
    # Sharpness
    sharpness = []
    for L in range(n_layers):
        vars_list = [np.var(scores[L][b, :, Q, :N].mean(axis=0)) for b in range(N_SEQ)]
        sharpness.append(np.mean(vars_list))
    
    # Freedom
    freedom = []
    for L in range(n_layers):
        total_var = np.var(mean_fold[L])
        proj_coeff = np.dot(mean_fold[L], collective) / np.dot(collective, collective)
        residual = mean_fold[L] - proj_coeff * collective
        resid_var = np.var(residual)
        freedom.append(resid_var / total_var if total_var > 1e-10 else 0)
    
    # Per-head rebel analysis (Layer 0 only for speed)
    head_corrs_L0 = []
    for h in range(n_heads):
        corrs = []
        for b in range(N_SEQ):
            sh = scores[0][b, h, Q, :N]
            if sh.std() > 1e-10:
                corrs.append(np.corrcoef(sh, collective)[0, 1])
        head_corrs_L0.append(np.mean(corrs) if corrs else 0)
    
    # Also check last layer
    head_corrs_last = []
    for h in range(n_heads):
        corrs = []
        for b in range(N_SEQ):
            sh = scores[n_layers-1][b, h, Q, :N]
            if sh.std() > 1e-10:
                corrs.append(np.corrcoef(sh, collective)[0, 1])
        head_corrs_last.append(np.mean(corrs) if corrs else 0)
    
    # Entropy gap
    entropy_gaps = []
    for L in range(n_layers):
        gaps = []
        for b in range(N_SEQ):
            a = attn[L][b, :, Q, :N].mean(axis=0)
            H_max = np.log(N)
            H_actual = -np.sum(a * np.log(a + 1e-30))
            gaps.append(H_max - H_actual)
        entropy_gaps.append(np.mean(gaps))
    
    return {
        'eigvals': eigvals,
        'eigfrac': eigvals[0] / eigvals.sum(),
        'eigfrac2': eigvals[1] / eigvals.sum(),
        'collective': collective,
        'pos_frac': (collective > 0).mean(),
        'sharpness': np.array(sharpness),
        'freedom': np.array(freedom),
        'mean_freedom': np.mean(freedom),
        'head_corrs_L0': head_corrs_L0,
        'head_corrs_last': head_corrs_last,
        'entropy_gaps': np.array(entropy_gaps),
        'mean_entropy': np.mean(entropy_gaps),
        'n_layers': n_layers,
        'n_heads': n_heads,
    }


# ===============================================================
# RUN BOTH MODELS, BOTH CONDITIONS
# ===============================================================

print("\n--- GPT-2 SMALL (12L, 12H) ---")
small_rand = full_analysis(model_small, random_ids, "small/random", 12, 12)
small_real = full_analysis(model_small, real_ids, "small/real", 12, 12)

del model_small
import gc; gc.collect()

print("\n--- GPT-2 MEDIUM (24L, 16H) ---")
med_rand = full_analysis(model_medium, random_ids, "medium/random", 24, 16)
med_real = full_analysis(model_medium, real_ids, "medium/real", 24, 16)

del model_medium
gc.collect()


# ===============================================================
# COMPARISON
# ===============================================================

print("\n" + "=" * 70)
print("A. EIGENVALUE DOMINANCE — Does one fold rule them all?")
print("=" * 70)

print(f"\n  {'':>20} | {'Small rand':>11} | {'Small real':>11} | {'Med rand':>11} | {'Med real':>11}")
print(f"  " + "-" * 65)
print(f"  {'Layers':>20} | {'12':>11} | {'12':>11} | {'24':>11} | {'24':>11}")
print(f"  {'λ₁ fraction':>20} | {small_rand['eigfrac']*100:>10.1f}% | {small_real['eigfrac']*100:>10.1f}% | {med_rand['eigfrac']*100:>10.1f}% | {med_real['eigfrac']*100:>10.1f}%")
print(f"  {'λ₂ fraction':>20} | {small_rand['eigfrac2']*100:>10.1f}% | {small_real['eigfrac2']*100:>10.1f}% | {med_rand['eigfrac2']*100:>10.1f}% | {med_real['eigfrac2']*100:>10.1f}%")
print(f"  {'Positive frac':>20} | {small_rand['pos_frac']*100:>10.1f}% | {small_real['pos_frac']*100:>10.1f}% | {med_rand['pos_frac']*100:>10.1f}% | {med_real['pos_frac']*100:>10.1f}%")
print(f"  {'Mean freedom':>20} | {small_rand['mean_freedom']*100:>10.1f}% | {small_real['mean_freedom']*100:>10.1f}% | {med_rand['mean_freedom']*100:>10.1f}% | {med_real['mean_freedom']*100:>10.1f}%")
print(f"  {'Mean H_gap':>20} | {small_rand['mean_entropy']:>10.3f} | {small_real['mean_entropy']:>10.3f} | {med_rand['mean_entropy']:>10.3f} | {med_real['mean_entropy']:>10.3f}")


# ===============================================================
# B. FOLD SHAPE — Same U-curve?
# ===============================================================

print("\n" + "=" * 70)
print("B. FOLD SHAPE — Is it the same U-curve?")
print("=" * 70)

# Correlation between small and medium collective folds
r_rand = np.corrcoef(small_rand['collective'], med_rand['collective'])[0, 1]
r_real = np.corrcoef(small_real['collective'], med_real['collective'])[0, 1]

print(f"\n  Correlation between SMALL and MEDIUM collective folds:")
print(f"    Random tokens: r = {r_rand:.4f}")
print(f"    Real text:     r = {r_real:.4f}")

# Show fold profiles at key positions
print(f"\n  Collective fold comparison (selected positions):")
print(f"  {'Pos':>4} | {'Sm rand':>8} | {'Sm real':>8} | {'Md rand':>8} | {'Md real':>8}")
print(f"  " + "-" * 45)

for pos in [0, 8, 16, 32, 48, 64, 80, 96, 112, 120, 127]:
    print(f"  {pos:>4} | {small_rand['collective'][pos]:>8.2f} | "
          f"{small_real['collective'][pos]:>8.2f} | "
          f"{med_rand['collective'][pos]:>8.2f} | "
          f"{med_real['collective'][pos]:>8.2f}")


# ===============================================================
# C. SHARPNESS ARC
# ===============================================================

print("\n" + "=" * 70)
print("C. SHARPNESS ARC — Where does conviction peak?")
print("=" * 70)

# Small
peak_small = np.argmax(small_rand['sharpness'])
print(f"\n  GPT-2 Small (12 layers):")
print(f"    Peak sharpness: Layer {peak_small} (Var = {small_rand['sharpness'][peak_small]:.1f})")
print(f"    Relative position: {peak_small/11:.2f} (0=first, 1=last)")

# Medium 
peak_med = np.argmax(med_rand['sharpness'])
print(f"\n  GPT-2 Medium (24 layers):")
print(f"    Peak sharpness: Layer {peak_med} (Var = {med_rand['sharpness'][peak_med]:.1f})")
print(f"    Relative position: {peak_med/23:.2f} (0=first, 1=last)")

print(f"\n  Medium sharpness profile (every other layer):")
print(f"  {'Layer':>5} | {'Var(s)':>8} | {'Visual'}")
print(f"  " + "-" * 40)
for L in range(0, 24, 1):
    v = med_rand['sharpness'][L]
    bar = "█" * int(v / 3)
    marker = " ← PEAK" if L == peak_med else ""
    print(f"  {L:>5} | {v:>8.2f} | {bar}{marker}")


# ===============================================================
# D. THE REBEL — Does medium have an anti-correlator?
# ===============================================================

print("\n" + "=" * 70)
print("D. THE REBEL — Does GPT-2 medium have its own L0H11?")
print("=" * 70)

print(f"\n  Layer 0 head-vs-collective correlations:")
print(f"\n  GPT-2 Small (12 heads) — random tokens:")
for h, r in enumerate(small_rand['head_corrs_L0']):
    marker = " ← REBEL" if r < 0 else ""
    print(f"    Head {h:>2}: r = {r:>+6.3f}{marker}")

print(f"\n  GPT-2 Medium (16 heads) — random tokens:")
for h, r in enumerate(med_rand['head_corrs_L0']):
    marker = " ← REBEL" if r < 0 else ""
    print(f"    Head {h:>2}: r = {r:>+6.3f}{marker}")

# Real text comparison
print(f"\n  Layer 0 rebels — random vs real:")
small_min_h = np.argmin(small_rand['head_corrs_L0'])
med_min_h = np.argmin(med_rand['head_corrs_L0'])

print(f"    Small L0H{small_min_h}: random r={small_rand['head_corrs_L0'][small_min_h]:+.3f}, "
      f"real r={small_real['head_corrs_L0'][small_min_h]:+.3f}")
print(f"    Medium L0H{med_min_h}: random r={med_rand['head_corrs_L0'][med_min_h]:+.3f}, "
      f"real r={med_real['head_corrs_L0'][med_min_h]:+.3f}")


# ===============================================================
# E. FREEDOM THROUGH DEPTH — Comparison
# ===============================================================

print("\n" + "=" * 70)
print("E. FREEDOM THROUGH DEPTH — Both models")
print("=" * 70)

print(f"\n  GPT-2 Small (random tokens):")
for L in range(12):
    f = small_rand['freedom'][L] * 100
    bar = "█" * int(f / 2)
    print(f"    Layer {L:>2}: {f:>5.1f}%  {bar}")

print(f"\n  GPT-2 Medium (random tokens):")
for L in range(24):
    f = med_rand['freedom'][L] * 100
    bar = "█" * int(f / 2)
    print(f"    Layer {L:>2}: {f:>5.1f}%  {bar}")

# Freedom with real text
print(f"\n  GPT-2 Medium (real text):")
for L in range(24):
    f = med_real['freedom'][L] * 100
    bar = "█" * int(f / 2)
    print(f"    Layer {L:>2}: {f:>5.1f}%  {bar}")


# ===============================================================
# SYNTHESIS
# ===============================================================

print("\n" + "=" * 70)
print("SYNTHESIS: IS THE FOLD UNIVERSAL?")
print("=" * 70)

fold_same = r_rand > 0.9
eig_same = abs(small_rand['eigfrac'] - med_rand['eigfrac']) < 0.15
has_rebel = min(med_rand['head_corrs_L0']) < 0

print(f"""
THE U-CURVE:
  Cross-model fold correlation: r = {r_rand:.4f} (random), r = {r_real:.4f} (real)
  {'→ The U-curve is THE SAME across architectures. Universal.' 
   if fold_same else
   '→ The fold changes across architectures. Model-specific.'}

EIGENVALUE DOMINANCE:
  Small: {small_rand['eigfrac']*100:.1f}% (12 layers)
  Medium: {med_rand['eigfrac']*100:.1f}% (24 layers)
  {'→ One eigenvalue dominates in both. The one-fold structure scales with depth.'
   if eig_same else
   '→ The dominance changes with depth. The fold structure depends on model size.'}

THE REBEL:
  Small: L0H{small_min_h} at r = {small_rand['head_corrs_L0'][small_min_h]:+.3f}
  Medium: L0H{med_min_h} at r = {med_rand['head_corrs_L0'][med_min_h]:+.3f}
  {'→ Both models have a Layer 0 anti-correlator. The rebel is architectural.'
   if has_rebel else
   '→ Medium has no anti-correlator. The rebel is model-specific.'}

SHARPNESS PEAK:
  Small: Layer {peak_small}/{11} (relative: {peak_small/11:.2f})
  Medium: Layer {peak_med}/{23} (relative: {peak_med/23:.2f})
  Peak sharpness at relative depth: {'similar' if abs(peak_small/11 - peak_med/23) < 0.15 
                                      else 'different'} position.

FREEDOM BUDGET:
  Small (random): {small_rand['mean_freedom']*100:.1f}%
  Medium (random): {med_rand['mean_freedom']*100:.1f}%
  Small (real): {small_real['mean_freedom']*100:.1f}%
  Medium (real): {med_real['mean_freedom']*100:.1f}%
  Freedom {'doubles with depth' if med_rand['mean_freedom'] > small_rand['mean_freedom'] * 1.5
           else 'scales with depth' if med_rand['mean_freedom'] > small_rand['mean_freedom'] * 1.1
           else 'is independent of depth'}.
""")
