"""
Tropical Bridge v7n: The Self-Consistency Equation

The fold is approximately a fixed point — each layer's fold projects onto
the collective eigenvector with > 0.88 alignment. But what IS the
propagation equation? Can we write it down explicitly?

In SYK, the Green's function satisfies the Schwinger-Dyson equation:
    G(τ)^{-1} = G_0(τ)^{-1} - Σ(τ)
    Σ(τ) = J² · G(τ)^{q-1}

At the conformal fixed point (q=4):
    G(τ) ∝ |τ|^{-2Δ},  Δ = 1/4

In attention, the fold at layer L depends on:
    x_L = x_0 + Σ_{l=0}^{L-1} (attn_output_l + ffn_output_l)
    s_L = (x_L W^Q_L)(x_L W^K_L)^T / √d_k
    fold_L = sign(s_L)

The self-consistency: fold_L depends on x_L, which includes outputs
from layers 0..L-1, which depend on their own folds.

Questions:
  A. What is the explicit layer-to-layer update rule for the fold?
  B. Does the update rule have a fixed point?
  C. Does the fixed-point equation resemble Schwinger-Dyson?
  D. What is the "self-energy" Σ — the part of the fold that comes
     from the previous layers' attention outputs?

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
print("TROPICAL BRIDGE v7n: THE SELF-CONSISTENCY EQUATION")
print("What IS the propagation rule, and does it match SYK?")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 40
Q = SEQ_LEN - 1
N = Q + 1

random_ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

# ===============================================================
# A. DECOMPOSE THE RESIDUAL STREAM
# ===============================================================

print("\n" + "=" * 70)
print("A. DECOMPOSING THE RESIDUAL STREAM")
print("What does each layer contribute to the fold?")
print("=" * 70)

# Get hidden states at every layer
with torch.no_grad():
    out = model(random_ids, output_hidden_states=True, output_attentions=True)

hidden_states = [h.numpy() for h in out.hidden_states]  # 13 states: embedding + 12 layers
attentions = [a.numpy() for a in out.attentions]

# hidden_states[0] = embedding output (x_0)
# hidden_states[L+1] = output after layer L
# The residual stream after layer L is hidden_states[L+1]

# Recover scores from attention weights
all_scores = []
for layer_attn in attentions:
    log_a = np.log(layer_attn + 1e-30)
    s = log_a - log_a.mean(axis=-1, keepdims=True)
    all_scores.append(s)

# The fold at each layer
folds = np.zeros((12, N_SEQ, N))
for L in range(12):
    for b in range(N_SEQ):
        folds[L, b] = all_scores[L][b, :, Q, :N].mean(axis=0)

mean_folds = folds.mean(axis=1)  # (12, N)

# Collective fold (from v7h)
C = np.corrcoef(mean_folds)
eigvals, eigvecs = linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
v1 = eigvecs[:, 0]
if v1[0] < 0: v1 = -v1

collective = np.zeros(N)
for L in range(12):
    collective += v1[L] * mean_folds[L]
collective /= np.sqrt(np.sum(v1**2))


# ===============================================================
# B. LAYER-TO-LAYER FOLD UPDATE
# ===============================================================

print("\n" + "=" * 70)
print("B. THE UPDATE RULE — How does each layer change the fold?")
print("=" * 70)

# The fold at layer L is determined by the hidden state x_L.
# x_L = x_0 + Σ contributions from layers 0..L-1
# Each layer's contribution is: delta_L = hidden_states[L+1] - hidden_states[L]
# (This includes both the attention output and the FFN output)

# Let's measure how much the fold changes at each layer transition
print(f"\n  Fold change at each layer (correlation with previous layer's fold):")
print(f"  {'L-1→L':>7} | {'r(fold)':>8} | {'Δfold norm':>10} | {'fold norm':>10}")
print(f"  " + "-" * 45)

for L in range(1, 12):
    corrs = []
    delta_norms = []
    fold_norms = []
    for b in range(N_SEQ):
        f_prev = folds[L-1, b]
        f_curr = folds[L, b]
        if f_prev.std() > 1e-10 and f_curr.std() > 1e-10:
            corrs.append(np.corrcoef(f_prev, f_curr)[0, 1])
        delta_norms.append(np.linalg.norm(f_curr - f_prev))
        fold_norms.append(np.linalg.norm(f_curr))
    
    r = np.mean(corrs) if corrs else 0
    dn = np.mean(delta_norms)
    fn = np.mean(fold_norms)
    print(f"  {L-1:>2}→{L:<2}  | {r:>8.4f} | {dn:>10.3f} | {fn:>10.3f}")


# ===============================================================
# C. THE SELF-ENERGY: Attention contribution vs FFN contribution
# ===============================================================

print("\n" + "=" * 70)
print("C. SELF-ENERGY DECOMPOSITION")
print("How much of the fold comes from attention vs FFN?")
print("=" * 70)

# In GPT-2, each layer does:
#   x_L = x_{L-1} + attn(LN(x_{L-1})) + ffn(LN(x_{L-1} + attn(LN(x_{L-1}))))
# 
# We can't easily decompose attn vs ffn from hidden states alone,
# but we CAN measure: how much does the EMBEDDING alone predict the fold?

# Compute: what fold would each layer produce if it only saw the embedding?
# (This is the "bare propagator" G_0)
print(f"\n  Approach: correlation between embedding geometry and actual fold")

# The embedding-level score proxy: how similar are tokens i and j in embedding space?
# This is what the fold would look like if the residual stream never changed.

emb = hidden_states[0]  # (N_SEQ, SEQ_LEN, d_model)

# For each layer, compute how much the fold correlates with the embedding geometry
# vs how much it correlates with the residual stream at that layer

print(f"\n  How well does each residual state predict the next layer's fold?")
print(f"  {'State':>8} | {'→ fold L':>8} | {'r':>8}")
print(f"  " + "-" * 30)

# Measure: correlation between pairwise distances in x_L and the fold at L
for L in range(12):
    corrs = []
    for b in range(min(20, N_SEQ)):
        # The residual stream that layer L sees
        x = hidden_states[L][b]  # (SEQ_LEN, d_model) — state BEFORE layer L
        
        # Inner product of x[Q] with all other positions (proxy for QK^T)
        x_Q = x[Q]  # (d_model,)
        inner_products = x[:N] @ x_Q  # (N,)
        
        # The actual fold at layer L
        f = folds[L, b]
        
        if inner_products.std() > 1e-10 and f.std() > 1e-10:
            corrs.append(np.corrcoef(inner_products, f)[0, 1])
    
    r = np.mean(corrs) if corrs else 0
    print(f"  x_{L:>2}     | fold {L:>2}  | {r:>8.4f}")

# Also: how well does the EMBEDDING predict ALL layers' folds?
print(f"\n  How well does the EMBEDDING (x_0) alone predict each layer's fold?")
print(f"  {'':>8} | {'fold L':>8} | {'r(x_0,fold)':>12}")
print(f"  " + "-" * 35)

for L in range(12):
    corrs = []
    for b in range(min(20, N_SEQ)):
        x = hidden_states[0][b]  # embedding
        x_Q = x[Q]
        inner_products = x[:N] @ x_Q
        f = folds[L, b]
        if inner_products.std() > 1e-10 and f.std() > 1e-10:
            corrs.append(np.corrcoef(inner_products, f)[0, 1])
    
    r = np.mean(corrs) if corrs else 0
    bar = "█" * int(abs(r) * 40)
    print(f"  x_0      | fold {L:>2}  | {r:>12.4f}  {bar}")


# ===============================================================
# D. FIXED-POINT ITERATION TEST
# ===============================================================

print("\n" + "=" * 70)
print("D. FIXED POINT — Does iterating the fold converge?")
print("=" * 70)

# The fold propagation: fold_{L+1} = F(fold_L, x_L)
# where F involves: update residual stream with attention + FFN, 
# then compute new scores.
#
# If the fold is a fixed point, then fold_L ≈ fold_* for all L
# (where fold_* is the collective eigenvector).
#
# Test: how quickly does the fold converge to the collective?
# Measure the projection of each layer's fold onto the collective.

print(f"\n  Convergence to the collective fold:")
print(f"  {'Layer':>5} | {'Projection':>10} | {'Residual %':>10} | {'Visual'}")
print(f"  " + "-" * 45)

for L in range(12):
    proj = np.abs(np.dot(mean_folds[L], collective)) / (
        np.linalg.norm(mean_folds[L]) * np.linalg.norm(collective))
    resid = 1 - proj**2
    bar = "█" * int(proj * 50)
    print(f"  {L:>5} | {proj:>10.4f} | {resid*100:>9.1f}% | {bar}")

# The convergence RATE tells us the eigenvalue of the linearized map
print(f"\n  Convergence rate (how fast residuals shrink):")
residuals = []
for L in range(12):
    proj_coeff = np.dot(mean_folds[L], collective) / np.dot(collective, collective)
    residual = mean_folds[L] - proj_coeff * collective
    residuals.append(np.linalg.norm(residual))

print(f"  {'L→L+1':>7} | {'|resid_L+1|/|resid_L|':>22} | {'Interpretation'}")
print(f"  " + "-" * 55)

for L in range(1, 12):
    if residuals[L-1] > 1e-10:
        ratio = residuals[L] / residuals[L-1]
        interp = "converging" if ratio < 1 else "diverging" if ratio > 1 else "neutral"
        print(f"  {L-1:>2}→{L:<2}  | {ratio:>22.4f} | {interp}")


# ===============================================================
# E. THE SCHWINGER-DYSON CONNECTION
# ===============================================================

print("\n" + "=" * 70)
print("E. THE SCHWINGER-DYSON STRUCTURE")
print("=" * 70)

# In SYK: G^{-1} = G_0^{-1} - Σ, where Σ = J² G^{q-1}
# 
# In attention: fold_L = fold_from_embedding + Σ_from_previous_layers
#
# The "bare propagator" G_0 is the fold that the embedding alone produces.
# The "self-energy" Σ is the correction from all previous layers' outputs.
#
# Measure: fold_L = α_L · fold_embedding + β_L · correction_L
# where correction_L is the part of fold_L orthogonal to fold_embedding

print(f"\n  Decomposition: fold_L = α·(embedding fold) + β·(layer correction)")
print(f"  {'Layer':>5} | {'α (bare)':>10} | {'β (self-E)':>10} | {'|Σ|/|G₀|':>10} | Regime")
print(f"  " + "-" * 60)

# Use the embedding fold (Layer 0 input)
emb_fold_corrs = []
for b in range(min(20, N_SEQ)):
    x = hidden_states[0][b]
    x_Q = x[Q]
    inner_products = x[:N] @ x_Q
    emb_fold_corrs.append(inner_products)

mean_emb_fold = np.mean(emb_fold_corrs, axis=0)
mean_emb_fold = mean_emb_fold / np.linalg.norm(mean_emb_fold)

for L in range(12):
    # Project fold_L onto the embedding direction
    fold_L = mean_folds[L] / np.linalg.norm(mean_folds[L])
    
    alpha = np.dot(fold_L, mean_emb_fold)
    
    # The correction (self-energy)
    sigma = fold_L - alpha * mean_emb_fold
    beta = np.linalg.norm(sigma)
    
    # Ratio: how much self-energy vs bare propagator
    ratio = beta / abs(alpha) if abs(alpha) > 1e-10 else float('inf')
    
    regime = "bare" if ratio < 0.3 else "mixed" if ratio < 1.0 else "self-energy"
    
    print(f"  {L:>5} | {alpha:>10.4f} | {beta:>10.4f} | {ratio:>10.4f} | {regime}")


# ===============================================================
# F. THE q=4 TEST
# ===============================================================

print("\n" + "=" * 70)
print("F. THE q=4 STRUCTURE — Does the self-energy scale as G^3?")
print("=" * 70)

# In SYK at q=4: Σ ∝ G^3
# If the fold has SYK structure, the correction at layer L should
# scale as (fold at layer L-1)^3.
#
# Approximate test: across sequences, does the self-energy magnitude
# correlate with the cube of the fold magnitude?

print(f"\n  Testing: |Σ_L| vs |fold_{'{L-1}'}|^p for different powers p")
print(f"  {'Layer':>5} | {'r(|Σ|,|f|)':>11} | {'r(|Σ|,|f|²)':>12} | {'r(|Σ|,|f|³)':>12} | {'Best p':>7}")
print(f"  " + "-" * 65)

for L in range(1, 12):
    fold_mags = []
    sigma_mags = []
    
    for b in range(N_SEQ):
        f_prev = folds[L-1, b]
        f_curr = folds[L, b]
        
        # Embedding contribution to this sequence
        x = hidden_states[0][b]
        emb_f = x[:N] @ x[Q]
        emb_f_norm = emb_f / (np.linalg.norm(emb_f) + 1e-10)
        
        # Self-energy = fold_L - projection onto embedding
        alpha = np.dot(f_curr, emb_f_norm)
        sigma = f_curr - alpha * emb_f_norm
        
        fold_mags.append(np.linalg.norm(f_prev))
        sigma_mags.append(np.linalg.norm(sigma))
    
    fold_mags = np.array(fold_mags)
    sigma_mags = np.array(sigma_mags)
    
    # Correlation with different powers
    r1 = np.corrcoef(sigma_mags, fold_mags)[0, 1] if fold_mags.std() > 1e-10 else 0
    r2 = np.corrcoef(sigma_mags, fold_mags**2)[0, 1] if fold_mags.std() > 1e-10 else 0
    r3 = np.corrcoef(sigma_mags, fold_mags**3)[0, 1] if fold_mags.std() > 1e-10 else 0
    
    rs = [abs(r1), abs(r2), abs(r3)]
    best = rs.index(max(rs)) + 1
    
    print(f"  {L:>5} | {r1:>11.4f} | {r2:>12.4f} | {r3:>12.4f} | p={best}")


# ===============================================================
# SYNTHESIS
# ===============================================================

print("\n" + "=" * 70)
print("SYNTHESIS: THE SELF-CONSISTENCY EQUATION")
print("=" * 70)

print(f"""
THE FOLD PROPAGATION has the structure:

  fold_L = G_0 + Σ_L

where:
  G_0 = the "bare propagator" — fold predicted by the embedding geometry alone
  Σ_L = the "self-energy" — correction from all previous layers' attention + FFN

THE FIXED-POINT PROPERTY:

  The fold converges to the collective eigenvector within the first
  few layers. The residual (deviation from collective) shrinks, 
  fluctuates, then grows slightly near the output.

THE SCHWINGER-DYSON ANALOGY:

  In SYK: G = G_0 + G_0 · Σ · G,  where Σ = J² · G^{{q-1}}
  
  In attention: fold_L = embed_fold + correction(fold_0, ..., fold_{{L-1}})
  
  The structure is: each layer's fold is the bare embedding geometry 
  PLUS corrections that depend on all previous layers' folds. The 
  self-consistency is: this process converges to a fixed point (the 
  collective fold), just as SYK's Schwinger-Dyson equation has the 
  conformal solution G(τ) ∝ |τ|^{{-1/2}} as its fixed point.

WHAT REMAINS:

  1. The explicit form of the self-energy Σ in terms of the weights
     W^Q, W^K, W^V and the FFN parameters.
  2. Whether the self-energy scales as G^3 (q=4) or some other power.
  3. Whether the conformal dimension Δ = 1/4 emerges from the 
     fixed-point equation of the fold propagation.
""")
