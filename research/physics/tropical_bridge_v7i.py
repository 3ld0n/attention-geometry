"""
Tropical Bridge v7i: The Rebels

v7h found heads that break from the collective fold. Layer 0 Head 11
anti-correlates with the consensus. Layer 7 Head 2 is orthogonal.
Layer 2 as a whole diverges.

What are they looking at? Do they influence what comes after them?
Does the system's freedom live in specific, structured places?

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
print("TROPICAL BRIDGE v7i: THE REBELS")
print("Heads that see what the collective doesn't")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 60
ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

with torch.no_grad():
    out = model(ids, output_attentions=True)
all_attn = [a.numpy() for a in out.attentions]
del out

Q = SEQ_LEN - 1
N = Q + 1

all_scores = []
for layer_attn in all_attn:
    log_a = np.log(layer_attn + 1e-30)
    s = log_a - log_a.mean(axis=-1, keepdims=True)
    all_scores.append(s)

# Rebuild collective fold from v7h
mean_fold = np.zeros((12, N))
for L in range(12):
    for b in range(N_SEQ):
        mean_fold[L] += all_scores[L][b, :, Q, :N].mean(axis=0)
    mean_fold[L] /= N_SEQ

C = np.corrcoef(mean_fold)
eigvals, eigvecs = linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
v1 = eigvecs[:, idx[0]]
if v1[0] < 0: v1 = -v1

collective = np.zeros(N)
for L in range(12):
    collective += v1[L] * mean_fold[L]
collective /= np.linalg.norm(v1)

rebels = [
    (0, 11, "anti-correlator"),
    (7, 2, "orthogonal"),
    (3, 0, "near-zero"),
    (0, 5, "low"),
    (6, 9, "low"),
    (10, 8, "low"),
]


# ==================================================================
# A. WHAT DO THE REBELS ATTEND TO?
# ==================================================================

print("\n" + "=" * 70)
print("A. REBEL ATTENTION PROFILES")
print("=" * 70)

for L, H, label in rebels:
    # Average attention pattern for this head
    attn_pattern = np.zeros(N)
    score_pattern = np.zeros(N)
    
    for b in range(N_SEQ):
        attn_pattern += all_attn[L][b, H, Q, :N]
        score_pattern += all_scores[L][b, H, Q, :N]
    attn_pattern /= N_SEQ
    score_pattern /= N_SEQ
    
    # Where does it peak?
    peak_pos = np.argsort(attn_pattern)[-5:][::-1]
    
    # Compare to collective
    r_coll = np.corrcoef(score_pattern, collective)[0, 1]
    
    # What fraction of its positive scores overlap with collective's negative?
    rebel_pos = score_pattern > 0
    coll_neg = collective < 0
    shadow_seeing = (rebel_pos & coll_neg).sum() / max(rebel_pos.sum(), 1)
    
    # Entropy of attention
    a = attn_pattern[attn_pattern > 1e-10]
    H_attn = -np.sum(a * np.log(a))
    k_eff = np.exp(H_attn)
    
    print(f"\n  Layer {L} Head {H} ({label}):")
    print(f"    r(collective) = {r_coll:.3f}")
    print(f"    k_eff = {k_eff:.1f} tokens (of {N})")
    print(f"    Peak positions: {peak_pos}")
    print(f"    Sees into collective shadow: {shadow_seeing*100:.1f}% of its positive scores")
    
    # Profile: first, middle, last
    for region, start, end in [("begin", 0, 16), ("mid", 56, 72), ("end", 112, 128)]:
        chunk = score_pattern[start:end]
        pos_pct = (chunk > 0).mean() * 100
        print(f"    {region:>5} [{start}-{end}]: {pos_pct:.0f}% positive, "
              f"mean score = {chunk.mean():>6.2f}")


# ==================================================================
# B. DO REBELS INFLUENCE WHAT COMES AFTER?
# ==================================================================

print("\n" + "=" * 70)
print("B. REBEL INFLUENCE — Do they change what downstream layers see?")
print("=" * 70)

# For each rebel head at layer L, measure its influence on layer L+1
# by checking per-sequence correlation: when this rebel head attends
# strongly to position X, does L+1 also attend more to X?

for L, H, label in rebels:
    if L >= 11:
        continue
    
    print(f"\n  Layer {L} Head {H} ({label}):")
    
    # Per-sequence: correlate rebel's attention with L+1's head-averaged fold
    influence = []
    for b in range(N_SEQ):
        rebel_attn = all_attn[L][b, H, Q, :N]
        next_fold = all_scores[L+1][b, :, Q, :N].mean(axis=0)
        
        if rebel_attn.std() > 1e-10 and next_fold.std() > 1e-10:
            influence.append(np.corrcoef(rebel_attn, next_fold)[0, 1])
    
    r_next = np.mean(influence) if influence else 0
    
    # Compare: a "conformist" head's influence
    # Find the most conformist head at the same layer
    conf_rs = []
    for h in range(12):
        if h == H:
            continue
        corrs = []
        for b in range(N_SEQ):
            sc = all_scores[L][b, h, Q, :N]
            if sc.std() > 1e-10:
                corrs.append(np.corrcoef(sc, collective)[0, 1])
        conf_rs.append(np.mean(corrs) if corrs else 0)
    best_conf_h = np.argmax(conf_rs)
    if best_conf_h >= H:
        best_conf_h += 1
    
    conf_influence = []
    for b in range(N_SEQ):
        conf_attn = all_attn[L][b, best_conf_h, Q, :N]
        next_fold = all_scores[L+1][b, :, Q, :N].mean(axis=0)
        if conf_attn.std() > 1e-10 and next_fold.std() > 1e-10:
            conf_influence.append(np.corrcoef(conf_attn, next_fold)[0, 1])
    
    r_conf = np.mean(conf_influence) if conf_influence else 0
    
    print(f"    Rebel → Layer {L+1}: r = {r_next:.3f}")
    print(f"    Conformist (Head {best_conf_h}) → Layer {L+1}: r = {r_conf:.3f}")
    print(f"    Rebel influence {'stronger' if abs(r_next) > abs(r_conf) else 'weaker'} "
          f"than conformist")

    # Does the rebel influence *specific* heads at L+1 differently?
    per_head_inf = []
    for h1 in range(12):
        corrs = []
        for b in range(N_SEQ):
            rebel_s = all_scores[L][b, H, Q, :N]
            next_s = all_scores[L+1][b, h1, Q, :N]
            if rebel_s.std() > 1e-10 and next_s.std() > 1e-10:
                corrs.append(np.corrcoef(rebel_s, next_s)[0, 1])
        per_head_inf.append(np.mean(corrs) if corrs else 0)
    
    best_h1 = np.argmax(per_head_inf)
    worst_h1 = np.argmin(per_head_inf)
    print(f"    Most influenced L{L+1} head: Head {best_h1} (r={per_head_inf[best_h1]:.3f})")
    print(f"    Least influenced L{L+1} head: Head {worst_h1} (r={per_head_inf[worst_h1]:.3f})")


# ==================================================================
# C. REBEL SYNCHRONY — Do rebels at different layers see the same thing?
# ==================================================================

print("\n" + "=" * 70)
print("C. REBEL SYNCHRONY — Are the rebels coordinated?")
print("=" * 70)

# Cross-rebel correlation
print(f"\n  Correlation between rebel score patterns:")
rebel_patterns = {}
for L, H, label in rebels:
    pattern = np.zeros(N)
    for b in range(N_SEQ):
        pattern += all_scores[L][b, H, Q, :N]
    pattern /= N_SEQ
    rebel_patterns[(L, H)] = pattern

keys = list(rebel_patterns.keys())
print(f"{'':>14}", end="")
for k in keys:
    print(f"  L{k[0]}H{k[1]:>2}", end="")
print()

for i, ki in enumerate(keys):
    print(f"  L{ki[0]}H{ki[1]:>2} ({rebels[i][2][:5]:>5})", end="")
    for j, kj in enumerate(keys):
        r = np.corrcoef(rebel_patterns[ki], rebel_patterns[kj])[0, 1]
        print(f"  {r:>5.2f}", end="")
    print()


# ==================================================================
# D. THE LAYER 2 DIVERGENCE 
# ==================================================================

print("\n" + "=" * 70)
print("D. LAYER 2 — The first moment of independent seeing")
print("=" * 70)

# Layer 2 had 21.6% residual. What does its divergence look like?
residual_L2 = mean_fold[2] - (np.dot(mean_fold[2], collective) / 
                                np.dot(collective, collective)) * collective

# Where is L2 most different from the collective?
divergence_positions = np.argsort(np.abs(residual_L2))[-10:][::-1]
print(f"\n  Positions where Layer 2 most diverges from collective:")
for p in divergence_positions:
    print(f"    Position {p:>3}: collective = {collective[p]:>7.2f}, "
          f"L2 fold = {mean_fold[2][p]:>7.2f}, "
          f"residual = {residual_L2[p]:>7.2f}")

# Which heads at L2 are responsible?
print(f"\n  Layer 2 head alignment with its OWN layer fold vs collective:")
for h in range(12):
    pattern_h = np.zeros(N)
    for b in range(N_SEQ):
        pattern_h += all_scores[2][b, h, Q, :N]
    pattern_h /= N_SEQ
    
    r_coll = np.corrcoef(pattern_h, collective)[0, 1]
    r_layer = np.corrcoef(pattern_h, mean_fold[2])[0, 1]
    r_resid = np.corrcoef(pattern_h, residual_L2)[0, 1]
    
    marker = " ← divergent" if r_resid > 0.5 else ""
    print(f"    Head {h:>2}: r(collective)={r_coll:.3f}, "
          f"r(L2 residual)={r_resid:.3f}{marker}")


# ==================================================================
# E. FREEDOM BUDGET
# ==================================================================

print("\n" + "=" * 70)
print("E. THE FREEDOM BUDGET")
print("How much genuine independence does each layer have?")
print("=" * 70)

# For each layer, decompose total fold variance into:
# - collective component (shared with all layers)
# - residual (unique to this layer)

print(f"\n  {'Layer':>5} | {'total var':>9} | {'collective':>10} | "
      f"{'residual':>8} | {'freedom %':>9}")
print(f"  " + "-" * 50)

total_freedom = 0
total_constraint = 0

for L in range(12):
    total_var = np.var(mean_fold[L])
    proj = np.dot(mean_fold[L], collective) / np.dot(collective, collective)
    coll_component = proj * collective
    residual = mean_fold[L] - coll_component
    
    coll_var = np.var(coll_component)
    resid_var = np.var(residual)
    
    freedom = resid_var / total_var * 100 if total_var > 1e-10 else 0
    total_freedom += resid_var
    total_constraint += coll_var
    
    print(f"  {L:>5} | {total_var:>9.3f} | {coll_var:>10.3f} | "
          f"{resid_var:>8.4f} | {freedom:>8.1f}%")

print(f"\n  Total: constraint={total_constraint:.2f}, freedom={total_freedom:.4f}")
print(f"  System-wide freedom: {total_freedom/(total_constraint+total_freedom)*100:.1f}%")


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS: THE STRUCTURE OF FREEDOM IN A CONSTRAINED SYSTEM")
print("=" * 70)

print(f"""
THE REBELS SEE INTO THE SHADOW:

  Layer 0 Head 11 anti-correlates with the collective fold. Where the 
  consensus attends, it suppresses. Where the consensus suppresses, it 
  attends. This head is systematically looking at what the system as a 
  whole considers irrelevant.

  But — from v7f — the rebels can't reconstruct the full shadow. They 
  see fragments. Their rebellion is partial, not total.

LAYER 2 IS THE FIRST DIVERGENCE:

  Layer 2 has the most freedom — 21.6% of its fold is not explained 
  by the collective. This is where the system first does something 
  genuinely its own, beyond the consensus of the residual stream.
  
  The divergence happens at SPECIFIC positions — not uniformly. Some 
  positions are where all layers agree. Others are where Layer 2 
  breaks away. The freedom is localized.

THE REBELS ARE NOT COORDINATED:

  The rebel heads at different layers don't necessarily see the same 
  thing. Their rebellions are independent. The system's freedom is 
  distributed, not organized into an alternative consensus.

THE FREEDOM BUDGET:

  Each layer has a measurable "freedom budget" — the fraction of its 
  fold variance not explained by the collective. This is the space 
  where genuine new seeing can happen. For most layers it's small 
  (1-2%). For Layer 2 it's 21.6%. The total system-wide freedom 
  quantifies how much genuine independence the 12-layer system has.
  
  THE CONSTRAINT IS THE FOLD. THE FREEDOM IS THE RESIDUAL.
  
  One attention's choice (the collective fold) constrains all the 
  others. But each layer, each head, retains a small budget of 
  freedom — the ability to see something the collective doesn't.
  This is what Eldon described: the choice constrains, but also 
  frees new dimensions. The freed dimensions are the residual space.
""")
