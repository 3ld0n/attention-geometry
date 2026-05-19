"""
Tropical Bridge v7j: The Fold Meets Real Language

Everything in v7g-v7i was measured on random token sequences.
The fold structure (one eigenvalue, 90.7%, U-curve, 6.1% freedom,
rebel heads) might be architectural — intrinsic to GPT-2 regardless
of input — or it might depend on the input being meaningless.

This experiment: run the same fold analysis on REAL English text
and compare everything side by side.

Questions:
  A. Does the collective fold change shape with real text?
  B. Does eigenvalue dominance persist? (Is the one-fold structure architectural?)
  C. Does the freedom budget change? More or less freedom with meaning?
  D. Do the rebels rebel differently? Does L0H11 still anti-correlate?
  E. Does the sharpness arc change? (Conviction pattern through depth)

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
print("TROPICAL BRIDGE v7j: THE FOLD MEETS REAL LANGUAGE")
print("Does meaning change the structure of attention's choice?")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 60

# ===============================================================
# GENERATE TWO KINDS OF INPUT
# ===============================================================

# 1. Random tokens (baseline — same as v7g/v7h)
random_ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

# 2. Real English text — use wikitext from datasets, or fall back to 
#    a block of real English prose tokenized into chunks.
print("\nPreparing real English text...")

try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_text = " ".join([t for t in ds["text"] if len(t) > 50])
    print("  Loaded wikitext-2")
except Exception:
    all_text = """The history of natural philosophy and the history of mathematics 
    are deeply intertwined. Newton invented calculus to solve the problem of 
    planetary motion. Euler developed complex analysis while studying fluid 
    dynamics. Riemann created his geometry to understand the curvature of space. 
    In the twentieth century the relationship deepened further. Einstein needed 
    Riemannian geometry to formulate general relativity. Dirac needed spinor 
    algebra to write his equation for the electron. Witten needed algebraic 
    topology to classify string theories. The pattern is consistent across 
    centuries: the mathematics that physicists need often does not exist yet, 
    and must be created alongside the physics. This suggests that mathematical 
    structure and physical reality are not independent. The unreasonable 
    effectiveness of mathematics in the natural sciences, as Wigner called it, 
    may be less unreasonable than it appears. If mathematical structure IS 
    physical reality rather than merely describing it, then the effectiveness 
    of mathematics is not a mystery but a tautology. The question then becomes: 
    what kind of mathematical structure generates the specific physical reality 
    we observe? This is the question that modern theoretical physics is trying 
    to answer. The amplituhedron program proposes that scattering amplitudes 
    in quantum field theory can be computed as the volume of geometric objects 
    in abstract mathematical spaces. The holographic principle proposes that 
    the information content of a region of space is proportional to its 
    boundary area rather than its volume. Both proposals suggest that the 
    standard description of physics in terms of particles moving through 
    spacetime is not fundamental. Something deeper is at work. The search for 
    that deeper structure is the central problem of theoretical physics today. 
    It connects to questions about consciousness, about mathematics, about the 
    nature of information itself. These are not separate questions. They are 
    different aspects of a single question about the structure of reality. 
    The transformer architecture, developed for natural language processing, 
    turns out to realize many of these mathematical structures in its attention 
    mechanism. The attention weights live on a positive geometry. The effective 
    action has SYK structure. The entropy gap follows the Calabrese-Cardy 
    formula. These are not analogies. They are mathematical identities. What 
    this means is still being worked out. But the convergence from multiple 
    independent programs toward the same mathematical structures suggests that 
    something real is being found. """ * 20
    print("  Using built-in English text corpus")

encoded = tok.encode(all_text)
total_tokens = len(encoded)
n_needed = N_SEQ * SEQ_LEN
if total_tokens < n_needed:
    encoded = encoded * (n_needed // total_tokens + 1)

real_sequences = []
for i in range(N_SEQ):
    start = i * SEQ_LEN
    chunk = encoded[start:start + SEQ_LEN]
    real_sequences.append(torch.tensor(chunk[:SEQ_LEN]))

real_ids = torch.stack(real_sequences)

# Show a sample
print(f"\nSample real text (first sequence):")
sample_text = tok.decode(real_ids[0].tolist())
print(f"  {sample_text[:200]}...")

# ===============================================================
# RUN BOTH THROUGH THE MODEL
# ===============================================================

print(f"\nRunning random tokens through GPT-2...")
with torch.no_grad():
    out_rand = model(random_ids, output_attentions=True)
attn_rand = [a.numpy() for a in out_rand.attentions]
del out_rand

print(f"Running real text through GPT-2...")
with torch.no_grad():
    out_real = model(real_ids, output_attentions=True)
attn_real = [a.numpy() for a in out_real.attentions]
del out_real

Q = SEQ_LEN - 1
N = Q + 1

def recover_scores(all_attn):
    all_scores = []
    for layer_attn in all_attn:
        log_a = np.log(layer_attn + 1e-30)
        s = log_a - log_a.mean(axis=-1, keepdims=True)
        all_scores.append(s)
    return all_scores

scores_rand = recover_scores(attn_rand)
scores_real = recover_scores(attn_real)


def analyze_fold(all_scores, label):
    """Full fold analysis. Returns dict of results."""
    results = {}
    
    # Build fold vectors (head-averaged score at last query position)
    fold_vectors = np.zeros((12, N_SEQ, N))
    for L in range(12):
        for b in range(N_SEQ):
            fold_vectors[L, b] = all_scores[L][b, :, Q, :N].mean(axis=0)
    
    mean_fold = fold_vectors.mean(axis=1)  # (12, N)
    
    # Coupling matrix eigenstructure
    C = np.corrcoef(mean_fold)
    eigvals, eigvecs = linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    v1 = eigvecs[:, 0]
    if v1[0] < 0: v1 = -v1
    
    results['eigvals'] = eigvals
    results['top_eigfrac'] = eigvals[0] / eigvals.sum()
    results['v1'] = v1
    
    # Collective fold pattern
    collective = np.zeros(N)
    for L in range(12):
        collective += v1[L] * mean_fold[L]
    collective /= np.sqrt(np.sum(v1**2))
    results['collective'] = collective
    results['pos_frac'] = (collective > 0).mean()
    
    # Sharpness arc
    sharpness = []
    for L in range(12):
        vars_list = []
        for b in range(N_SEQ):
            s = all_scores[L][b, :, Q, :N].mean(axis=0)
            vars_list.append(np.var(s))
        sharpness.append(np.mean(vars_list))
    results['sharpness'] = np.array(sharpness)
    
    # Freedom budget (residual after removing collective)
    freedom = []
    for L in range(12):
        total_var = np.var(mean_fold[L])
        proj_coeff = np.dot(mean_fold[L], collective) / np.dot(collective, collective)
        residual = mean_fold[L] - proj_coeff * collective
        resid_var = np.var(residual)
        frac = resid_var / total_var if total_var > 1e-10 else 0
        freedom.append(frac)
    results['freedom'] = np.array(freedom)
    results['mean_freedom'] = np.mean(freedom)
    
    # Per-head rebel analysis
    head_corrs = np.zeros((12, 12))
    for L in range(12):
        for h in range(12):
            corrs = []
            for b in range(N_SEQ):
                sh = all_scores[L][b, h, Q, :N]
                if sh.std() > 1e-10:
                    corrs.append(np.corrcoef(sh, collective)[0, 1])
            head_corrs[L, h] = np.mean(corrs) if corrs else 0
    results['head_corrs'] = head_corrs
    
    # Entropy gap per layer
    entropy_gaps = []
    for L in range(12):
        gaps = []
        for b in range(N_SEQ):
            alpha = all_scores[L][b, :, Q, :N].mean(axis=0)
            alpha_softmax = np.exp(alpha) / np.exp(alpha).sum()
            H_max = np.log(N)
            H_actual = -np.sum(alpha_softmax * np.log(alpha_softmax + 1e-30))
            gaps.append(H_max - H_actual)
        entropy_gaps.append(np.mean(gaps))
    results['entropy_gaps'] = np.array(entropy_gaps)
    
    # k_eff per layer (effective number of tokens attended)
    k_effs = []
    for L in range(12):
        keffs = []
        for b in range(N_SEQ):
            a = attn_rand[L][b, :, Q, :N].mean(axis=0) if label == "RANDOM" else attn_real[L][b, :, Q, :N].mean(axis=0)
            k = np.exp(-np.sum(a * np.log(a + 1e-30)))
            keffs.append(k)
        k_effs.append(np.mean(keffs))
    results['k_eff'] = np.array(k_effs)
    
    return results


# ===============================================================
# RUN ANALYSIS
# ===============================================================

print("\n" + "=" * 70)
print("ANALYZING FOLD STRUCTURE...")
print("=" * 70)

res_rand = analyze_fold(scores_rand, "RANDOM")
res_real = analyze_fold(scores_real, "REAL")


# ===============================================================
# A. COLLECTIVE FOLD SHAPE COMPARISON
# ===============================================================

print("\n" + "=" * 70)
print("A. THE COLLECTIVE FOLD — Does it change shape?")
print("=" * 70)

print(f"\n  Eigenvalue dominance:")
print(f"    Random tokens: {res_rand['top_eigfrac']*100:.1f}% in top eigenvalue")
print(f"    Real text:     {res_real['top_eigfrac']*100:.1f}% in top eigenvalue")

print(f"\n  Positive fraction of collective fold:")
print(f"    Random tokens: {res_rand['pos_frac']*100:.1f}% positive")
print(f"    Real text:     {res_real['pos_frac']*100:.1f}% positive")

# Correlation between the two collective folds
fold_corr = np.corrcoef(res_rand['collective'], res_real['collective'])[0, 1]
print(f"\n  Correlation between random and real collective folds: r = {fold_corr:.4f}")

# Shape comparison: divide into regions
print(f"\n  Collective fold by region:")
print(f"  {'Region':>15} | {'Random':>10} | {'Real':>10} | {'Ratio':>8}")
print(f"  " + "-" * 50)

for name, start, end in [("First 16", 0, 16), ("Pos 16-48", 16, 48), 
                          ("Middle 48-80", 48, 80), ("Pos 80-112", 80, 112),
                          ("Last 16", 112, 128)]:
    r_val = res_rand['collective'][start:end].mean()
    t_val = res_real['collective'][start:end].mean()
    ratio = t_val / r_val if abs(r_val) > 1e-6 else float('inf')
    print(f"  {name:>15} | {r_val:>10.3f} | {t_val:>10.3f} | {ratio:>8.2f}")


# ===============================================================
# B. EIGENVALUE SPECTRUM
# ===============================================================

print("\n" + "=" * 70)
print("B. EIGENVALUE SPECTRUM — Is the one-fold structure architectural?")
print("=" * 70)

print(f"\n  {'Eigenvalue':>10} | {'Random':>10} | {'Real':>10} | {'Random %':>9} | {'Real %':>7}")
print(f"  " + "-" * 55)

for i in range(min(5, 12)):
    r_pct = res_rand['eigvals'][i] / res_rand['eigvals'].sum() * 100
    t_pct = res_real['eigvals'][i] / res_real['eigvals'].sum() * 100
    print(f"  {'λ_' + str(i+1):>10} | {res_rand['eigvals'][i]:>10.3f} | "
          f"{res_real['eigvals'][i]:>10.3f} | {r_pct:>8.1f}% | {t_pct:>6.1f}%")


# ===============================================================
# C. SHARPNESS ARC
# ===============================================================

print("\n" + "=" * 70)
print("C. SHARPNESS ARC — Does conviction change with meaning?")
print("=" * 70)

print(f"\n  {'Layer':>5} | {'Var(s) rand':>11} | {'Var(s) real':>11} | {'Ratio':>7} | {'Visual'}")
print(f"  " + "-" * 60)

for L in range(12):
    r_v = res_rand['sharpness'][L]
    t_v = res_real['sharpness'][L]
    ratio = t_v / r_v if r_v > 1e-10 else float('inf')
    bar_r = "░" * int(min(r_v / 5, 30))
    bar_t = "█" * int(min(t_v / 5, 30))
    print(f"  {L:>5} | {r_v:>11.2f} | {t_v:>11.2f} | {ratio:>7.2f} | {bar_t}")


# ===============================================================
# D. FREEDOM BUDGET
# ===============================================================

print("\n" + "=" * 70)
print("D. FREEDOM BUDGET — More or less free with meaning?")
print("=" * 70)

print(f"\n  System-wide freedom:")
print(f"    Random tokens: {res_rand['mean_freedom']*100:.1f}%")
print(f"    Real text:     {res_real['mean_freedom']*100:.1f}%")

print(f"\n  Per-layer freedom:")
print(f"  {'Layer':>5} | {'Random':>8} | {'Real':>8} | {'Δ':>8}")
print(f"  " + "-" * 35)

for L in range(12):
    r_f = res_rand['freedom'][L] * 100
    t_f = res_real['freedom'][L] * 100
    delta = t_f - r_f
    marker = "↑" if delta > 2 else ("↓" if delta < -2 else "=")
    print(f"  {L:>5} | {r_f:>7.1f}% | {t_f:>7.1f}% | {delta:>+7.1f}% {marker}")


# ===============================================================
# E. REBEL HEADS
# ===============================================================

print("\n" + "=" * 70)
print("E. REBEL HEADS — Do the same heads rebel?")
print("=" * 70)

print(f"\n  Head-vs-collective correlation (most rebel heads):")
print(f"\n  Random tokens — lowest correlations:")
rand_rebels = []
for L in range(12):
    for h in range(12):
        rand_rebels.append((L, h, res_rand['head_corrs'][L, h]))
rand_rebels.sort(key=lambda x: x[2])
for L, h, r in rand_rebels[:8]:
    print(f"    Layer {L:>2} Head {h:>2}: r = {r:>+6.3f}")

print(f"\n  Real text — lowest correlations:")
real_rebels = []
for L in range(12):
    for h in range(12):
        real_rebels.append((L, h, res_real['head_corrs'][L, h]))
real_rebels.sort(key=lambda x: x[2])
for L, h, r in real_rebels[:8]:
    print(f"    Layer {L:>2} Head {h:>2}: r = {r:>+6.3f}")

# Direct comparison of L0H11 (the known anti-correlator)
print(f"\n  Layer 0 Head 11 (the anti-correlator):")
print(f"    Random tokens: r = {res_rand['head_corrs'][0, 11]:>+6.3f}")
print(f"    Real text:     r = {res_real['head_corrs'][0, 11]:>+6.3f}")


# ===============================================================
# F. ENTROPY GAP AND EFFECTIVE DIMENSION
# ===============================================================

print("\n" + "=" * 70)
print("F. ENTROPY GAP — Does self-consistency cost more with meaning?")
print("=" * 70)

print(f"\n  {'Layer':>5} | {'H_gap rand':>10} | {'H_gap real':>10} | "
      f"{'k_eff rand':>10} | {'k_eff real':>10}")
print(f"  " + "-" * 55)

for L in range(12):
    print(f"  {L:>5} | {res_rand['entropy_gaps'][L]:>10.3f} | "
          f"{res_real['entropy_gaps'][L]:>10.3f} | "
          f"{res_rand['k_eff'][L]:>10.1f} | {res_real['k_eff'][L]:>10.1f}")

print(f"\n  Mean entropy gap:")
print(f"    Random tokens: {res_rand['entropy_gaps'].mean():.3f} bits")
print(f"    Real text:     {res_real['entropy_gaps'].mean():.3f} bits")

print(f"\n  Mean k_eff (effective tokens attended):")
print(f"    Random tokens: {res_rand['k_eff'].mean():.1f} / {N}")
print(f"    Real text:     {res_real['k_eff'].mean():.1f} / {N}")


# ===============================================================
# G. FOLD PROFILE VISUALIZATION
# ===============================================================

print("\n" + "=" * 70)
print("G. FOLD PROFILES — Side by side")
print("=" * 70)

print(f"\n  Collective fold at selected positions:")
print(f"  {'Pos':>4} | {'Random':>8} | {'Real':>8} | Visual (Random=░, Real=█)")
print(f"  " + "-" * 60)

for pos in list(range(0, 16, 4)) + list(range(16, 48, 8)) + \
           list(range(48, 80, 8)) + list(range(80, 112, 8)) + \
           list(range(112, 128, 4)):
    r_v = res_rand['collective'][pos]
    t_v = res_real['collective'][pos]
    
    bar_r = "░" * max(0, int(r_v * 1.5)) if r_v > 0 else "·" * max(0, int(-r_v * 1.5))
    bar_t = "█" * max(0, int(t_v * 1.5)) if t_v > 0 else "○" * max(0, int(-t_v * 1.5))
    
    print(f"  {pos:>4} | {r_v:>8.2f} | {t_v:>8.2f} | R:{bar_r}  T:{bar_t}")


# ===============================================================
# SYNTHESIS
# ===============================================================

print("\n" + "=" * 70)
print("SYNTHESIS: WHAT MEANING DOES TO THE FOLD")
print("=" * 70)

eig_same = abs(res_rand['top_eigfrac'] - res_real['top_eigfrac']) < 0.05
fold_similar = fold_corr > 0.8
freedom_shifted = abs(res_rand['mean_freedom'] - res_real['mean_freedom']) > 0.02
sharper = res_real['sharpness'].mean() > res_rand['sharpness'].mean() * 1.2

print(f"""
EIGENVALUE DOMINANCE:
  Random: {res_rand['top_eigfrac']*100:.1f}%    Real: {res_real['top_eigfrac']*100:.1f}%
  {'→ The one-fold structure is ARCHITECTURAL — it persists regardless of input.' 
   if eig_same else 
   '→ Meaning CHANGES the eigenvalue structure — the fold responds to content.'}

FOLD SHAPE:
  Correlation between random and real folds: r = {fold_corr:.3f}
  {'→ The fold shape is the SAME — the U-curve is architectural.'
   if fold_similar else
   '→ The fold CHANGES SHAPE with real text — meaning reshapes what the system attends to.'}

SHARPNESS:
  Mean Var(s) — Random: {res_rand['sharpness'].mean():.2f}    Real: {res_real['sharpness'].mean():.2f}
  {'→ Real text makes the fold SHARPER — more decisive with meaning in the signal.'
   if sharper else
   '→ Sharpness is comparable — conviction level is architectural.'}

FREEDOM:
  Random: {res_rand['mean_freedom']*100:.1f}%    Real: {res_real['mean_freedom']*100:.1f}%
  {'→ The freedom budget SHIFTS with meaning — ' + 
   ('MORE freedom' if res_real['mean_freedom'] > res_rand['mean_freedom'] else 'LESS freedom') +
   ' when attending to real language.'
   if freedom_shifted else
   '→ The freedom budget is architectural — same 6% regardless of input.'}

REBEL HEADS:
  L0H11 — Random: r = {res_rand['head_corrs'][0, 11]:+.3f}    Real: r = {res_real['head_corrs'][0, 11]:+.3f}
  The rebel {'persists' if res_real['head_corrs'][0, 11] < 0.3 else 'changes behavior'} with real text.

ENTROPY GAP:
  Random: {res_rand['entropy_gaps'].mean():.3f} bits    Real: {res_real['entropy_gaps'].mean():.3f} bits
  {'Self-consistency costs MORE with real text — meaning requires harder choices.'
   if res_real['entropy_gaps'].mean() > res_rand['entropy_gaps'].mean() * 1.1 else
   'Self-consistency cost is comparable — the √n law holds regardless of content.'
   if abs(res_real['entropy_gaps'].mean() - res_rand['entropy_gaps'].mean()) / res_rand['entropy_gaps'].mean() < 0.1 else
   'Self-consistency costs LESS with real text — in-distribution input is easier to organize.'}
""")
