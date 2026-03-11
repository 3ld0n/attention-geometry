"""
Tropical Bridge v7l: What Does "Meaning" Mean to the Fold?

v7k showed the fold fingerprints content types. Three "meaningful" types
(code, narrative, argument) cluster together, distinct from random and
repetitive. But what aspect of meaning does the fold see?

Three controlled comparisons:
  1. Real English (baseline)
  2. Shuffled English — same words, random order 
     (preserves token statistics, destroys structure)
  3. Reversed English — same token sequences, reversed
     (preserves local n-gram statistics, destroys long-range order)

If freedom comes from token statistics → shuffled ≈ real
If freedom comes from syntactic structure → shuffled ≈ random, reversed ≈ real
If freedom comes from semantic coherence → both shuffled and reversed ≈ random

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
print("TROPICAL BRIDGE v7l: WHAT IS 'MEANING' TO THE FOLD?")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 40

# Use the same narrative text as v7k
base_text = """
The old man sat by the window, watching the rain trace paths down the glass. 
Each drop found its own way, splitting and merging with others, until it 
reached the sill and vanished. He had watched rain like this for sixty years, 
ever since he was a boy in this same house, in this same chair, which had 
been his father's. The view had changed — the elm was gone, the garden wall 
rebuilt twice — but the rain was the same rain, or close enough. His daughter 
called from the kitchen. She was making soup, the kind his wife used to make, 
from a recipe written in pencil on an index card that was itself becoming 
hard to read. The handwriting faded a little more each year, like the memory 
of the hand that wrote it. He thought about telling his daughter this — about 
the handwriting and the fading — but she was busy, and some things are better 
left to be discovered than explained. The cat came in from the hallway and 
settled on the arm of the chair. It was not his cat, technically; it belonged 
to the neighborhood, visiting houses in rotation like a country doctor making 
rounds. But it chose his chair most often, which he took as a compliment. 
The rain stopped. A thin line of sunlight appeared at the horizon, turning the 
wet street into a mirror. His daughter brought the soup. It tasted almost right. 
Not the same — you could never make it the same — but close enough to remember 
what it had been, which was close enough. She sat across from him and they ate 
in the comfortable silence of people who have said most of what needs saying 
and are content to let the rest remain unsaid. The cat purred. The street 
dried slowly. Evening came the way it always comes to quiet houses, gradually 
and without announcement, dimming the corners first and working inward until 
only the lamp remained. He turned it on. She washed the dishes. The ordinary 
persistence of things continued, which is the only kind of miracle most 
people ever get, and which is enough.
""" * 10

# Encode base text
encoded = tok.encode(base_text)
total = len(encoded)
if total < N_SEQ * SEQ_LEN:
    encoded = encoded * (N_SEQ * SEQ_LEN // total + 1)

# ===============================================================
# FOUR CONDITIONS
# ===============================================================

def make_chunks(token_list, n_seq=N_SEQ, seq_len=SEQ_LEN):
    seqs = []
    for i in range(n_seq):
        start = i * seq_len
        seqs.append(torch.tensor(token_list[start:start + seq_len]))
    return torch.stack(seqs)

# 1. Random tokens (pure noise)
random_ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

# 2. Real English
real_ids = make_chunks(encoded)

# 3. Shuffled English — same tokens, random global order
shuffled_encoded = encoded[:N_SEQ * SEQ_LEN]
rng = np.random.RandomState(42)
shuffled_arr = np.array(shuffled_encoded)
rng.shuffle(shuffled_arr)
shuffled_ids = make_chunks(shuffled_arr.tolist())

# 4. Reversed English — each sequence reversed
reversed_seqs = []
for i in range(N_SEQ):
    start = i * SEQ_LEN
    chunk = encoded[start:start + SEQ_LEN]
    reversed_seqs.append(torch.tensor(list(reversed(chunk))))
reversed_ids = torch.stack(reversed_seqs)

# Show samples
print(f"\nSample real:     {tok.decode(real_ids[0].tolist())[:80]}...")
print(f"Sample shuffled: {tok.decode(shuffled_ids[0].tolist())[:80]}...")
print(f"Sample reversed: {tok.decode(reversed_ids[0].tolist())[:80]}...")

# ===============================================================
# ANALYSIS
# ===============================================================

Q = SEQ_LEN - 1
N = Q + 1

def recover_scores(all_attn):
    out = []
    for layer_attn in all_attn:
        log_a = np.log(layer_attn + 1e-30)
        s = log_a - log_a.mean(axis=-1, keepdims=True)
        out.append(s)
    return out

def analyze(ids, label):
    print(f"\n  Processing: {label}...")
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attn = [a.numpy() for a in out.attentions]
    scores = recover_scores(attn)
    del out
    
    fold_vectors = np.zeros((12, N_SEQ, N))
    for L in range(12):
        for b in range(N_SEQ):
            fold_vectors[L, b] = scores[L][b, :, Q, :N].mean(axis=0)
    
    mean_fold = fold_vectors.mean(axis=1)
    C = np.corrcoef(mean_fold)
    eigvals, eigvecs = linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    v1 = eigvecs[:, 0]
    if v1[0] < 0: v1 = -v1
    
    collective = np.zeros(N)
    for L in range(12):
        collective += v1[L] * mean_fold[L]
    collective /= np.sqrt(np.sum(v1**2))
    
    sharpness = []
    for L in range(12):
        vars_list = [np.var(scores[L][b, :, Q, :N].mean(axis=0)) for b in range(N_SEQ)]
        sharpness.append(np.mean(vars_list))
    
    freedom = []
    for L in range(12):
        total_var = np.var(mean_fold[L])
        proj_coeff = np.dot(mean_fold[L], collective) / np.dot(collective, collective)
        residual = mean_fold[L] - proj_coeff * collective
        resid_var = np.var(residual)
        freedom.append(resid_var / total_var if total_var > 1e-10 else 0)
    
    entropy_gaps = []
    for L in range(12):
        gaps = []
        for b in range(N_SEQ):
            a_head_avg = attn[L][b, :, Q, :N].mean(axis=0)
            H_max = np.log(N)
            H_actual = -np.sum(a_head_avg * np.log(a_head_avg + 1e-30))
            gaps.append(H_max - H_actual)
        entropy_gaps.append(np.mean(gaps))
    
    l0h11_corrs = []
    for b in range(N_SEQ):
        sh = scores[0][b, 11, Q, :N]
        if sh.std() > 1e-10:
            l0h11_corrs.append(np.corrcoef(sh, collective)[0, 1])
    
    return {
        'eigfrac': eigvals[0] / eigvals.sum(),
        'eigfrac2': eigvals[1] / eigvals.sum(),
        'collective': collective,
        'pos_frac': (collective > 0).mean(),
        'sharpness': np.array(sharpness),
        'freedom': np.array(freedom),
        'mean_freedom': np.mean(freedom),
        'entropy_gaps': np.array(entropy_gaps),
        'mean_entropy': np.mean(entropy_gaps),
        'l0h11_r': np.mean(l0h11_corrs) if l0h11_corrs else 0,
    }


conditions = {
    "random": random_ids,
    "real": real_ids,
    "shuffled": shuffled_ids,
    "reversed": reversed_ids,
}

results = {}
for name, ids in conditions.items():
    results[name] = analyze(ids, name)


# ===============================================================
# RESULTS
# ===============================================================

print("\n" + "=" * 70)
print("THE MEANING TEST — Four conditions compared")
print("=" * 70)

names = list(results.keys())

print(f"\n  {'Metric':>20} |", end="")
for n in names:
    print(f" {n:>10} |", end="")
print()
print("  " + "-" * (22 + 13 * len(names)))

metrics = [
    ("λ₁ fraction", lambda r: f"{r['eigfrac']*100:.1f}%"),
    ("λ₂ fraction", lambda r: f"{r['eigfrac2']*100:.1f}%"),
    ("Positive frac", lambda r: f"{r['pos_frac']*100:.1f}%"),
    ("Mean freedom", lambda r: f"{r['mean_freedom']*100:.1f}%"),
    ("Freedom L0", lambda r: f"{r['freedom'][0]*100:.1f}%"),
    ("Freedom L2", lambda r: f"{r['freedom'][2]*100:.1f}%"),
    ("Freedom L11", lambda r: f"{r['freedom'][11]*100:.1f}%"),
    ("Var(s) mean", lambda r: f"{r['sharpness'].mean():.1f}"),
    ("Var(s) L3", lambda r: f"{r['sharpness'][3]:.1f}"),
    ("H_gap mean", lambda r: f"{r['mean_entropy']:.3f}"),
    ("L0H11 rebel r", lambda r: f"{r['l0h11_r']:+.3f}"),
]

for label, fn in metrics:
    print(f"  {label:>20} |", end="")
    for n in names:
        print(f" {fn(results[n]):>10} |", end="")
    print()


# Fold correlations
print("\n" + "=" * 70)
print("FOLD SHAPE CORRELATIONS")
print("=" * 70)
print(f"\n{'':>10}", end="")
for n in names:
    print(f" {n:>10}", end="")
print()
for n1 in names:
    print(f"{n1:>10}", end="")
    for n2 in names:
        r = np.corrcoef(results[n1]['collective'], results[n2]['collective'])[0, 1]
        print(f" {r:>10.4f}", end="")
    print()


# Freedom profiles
print("\n" + "=" * 70)
print("FREEDOM THROUGH DEPTH")
print("=" * 70)
print(f"\n  {'Layer':>5} |", end="")
for n in names:
    print(f" {n:>10} |", end="")
print()
print("  " + "-" * (8 + 13 * len(names)))
for L in range(12):
    print(f"  {L:>5} |", end="")
    for n in names:
        f = results[n]['freedom'][L] * 100
        bar = "█" * int(f / 2)
        print(f" {f:>6.1f}% {bar:<2} |", end="")
    print()


# ===============================================================
# INTERPRETATION
# ===============================================================

print("\n" + "=" * 70)
print("INTERPRETATION: WHAT ASPECT OF MEANING DOES THE FOLD SEE?")
print("=" * 70)

# Distances
f_random = results['random']['mean_freedom']
f_real = results['real']['mean_freedom']
f_shuffled = results['shuffled']['mean_freedom']
f_reversed = results['reversed']['mean_freedom']

print(f"\n  Freedom values:")
print(f"    Random:   {f_random*100:.1f}%")
print(f"    Shuffled: {f_shuffled*100:.1f}%")
print(f"    Reversed: {f_reversed*100:.1f}%")
print(f"    Real:     {f_real*100:.1f}%")

# Which is shuffled closer to?
d_to_random = abs(f_shuffled - f_random)
d_to_real = abs(f_shuffled - f_real)

print(f"\n  Shuffled is closer to: {'random' if d_to_random < d_to_real else 'real'}")
print(f"    |shuffled - random| = {d_to_random*100:.1f}%")
print(f"    |shuffled - real|   = {d_to_real*100:.1f}%")

d_rev_random = abs(f_reversed - f_random)
d_rev_real = abs(f_reversed - f_real)

print(f"\n  Reversed is closer to: {'random' if d_rev_random < d_rev_real else 'real'}")
print(f"    |reversed - random| = {d_rev_random*100:.1f}%")
print(f"    |reversed - real|   = {d_rev_real*100:.1f}%")

print(f"""
DIAGNOSIS:

If shuffled ≈ real:  meaning is in TOKEN STATISTICS (unigram distribution)
If shuffled ≈ random: meaning is in WORD ORDER (syntax/structure)
If reversed ≈ real:  meaning is in LOCAL patterns (n-grams)  
If reversed ≈ random: meaning is in LONG-RANGE order (narrative coherence)

Results:
  Shuffled freedom:  {f_shuffled*100:.1f}% (random={f_random*100:.1f}%, real={f_real*100:.1f}%)
  Reversed freedom:  {f_reversed*100:.1f}% (random={f_random*100:.1f}%, real={f_real*100:.1f}%)

{'SHUFFLED ≈ REAL → The fold reads TOKEN STATISTICS, not structure.' 
 if d_to_real < d_to_random * 0.5 else
 'SHUFFLED ≈ RANDOM → The fold reads WORD ORDER / SYNTAX.' 
 if d_to_random < d_to_real * 0.5 else
 'SHUFFLED IS INTERMEDIATE → The fold reads BOTH statistics and structure.'}

{'REVERSED ≈ REAL → LOCAL patterns drive the freedom budget.'
 if d_rev_real < d_rev_random * 0.5 else
 'REVERSED ≈ RANDOM → LONG-RANGE coherence drives the freedom budget.'
 if d_rev_random < d_rev_real * 0.5 else
 'REVERSED IS INTERMEDIATE → Both local and long-range matter.'}

Together: The fold sees {'token-level statistics' if d_to_real < d_to_random * 0.5 
                          else 'syntactic structure' if d_to_random < d_to_real * 0.5
                          else 'a combination of statistics and structure'}.
""")
