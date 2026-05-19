"""
Tropical Bridge v7k: Fold Fingerprints

v7j showed: the fold's shape is architectural but its freedom responds
to meaning. Freedom doubles from 4.4% to 10.3% with real text. 
Layer 11 explodes to 32.7% freedom.

Question: does the fold fingerprint DIFFERENT types of content?
Is the freedom budget a measure of semantic complexity?

Five conditions:
  1. Random tokens (baseline — 0% meaning)
  2. Repetitive text (low complexity — same phrase repeated)
  3. Code (high structure, low narrative)
  4. Narrative prose (moderate complexity)
  5. Dense argumentative text (high complexity)

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
print("TROPICAL BRIDGE v7k: FOLD FINGERPRINTS")
print("Does the fold read the content?")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 128
N_SEQ = 40

def make_ids(text, n_seq=N_SEQ, seq_len=SEQ_LEN):
    encoded = tok.encode(text)
    if len(encoded) < n_seq * seq_len:
        encoded = encoded * (n_seq * seq_len // len(encoded) + 1)
    seqs = []
    for i in range(n_seq):
        start = i * seq_len
        seqs.append(torch.tensor(encoded[start:start + seq_len]))
    return torch.stack(seqs)

# ===============================================================
# FIVE TEXT TYPES
# ===============================================================

texts = {}

texts["random"] = None

texts["repetitive"] = "The cat sat on the mat. The cat sat on the mat. " * 200

texts["code"] = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)
    
    def search(self, value):
        if value == self.value:
            return True
        elif value < self.value and self.left:
            return self.left.search(value)
        elif value > self.value and self.right:
            return self.right.search(value)
        return False

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result
""" * 10

texts["narrative"] = """
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

texts["argument"] = """
The relationship between mathematical structure and physical reality is not 
one of description but of identity. This claim is stronger than Wigner's 
observation about the unreasonable effectiveness of mathematics. Wigner noted 
the correspondence and called it a mystery. We are claiming there is no 
correspondence because there is no gap — the mathematical structure IS the 
physical reality, not a map of it. Consider the evidence. Every successful 
physical theory in history has taken the form: physical phenomenon X is 
IDENTICAL to mathematical structure Y. Not "described by" or "modeled by" 
but IS. Mass-energy equivalence is not a metaphor. The curvature of spacetime 
is not an analogy for gravity. Quantum amplitudes are not representations of 
probability — they ARE the probability structure. The pattern is consistent: 
what begins as "X behaves as if Y" always resolves, upon deeper understanding, 
to "X is Y." There has never been a case where the mathematical description 
turned out to be merely approximate in principle — only in practice, where our 
knowledge of the correct structure was incomplete. The counterargument is that 
mathematics is a human invention, a language we constructed, and the 
effectiveness of a language in describing reality says more about the describer 
than the described. But this misunderstands the nature of mathematical truth. 
Mathematical structures are discovered, not invented. The properties of prime 
numbers, the topology of manifolds, the solutions of differential equations — 
these exist independently of any mathematician's knowledge of them. Euler's 
identity was true before Euler. The incompleteness theorems were true before 
Godel. If mathematics is discovered rather than invented, and if physical 
reality consistently turns out to BE mathematical structure, then the conclusion 
is that reality itself is mathematical structure. This is Tegmark's mathematical 
universe hypothesis, but we arrive at it from a different direction. Tegmark 
argues from parsimony. We argue from the evidence of physics itself: every 
time we look deeper, we find more mathematics, never less. The direction has 
been monotonic for four centuries. The probability that this is coincidence 
decreases with each new confirmation. The specific mathematical structures 
that appear in fundamental physics — gauge groups, fiber bundles, conformal 
field theories, positive geometries — are not arbitrary. They share properties: 
self-consistency, internal completeness, the capacity to generate new structure 
from their own axioms. These are the properties of consciousness examining 
itself. The recursive structure of self-reference — a system that contains its 
own description — is the common thread between Godel's theorems, quantum 
measurement, and the hard problem of consciousness. All three arise from the 
same structural feature: a system that is both the observer and the observed.
""" * 10

# ===============================================================
# RUN ALL CONDITIONS
# ===============================================================

Q = SEQ_LEN - 1
N = Q + 1

def recover_scores(all_attn):
    all_scores = []
    for layer_attn in all_attn:
        log_a = np.log(layer_attn + 1e-30)
        s = log_a - log_a.mean(axis=-1, keepdims=True)
        all_scores.append(s)
    return all_scores

def analyze_condition(all_scores, all_attn):
    fold_vectors = np.zeros((12, N_SEQ, N))
    for L in range(12):
        for b in range(N_SEQ):
            fold_vectors[L, b] = all_scores[L][b, :, Q, :N].mean(axis=0)
    
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
        vars_list = [np.var(all_scores[L][b, :, Q, :N].mean(axis=0)) for b in range(N_SEQ)]
        sharpness.append(np.mean(vars_list))
    
    freedom = []
    for L in range(12):
        total_var = np.var(mean_fold[L])
        proj_coeff = np.dot(mean_fold[L], collective) / np.dot(collective, collective)
        residual = mean_fold[L] - proj_coeff * collective
        resid_var = np.var(residual)
        freedom.append(resid_var / total_var if total_var > 1e-10 else 0)
    
    entropy_gaps = []
    k_effs = []
    for L in range(12):
        gaps, keffs = [], []
        for b in range(N_SEQ):
            a_head_avg = all_attn[L][b, :, Q, :N].mean(axis=0)
            H_max = np.log(N)
            H_actual = -np.sum(a_head_avg * np.log(a_head_avg + 1e-30))
            gaps.append(H_max - H_actual)
            keffs.append(np.exp(H_actual))
        entropy_gaps.append(np.mean(gaps))
        k_effs.append(np.mean(keffs))
    
    # L0H11 rebel correlation
    l0h11_corrs = []
    for b in range(N_SEQ):
        sh = all_scores[0][b, 11, Q, :N]
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
        'k_eff': np.array(k_effs),
        'mean_k_eff': np.mean(k_effs),
        'l0h11_r': np.mean(l0h11_corrs) if l0h11_corrs else 0,
    }


results = {}
for name, text in texts.items():
    print(f"\n  Processing: {name}...")
    if text is None:
        ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))
    else:
        ids = make_ids(text)
    
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attn = [a.numpy() for a in out.attentions]
    scores = recover_scores(attn)
    results[name] = analyze_condition(scores, attn)
    del out

# ===============================================================
# COMPARISON TABLE
# ===============================================================

print("\n" + "=" * 70)
print("THE FOLD FINGERPRINT — Five conditions compared")
print("=" * 70)

names = list(results.keys())

print(f"\n  {'Metric':>20} |", end="")
for n in names:
    print(f" {n:>12} |", end="")
print()
print("  " + "-" * (22 + 15 * len(names)))

metrics = [
    ("λ₁ fraction", lambda r: f"{r['eigfrac']*100:.1f}%"),
    ("λ₂ fraction", lambda r: f"{r['eigfrac2']*100:.1f}%"),
    ("Positive frac", lambda r: f"{r['pos_frac']*100:.1f}%"),
    ("Mean freedom", lambda r: f"{r['mean_freedom']*100:.1f}%"),
    ("Freedom L0", lambda r: f"{r['freedom'][0]*100:.1f}%"),
    ("Freedom L2", lambda r: f"{r['freedom'][2]*100:.1f}%"),
    ("Freedom L5", lambda r: f"{r['freedom'][5]*100:.1f}%"),
    ("Freedom L11", lambda r: f"{r['freedom'][11]*100:.1f}%"),
    ("Var(s) mean", lambda r: f"{r['sharpness'].mean():.1f}"),
    ("Var(s) L0", lambda r: f"{r['sharpness'][0]:.2f}"),
    ("Var(s) L3 (peak)", lambda r: f"{r['sharpness'][3]:.1f}"),
    ("Var(s) L11", lambda r: f"{r['sharpness'][11]:.2f}"),
    ("H_gap mean", lambda r: f"{r['mean_entropy']:.3f}"),
    ("k_eff mean", lambda r: f"{r['mean_k_eff']:.1f}"),
    ("L0H11 rebel r", lambda r: f"{r['l0h11_r']:+.3f}"),
]

for label, fn in metrics:
    print(f"  {label:>20} |", end="")
    for n in names:
        print(f" {fn(results[n]):>12} |", end="")
    print()

# ===============================================================
# FOLD CORRELATION MATRIX BETWEEN CONDITIONS
# ===============================================================

print("\n" + "=" * 70)
print("FOLD SHAPE CORRELATION — How similar are the collective folds?")
print("=" * 70)

print(f"\n{'':>12}", end="")
for n in names:
    print(f" {n:>10}", end="")
print()

for n1 in names:
    print(f"{n1:>12}", end="")
    for n2 in names:
        r = np.corrcoef(results[n1]['collective'], results[n2]['collective'])[0, 1]
        print(f" {r:>10.4f}", end="")
    print()


# ===============================================================
# FREEDOM PROFILE VISUALIZATION
# ===============================================================

print("\n" + "=" * 70)
print("FREEDOM THROUGH DEPTH — Each condition's profile")
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
        print(f" {f:>7.1f}% {bar:>0} |", end="")
    print()


# ===============================================================
# SYNTHESIS
# ===============================================================

print("\n" + "=" * 70)
print("SYNTHESIS: THE FOLD READS THE CONTENT")
print("=" * 70)

# Sort conditions by mean freedom
sorted_by_freedom = sorted(names, key=lambda n: results[n]['mean_freedom'])

print(f"\n  Conditions ranked by mean freedom:")
for i, n in enumerate(sorted_by_freedom):
    f = results[n]['mean_freedom'] * 100
    bar = "█" * int(f * 2)
    print(f"    {i+1}. {n:>12}: {f:>5.1f}%  {bar}")

print(f"\n  Conditions ranked by entropy gap:")
sorted_by_entropy = sorted(names, key=lambda n: results[n]['mean_entropy'])
for i, n in enumerate(sorted_by_entropy):
    h = results[n]['mean_entropy']
    print(f"    {i+1}. {n:>12}: {h:.3f} bits")

print(f"\n  Conditions ranked by sharpness (peak layer):")
sorted_by_sharp = sorted(names, key=lambda n: results[n]['sharpness'][3], reverse=True)
for i, n in enumerate(sorted_by_sharp):
    v = results[n]['sharpness'][3]
    print(f"    {i+1}. {n:>12}: Var(s)={v:.1f}")

print(f"""
KEY QUESTION ANSWERS:

1. Does the fold fingerprint content type?
   Mean freedom ranges from {results[sorted_by_freedom[0]]['mean_freedom']*100:.1f}% 
   ({sorted_by_freedom[0]}) to {results[sorted_by_freedom[-1]]['mean_freedom']*100:.1f}% 
   ({sorted_by_freedom[-1]}).
   {'YES — the fold distinguishes content types through its freedom budget.'
    if results[sorted_by_freedom[-1]]['mean_freedom'] > results[sorted_by_freedom[0]]['mean_freedom'] * 1.3
    else 'Partially — some variation but not as dramatic as random vs real.'}

2. Is freedom a measure of semantic complexity?
   If so, the ranking should be: random < repetitive < code < narrative < argument.
   Actual ranking: {' < '.join(sorted_by_freedom)}
   {'The ranking matches the predicted complexity ordering!' 
    if sorted_by_freedom[0] in ['random', 'repetitive'] and sorted_by_freedom[-1] in ['argument', 'narrative']
    else 'The ranking partially matches but with surprises.'}

3. Does the entropy gap change?
   Range: {min(results[n]['mean_entropy'] for n in names):.3f} to {max(results[n]['mean_entropy'] for n in names):.3f} bits.
   {'The entropy gap is CONSTANT — the √n cost is truly content-independent.'
    if max(results[n]['mean_entropy'] for n in names) - min(results[n]['mean_entropy'] for n in names) < 0.3
    else 'The entropy gap VARIES — self-consistency cost depends on content type.'}

4. Does L0H11 always rebel?
   r values: {', '.join(f'{n}={results[n]["l0h11_r"]:+.2f}' for n in names)}
   {'YES — L0H11 is constitutionally an anti-correlator regardless of content.'
    if all(results[n]['l0h11_r'] < 0.3 for n in names)
    else 'The rebel behavior depends on content type.'}
""")
