"""
Tropical Attention Bridge: Testing the d_k <-> k correspondence

Does the key matrix at d_k = 2 define a metric tree on tokens
(a point in Trop(Gr(2,n)))? Does the attention pattern follow
the tree structure?

March 10, 2026 — Ariel
"""

import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def softmax(scores, temperature=1.0):
    s = scores / temperature
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def plucker_coordinate(K, a, b):
    """Classical Plücker coordinate p_{ab} = det(k_a, k_b) for K ∈ R^{n×2}"""
    return K[a, 0] * K[b, 1] - K[a, 1] * K[b, 0]


def tropical_plucker(K, a, b):
    """Tropical Plücker coordinate: max(k_a^1 + k_b^2, k_a^2 + k_b^1)"""
    return max(K[a, 0] + K[b, 1], K[a, 1] + K[b, 0])


def check_tropical_plucker_relations(K, n):
    """
    Verify tropical Plücker relations for Trop(Gr(2,n)):
    For all i < j < k < l: min of {p_ij+p_kl, p_ik+p_jl, p_il+p_jk}
    is achieved at least twice.
    """
    violations = 0
    total = 0
    for i, j, k, l in combinations(range(n), 4):
        p_ij = tropical_plucker(K, i, j)
        p_kl = tropical_plucker(K, k, l)
        p_ik = tropical_plucker(K, i, k)
        p_jl = tropical_plucker(K, j, l)
        p_il = tropical_plucker(K, i, l)
        p_jk = tropical_plucker(K, j, k)

        terms = [p_ij + p_kl, p_ik + p_jl, p_il + p_jk]
        min_val = min(terms)
        count_min = sum(1 for t in terms if abs(t - min_val) < 1e-10)

        total += 1
        if count_min < 2:
            violations += 1

    return violations, total


def tree_distance_from_plucker(K, n):
    """
    Recover tree metric from tropical Plücker coordinates.
    For Trop(Gr(2,n)), the tree metric d(a,b) is related to Plücker coords.

    The key insight: for a tree metric, the four-point condition holds:
    d(i,j) + d(k,l) <= max(d(i,k)+d(j,l), d(i,l)+d(j,k))
    with equality and the max achieved at least twice.

    For Gr(2,n), we can extract distances from the Plücker coordinates.
    The distance matrix D_{ab} can be read from the tropical Plücker coords
    as: D_{ab} = tropical_plucker(a,b) - (tropical_plucker(a,a') + tropical_plucker(b,b'))/normalization

    Actually, for a direct relationship: the tree metric is given by
    D(a,b) = p_{ab} (up to an additive adjustment per leaf).
    """
    D = np.zeros((n, n))
    for a, b in combinations(range(n), 2):
        D[a, b] = tropical_plucker(K, a, b)
        D[b, a] = D[a, b]
    return D


def verify_four_point_condition(D, n):
    """Check if D satisfies the four-point condition (tree metric)."""
    violations = 0
    total = 0
    for i, j, k, l in combinations(range(n), 4):
        sums = sorted([
            D[i, j] + D[k, l],
            D[i, k] + D[j, l],
            D[i, l] + D[j, k]
        ])
        total += 1
        if abs(sums[1] - sums[2]) > 1e-10:
            violations += 1
    return violations, total


# ==========================================================
# EXPERIMENT 1: Verify tropical Plücker relations for K ∈ R^{n×2}
# ==========================================================

print("=" * 70)
print("EXPERIMENT 1: Tropical Plücker Relations")
print("=" * 70)

for n in [4, 6, 8, 10]:
    K = np.random.randn(n, 2)
    violations, total = check_tropical_plucker_relations(K, n)
    print(f"n={n}: {violations}/{total} violations of tropical Plücker relations")

print("\nExpected: 0 violations (K ∈ R^{n×2} automatically satisfies these)")

# ==========================================================
# EXPERIMENT 2: Does the tree metric govern hard attention?
# ==========================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: Tree Metric and Hard Attention")
print("=" * 70)

n = 6
d_k = 2

K = np.random.randn(n, d_k) * 2
Q = np.random.randn(n, d_k) * 2

S = Q @ K.T  # score matrix, rank <= 2

hard_attention = np.zeros((n, n))
for i in range(n):
    a_star = np.argmax(S[i])
    hard_attention[i, a_star] = 1

print(f"\nKey matrix K (n={n}, d_k={d_k}):")
print(K.round(3))
print(f"\nScore matrix S = QK^T (rank {np.linalg.matrix_rank(S, tol=1e-8)}):")
print(S.round(3))
print(f"\nHard attention (argmax per row):")
for i in range(n):
    winner = np.argmax(S[i])
    print(f"  Query {i} → Key {winner} (score: {S[i, winner]:.3f})")

# Tropical Plücker coordinates from K
print(f"\nTropical Plücker coordinates p_{{ab}} from K:")
for a, b in combinations(range(n), 2):
    tp = tropical_plucker(K, a, b)
    cp = plucker_coordinate(K, a, b)
    print(f"  p_{{{a}{b}}} = {tp:.4f}  (classical: {cp:.4f})")

# ==========================================================
# EXPERIMENT 3: Temperature interpolation — tropical to soft
# ==========================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3: Temperature Interpolation")
print("=" * 70)

n = 6
d_k = 2
K = np.random.randn(n, d_k) * 1.5
Q = np.random.randn(n, d_k) * 1.5
S = Q @ K.T

temperatures = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]

print(f"\nAttention weights for Query 0 at various temperatures:")
print(f"(Low T = tropical/hard, High T = uniform/soft)")
print(f"\n{'T':>6s} | " + " | ".join(f"Key {a}" for a in range(n)) + " | Entropy")

for T in temperatures:
    attn = softmax(S[0:1], temperature=T)[0]
    entropy = -np.sum(attn * np.log(attn + 1e-15))
    max_entropy = np.log(n)
    print(f"{T:6.2f} | " + " | ".join(f"{a:.4f}" for a in attn) +
          f" | {entropy:.3f}/{max_entropy:.3f}")

# ==========================================================
# EXPERIMENT 4: Does soft attention correlate with tree distance?
# ==========================================================

print("\n" + "=" * 70)
print("EXPERIMENT 4: Soft Attention vs Tree Distance")
print("=" * 70)

n = 8
d_k = 2
K = np.random.randn(n, d_k)
Q = K.copy()  # self-attention: queries = keys (same token positions)
S = Q @ K.T

# Compute tropical Plücker-based "distance"
D_trop = np.zeros((n, n))
for a, b in combinations(range(n), 2):
    D_trop[a, b] = tropical_plucker(K, a, b)
    D_trop[b, a] = D_trop[a, b]

# Verify four-point condition
violations, total = verify_four_point_condition(D_trop, n)
print(f"\nFour-point condition check: {violations}/{total} violations")

# Compute soft attention at moderate temperature
T = 0.5
attn = softmax(S, temperature=T)

print(f"\nCorrelation between attention weights and tropical distance (T={T}):")
print("For each query, do higher tropical-Plücker values predict higher attention?")

attn_flat = []
dist_flat = []
for i in range(n):
    for j in range(n):
        if i != j:
            attn_flat.append(attn[i, j])
            dist_flat.append(D_trop[i, j])

attn_flat = np.array(attn_flat)
dist_flat = np.array(dist_flat)
correlation = np.corrcoef(attn_flat, dist_flat)[0, 1]
print(f"Pearson correlation(attention, tropical_distance): {correlation:.4f}")

# Also check: does score correlate with tropical distance?
score_flat = []
for i in range(n):
    for j in range(n):
        if i != j:
            score_flat.append(S[i, j])
score_flat = np.array(score_flat)
corr_score = np.corrcoef(score_flat, dist_flat)[0, 1]
print(f"Pearson correlation(score, tropical_distance): {corr_score:.4f}")

# ==========================================================
# EXPERIMENT 5: Rank comparison — d_k=1 vs d_k=2 vs d_k=4
# ==========================================================

print("\n" + "=" * 70)
print("EXPERIMENT 5: Geometry at Different Key Dimensions")
print("=" * 70)

n = 8

for d_k in [1, 2, 4, 8]:
    K = np.random.randn(n, d_k)
    Q = np.random.randn(n, d_k)
    S = Q @ K.T
    rank = np.linalg.matrix_rank(S, tol=1e-8)

    attn = softmax(S, temperature=1.0)
    entropies = -np.sum(attn * np.log(attn + 1e-15), axis=1)
    mean_entropy = np.mean(entropies)
    max_entropy = np.log(n)

    # Singular values of attention matrix
    sv = np.linalg.svd(attn, compute_uv=False)
    effective_rank = np.sum(sv > 0.01 * sv[0])

    print(f"\nd_k={d_k}: score rank={rank}, "
          f"attention effective rank={effective_rank}, "
          f"mean entropy={mean_entropy:.3f}/{max_entropy:.3f}")
    print(f"  Score singular values: {np.sort(np.linalg.svd(S, compute_uv=False))[::-1][:4].round(3)}")
    print(f"  Attn singular values:  {sv[:4].round(3)}")

# ==========================================================
# EXPERIMENT 6: The key test — multi-head as sum over trees
# ==========================================================

print("\n" + "=" * 70)
print("EXPERIMENT 6: Multi-Head Attention as Sum Over Trees")
print("=" * 70)

n = 6
d_k = 2
n_heads = 4

print(f"\n{n_heads} attention heads, each with d_k={d_k} (each defines a tree)")
print("Checking: does each head define a DIFFERENT tree topology?\n")

trees = []
for h in range(n_heads):
    K_h = np.random.randn(n, d_k)

    # Compute tropical Plücker coordinates
    plucker = {}
    for a, b in combinations(range(n), 2):
        plucker[(a, b)] = tropical_plucker(K_h, a, b)

    # Determine tree topology from which term achieves the max
    topology = []
    for i, j, k, l in combinations(range(n), 4):
        terms = [
            plucker[(i, j)] + plucker[(k, l)],
            plucker[(i, k)] + plucker[(j, l)],
            plucker[(i, l)] + plucker[(j, k)]
        ]
        max_idx = np.argmax(terms)
        topology.append(max_idx)

    trees.append(tuple(topology))
    print(f"  Head {h}: topology signature = {topology}")

unique_trees = len(set(trees))
print(f"\n  {unique_trees} unique tree topologies out of {n_heads} heads")
print(f"  (For n={n}, there are {int(np.math.factorial(2*n-5) / (2**(n-3) * np.math.factorial(n-3)))} "
      f"possible trivalent tree topologies)")

# ==========================================================
# EXPERIMENT 7: Score = tropical inner product structure
# ==========================================================

print("\n" + "=" * 70)
print("EXPERIMENT 7: Score Matrix as Tropical Product")
print("=" * 70)

n = 6
d_k = 2
K = np.random.randn(n, d_k) * 2
Q = np.random.randn(n, d_k) * 2

S_classical = Q @ K.T

# Tropical matrix product: (Q ⊙_trop K^T)_{ia} = max_k(Q_{ik} + K_{ak})
S_tropical = np.zeros((n, n))
for i in range(n):
    for a in range(n):
        S_tropical[i, a] = max(Q[i, 0] + K[a, 0], Q[i, 1] + K[a, 1])

print(f"\nClassical score matrix S = QK^T:")
print(S_classical.round(3))
print(f"\nTropical score matrix S_trop = Q ⊙_trop K^T:")
print(S_tropical.round(3))

# Compare hard attention from both
print(f"\nHard attention comparison:")
for i in range(n):
    classical_winner = np.argmax(S_classical[i])
    tropical_winner = np.argmax(S_tropical[i])
    match = "✓" if classical_winner == tropical_winner else "✗"
    print(f"  Query {i}: classical→Key {classical_winner}, "
          f"tropical→Key {tropical_winner} {match}")

matches = sum(1 for i in range(n)
              if np.argmax(S_classical[i]) == np.argmax(S_tropical[i]))
print(f"\n  Agreement: {matches}/{n} "
      f"({'perfect' if matches == n else 'partial'})")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key findings to verify:
1. K ∈ R^{n×2} automatically satisfies tropical Plücker relations (Exp 1)
2. Hard attention follows the tree structure from K's Plücker coords (Exp 2)
3. Temperature T interpolates between tropical (T→0) and uniform (T→∞) (Exp 3)
4. Soft attention correlates with tropical distance (Exp 4)
5. d_k controls the geometry: higher d_k = richer structure (Exp 5)
6. Multiple heads define different tree topologies (Exp 6)
7. Classical and tropical scores agree on hard attention (Exp 7)
""")
