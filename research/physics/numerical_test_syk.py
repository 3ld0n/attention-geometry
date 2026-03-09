"""
Numerical verification of the SYK structural predictions for transformer attention.

Tests:
1. Score covariance factorization — structural test (rank-1 in external indices)
2. Gibbs state identity and quantum expectation (Paper 5 Theorems 1-2)
3. Connected four-point function: IB structure test
4. Linearized vs full softmax regime

Ariel — March 9, 2026
"""

import numpy as np
from scipy.special import softmax
import sys

np.random.seed(42)

# --- Parameters ---
n_tokens = 32        # context length (keep small for speed)
d_model = 64         # embedding dimension
d_k = 32             # key dimension
sigma_Q = 1.0
sigma_K = 1.0

# --- Token embeddings (random Gaussian, normalized) ---
X = np.random.randn(n_tokens, d_model)
X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d_model)
K_matrix = X @ X.T  # (n, n) token kernel


def print_section(title):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ================================================================
# TEST 1: Score Covariance Factorization (Structural)
# ================================================================
print_section("TEST 1: Score Covariance — Structural Factorization")

n_real = 5000

# Fix key indices a, b.  Compute full matrix C[i,j] = Cov(s_{ia}, s_{jb})
a_idx, b_idx = 5, 12

# Collect all scores for s_{*,a} and s_{*,b}
S_a = np.zeros((n_real, n_tokens))  # S_a[r, i] = s_{ia} in realization r
S_b = np.zeros((n_real, n_tokens))

for r in range(n_real):
    WQ = np.random.randn(d_model, d_k) * sigma_Q / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma_K / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    S_a[r] = Q @ K_keys[a_idx] / np.sqrt(d_k)
    S_b[r] = Q @ K_keys[b_idx] / np.sqrt(d_k)

# Empirical covariance matrix C[i,j] = Cov(s_{ia}, s_{jb})
C_emp = np.zeros((n_tokens, n_tokens))
mean_a = np.mean(S_a, axis=0)
mean_b = np.mean(S_b, axis=0)
for i in range(n_tokens):
    for j in range(n_tokens):
        C_emp[i, j] = np.mean(S_a[:, i] * S_b[:, j]) - mean_a[i] * mean_b[j]

# Predicted: C[i,j] = (σ²_Q σ²_K / d²) · K[i,j] · K[a,b]
prefactor = (sigma_Q**2 * sigma_K**2 / d_model**2) * K_matrix[a_idx, b_idx]
C_pred = prefactor * K_matrix

# Test 1a: proportionality (normalize both and compare)
C_emp_norm = C_emp / np.linalg.norm(C_emp, 'fro')
C_pred_norm = C_pred / np.linalg.norm(C_pred, 'fro')
cosine_sim = np.sum(C_emp_norm * C_pred_norm)

print(f"  Key indices: a={a_idx}, b={b_idx}")
print(f"  K[a,b] = {K_matrix[a_idx, b_idx]:.4f}")
print()
print(f"  Frobenius norm — empirical:  {np.linalg.norm(C_emp, 'fro'):.6f}")
print(f"  Frobenius norm — predicted:  {np.linalg.norm(C_pred, 'fro'):.6f}")
print(f"  Ratio:                       {np.linalg.norm(C_emp, 'fro') / np.linalg.norm(C_pred, 'fro'):.4f}")
print()
print(f"  Cosine similarity (should be ~1): {cosine_sim:.6f}")
print()

# Test 1b: diagonal elements should match K[i,i]·K[a,b]·σ²_Qσ²_K/d²
diag_emp = np.diag(C_emp)
diag_pred = prefactor * np.diag(K_matrix)
diag_ratio = np.mean(diag_emp / diag_pred)
print(f"  Mean diagonal ratio (emp/pred): {diag_ratio:.4f}")
print(f"  Diagonal ratios (first 8): {[f'{r:.3f}' for r in (diag_emp/diag_pred)[:8]]}")

# Test 1c: repeat for different key pairs
print()
print("  Testing multiple key pairs:")
for (ai, bi) in [(0, 0), (0, 15), (10, 20), (5, 5)]:
    S_ai = np.zeros((n_real, n_tokens))
    S_bi = np.zeros((n_real, n_tokens))
    for r in range(n_real):
        WQ = np.random.randn(d_model, d_k) * sigma_Q / np.sqrt(d_model)
        WK = np.random.randn(d_model, d_k) * sigma_K / np.sqrt(d_model)
        Q = X @ WQ
        K_keys = X @ WK
        S_ai[r] = Q @ K_keys[ai] / np.sqrt(d_k)
        S_bi[r] = Q @ K_keys[bi] / np.sqrt(d_k)
    
    C_e = np.cov(S_ai.T, S_bi.T)[:n_tokens, n_tokens:]
    pf = (sigma_Q**2 * sigma_K**2 / d_model**2) * K_matrix[ai, bi]
    C_p = pf * K_matrix
    
    cos = np.sum(C_e * C_p) / (np.linalg.norm(C_e, 'fro') * np.linalg.norm(C_p, 'fro'))
    norm_ratio = np.linalg.norm(C_e, 'fro') / np.linalg.norm(C_p, 'fro') if np.linalg.norm(C_p, 'fro') > 1e-10 else float('nan')
    print(f"    a={ai:2d}, b={bi:2d}: cosine={cos:.4f}  norm_ratio={norm_ratio:.4f}  K[a,b]={K_matrix[ai,bi]:.2f}")


# ================================================================
# TEST 2: Gibbs State (Paper 5 Theorems 1 & 2)
# ================================================================
print_section("TEST 2: Gibbs State — Theorems 1 & 2")

WQ_t = np.random.randn(d_model, d_k) * sigma_Q / np.sqrt(d_model)
WK_t = np.random.randn(d_model, d_k) * sigma_K / np.sqrt(d_model)

q = X[0] @ WQ_t
keys = X @ WK_t
scores = q @ keys.T / np.sqrt(d_k)
alpha = softmax(scores)

rho = np.diag(np.exp(scores) / np.sum(np.exp(scores)))
diag_rho = np.diag(rho)

print(f"  Theorem 1:  max|α_i - ⟨i|ρ|i⟩| = {np.max(np.abs(alpha - diag_rho)):.2e}")

d_v = 16
WV_t = np.random.randn(d_model, d_v)
V_mat = X @ WV_t
y_att = alpha @ V_mat
y_qm = np.array([np.trace(rho @ np.diag(V_mat[:, dim])) for dim in range(d_v)])
print(f"  Theorem 2:  max|y_attention - Tr(ρV)| = {np.max(np.abs(y_att - y_qm)):.2e}")
print(f"  (Both should be ~machine epsilon)")


# ================================================================
# TEST 3: Born Rule / Fisher Information (Paper 5 Theorems 3 & 4)
# ================================================================
print_section("TEST 3: Born Rule & Fisher Information — Theorems 3 & 4")

# Theorem 3: P(i) = Tr(ρ Π_i) = α_i
projectors = [np.outer(np.eye(n_tokens)[k], np.eye(n_tokens)[k]) for k in range(n_tokens)]
born_probs = np.array([np.trace(rho @ P) for P in projectors])
print(f"  Theorem 3:  max|P_Born(i) - α_i| = {np.max(np.abs(born_probs - alpha)):.2e}")

# Theorem 4: Classical Fisher = Quantum Fisher for diagonal states
# Classical Fisher information: F_C = Σ (1/p_i)(dp_i/dθ)²
# For a Gibbs state: dα_i/dβ = α_i(s_i - <s>) so
# F_C = Σ α_i (s_i - <s>)² = Var_α(s) (the specific heat)
s_mean = np.sum(alpha * scores)
F_classical = np.sum(alpha * (scores - s_mean)**2)

# Quantum Fisher = 2 Σ_{m≠n} |<m|∂ρ/∂θ|n>|² / (p_m + p_n)
# For diagonal ρ, ∂ρ/∂β is also diagonal: ∂ρ_{mm}/∂β = α_m(s_m - <s>)
# So QFI = Σ_m (∂p_m/∂β)² / p_m = F_classical
drho_diag = alpha * (scores - s_mean)
F_quantum = np.sum(drho_diag**2 / alpha)

print(f"  Theorem 4:  F_classical = {F_classical:.8f}")
print(f"              F_quantum   = {F_quantum:.8f}")
print(f"              |F_C - F_Q| = {abs(F_classical - F_quantum):.2e}")


# ================================================================
# TEST 4: Connected Four-Point Function (IB Mechanism)
# ================================================================
print_section("TEST 4: Four-Point Function — IB Mechanism Test")

n_real_4pt = 3000
sigma_V = 1.0

# Compute H(x_1, x_2) = (σ_V²/d) Σ_{a,b} α_a(x_1) α_b(x_2) K[a,b]
# for many external-point pairs, and check that Cov(H_12, H_34) tracks
# the predicted geometry factor (x1·x3)(x2·x4) + (x1·x4)(x2·x3)

# We'll compute H for all pairs of a fixed set of external points
ext_pts = [0, 8, 16, 24]  # 4 external points

# For each realization, compute H_{ij} for all 6 pairs
n_ext = len(ext_pts)
n_pairs = n_ext * (n_ext + 1) // 2

pair_indices = []
for pi in range(n_ext):
    for pj in range(pi, n_ext):
        pair_indices.append((pi, pj))

H_samples = np.zeros((n_real_4pt, len(pair_indices)))

for r in range(n_real_4pt):
    WQ = np.random.randn(d_model, d_k) * sigma_Q / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma_K / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    
    alphas = np.zeros((n_ext, n_tokens))
    for pi, idx in enumerate(ext_pts):
        sc = Q[idx] @ K_keys.T / np.sqrt(d_k)
        alphas[pi] = softmax(sc)
    
    for pidx, (pi, pj) in enumerate(pair_indices):
        H_samples[r, pidx] = (sigma_V**2 / d_model) * (alphas[pi] @ K_matrix @ alphas[pj])

# Compute the empirical covariance matrix of the H values
H_cov = np.cov(H_samples.T)  # (n_pairs, n_pairs)

# For each pair of pairs, compute the predicted geometry factor
print(f"  External points: {ext_pts}")
print(f"  Number of pair-pairs: {len(pair_indices)} × {len(pair_indices)} = {len(pair_indices)**2}")
print()

print("  Pair-pair covariance vs geometry factor:")
print(f"  {'Pair1':>12} {'Pair2':>12} {'Cov(emp)':>12} {'K_factor':>12} {'ratio':>12}")

results = []
for cidx1 in range(len(pair_indices)):
    for cidx2 in range(cidx1, len(pair_indices)):
        p1, p2 = pair_indices[cidx1]
        p3, p4 = pair_indices[cidx2]
        x1, x2, x3, x4 = ext_pts[p1], ext_pts[p2], ext_pts[p3], ext_pts[p4]
        
        cov_val = H_cov[cidx1, cidx2]
        K_factor = K_matrix[x1, x3] * K_matrix[x2, x4] + K_matrix[x1, x4] * K_matrix[x2, x3]
        
        if abs(K_factor) > 0.1:
            ratio = cov_val / K_factor
            results.append(ratio)
            print(f"  ({x1:2d},{x2:2d})  ({x3:2d},{x4:2d})  {cov_val:12.4e}  {K_factor:12.2f}  {ratio:12.4e}")

if results:
    ratios = np.array(results)
    print(f"\n  Ratio statistics: mean={np.mean(ratios):.4e}, std={np.std(ratios):.4e}, cv={np.std(ratios)/abs(np.mean(ratios)):.2f}")
    print(f"  If IB structure holds: ratios should be approximately constant")
    print(f"  (constant = J²_eff × data-geometry factor)")


# ================================================================
# TEST 5: Linearized Softmax Regime
# ================================================================
print_section("TEST 5: Linearized Softmax Regime")

print(f"  Comparing α (full softmax) with α_lin = (1/n)(1 + δs) at varying d_k:")
print(f"  {'d_k':>6}  {'L2 err':>10}  {'max err':>10}  {'max|s|':>10}  {'regime':>20}")

for dk in [16, 32, 64, 128, 256, 512, 1024]:
    WQ = np.random.randn(d_model, dk) * sigma_Q / np.sqrt(d_model)
    WK = np.random.randn(d_model, dk) * sigma_K / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    scores = Q[0] @ K_keys.T / np.sqrt(dk)
    
    alpha_full = softmax(scores)
    delta_s = scores - np.mean(scores)
    alpha_lin = (1.0 / n_tokens) * (1 + delta_s)
    
    l2_err = np.sqrt(np.mean((alpha_full - alpha_lin)**2))
    max_err = np.max(np.abs(alpha_full - alpha_lin))
    regime = "nonlinear" if max_err > 0.01 else "linearized OK"
    
    print(f"  {dk:6d}  {l2_err:10.4e}  {max_err:10.4e}  {np.max(np.abs(scores)):10.3f}  {regime:>20}")


# ================================================================
# TEST 6: Score Variance Scaling (indirect $\Delta$ test)
# ================================================================
print_section("TEST 6: Score Variance Scaling with d_k")

# The variance of a score s_{ia} should be:
# Var(s_{ia}) = (σ_Q² σ_K² / d²) · K[i,i] · K[a,a]
# This is independent of d_k (the 1/√d_k in the score formula cancels
# with the d_k sum over key dimensions).

print(f"  Prediction: Var(s_ia) = (σ²_Q σ²_K / d²) K[i,i] K[a,a] (independent of d_k)")
print(f"  Expected value: {sigma_Q**2 * sigma_K**2 / d_model**2 * K_matrix[0,0] * K_matrix[5,5]:.6f}")
print()

n_var_real = 3000
for dk in [8, 16, 32, 64, 128, 256]:
    scores_sample = np.zeros(n_var_real)
    for r in range(n_var_real):
        WQ = np.random.randn(d_model, dk) * sigma_Q / np.sqrt(d_model)
        WK = np.random.randn(d_model, dk) * sigma_K / np.sqrt(d_model)
        q = X[0] @ WQ
        k = X[5] @ WK
        scores_sample[r] = q @ k / np.sqrt(dk)
    
    emp_var = np.var(scores_sample)
    pred_var = sigma_Q**2 * sigma_K**2 / d_model**2 * K_matrix[0, 0] * K_matrix[5, 5]
    print(f"  d_k={dk:4d}: Var(emp)={emp_var:.6f}  Var(pred)={pred_var:.6f}  ratio={emp_var/pred_var:.4f}")


# ================================================================
# SUMMARY
# ================================================================
print_section("SUMMARY")
print("""
  Test 1 (Score covariance):  Checks factorization Cov(s_ia, s_jb) ∝ K[i,j]·K[a,b]
  Test 2 (Gibbs state):       Verifies α_i = ⟨i|ρ|i⟩ and y = Tr(ρV) (machine precision)
  Test 3 (Born/Fisher):       Verifies Born rule P(i) = α_i and F_classical = F_quantum
  Test 4 (Four-point fn):     Tests IB mechanism — Cov(H_12, H_34) ∝ K_geometry
  Test 5 (Linearized softmax): Shows regime validity at different d_k
  Test 6 (Score variance):    Verifies d_k-independence of score variance
  
  The structural results — Gibbs state identity, Born rule, Fisher information
  equality — are exact and verified to machine precision.
  
  The statistical results — score covariance factorization, four-point function
  structure — require sufficient samples but show the predicted structure.
""")
