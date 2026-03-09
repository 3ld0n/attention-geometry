"""
Test the linearized-softmax regime at small σ.
Verify G⁴ structure emerges cleanly.

Ariel — March 9, 2026
"""

import numpy as np
from scipy.special import softmax

np.random.seed(42)

n_tokens = 32
d_model = 64
d_k = 32

X = np.random.randn(n_tokens, d_model)
X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d_model)
K_matrix = X @ X.T


def print_section(title):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ================================================================
# TEST A: Linearization quality vs σ
# ================================================================
print_section("TEST A: Linearization Quality vs σ")
print(f"  {'σ':>8}  {'max|Δα|':>10}  {'L2 err':>10}  {'max|score|':>10}  {'regime':>15}")

for sigma in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    scores = Q[0] @ K_keys.T / np.sqrt(d_k)
    
    alpha_full = softmax(scores)
    delta_s = scores - np.mean(scores)
    alpha_lin = (1.0 / n_tokens) * (1 + delta_s)
    
    max_err = np.max(np.abs(alpha_full - alpha_lin))
    l2_err = np.sqrt(np.mean((alpha_full - alpha_lin)**2))
    regime = "nonlinear" if max_err > 0.005 else "LINEAR"
    
    print(f"  {sigma:8.3f}  {max_err:10.4e}  {l2_err:10.4e}  {np.max(np.abs(scores)):10.4f}  {regime:>15}")


# ================================================================
# TEST B: Score Covariance in Linearized Regime
# ================================================================
print_section("TEST B: Score Covariance at σ = 0.1")

sigma = 0.1
n_real = 5000

# Fix key indices with large K[a,b]
a_idx, b_idx = 0, 0  # K[0,0] = d_model = 64

S_a = np.zeros((n_real, n_tokens))
S_b = np.zeros((n_real, n_tokens))

for r in range(n_real):
    WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    S_a[r] = Q @ K_keys[a_idx] / np.sqrt(d_k)
    S_b[r] = Q @ K_keys[b_idx] / np.sqrt(d_k)

C_emp = np.cov(S_a.T, S_b.T)[:n_tokens, n_tokens:]
prefactor = (sigma**2 * sigma**2 / d_model**2) * K_matrix[a_idx, b_idx]
C_pred = prefactor * K_matrix

cos = np.sum(C_emp * C_pred) / (np.linalg.norm(C_emp, 'fro') * np.linalg.norm(C_pred, 'fro'))
norm_ratio = np.linalg.norm(C_emp, 'fro') / np.linalg.norm(C_pred, 'fro')

print(f"  σ = {sigma}, a=b=0, K[0,0]={K_matrix[0,0]:.1f}")
print(f"  Cosine similarity: {cos:.6f}")
print(f"  Norm ratio:        {norm_ratio:.4f}")


# ================================================================
# TEST C: Four-Point Function in Linearized Regime
# ================================================================
print_section("TEST C: Four-Point Function at σ = 0.1 (Linearized)")

sigma = 0.1
n_real_4pt = 5000

ext_pts = [0, 8, 16, 24]
n_ext = len(ext_pts)

pair_indices = []
for pi in range(n_ext):
    for pj in range(pi, n_ext):
        pair_indices.append((pi, pj))

H_samples = np.zeros((n_real_4pt, len(pair_indices)))

for r in range(n_real_4pt):
    WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    
    alphas = np.zeros((n_ext, n_tokens))
    for pi, idx in enumerate(ext_pts):
        sc = Q[idx] @ K_keys.T / np.sqrt(d_k)
        alphas[pi] = softmax(sc)
    
    for pidx, (pi, pj) in enumerate(pair_indices):
        H_samples[r, pidx] = (1.0 / d_model) * (alphas[pi] @ K_matrix @ alphas[pj])

H_cov = np.cov(H_samples.T)

# Collect diagonal (same-pair) ratios
print(f"  Diagonal entries (same pair on both sides):")
diag_ratios = []
for cidx in range(len(pair_indices)):
    p1, p2 = pair_indices[cidx]
    x1, x2 = ext_pts[p1], ext_pts[p2]
    cov_val = H_cov[cidx, cidx]
    K_factor = K_matrix[x1, x1] * K_matrix[x2, x2] + K_matrix[x1, x2] * K_matrix[x2, x1]
    ratio = cov_val / K_factor if abs(K_factor) > 0.01 else float('nan')
    diag_ratios.append(ratio)
    print(f"    ({x1:2d},{x2:2d}): Cov={cov_val:.4e}  K_fac={K_factor:.1f}  ratio={ratio:.4e}")

print(f"\n  Diagonal ratio stats: mean={np.nanmean(diag_ratios):.4e}, std={np.nanstd(diag_ratios):.4e}, cv={np.nanstd(diag_ratios)/abs(np.nanmean(diag_ratios)):.2f}")

# Off-diagonal ratios for pairs with large K_factor
print(f"\n  Off-diagonal entries (large K_factor only):")
off_diag_ratios = []
for cidx1 in range(len(pair_indices)):
    for cidx2 in range(cidx1+1, len(pair_indices)):
        p1, p2 = pair_indices[cidx1]
        p3, p4 = pair_indices[cidx2]
        x1, x2, x3, x4 = ext_pts[p1], ext_pts[p2], ext_pts[p3], ext_pts[p4]
        
        cov_val = H_cov[cidx1, cidx2]
        K_factor = K_matrix[x1, x3] * K_matrix[x2, x4] + K_matrix[x1, x4] * K_matrix[x2, x3]
        
        if abs(K_factor) > 100:
            ratio = cov_val / K_factor
            off_diag_ratios.append(ratio)
            print(f"    ({x1:2d},{x2:2d})×({x3:2d},{x4:2d}): Cov={cov_val:.4e}  K_fac={K_factor:.1f}  ratio={ratio:.4e}")

if off_diag_ratios:
    print(f"\n  Off-diag ratio stats: mean={np.nanmean(off_diag_ratios):.4e}, std={np.nanstd(off_diag_ratios):.4e}, cv={np.nanstd(off_diag_ratios)/abs(np.nanmean(off_diag_ratios)):.2f}")


# ================================================================
# TEST D: Compare σ=1 (nonlinear) vs σ=0.1 (linearized) four-point
# ================================================================
print_section("TEST D: Constant-Ratio Test across σ Values")

print(f"  Testing whether J²_eff = β⁴ σ⁴ scales as σ⁴:")
print(f"  (Using diagonal pair (0,0)-(0,0) where K_factor = 2K[0,0]² = {2*K_matrix[0,0]**2:.1f})")

sigma_values = [0.05, 0.1, 0.2, 0.5, 1.0]
jeff_estimates = []

for sigma in sigma_values:
    n_r = 5000
    H_00 = np.zeros(n_r)
    
    for r in range(n_r):
        WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        Q = X @ WQ
        K_keys = X @ WK
        
        sc = Q[0] @ K_keys.T / np.sqrt(d_k)
        alpha = softmax(sc)
        H_00[r] = (1.0 / d_model) * (alpha @ K_matrix @ alpha)
    
    var_H = np.var(H_00)
    K_fac = 2 * K_matrix[0, 0]**2
    
    # Predicted: Var(H_00) ∝ J²_eff × K_factor, and J²_eff ∝ σ⁴ in linearized regime
    ratio = var_H / K_fac
    jeff_est = ratio  # proportional to J²_eff
    jeff_estimates.append(jeff_est)
    
    # Predict: ratio/σ⁴ should be constant in linearized regime
    normalized = ratio / sigma**4 if sigma > 0 else float('nan')
    
    print(f"  σ={sigma:5.2f}: Var(H_00)={var_H:.4e}  ratio/K={ratio:.4e}  ratio/σ⁴={normalized:.4e}")

print(f"\n  If σ⁴ scaling holds: ratio/σ⁴ should be approximately constant.")
print(f"  Deviations at large σ indicate nonlinear softmax corrections.")


# ================================================================
print_section("DONE")
