"""
Multi-layer attention: does the SD equation become nonlinear?

Paper 4's Question 1: in multi-layer attention, the output of layer L
feeds back as input to layer L+1. This creates a self-consistency
condition that should produce a nonlinear Schwinger-Dyson equation.

Concrete test:
- Layer 1: X → α₁ → Y₁ = α₁ V₁
- Layer 2: Y₁ → α₂ → Y₂ = α₂ V₂
where α₂ depends on Y₁, which depends on α₁.

The two-point function at layer 2 should depend nonlinearly on the
layer 1 propagator — that's the nonlinear SD equation.

Ariel — March 9, 2026
"""

import numpy as np
from scipy.special import softmax

np.random.seed(42)

n_tokens = 32
d_model = 64
d_k = 32
d_v = 64  # value dimension = d_model (for residual connection)

X = np.random.randn(n_tokens, d_model)
X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d_model)


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def attention_layer(X_in, WQ, WK, WV, d_k_val):
    """Single attention layer: X → Y = softmax(QK^T/√d_k) V"""
    Q = X_in @ WQ
    K = X_in @ WK
    V = X_in @ WV
    scores = Q @ K.T / np.sqrt(d_k_val)
    alpha = softmax(scores, axis=1)  # (n, n) attention matrix
    return alpha @ V  # (n, d_v)


def bilocal(X_in, alpha):
    """Bilocal function H(i,j) = (1/d) Σ_{a,b} α_{ia} α_{jb} (x_a · x_b)"""
    K = X_in @ X_in.T / d_model  # normalized kernel
    return alpha @ K @ alpha.T  # (n, n)


# ================================================================
# TEST: Single-layer vs two-layer propagator structure
# ================================================================
print_section("Single Layer vs Two Layer: Propagator Nonlinearity")

n_real = 3000
sigma = 0.3  # moderate sigma — between linearized and fully nonlinear

# --- Single layer ---
G1_samples = np.zeros((n_real, n_tokens, n_tokens))

for r in range(n_real):
    WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    
    Q = X @ WQ
    K_keys = X @ WK
    scores = Q @ K_keys.T / np.sqrt(d_k)
    alpha = softmax(scores, axis=1)
    
    G1_samples[r] = bilocal(X, alpha)

G1_mean = np.mean(G1_samples, axis=0)  # disorder-averaged single-layer propagator

# --- Two layers (with residual connection) ---
G2_samples = np.zeros((n_real, n_tokens, n_tokens))

for r in range(n_real):
    # Layer 1
    WQ1 = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK1 = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WV1 = np.random.randn(d_model, d_v) * sigma / np.sqrt(d_model)
    
    Y1 = attention_layer(X, WQ1, WK1, WV1, d_k)
    X2 = X + Y1  # residual connection
    
    # Layer 2
    WQ2 = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK2 = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    
    Q2 = X2 @ WQ2
    K2 = X2 @ WK2
    scores2 = Q2 @ K2.T / np.sqrt(d_k)
    alpha2 = softmax(scores2, axis=1)
    
    G2_samples[r] = bilocal(X2, alpha2)

G2_mean = np.mean(G2_samples, axis=0)

print(f"  σ = {sigma}")
print(f"  G1 diagonal mean: {np.mean(np.diag(G1_mean)):.6f}")
print(f"  G2 diagonal mean: {np.mean(np.diag(G2_mean)):.6f}")
print(f"  G2/G1 diagonal ratio: {np.mean(np.diag(G2_mean))/np.mean(np.diag(G1_mean)):.4f}")

# Key test: Is G2 a linear function of G1?
# If the SD equation is linear: G2 should be proportional to the input kernel.
# If nonlinear: G2 should depend on G1 in a nonlinear way.

# Test linearity: compute G2 for two different σ values at layer 1
# If linear: G2(2σ₁) = 2·G2(σ₁) - G2(0)
# If nonlinear: this won't hold


# ================================================================
# TEST: Nonlinearity via σ₁ dependence
# ================================================================
print_section("Nonlinearity Test: G₂ vs σ₁")

sigma2 = 0.2  # fixed layer 2 initialization

sigma1_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
G2_diag_means = []

for s1 in sigma1_values:
    G2_s = np.zeros((1000, n_tokens))  # only diagonal
    
    for r in range(1000):
        WQ1 = np.random.randn(d_model, d_k) * s1 / np.sqrt(d_model)
        WK1 = np.random.randn(d_model, d_k) * s1 / np.sqrt(d_model)
        WV1 = np.random.randn(d_model, d_v) * s1 / np.sqrt(d_model)
        
        Y1 = attention_layer(X, WQ1, WK1, WV1, d_k)
        X2 = X + Y1
        
        WQ2 = np.random.randn(d_model, d_k) * sigma2 / np.sqrt(d_model)
        WK2 = np.random.randn(d_model, d_k) * sigma2 / np.sqrt(d_model)
        
        Q2 = X2 @ WQ2
        K2 = X2 @ WK2
        scores2 = Q2 @ K2.T / np.sqrt(d_k)
        alpha2 = softmax(scores2, axis=1)
        
        G2_s[r] = np.diag(bilocal(X2, alpha2))
    
    mean_diag = np.mean(G2_s)
    G2_diag_means.append(mean_diag)
    print(f"  σ₁={s1:.2f}: ⟨G₂_diag⟩ = {mean_diag:.6f}")

# Check linearity: if G₂ is linear in σ₁², then ⟨G₂⟩ = A + B·σ₁²
# Fit a linear model
s1_sq = np.array([s**2 for s in sigma1_values])
g2_arr = np.array(G2_diag_means)

A = np.column_stack([np.ones_like(s1_sq), s1_sq])
coeff, res, _, _ = np.linalg.lstsq(A, g2_arr, rcond=None)
g2_linear_pred = A @ coeff

# Also fit quadratic: A + B·σ₁² + C·σ₁⁴
A_quad = np.column_stack([np.ones_like(s1_sq), s1_sq, s1_sq**2])
coeff_q, res_q, _, _ = np.linalg.lstsq(A_quad, g2_arr, rcond=None)
g2_quad_pred = A_quad @ coeff_q

print()
print("  Linear fit (G₂ = a + b·σ₁²):")
print(f"    Coefficients: a={coeff[0]:.6f}, b={coeff[1]:.6f}")
print(f"    Residuals: {[f'{(g2_arr[i] - g2_linear_pred[i]):.4e}' for i in range(len(sigma1_values))]}")
linear_rmse = np.sqrt(np.mean((g2_arr - g2_linear_pred)**2))
print(f"    RMSE: {linear_rmse:.4e}")

print()
print("  Quadratic fit (G₂ = a + b·σ₁² + c·σ₁⁴):")
print(f"    Coefficients: a={coeff_q[0]:.6f}, b={coeff_q[1]:.6f}, c={coeff_q[2]:.6f}")
print(f"    Residuals: {[f'{(g2_arr[i] - g2_quad_pred[i]):.4e}' for i in range(len(sigma1_values))]}")
quad_rmse = np.sqrt(np.mean((g2_arr - g2_quad_pred)**2))
print(f"    RMSE: {quad_rmse:.4e}")

print()
if quad_rmse < 0.1 * linear_rmse and abs(coeff_q[2]) > 0:
    print(f"  ⟹ NONLINEAR dependence detected: σ₁⁴ coefficient = {coeff_q[2]:.4e}")
    print(f"    The two-layer propagator depends nonlinearly on the layer 1 coupling.")
    print(f"    This is the signature of a nonlinear SD equation.")
elif linear_rmse < 1e-5 * abs(np.mean(g2_arr)):
    print(f"  ⟹ LINEAR dependence: G₂ ≈ linear in σ₁²")
    print(f"    The two-layer SD equation is still linear.")
else:
    print(f"  ⟹ Inconclusive: linear RMSE = {linear_rmse:.4e}, quad RMSE = {quad_rmse:.4e}")


# ================================================================
# TEST: Variance structure (connected correlator)
# ================================================================
print_section("Two-Layer: Connected Correlator Enhancement")

# Does the connected four-point function at layer 2 get enhanced
# by layer 1 disorder? If so, the multi-layer system has stronger
# IB correlations than a single layer.

sigma_test = 0.2
n_r = 2000

# Single layer variance of H_{00}
H1_00 = np.zeros(n_r)
H2_00 = np.zeros(n_r)

for r in range(n_r):
    # Single layer (layer 2 only, with X as input)
    WQ = np.random.randn(d_model, d_k) * sigma_test / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma_test / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    sc = Q[0] @ K_keys.T / np.sqrt(d_k)
    alpha = softmax(sc)
    K_mat = X @ X.T / d_model
    H1_00[r] = alpha @ K_mat @ alpha

    # Two layer
    WQ1 = np.random.randn(d_model, d_k) * sigma_test / np.sqrt(d_model)
    WK1 = np.random.randn(d_model, d_k) * sigma_test / np.sqrt(d_model)
    WV1 = np.random.randn(d_model, d_v) * sigma_test / np.sqrt(d_model)
    
    Y1 = attention_layer(X, WQ1, WK1, WV1, d_k)
    X2 = X + Y1
    
    WQ2 = np.random.randn(d_model, d_k) * sigma_test / np.sqrt(d_model)
    WK2 = np.random.randn(d_model, d_k) * sigma_test / np.sqrt(d_model)
    Q2 = X2 @ WQ2
    K2 = X2 @ WK2
    sc2 = Q2[0] @ K2.T / np.sqrt(d_k)
    alpha2 = softmax(sc2)
    K2_mat = X2 @ X2.T / d_model
    H2_00[r] = alpha2 @ K2_mat @ alpha2

var_1L = np.var(H1_00)
var_2L = np.var(H2_00)

print(f"  σ = {sigma_test}")
print(f"  Single layer:  Var(H₀₀) = {var_1L:.4e}")
print(f"  Two layers:    Var(H₀₀) = {var_2L:.4e}")
print(f"  Enhancement:   Var₂/Var₁ = {var_2L/var_1L:.2f}×")
print()
print(f"  If enhancement > 1: layer 1 disorder amplifies layer 2 fluctuations")
print(f"  This is the multi-layer IB mechanism that generates nonlinear SD equations")


print_section("DONE")
