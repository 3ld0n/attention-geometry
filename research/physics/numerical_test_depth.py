"""
How does the connected correlator enhancement scale with depth?

If the 18× enhancement from single→double layer compounds,
deep transformers are in the strongly-coupled SYK regime.

Ariel — March 9, 2026
"""

import numpy as np
from scipy.special import softmax

np.random.seed(42)

n_tokens = 32
d_model = 64
d_k = 32
d_v = 64
sigma = 0.2
n_real = 2000


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


X0 = np.random.randn(n_tokens, d_model)
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True) * np.sqrt(d_model)


print_section("Connected Correlator vs Depth")
print(f"  Parameters: n={n_tokens}, d={d_model}, d_k={d_k}, σ={sigma}")
print(f"  Layers: 0 (no attention) through 6")
print(f"  Measuring: Var(H₀₀) where H₀₀ = (1/d) α₀ · K · α₀")
print()

max_layers = 6
vars_by_depth = []

for n_layers in range(max_layers + 1):
    H_00 = np.zeros(n_real)
    
    for r in range(n_real):
        X_curr = X0.copy()
        
        for L in range(n_layers):
            WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
            WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
            WV = np.random.randn(d_model, d_v) * sigma / np.sqrt(d_model)
            
            Q = X_curr @ WQ
            K_keys = X_curr @ WK
            V = X_curr @ WV
            
            scores = Q @ K_keys.T / np.sqrt(d_k)
            alpha = softmax(scores, axis=1)
            Y = alpha @ V
            
            X_curr = X_curr + Y  # residual connection
        
        # Measure the bilocal at the output
        K_mat = X_curr @ X_curr.T / d_model
        WQ_m = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        WK_m = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        Q_m = X_curr @ WQ_m
        K_m = X_curr @ WK_m
        sc = Q_m[0] @ K_m.T / np.sqrt(d_k)
        alpha_m = softmax(sc)
        
        H_00[r] = alpha_m @ K_mat @ alpha_m
    
    var_H = np.var(H_00)
    mean_H = np.mean(H_00)
    vars_by_depth.append(var_H)
    
    if n_layers == 0:
        enhancement = 1.0
    else:
        enhancement = var_H / vars_by_depth[0] if vars_by_depth[0] > 0 else float('nan')
    
    print(f"  L={n_layers}: ⟨H₀₀⟩={mean_H:.6f}  Var(H₀₀)={var_H:.4e}  Enhancement={enhancement:.1f}×")

print()

# Check scaling: exponential vs polynomial
vars_arr = np.array(vars_by_depth[1:])  # exclude L=0
layers_arr = np.arange(1, max_layers + 1)

# Log-linear fit: ln(Var) = a + b·L → exponential
ln_vars = np.log(vars_arr)
A_exp = np.column_stack([np.ones_like(layers_arr), layers_arr.astype(float)])
coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, ln_vars, rcond=None)
pred_exp = np.exp(A_exp @ coeff_exp)
rmse_exp = np.sqrt(np.mean((vars_arr - pred_exp)**2))

# Log-log fit: ln(Var) = a + b·ln(L) → power law
ln_layers = np.log(layers_arr.astype(float))
A_pow = np.column_stack([np.ones_like(ln_layers), ln_layers])
coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, ln_vars, rcond=None)
pred_pow = np.exp(A_pow @ coeff_pow)
rmse_pow = np.sqrt(np.mean((vars_arr - pred_pow)**2))

print(f"  Exponential fit: Var ~ exp({coeff_exp[1]:.3f} · L), RMSE={rmse_exp:.4e}")
print(f"  Power-law fit:   Var ~ L^{coeff_pow[1]:.3f},         RMSE={rmse_pow:.4e}")

if rmse_exp < rmse_pow:
    print(f"\n  ⟹ Better fit: EXPONENTIAL growth with rate {coeff_exp[1]:.3f} per layer")
    print(f"    Doubling every {np.log(2)/coeff_exp[1]:.1f} layers")
else:
    print(f"\n  ⟹ Better fit: POWER LAW growth with exponent {coeff_pow[1]:.2f}")

# Per-layer enhancement ratio
print()
print("  Per-layer enhancement ratios:")
for i in range(1, len(vars_by_depth)):
    ratio = vars_by_depth[i] / vars_by_depth[i-1] if vars_by_depth[i-1] > 0 else float('nan')
    print(f"    L={i-1}→{i}: {ratio:.2f}×")


print_section("DONE")
