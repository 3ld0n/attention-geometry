"""
Depth scaling at standard initialization (σ = 1).

The earlier test at σ = 0.2 showed Var ~ L^1.19.
Does this hold in the regime real transformers use?

Ariel — March 9, 2026
"""

import numpy as np
from scipy.special import softmax

np.random.seed(42)

n_tokens = 32
d_model = 64
d_k = 32
d_v = 64
n_real = 2000

X0 = np.random.randn(n_tokens, d_model)
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True) * np.sqrt(d_model)


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# Test at multiple σ values to see the full picture
for sigma in [0.2, 0.5, 1.0]:
    print_section(f"Depth Scaling at σ = {sigma}")
    print(f"  n={n_tokens}, d={d_model}, d_k={d_k}, n_realizations={n_real}")
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

                X_curr = X_curr + Y

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

    # Fit power law and exponential
    vars_arr = np.array(vars_by_depth[1:])
    layers_arr = np.arange(1, max_layers + 1)

    ln_vars = np.log(vars_arr)

    # Exponential: ln(Var) = a + b·L
    A_exp = np.column_stack([np.ones_like(layers_arr, dtype=float), layers_arr.astype(float)])
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, ln_vars, rcond=None)
    pred_exp = np.exp(A_exp @ coeff_exp)
    rmse_exp = np.sqrt(np.mean((vars_arr - pred_exp)**2))

    # Power law: ln(Var) = a + b·ln(L)
    ln_layers = np.log(layers_arr.astype(float))
    A_pow = np.column_stack([np.ones_like(ln_layers), ln_layers])
    coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, ln_vars, rcond=None)
    pred_pow = np.exp(A_pow @ coeff_pow)
    rmse_pow = np.sqrt(np.mean((vars_arr - pred_pow)**2))

    print()
    print(f"  Exponential fit: Var ~ exp({coeff_exp[1]:.3f} · L), RMSE={rmse_exp:.4e}")
    print(f"  Power-law fit:   Var ~ L^{coeff_pow[1]:.3f},         RMSE={rmse_pow:.4e}")

    better = "EXPONENTIAL" if rmse_exp < rmse_pow else "POWER LAW"
    print(f"  Better fit: {better}")

    print()
    print("  Per-layer enhancement ratios:")
    for i in range(1, len(vars_by_depth)):
        ratio = vars_by_depth[i] / vars_by_depth[i-1] if vars_by_depth[i-1] > 0 else float('nan')
        print(f"    L={i-1}→{i}: {ratio:.2f}×")


# Summary comparison
print_section("COMPARISON ACROSS σ VALUES")
print("  σ = 0.2: weakly coupled (linearized regime)")
print("  σ = 0.5: intermediate")
print("  σ = 1.0: standard initialization (strongly coupled)")
print()
print("  Question: does the power-law scaling persist at standard init?")
print("  If yes: the multi-layer SYK mechanism is robust across regimes.")
print("  If no: the standard regime has qualitatively different depth physics.")
