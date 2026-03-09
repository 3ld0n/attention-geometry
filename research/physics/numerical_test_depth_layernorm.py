"""
Does LayerNorm tame the exponential blowup?

Without normalization: σ=1 gives exponential growth with accelerating
per-layer ratios. Real transformers use LayerNorm/RMSNorm.

Hypothesis: LayerNorm keeps the system near a conformal fixed point,
converting exponential growth into power-law (or saturating) growth.

In SYK terms: LayerNorm is the UV cutoff that prevents the effective
coupling from diverging with depth.

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
sigma = 1.0

X0 = np.random.randn(n_tokens, d_model)
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True) * np.sqrt(d_model)


def layer_norm(X, eps=1e-5):
    """Standard LayerNorm (per-token normalization, no learnable params)."""
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ================================================================
# Without LayerNorm (reproducing the σ=1 result)
# ================================================================
print_section("σ = 1.0 WITHOUT LayerNorm")

max_layers = 8
vars_no_ln = []

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

            X_curr = X_curr + Y  # residual only

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
    vars_no_ln.append(var_H)

    enh = var_H / vars_no_ln[0] if vars_no_ln[0] > 0 else float('nan')
    per_layer = var_H / vars_no_ln[max(0, n_layers-1)] if n_layers > 0 and vars_no_ln[n_layers-1] > 0 else 0
    print(f"  L={n_layers}: ⟨H⟩={mean_H:.4f}  Var={var_H:.4e}  Enh={enh:.0f}×  per-layer={per_layer:.2f}×")


# ================================================================
# With LayerNorm (Pre-LN style: LN before attention)
# ================================================================
print_section("σ = 1.0 WITH LayerNorm (Pre-LN)")

vars_ln = []

for n_layers in range(max_layers + 1):
    H_00 = np.zeros(n_real)

    for r in range(n_real):
        X_curr = X0.copy()

        for L in range(n_layers):
            X_normed = layer_norm(X_curr)

            WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
            WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
            WV = np.random.randn(d_model, d_v) * sigma / np.sqrt(d_model)

            Q = X_normed @ WQ
            K_keys = X_normed @ WK
            V = X_normed @ WV

            scores = Q @ K_keys.T / np.sqrt(d_k)
            alpha = softmax(scores, axis=1)
            Y = alpha @ V

            X_curr = X_curr + Y  # residual + LN

        K_mat = X_curr @ X_curr.T / d_model
        X_normed = layer_norm(X_curr)
        WQ_m = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        WK_m = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        Q_m = X_normed @ WQ_m
        K_m = X_normed @ WK_m
        sc = Q_m[0] @ K_m.T / np.sqrt(d_k)
        alpha_m = softmax(sc)

        H_00[r] = alpha_m @ K_mat @ alpha_m

    var_H = np.var(H_00)
    mean_H = np.mean(H_00)
    vars_ln.append(var_H)

    enh = var_H / vars_ln[0] if vars_ln[0] > 0 else float('nan')
    per_layer = var_H / vars_ln[max(0, n_layers-1)] if n_layers > 0 and vars_ln[n_layers-1] > 0 else 0
    print(f"  L={n_layers}: ⟨H⟩={mean_H:.4f}  Var={var_H:.4e}  Enh={enh:.0f}×  per-layer={per_layer:.2f}×")


# ================================================================
# Fit and compare
# ================================================================
print_section("COMPARISON")

for label, vars_list in [("No LN", vars_no_ln), ("With LN", vars_ln)]:
    vars_arr = np.array(vars_list[1:])
    layers_arr = np.arange(1, max_layers + 1)
    ln_vars = np.log(vars_arr)

    A_exp = np.column_stack([np.ones_like(layers_arr, dtype=float), layers_arr.astype(float)])
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, ln_vars, rcond=None)

    ln_layers = np.log(layers_arr.astype(float))
    A_pow = np.column_stack([np.ones_like(ln_layers), ln_layers])
    coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, ln_vars, rcond=None)

    pred_exp = np.exp(A_exp @ coeff_exp)
    pred_pow = np.exp(A_pow @ coeff_pow)
    rmse_exp = np.sqrt(np.mean((vars_arr - pred_exp)**2))
    rmse_pow = np.sqrt(np.mean((vars_arr - pred_pow)**2))

    better = "exponential" if rmse_exp < rmse_pow else "power law"
    print(f"  {label:8s}: exp rate={coeff_exp[1]:.3f}, power={coeff_pow[1]:.3f}, better={better}")

print()
print("  If LayerNorm converts exponential→power law:")
print("    LN acts as the UV cutoff that keeps the system near the conformal fixed point")
print("    This would be a physical explanation of why LayerNorm works")
