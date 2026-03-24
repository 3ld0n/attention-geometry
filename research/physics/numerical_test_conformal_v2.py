"""
Conformal Fixed Point Test v2:
Measure the CONNECTED two-point correlator (variance across disorder realizations)
as a function of position separation.

The mean two-point function homogenizes with depth (all tokens converge).
The SYK physics lives in the fluctuations — the disorder-induced variance.
If the conformal fixed point governs, Var_disorder[G(Δx)] should show
power-law scaling in Δx.

Ariel — March 24, 2026
"""

import numpy as np
from scipy.special import softmax

np.random.seed(42)

n_tokens = 128
d_model = 128
d_k = 64
d_v = 128
sigma = 1.0
n_realizations = 500

max_dx = n_tokens // 2


def layer_norm(X, eps=1e-5):
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def attention_layer(X):
    WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WV = np.random.randn(d_model, d_v) * sigma / np.sqrt(d_model)
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    scores = Q @ K.T / np.sqrt(d_k)
    alpha = softmax(scores, axis=1)
    return layer_norm(X + alpha @ V)


# Fourier feature embeddings
n_freq = d_model // 2
omega = np.random.randn(n_freq) * 0.3
positions = np.arange(n_tokens, dtype=float)

X0 = np.zeros((n_tokens, d_model))
for j in range(n_freq):
    X0[:, 2*j] = np.cos(omega[j] * positions)
    X0[:, 2*j+1] = np.sin(omega[j] * positions)
X0 *= np.sqrt(2.0 / d_model)
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True) * np.sqrt(d_model)


def compute_bilocal_per_dx(X):
    """For each Δx, compute the average bilocal H(Δx) = (1/(n-Δx)) Σ_i K_out[i, i+Δx]."""
    K_out = X @ X.T / d_model
    H = np.zeros(max_dx)
    for dx in range(max_dx):
        H[dx] = np.mean([K_out[i, i + dx] for i in range(n_tokens - dx)])
    return H


def fit_power_law(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (y_arr > 1e-30)
    if np.sum(mask) < 5:
        return None, None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask])
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    exponent = -coeffs[1]
    return exponent, exponent / 2, R2


depths = [0, 1, 2, 4, 8, 12]
dx_arr = np.arange(max_dx)

print("=" * 80)
print("  CONNECTED CORRELATOR TEST (Variance across disorder realizations)")
print(f"  n_tokens={n_tokens}, d_model={d_model}, d_k={d_k}, σ={sigma}")
print(f"  {n_realizations} realizations, Fourier feature embeddings")
print("=" * 80)


# === Main: collect H(Δx) for each realization, compute variance ===
for L in depths:
    all_H = np.zeros((n_realizations, max_dx))

    for r in range(n_realizations):
        X = X0.copy()
        for _ in range(L):
            X = attention_layer(X)
        all_H[r] = compute_bilocal_per_dx(X)

    mean_H = np.mean(all_H, axis=0)
    var_H = np.var(all_H, axis=0)

    print(f"\n  L={L}:")
    print(f"    Mean  G(0)={mean_H[0]:.4f}  G(4)={mean_H[4]:.4f}  G(16)={mean_H[16]:.4f}  G(32)={mean_H[32]:.4f}")
    print(f"    Var   G(0)={var_H[0]:.4e}  G(4)={var_H[4]:.4e}  G(16)={var_H[16]:.4e}  G(32)={var_H[32]:.4e}")

    if L == 0:
        print("    (L=0: no random weights, variance is numerical noise)")
        continue

    # Fit power law to Var(Δx) — this is the connected correlator
    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, var_H, cutoff_low=2)

    if exp_pw is not None:
        print(f"    Power law fit to Var(Δx): exponent = {exp_pw:.4f}  (Δ_eff = {delta_pw:.4f})  R² = {R2_pw:.4f}")

    # Also fit to the NORMALIZED connected correlator: Var(Δx) / Var(0)
    if var_H[0] > 0:
        var_norm = var_H / var_H[0]
        exp_n, delta_n, R2_n = fit_power_law(dx_arr, var_norm, cutoff_low=2)
        if exp_n is not None:
            print(f"    Normalized Var(Δx)/Var(0): exponent = {exp_n:.4f}  (Δ_eff = {delta_n:.4f})  R² = {R2_n:.4f}")

    # Key diagnostic: how does variance decay with separation?
    if var_H[4] > 0:
        print(f"    Var ratios: V(8)/V(4)={var_H[8]/var_H[4]:.4f}  V(16)/V(4)={var_H[16]/var_H[4]:.4f}  V(32)/V(4)={var_H[32]/var_H[4]:.4f}")
        # SYK Δ=1/4 prediction for the FOUR-point connected function is more complex,
        # but for the two-point variance, power law ~ Δx^{-4Δ} = Δx^{-1} for Δ=1/4
        print(f"    (Δx^{{-1}} pred: V(8)/V(4)={4/8:.4f}  V(16)/V(4)={4/16:.4f}  V(32)/V(4)={4/32:.4f})")
        print(f"    (Δx^{{-0.5}} pred: V(8)/V(4)={(4/8)**0.5:.4f}  V(16)/V(4)={(4/16)**0.5:.4f}  V(32)/V(4)={(4/32)**0.5:.4f})")


# === Summary ===
print()
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
print()
print("  The variance of G(Δx) across realizations is the connected correlator.")
print("  If the SYK conformal fixed point governs, this should show power-law decay.")
print("  The exponent depends on the conformal dimension Δ.")
print()
print("  For Δ = 1/4 (SYK q=4, D=1):")
print("    - Two-point propagator: G(Δx) ~ Δx^{-2Δ} = Δx^{-0.5}")
print("    - Variance of two-point: Var(Δx) ~ Δx^{-4Δ} = Δx^{-1} (if propagator fluctuations)")
print("    - Four-point connected: more complex structure (ladder diagrams)")
print()
