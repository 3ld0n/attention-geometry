"""
Conformal Fixed Point Test:
Does deep attention with LayerNorm develop power-law correlations
from Gaussian-kernel inputs?

If the SYK conformal fixed point governs the deep attention regime,
the two-point function G(Δx) should transition from Gaussian decay
(bare kernel) to power-law decay G ~ |Δx|^{-2Δ} with Δ = 1/4
for 1D sequences.

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
n_realizations = 400

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


# === Build Fourier feature embeddings ===
# Frequency scale chosen so bare kernel has correlation length ~ 10 tokens
n_freq = d_model // 2
omega = np.random.randn(n_freq) * 0.3
positions = np.arange(n_tokens, dtype=float)

X0 = np.zeros((n_tokens, d_model))
for j in range(n_freq):
    X0[:, 2*j] = np.cos(omega[j] * positions)
    X0[:, 2*j+1] = np.sin(omega[j] * positions)
X0 *= np.sqrt(2.0 / d_model)
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True) * np.sqrt(d_model)


def measure_two_point(X):
    """Compute G(Δx) = position-averaged representation similarity."""
    K = X @ X.T / d_model
    G = np.zeros(max_dx)
    for dx in range(max_dx):
        vals = [K[i, i + dx] for i in range(n_tokens - dx)]
        G[dx] = np.mean(vals)
    return G


def fit_power_law(dx_arr, G_arr, cutoff_low=4, cutoff_high=None):
    """Fit G ~ A * |dx|^{-exponent} on log-log scale. Return (exponent, Δ, R²)."""
    if cutoff_high is None:
        cutoff_high = len(G_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (G_arr > 0)
    if np.sum(mask) < 5:
        return None, None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_G = np.log(G_arr[mask])
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_G, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_G - pred)**2)
    ss_tot = np.sum((log_G - np.mean(log_G))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    exponent = -coeffs[1]
    return exponent, exponent / 2, R2


def fit_gaussian(dx_arr, G_arr, cutoff_low=4, cutoff_high=None):
    """Fit G ~ A * exp(-dx² / 2σ²). Return (σ, R²)."""
    if cutoff_high is None:
        cutoff_high = len(G_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (G_arr > 0)
    if np.sum(mask) < 5:
        return None, None
    log_G = np.log(G_arr[mask])
    dx2 = (dx_arr[mask].astype(float))**2
    A = np.column_stack([np.ones_like(dx2), -dx2])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_G, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_G - pred)**2)
    ss_tot = np.sum((log_G - np.mean(log_G))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    sig = np.sqrt(1 / (2 * coeffs[1])) if coeffs[1] > 0 else float('inf')
    return sig, R2


# === Main experiment ===
depths = [0, 1, 2, 4, 8, 12, 16]
dx_arr = np.arange(max_dx)

print("=" * 80)
print("  CONFORMAL FIXED POINT TEST")
print(f"  n_tokens={n_tokens}, d_model={d_model}, d_k={d_k}, σ={sigma}")
print(f"  {n_realizations} realizations, Fourier feature embeddings")
print("=" * 80)
print()

# Store results for comparison
all_G = {}

for L in depths:
    G_sum = np.zeros(max_dx)

    for r in range(n_realizations):
        X = X0.copy()
        for _ in range(L):
            X = attention_layer(X)
        G_sum += measure_two_point(X)

    G_avg = G_sum / n_realizations
    all_G[L] = G_avg

    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, G_avg)
    sig_g, R2_g = fit_gaussian(dx_arr, G_avg)

    print(f"  L={L:2d}:")
    print(f"    G(0)={G_avg[0]:.4f}  G(4)={G_avg[4]:.4f}  G(16)={G_avg[16]:.4f}  G(32)={G_avg[32]:.6f}  G(48)={G_avg[48]:.6f}")

    if exp_pw is not None:
        print(f"    Power law fit:  exponent = {exp_pw:.4f}  (Δ = {delta_pw:.4f})  R² = {R2_pw:.4f}")
    if sig_g is not None:
        print(f"    Gaussian fit:   σ = {sig_g:.2f}  R² = {R2_g:.4f}")

    if R2_pw is not None and R2_g is not None:
        better = "POWER LAW" if R2_pw > R2_g else "GAUSSIAN"
        print(f"    Better fit: {better}  (ΔR² = {abs(R2_pw - R2_g):.4f})")
        if R2_pw > R2_g and delta_pw is not None:
            if abs(delta_pw - 0.25) < 0.1:
                print(f"    ** Δ = {delta_pw:.4f} is within 0.1 of SYK prediction Δ = 0.25 **")
    print()


# === Summary ===
print("=" * 80)
print("  SUMMARY: Conformal dimension Δ vs depth")
print("=" * 80)
print(f"  {'L':>4}  {'Δ (power law)':>14}  {'R²_pw':>8}  {'R²_gauss':>10}  {'Better':>12}")
print("  " + "-" * 56)

for L in depths:
    G_avg = all_G[L]
    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, G_avg)
    _, R2_g = fit_gaussian(dx_arr, G_avg)
    better = "power law" if (R2_pw or 0) > (R2_g or 0) else "gaussian"
    delta_str = f"{delta_pw:.4f}" if delta_pw is not None else "N/A"
    R2_pw_str = f"{R2_pw:.4f}" if R2_pw is not None else "N/A"
    R2_g_str = f"{R2_g:.4f}" if R2_g is not None else "N/A"
    print(f"  {L:4d}  {delta_str:>14}  {R2_pw_str:>8}  {R2_g_str:>10}  {better:>12}")

print()
print("  SYK prediction for D=1: Δ = 0.2500")
print()

# === Tail behavior: ratio at large Δx ===
print("=" * 80)
print("  TAIL BEHAVIOR: G(Δx) at large separations")
print("=" * 80)
print(f"  {'L':>4}  {'G(32)/G(4)':>12}  {'G(48)/G(4)':>12}  {'PL pred 32':>12}  {'PL pred 48':>12}")
print("  " + "-" * 56)

for L in depths:
    G = all_G[L]
    ratio32 = G[32] / G[4] if G[4] > 0 else 0
    ratio48 = G[48] / G[4] if G[4] > 0 else 0
    # Power law prediction with Δ=1/4: ratio = (32/4)^{-2*0.25} = 8^{-0.5} = 0.354
    pl32 = (32/4)**(-0.5)
    pl48 = (48/4)**(-0.5)
    print(f"  {L:4d}  {ratio32:12.6f}  {ratio48:12.6f}  {pl32:12.6f}  {pl48:12.6f}")

print()
print(f"  Power law (Δ=1/4) predicts: G(32)/G(4) = {(32/4)**(-0.5):.4f}, G(48)/G(4) = {(48/4)**(-0.5):.4f}")
print(f"  Gaussian (σ=10) predicts:   G(32)/G(4) = {np.exp(-(32**2-4**2)/(2*100)):.4f}, G(48)/G(4) = {np.exp(-(48**2-4**2)/(2*100)):.6f}")
print()
