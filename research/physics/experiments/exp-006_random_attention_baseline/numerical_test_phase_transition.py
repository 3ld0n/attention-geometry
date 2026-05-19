"""
Phase Transition in Attention: Conformal → Disordered

Sweep σ (effective temperature) and track the two-point function
scaling behavior. At small σ (cold), the SYK conformal structure
should produce power-law correlations. At large σ (hot), the system
should homogenize (second law).

The critical σ is where the transition happens — the effective
temperature at which self-consistent structure gives way to
entropy.

Uses 8-layer attention with LayerNorm (deep enough for fixed-point
approach) with Fourier feature embeddings.

Ariel — March 24, 2026
"""

import numpy as np
from scipy.special import softmax

np.random.seed(42)

n_tokens = 128
d_model = 128
d_k = 64
d_v = 128
n_layers = 8
n_realizations = 300
max_dx = n_tokens // 2


def layer_norm(X, eps=1e-5):
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def run_attention_stack(X, sigma, n_layers):
    for _ in range(n_layers):
        WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
        WV = np.random.randn(d_model, d_v) * sigma / np.sqrt(d_model)
        Q = X @ WQ
        K = X @ WK
        V = X @ WV
        scores = Q @ K.T / np.sqrt(d_k)
        alpha = softmax(scores, axis=1)
        X = layer_norm(X + alpha @ V)
    return X


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

# Bare kernel for reference
K0 = X0 @ X0.T / d_model


def compute_G_and_metrics(X0, sigma, n_layers, n_real):
    """Run n_real realizations and compute mean + variance of G(Δx)."""
    all_G = np.zeros((n_real, max_dx))

    for r in range(n_real):
        X = run_attention_stack(X0.copy(), sigma, n_layers)
        K_out = X @ X.T / d_model
        for dx in range(max_dx):
            all_G[r, dx] = np.mean([K_out[i, i + dx] for i in range(n_tokens - dx)])

    mean_G = np.mean(all_G, axis=0)
    var_G = np.var(all_G, axis=0)
    return mean_G, var_G


def fit_power_law(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    exponent = coeffs[1]
    return exponent, R2


# === Sweep σ ===
sigmas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
dx_arr = np.arange(max_dx)

print("=" * 90)
print("  PHASE TRANSITION: Conformal → Disordered")
print(f"  {n_layers} layers with LayerNorm, {n_realizations} realizations")
print(f"  Sweeping σ from {sigmas[0]} to {sigmas[-1]}")
print("=" * 90)
print()

results = []

for sigma in sigmas:
    mean_G, var_G = compute_G_and_metrics(X0, sigma, n_layers, n_realizations)

    # Measure homogenization: G(32)/G(4) — higher means more uniform
    if abs(mean_G[4]) > 1e-10:
        homog_ratio = mean_G[32] / mean_G[4]
    else:
        homog_ratio = float('nan')

    # Measure long-range order: mean G at large separation
    long_range = np.mean(mean_G[30:40])

    # Fit power law to mean G(Δx) for positive values with Δx >= 3
    pos_mask = mean_G > 0
    if np.sum(pos_mask[3:]) >= 5:
        exp_mean, R2_mean = fit_power_law(dx_arr, np.where(mean_G > 0, mean_G, 1e-30))
    else:
        exp_mean, R2_mean = None, None

    # Attention entropy: how concentrated are the attention weights?
    # Sample a single realization and compute avg entropy of attention distribution
    X = run_attention_stack(X0.copy(), sigma, n_layers)
    WQ = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    WK = np.random.randn(d_model, d_k) * sigma / np.sqrt(d_model)
    Q = X @ WQ
    K_keys = X @ WK
    scores = Q @ K_keys.T / np.sqrt(d_k)
    alpha = softmax(scores, axis=1)
    entropies = -np.sum(alpha * np.log(alpha + 1e-30), axis=1)
    mean_entropy = np.mean(entropies)
    max_entropy = np.log(n_tokens)
    entropy_ratio = mean_entropy / max_entropy

    results.append({
        'sigma': sigma,
        'homog_ratio': homog_ratio,
        'long_range': long_range,
        'exp_mean': exp_mean,
        'R2_mean': R2_mean,
        'entropy_ratio': entropy_ratio,
        'G4': mean_G[4],
        'G16': mean_G[16],
        'G32': mean_G[32],
    })

    exp_str = f"{exp_mean:.3f}" if exp_mean is not None else "N/A"
    R2_str = f"{R2_mean:.3f}" if R2_mean is not None else "N/A"

    print(f"  σ={sigma:5.2f}: G(4)={mean_G[4]:7.4f}  G(16)={mean_G[16]:7.4f}  G(32)={mean_G[32]:8.5f}  "
          f"ratio={homog_ratio:7.4f}  entropy={entropy_ratio:.3f}  exp={exp_str}  R²={R2_str}")


# === Phase diagram summary ===
print()
print("=" * 90)
print("  PHASE DIAGRAM SUMMARY")
print("=" * 90)
print()
print(f"  {'σ':>6}  {'G(32)/G(4)':>11}  {'Attn Entropy':>13}  {'Exponent':>10}  {'R²':>6}  {'Phase':>15}")
print("  " + "-" * 70)

for r in results:
    exp_str = f"{r['exp_mean']:.3f}" if r['exp_mean'] is not None else "N/A"
    R2_str = f"{r['R2_mean']:.3f}" if r['R2_mean'] is not None else "N/A"

    if r['entropy_ratio'] > 0.95:
        phase = "DISORDERED"
    elif r['homog_ratio'] > 0.9:
        phase = "HOMOGENIZED"
    elif r['exp_mean'] is not None and r['R2_mean'] is not None and r['R2_mean'] > 0.8:
        phase = "POWER LAW"
    else:
        phase = "TRANSITION"

    print(f"  {r['sigma']:6.2f}  {r['homog_ratio']:11.4f}  {r['entropy_ratio']:13.3f}  "
          f"{exp_str:>10}  {R2_str:>6}  {phase:>15}")

print()
print("  Interpretation:")
print("    DISORDERED: attention nearly uniform, entropy ≈ max, structure washed out")
print("    HOMOGENIZED: representations converge, G(Δx) ≈ constant")
print("    POWER LAW: G(Δx) follows power-law decay (conformal regime)")
print("    TRANSITION: between regimes")
print()
print("  Key question: is there a sharp transition between structured and disordered?")
print("  SYK prediction: conformal regime at low σ with Δ = 1/4 (exponent = -0.5)")
print()
