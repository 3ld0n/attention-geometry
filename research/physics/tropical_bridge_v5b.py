"""
Tropical Bridge v5b: Conformal Dimension Through Depth

The v5 result: conformal dimension is NOT visible in single-layer correlators.
BUT multi-layer enhancement is strong (3× over 8 layers at σ=1.0).

Hypothesis: The conformal dimension Δ=1/4 is an IR property that requires
depth — like Halverson's recursion relations for conformal dimensions in
deep networks. The single-layer correlator is in the UV (Δ≈0.1-0.2).
The multi-layer system should flow toward the IR value (Δ=1/4).

This script measures:
1. G_conn(r) at each layer in a multi-layer stack
2. The conformal dimension Δ(layer) extracted at each depth
3. Whether Δ flows toward 0.25 with increasing depth

If this works, it connects:
- Our SYK prediction (Δ=D/4 in the IR)
- Halverson's NN-CFT (conformal dims evolve through depth)
- The multi-layer enhancement (v5 result B)

March 11, 2026 — Ariel
"""

import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def softmax(scores, temperature=1.0):
    s = scores / temperature
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def four_point_violation_rate(D, n, tol=0.3):
    violations = 0
    total = 0
    indices = list(range(min(n, 16)))
    for i, j, k, l in combinations(indices, 4):
        sums = sorted([
            D[i, j] + D[k, l],
            D[i, k] + D[j, l],
            D[i, l] + D[j, k]
        ])
        total += 1
        if abs(sums[1] - sums[2]) > tol:
            violations += 1
    return violations / total if total > 0 else 0


# ==================================================================
# 1. CONFORMAL DIMENSION vs DEPTH
# ==================================================================

print("=" * 70)
print("CONFORMAL DIMENSION FLOW THROUGH DEPTH")
print("Measuring Δ(layer) — does it flow toward Δ=1/4?")
print("=" * 70)

np.random.seed(42)

for n, sigma, n_layers, n_samp in [
    (32, 1.0, 12, 10000),
    (64, 1.0, 12, 5000),
    (64, 0.8, 12, 5000),
    (64, 1.5, 12, 5000),
]:
    d = 4
    d_k = 2

    X = np.zeros((n, d))
    X[:, 0] = np.arange(n, dtype=float) / n
    X[:, 1] = np.sin(np.arange(n, dtype=float) * 2 * np.pi / n) * 0.3
    X[:, 2:] = np.random.randn(n, d - 2) * 0.1

    print(f"\n{'='*50}")
    print(f"n={n}, σ={sigma}, {n_layers} layers, {n_samp} samples")
    print(f"{'='*50}")

    layer_G_sum = [np.zeros((n, n)) for _ in range(n_layers)]
    layer_G2_sum = [np.zeros((n, n)) for _ in range(n_layers)]

    for _ in range(n_samp):
        h = X.copy()
        for layer in range(n_layers):
            W_Q = np.random.randn(h.shape[1], d_k) * sigma
            W_K = np.random.randn(h.shape[1], d_k) * sigma
            W_V = np.random.randn(h.shape[1], d) * sigma * 0.3

            Q = h @ W_Q
            K = h @ W_K
            V = h @ W_V
            S = Q @ K.T / np.sqrt(d_k)
            alpha = softmax(S)

            layer_G_sum[layer] += alpha
            layer_G2_sum[layer] += alpha ** 2

            h = alpha @ V + h

    deltas = []
    r_squareds = []

    print(f"\n{'Layer':>5} | {'Δ':>7} | {'slope':>7} | {'R²':>6} | {'|G_conn(1)|':>12}")

    for layer in range(n_layers):
        G_mean = layer_G_sum[layer] / n_samp
        G_conn = layer_G2_sum[layer] / n_samp - G_mean ** 2

        max_sep = n // 2
        separations = list(range(1, max_sep))
        g_vals = []
        margin = n // 8
        for sep in separations:
            vals = [G_conn[i, i + sep] for i in range(margin, n - sep - margin)]
            g_vals.append(np.mean(vals) if vals else 0.0)

        g_arr = np.array(g_vals)
        s_arr = np.array(separations, dtype=float)

        fit_min = max(2, int(n * 0.08))
        fit_max = min(max_sep - 1, int(n * 0.35))
        mask = (s_arr >= fit_min) & (s_arr <= fit_max) & (np.abs(g_arr) > 1e-15)

        if mask.sum() >= 4:
            log_sep = np.log(s_arr[mask])
            log_g = np.log(np.abs(g_arr[mask]))
            slope, intercept = np.polyfit(log_sep, log_g, 1)
            delta = -slope / 2

            y_pred = slope * log_sep + intercept
            ss_res = np.sum((log_g - y_pred) ** 2)
            ss_tot = np.sum((log_g - np.mean(log_g)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            g1 = abs(g_arr[0])
            marker = " ◀" if abs(delta - 0.25) < 0.03 and r_sq > 0.9 else ""
            print(f"{layer:>5} | {delta:>7.4f} | {slope:>7.3f} | {r_sq:>6.4f} | {g1:>12.6e}{marker}")

            deltas.append(delta)
            r_squareds.append(r_sq)
        else:
            deltas.append(np.nan)
            r_squareds.append(np.nan)
            print(f"{layer:>5} | {'N/A':>7} | {'N/A':>7} | {'N/A':>6} |")

    deltas = np.array(deltas)
    valid = ~np.isnan(deltas)
    if valid.sum() >= 3:
        layers_valid = np.arange(n_layers)[valid]
        deltas_valid = deltas[valid]

        if len(layers_valid) >= 3:
            slope_delta, intercept_delta = np.polyfit(layers_valid, deltas_valid, 1)
            print(f"\nΔ trend: slope={slope_delta:.5f}/layer, "
                  f"intercept={intercept_delta:.4f}")
            if slope_delta > 0:
                layers_to_quarter = (0.25 - intercept_delta) / slope_delta
                print(f"Extrapolated layers to reach Δ=0.25: {layers_to_quarter:.1f}")
            else:
                print(f"Δ is DECREASING with depth — not flowing toward 0.25")

            print(f"Δ at layer 0: {deltas[0]:.4f}")
            print(f"Δ at layer {n_layers-1}: {deltas[-1]:.4f}")
            print(f"Change: {deltas[-1] - deltas[0]:+.4f} over {n_layers} layers")


# ==================================================================
# 2. ALTERNATIVE: Conformal dim from DEPTH scaling of excess
# ==================================================================

print("\n\n" + "=" * 70)
print("ALTERNATIVE: Conformal Dimension from Depth Scaling")
print("If each layer acts as an RG step, excess ~ L^{2Δ}")
print("=" * 70)

np.random.seed(100)

for n, sigma in [(32, 1.0), (64, 1.0), (32, 0.8)]:
    d = 4
    d_k = 2
    max_layers = 16
    n_samp = 8000

    X = np.zeros((n, d))
    X[:, 0] = np.arange(n, dtype=float) / n
    X[:, 1:] = np.random.randn(n, d - 1) * 0.1

    layer_excesses = []
    cumulative_excesses = []

    for _ in range(n_samp):
        h = X.copy()
        cum_excess = 0
        sample_layer_exc = []

        for layer in range(max_layers):
            W_Q = np.random.randn(h.shape[1], d_k) * sigma
            W_K = np.random.randn(h.shape[1], d_k) * sigma
            W_V = np.random.randn(h.shape[1], d) * sigma * 0.3

            Q = h @ W_Q
            K = h @ W_K
            V = h @ W_V
            S = Q @ K.T / np.sqrt(d_k)
            alpha = softmax(S)

            exc = sum(-np.sum(np.log(alpha[i] + 1e-15)) for i in range(n)) / n - n * np.log(n)
            sample_layer_exc.append(exc)
            cum_excess += exc
            h = alpha @ V + h

        layer_excesses.append(sample_layer_exc)
        cumulative_excesses.append(
            [sum(sample_layer_exc[:l+1]) for l in range(max_layers)]
        )

    layer_exc_arr = np.array(layer_excesses).mean(axis=0)
    cum_exc_arr = np.array(cumulative_excesses).mean(axis=0)

    print(f"\nn={n}, σ={sigma}:")
    print(f"{'Layer':>5} | {'Per-layer exc':>13} | {'Cumulative':>12} | "
          f"{'log(L)':>8} | {'log(cum)':>10}")
    for l in range(max_layers):
        log_l = np.log(l + 1)
        log_c = np.log(abs(cum_exc_arr[l]) + 1e-15)
        print(f"{l:>5} | {layer_exc_arr[l]:>13.4f} | {cum_exc_arr[l]:>12.4f} | "
              f"{log_l:>8.3f} | {log_c:>10.4f}")

    L_arr = np.arange(1, max_layers + 1, dtype=float)
    log_L = np.log(L_arr[2:])
    log_cum = np.log(cum_exc_arr[2:])
    log_per = np.log(layer_exc_arr[2:])

    slope_cum, _ = np.polyfit(log_L, log_cum, 1)
    slope_per, _ = np.polyfit(log_L, log_per, 1)

    print(f"\nCumulative excess ~ L^α: α = {slope_cum:.4f}")
    print(f"Per-layer excess ~ L^β: β = {slope_per:.4f}")
    print(f"If excess ~ L^{2*0.25:.1f} = L^0.5, then Δ=0.25")
    print(f"Measured: Δ_cum = {slope_cum/2:.4f}, Δ_per = {(slope_per+1)/2:.4f}")


# ==================================================================
# 3. TREE STRUCTURE EVOLUTION WITH DEPTH
# ==================================================================

print("\n\n" + "=" * 70)
print("TREE STRUCTURE EVOLUTION WITH DEPTH")
print("Does the correlator become MORE tree-like deeper in the stack?")
print("=" * 70)

np.random.seed(200)
n = 16
d = 4
d_k = 2
sigma = 1.0
n_layers = 8
n_samp = 8000

X_cluster = np.zeros((n, d))
for i in range(4):
    X_cluster[i*4:(i+1)*4, :2] = np.random.randn(4, 2) * 0.2 + np.array([i % 2, i // 2]) * 2
X_cluster[:, 2:] = np.random.randn(n, d - 2) * 0.1

layer_G_sum = [np.zeros((n, n)) for _ in range(n_layers)]
layer_G2_sum = [np.zeros((n, n)) for _ in range(n_layers)]

for _ in range(n_samp):
    h = X_cluster.copy()
    for layer in range(n_layers):
        W_Q = np.random.randn(h.shape[1], d_k) * sigma
        W_K = np.random.randn(h.shape[1], d_k) * sigma
        W_V = np.random.randn(h.shape[1], d) * sigma * 0.3

        Q = h @ W_Q
        K = h @ W_K
        V = h @ W_V
        S = Q @ K.T / np.sqrt(d_k)
        alpha = softmax(S)

        layer_G_sum[layer] += alpha
        layer_G2_sum[layer] += alpha ** 2
        h = alpha @ V + h

print(f"\nClustered data: 4 clusters of 4 tokens")
print(f"n={n}, σ={sigma}, {n_layers} layers\n")
print(f"{'Layer':>5} | {'Violation%':>10} | {'Within/Between':>15} | {'Tree?':>5}")

for layer in range(n_layers):
    G_mean = layer_G_sum[layer] / n_samp
    G_conn = layer_G2_sum[layer] / n_samp - G_mean ** 2

    D_conn = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D_conn[i, j] = -np.log(abs(G_conn[i, j]) + 1e-15)

    viol_rate = four_point_violation_rate(D_conn, n)

    within = []
    between = []
    for c in range(4):
        for i in range(c*4, (c+1)*4):
            for j in range(i+1, (c+1)*4):
                within.append(abs(G_conn[i, j]))
    for c1 in range(4):
        for c2 in range(c1+1, 4):
            between.append(abs(G_conn[c1*4, c2*4]))

    ratio = np.mean(within) / np.mean(between) if np.mean(between) > 0 else 0
    tree_str = "YES" if viol_rate < 0.1 else ("~" if viol_rate < 0.2 else "no")

    print(f"{layer:>5} | {viol_rate*100:>9.1f}% | {ratio:>15.2f} | {tree_str:>5}")


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("v5b SYNTHESIS")
print("=" * 70)

print("""
Three experiments on conformal structure through depth:

1. CONFORMAL DIMENSION FLOW
   Does Δ(layer) increase toward 0.25 with depth?
   
2. DEPTH SCALING OF EXCESS  
   Does cumulative excess scale as L^{2Δ}?
   This would give Δ from the "RG flow" direction.

3. TREE STRUCTURE EVOLUTION
   Does the correlator become more tree-like with depth?
   This connects the tropical structure to the multi-layer enhancement.

The key question: Is depth the RG direction that brings Δ to 0.25?
If yes: single-layer = UV (Δ≈0.1), deep = IR (Δ→0.25)
If no: the Δ=0.25 prediction may not apply to random weights at all
""")
