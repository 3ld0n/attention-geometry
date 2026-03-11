"""
Tropical Bridge v5: The Three Open Edges

Pushing on the three results that would strengthen the paper:

A. CONFORMAL DIMENSION — v4 got Δ=0.113, expected 0.25 for D=1.
   Diagnosis: σ=0.5 is too perturbative, n=32 has finite-size effects.
   Strategy: σ ∈ {1.0, 1.5, 2.0}, n ∈ {64, 128}, wider fitting windows.

B. MULTI-LAYER ENHANCEMENT — v4 tested at σ=0.2 where the effect is invisible.
   Strategy: σ~1.0 with residual connections, track per-layer excess growth.

C. MULTI-HEAD AS HIGHER GRASSMANNIAN — does H-head attention compute on Gr+(H,n)?
   Strategy: compare single-head vs multi-head canonical forms, test for
   higher-dimensional positive geometry structure.

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
# A. CONFORMAL DIMENSION — The Make-or-Break Test
# ==================================================================

print("=" * 70)
print("A. CONFORMAL DIMENSION — Stronger Coupling, Larger Systems")
print("=" * 70)
print()
print("Theory: G_conn(r) ~ r^{-2Δ} with Δ = D/(4) = 1/4 for D=1")
print("v4 got Δ=0.113 at σ=0.5, n=32. Pushing harder.\n")

np.random.seed(42)

results_A = []

for n in [32, 64, 128]:
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        d_k = 2

        X = np.zeros((n, 2))
        X[:, 0] = np.arange(n, dtype=float) / n
        X[:, 1] = 0.0

        n_samp = max(5000, 30000 // (n // 32))

        G_sum = np.zeros((n, n))
        G2_sum = np.zeros((n, n))

        for _ in range(n_samp):
            W_Q = np.random.randn(2, d_k) * sigma
            W_K = np.random.randn(2, d_k) * sigma
            Q = X @ W_Q
            K = X @ W_K
            S = Q @ K.T / np.sqrt(d_k)
            alpha = softmax(S)
            G_sum += alpha
            G2_sum += alpha ** 2

        G_mean = G_sum / n_samp
        G_conn = G2_sum / n_samp - G_mean ** 2

        max_sep = n // 2
        separations = list(range(1, max_sep))
        g_conn_by_sep = []

        for sep in separations:
            vals = []
            margin = n // 8
            for i in range(margin, n - sep - margin):
                vals.append(G_conn[i, i + sep])
            if vals:
                g_conn_by_sep.append(np.mean(vals))
            else:
                g_conn_by_sep.append(0.0)

        g_conn_arr = np.array(g_conn_by_sep)
        sep_arr = np.array(separations, dtype=float)

        for fit_min_frac, fit_max_frac, label in [
            (0.05, 0.25, "near"),
            (0.10, 0.40, "mid"),
            (0.05, 0.40, "wide"),
        ]:
            fit_min = max(2, int(n * fit_min_frac))
            fit_max = min(max_sep - 1, int(n * fit_max_frac))

            mask = (sep_arr >= fit_min) & (sep_arr <= fit_max) & (np.abs(g_conn_arr) > 1e-15)

            if mask.sum() >= 4:
                log_sep = np.log(sep_arr[mask])
                log_g = np.log(np.abs(g_conn_arr[mask]))

                slope, intercept = np.polyfit(log_sep, log_g, 1)
                delta_measured = -slope / 2

                y_pred = slope * log_sep + intercept
                ss_res = np.sum((log_g - y_pred) ** 2)
                ss_tot = np.sum((log_g - np.mean(log_g)) ** 2)
                r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                results_A.append({
                    'n': n, 'sigma': sigma, 'fit': label,
                    'delta': delta_measured, 'slope': slope, 'r_sq': r_sq,
                    'fit_pts': mask.sum()
                })

print(f"{'n':>4} | {'σ':>4} | {'fit':>5} | {'Δ':>7} | {'slope':>7} | {'R²':>6} | {'pts':>3}")
print("-" * 55)
for r in results_A:
    marker = " ◀" if abs(r['delta'] - 0.25) < 0.03 and r['r_sq'] > 0.95 else ""
    print(f"{r['n']:>4} | {r['sigma']:>4.1f} | {r['fit']:>5} | "
          f"{r['delta']:>7.4f} | {r['slope']:>7.3f} | {r['r_sq']:>6.4f} | "
          f"{r['fit_pts']:>3}{marker}")

print()
best = max(results_A, key=lambda r: r['r_sq'] if abs(r['delta'] - 0.25) < 0.1 else 0)
print(f"Best candidate: n={best['n']}, σ={best['sigma']}, "
      f"Δ={best['delta']:.4f} (R²={best['r_sq']:.4f})")
print(f"Target: Δ=0.25 for D=1")


# ==================================================================
# A2. DETAILED CORRELATOR PROFILE — best parameters
# ==================================================================

print("\n" + "=" * 70)
print("A2. Detailed Correlator Profile at Best Parameters")
print("=" * 70)

for n, sigma in [(64, 1.5), (128, 1.5), (64, 2.0), (128, 2.0)]:
    d_k = 2
    X = np.zeros((n, 2))
    X[:, 0] = np.arange(n, dtype=float) / n

    n_samp = max(5000, 20000 // (n // 32))

    G_sum = np.zeros((n, n))
    G2_sum = np.zeros((n, n))

    for _ in range(n_samp):
        W_Q = np.random.randn(2, d_k) * sigma
        W_K = np.random.randn(2, d_k) * sigma
        Q = X @ W_Q
        K = X @ W_K
        S = Q @ K.T / np.sqrt(d_k)
        alpha = softmax(S)
        G_sum += alpha
        G2_sum += alpha ** 2

    G_mean = G_sum / n_samp
    G_conn = G2_sum / n_samp - G_mean ** 2

    max_sep = n // 2
    separations = list(range(1, max_sep))
    g_vals = []
    margin = n // 8
    for sep in separations:
        vals = [G_conn[i, i + sep] for i in range(margin, n - sep - margin)]
        g_vals.append(np.mean(vals) if vals else 0.0)

    g_arr = np.array(g_vals)
    s_arr = np.array(separations, dtype=float)

    print(f"\nn={n}, σ={sigma}:")
    print(f"{'sep':>5} | {'G_conn':>14} | {'log(sep)':>8} | {'log|G_conn|':>12}")
    for i in range(0, min(len(separations), 20), max(1, len(separations) // 20)):
        sep = separations[i]
        gc = g_arr[i]
        print(f"{sep:>5} | {gc:>14.8f} | {np.log(sep):>8.3f} | "
              f"{np.log(abs(gc) + 1e-15):>12.4f}")


# ==================================================================
# B. MULTI-LAYER ENHANCEMENT at Real Coupling
# ==================================================================

print("\n" + "=" * 70)
print("B. MULTI-LAYER CANONICAL FORM ENHANCEMENT")
print("σ=1.0 with residual connections — does per-layer excess grow?")
print("=" * 70)

np.random.seed(123)
n = 16
d = 4
d_k = 2
n_samp = 15000
baseline_per_layer = n * np.log(n)

for sigma in [0.5, 1.0, 1.5]:
    print(f"\n--- σ = {sigma} ---")
    print(f"{'layers':>6} | {'total excess':>12} | {'per-layer':>10} | "
          f"{'ratio to L=1':>12} | {'enhancement':>11}")

    per_layer_L1 = None

    for n_layers in [1, 2, 3, 4, 6, 8]:
        X = np.random.randn(n, d)

        excesses = []
        per_layer_excesses = []

        for _ in range(n_samp):
            h = X.copy()
            layer_excesses = []

            for layer in range(n_layers):
                W_Q = np.random.randn(h.shape[1], d_k) * sigma
                W_K = np.random.randn(h.shape[1], d_k) * sigma
                W_V = np.random.randn(h.shape[1], d) * sigma * 0.3

                Q = h @ W_Q
                K = h @ W_K
                V = h @ W_V
                S = Q @ K.T / np.sqrt(d_k)
                alpha = softmax(S)

                layer_log_omega = 0.0
                for i in range(n):
                    layer_log_omega += -np.sum(np.log(alpha[i] + 1e-15))

                layer_excesses.append(layer_log_omega / n - baseline_per_layer / n)

                h = alpha @ V + h

            excesses.append(sum(layer_excesses))
            per_layer_excesses.append(layer_excesses)

        mean_total = np.mean(excesses)
        mean_per_layer = mean_total / n_layers

        per_layer_arr = np.array(per_layer_excesses)
        layer_means = per_layer_arr.mean(axis=0)

        if n_layers == 1:
            per_layer_L1 = mean_per_layer

        ratio = mean_per_layer / per_layer_L1 if per_layer_L1 and per_layer_L1 != 0 else 0

        last_layer = layer_means[-1]
        first_layer = layer_means[0]
        enhancement = last_layer / first_layer if first_layer != 0 else 0

        print(f"{n_layers:>6} | {mean_total:>12.6f} | {mean_per_layer:>10.6f} | "
              f"{ratio:>12.3f} | {enhancement:>11.3f}")

    if n_layers >= 4:
        print(f"\n  Per-layer breakdown (last run, {n_layers} layers):")
        for l_idx, lm in enumerate(layer_means):
            bar = "█" * max(1, int(lm / layer_means[0] * 20)) if layer_means[0] > 0 else ""
            print(f"    Layer {l_idx}: excess={lm:.6f}  {bar}")


# ==================================================================
# C. MULTI-HEAD AS HIGHER GRASSMANNIAN
# ==================================================================

print("\n" + "=" * 70)
print("C. MULTI-HEAD ATTENTION → Gr+(H, n)?")
print("Does H-head attention compute on a higher Grassmannian?")
print("=" * 70)

np.random.seed(456)
n = 16
d = 8
n_samp = 15000

print(f"\nTest: compare single-head (d_k=H) vs H-head (d_k=1 each)")
print(f"If multi-head lifts to Gr+(H,n), combined canonical form should differ.\n")

for sigma in [0.5, 1.0, 1.5]:
    print(f"--- σ = {sigma} ---")
    X = np.random.randn(n, d)

    for H in [1, 2, 4, 8]:
        single_head_dk = H
        multi_head_dk = 1

        single_excesses = []
        multi_excesses = []
        multi_combined_forms = []

        for _ in range(n_samp):
            # --- Single head with d_k = H ---
            W_Q = np.random.randn(d, single_head_dk) * sigma
            W_K = np.random.randn(d, single_head_dk) * sigma
            Q = X @ W_Q
            K = X @ W_K
            S = Q @ K.T / np.sqrt(single_head_dk)
            alpha = softmax(S)
            log_omega = sum(-np.sum(np.log(alpha[i] + 1e-15)) for i in range(n))
            single_excesses.append(log_omega / n - n * np.log(n) / n)

            # --- H heads, each with d_k = 1 ---
            total_multi_log_omega = 0.0
            head_alphas = []
            for h in range(H):
                W_Q_h = np.random.randn(d, multi_head_dk) * sigma
                W_K_h = np.random.randn(d, multi_head_dk) * sigma
                Q_h = X @ W_Q_h
                K_h = X @ W_K_h
                S_h = Q_h @ K_h.T / np.sqrt(multi_head_dk)
                alpha_h = softmax(S_h)
                head_alphas.append(alpha_h)
                for i in range(n):
                    total_multi_log_omega += -np.sum(np.log(alpha_h[i] + 1e-15))

            multi_excesses.append(total_multi_log_omega / (n * H) - n * np.log(n) / n)

            if H > 1:
                plucker_like = 0.0
                for h1, h2 in combinations(range(H), 2):
                    for i in range(n):
                        for j in range(i + 1, min(i + 4, n)):
                            det = (head_alphas[h1][i, j] * head_alphas[h2][i, i]
                                   - head_alphas[h1][i, i] * head_alphas[h2][i, j])
                            plucker_like += abs(det)
                multi_combined_forms.append(plucker_like)

        s_mean = np.mean(single_excesses)
        m_mean = np.mean(multi_excesses)
        ratio = m_mean / s_mean if s_mean != 0 else 0

        plucker_str = ""
        if multi_combined_forms:
            plucker_str = f"  Plücker={np.mean(multi_combined_forms):.6f}"

        print(f"  H={H}: single(dk={H})={s_mean:.6f}  "
              f"multi(H×dk=1)={m_mean:.6f}  ratio={ratio:.3f}{plucker_str}")

    print()


# ==================================================================
# C2. PLÜCKER POSITIVITY TEST
# ==================================================================

print("=" * 70)
print("C2. Are the Plücker-like coordinates positive? (Gr+ test)")
print("=" * 70)

np.random.seed(789)
n = 8
d = 4
sigma = 1.0
H_test = 2

n_samp = 10000
positive_count = 0
total_minors = 0

for _ in range(n_samp):
    X = np.random.randn(n, d)

    heads = []
    for h in range(H_test):
        W_Q = np.random.randn(d, 1) * sigma
        W_K = np.random.randn(d, 1) * sigma
        S = (X @ W_Q) @ (X @ W_K).T
        alpha = softmax(S)
        heads.append(alpha)

    for i in range(n):
        alpha_matrix = np.array([heads[h][i] for h in range(H_test)])
        for j1, j2 in combinations(range(n), 2):
            minor = alpha_matrix[0, j1] * alpha_matrix[1, j2] - alpha_matrix[0, j2] * alpha_matrix[1, j1]
            total_minors += 1
            if minor > 0:
                positive_count += 1

pos_frac = positive_count / total_minors if total_minors > 0 else 0
print(f"\nH={H_test} heads, n={n} tokens, σ={sigma}")
print(f"Plücker-like 2×2 minors: {positive_count}/{total_minors} positive ({pos_frac:.1%})")
print(f"If all positive → attention lives in Gr+(H,n)")
print(f"If ~50% positive → no positivity structure (random signs)")

if pos_frac > 0.9:
    print(f"\n→ STRONG POSITIVITY: {pos_frac:.1%} of minors positive!")
    print(f"  Multi-head attention may genuinely compute on Gr+(H,n)")
elif pos_frac > 0.6:
    print(f"\n→ PARTIAL POSITIVITY: {pos_frac:.1%} of minors positive")
    print(f"  Some positive structure but not full Gr+(H,n)")
else:
    print(f"\n→ NO POSITIVITY: {pos_frac:.1%} of minors positive")
    print(f"  Multi-head does NOT naturally live on Gr+(H,n)")


# ==================================================================
# D. SIGMA SCALING — Verify σ⁴ at stronger coupling
# ==================================================================

print("\n" + "=" * 70)
print("D. σ⁴ Scaling Verification at Stronger Coupling")
print("Does the quartic scaling persist beyond perturbative regime?")
print("=" * 70)

np.random.seed(42)
n = 16
d_k = 2
d = 4
X = np.random.randn(n, d)
n_samp = 20000

sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
excesses = []

for sigma in sigmas:
    excess_sum = 0
    for _ in range(n_samp):
        W_Q = np.random.randn(d, d_k) * sigma
        W_K = np.random.randn(d, d_k) * sigma
        Q = X @ W_Q
        K = X @ W_K
        S = Q @ K.T / np.sqrt(d_k)
        alpha = softmax(S)
        for i in range(n):
            excess_sum += -np.sum(np.log(alpha[i] + 1e-15)) - n * np.log(n)

    mean_excess = excess_sum / (n_samp * n)
    excesses.append(mean_excess)

exc_arr = np.array(excesses)
sig_arr = np.array(sigmas)

print(f"\n{'σ':>6} | {'excess':>12} | {'excess/σ⁴':>12} | {'excess/σ²':>12}")
print("-" * 55)
for i, sigma in enumerate(sigmas):
    e = exc_arr[i]
    e4 = e / sigma**4 if sigma > 0 else 0
    e2 = e / sigma**2 if sigma > 0 else 0
    print(f"{sigma:>6.2f} | {e:>12.6f} | {e4:>12.4f} | {e2:>12.4f}")

mask_pert = sig_arr <= 0.3
if mask_pert.sum() >= 3:
    log_s = np.log(sig_arr[mask_pert])
    log_e = np.log(exc_arr[mask_pert])
    slope_pert, _ = np.polyfit(log_s, log_e, 1)
    print(f"\nPerturbative regime (σ ≤ 0.3): power law exponent = {slope_pert:.3f}")

mask_strong = sig_arr >= 1.0
if mask_strong.sum() >= 3:
    log_s = np.log(sig_arr[mask_strong])
    log_e = np.log(np.abs(exc_arr[mask_strong]) + 1e-15)
    slope_strong, _ = np.polyfit(log_s, log_e, 1)
    print(f"Strong coupling (σ ≥ 1.0): power law exponent = {slope_strong:.3f}")
    print(f"\nIf exponent changes from ~4 to ~2: crossover from G⁴ to saturated regime")
    print(f"If exponent stays ~4: quartic vertex dominates even at strong coupling")


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS — What v5 Found")
print("=" * 70)

print("""
Results to assess:

A. CONFORMAL DIMENSION
   - Does Δ approach 0.25 at stronger coupling?
   - Does the power law improve (R² → 1)?
   - Which parameter regime is cleanest?

B. MULTI-LAYER ENHANCEMENT
   - Does per-layer excess grow with depth at σ~1?
   - If so, layers genuinely amplify the G⁴ signature
   - If not, the random-weight regime may not be deep enough

C. MULTI-HEAD → Gr+(H,n)
   - Are Plücker-like minors positive? (Gr+ test)
   - Does multi-head canonical form differ systematically from single-head?
   - This would connect H-head attention to the amplituhedron at k=H

D. σ⁴ SCALING
   - Does the quartic persist at strong coupling?
   - Or does a crossover reveal new physics?

See AMPLITUHEDRON_CONNECTIONS.md for the full research context.
""")
