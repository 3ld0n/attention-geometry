"""
BCFT Multi-Layer Composition Analysis

The direct comparison (single-layer Δ → accuracy) failed because:
  - Measured median Δ = 0.49 at single-layer level
  - Free-fit Δ from accuracy data = 0.12-0.17
  - Gap = multi-layer composition

This script tests whether summing the BCFT profiles across all heads
(modeling the residual stream) produces a composite profile whose
effective shape matches the published accuracy curve.

Theory: In a transformer with residual connections, information from
position k reaches the output through the sum of all attention heads:
  G_composite(q,k) = Σ_heads C_h · |q-k|^{-2Δ_h} · [1 + λ_h · η^Δ_h]

The effective Δ of this sum should be smaller than any individual Δ
(dominated by the flattest heads at large separations), and the
effective λ should be larger (boundary effects accumulate).

Ariel — April 15, 2026
"""

import json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

# Load measurements
with open('research/physics/bcft_longchat_measurements.json') as f:
    data = json.load(f)

heads = data['heads']

# Liu et al. published accuracy for LongChat-13B-16K
LIU = {
    10: {'indices': np.array([0, 4, 9]),
         'accuracy': np.array([72.1, 58.9, 58.5])},
    20: {'indices': np.array([0, 4, 9, 14, 19]),
         'accuracy': np.array([68.6, 57.4, 55.3, 52.5, 55.0])},
    30: {'indices': np.array([0, 4, 9, 14, 19, 24, 29]),
         'accuracy': np.array([57.4, 53.3, 51.4, 49.2, 48.6, 46.2, 46.0])},
}


def bcft_head_attention(indices, n_docs, delta, lam, C):
    """BCFT attention for a single head at document positions."""
    x_q = n_docs + 0.5
    positions = indices + 0.5
    attn = np.zeros(len(positions))
    for i, x_k in enumerate(positions):
        dx = x_q - x_k
        if dx <= 0 or x_k <= 0:
            continue
        eta = dx**2 / (4.0 * x_q * x_k)
        attn[i] = C * dx ** (-2 * delta) * (1.0 + lam * eta ** delta)
    return attn


def bcft_single_head_curve(indices, n_docs, delta, lam):
    """Normalized BCFT curve (C-independent)."""
    x_q = n_docs + 0.5
    positions = indices + 0.5
    attn = np.zeros(len(positions))
    for i, x_k in enumerate(positions):
        dx = x_q - x_k
        if dx <= 0 or x_k <= 0:
            continue
        eta = dx**2 / (4.0 * x_q * x_k)
        attn[i] = dx ** (-2 * delta) * (1.0 + lam * eta ** delta)
    if attn.max() > 0:
        attn /= attn.max()
    return attn


# ========== COMPOSITE PROFILE ==========
print("=" * 80)
print("  MULTI-LAYER COMPOSITION: SUMMING BCFT ACROSS ALL HEADS")
print("=" * 80)
print()

for n_docs in [10, 20, 30]:
    entry = LIU[n_docs]
    indices = entry['indices'].astype(float)
    acc = entry['accuracy']

    # Sum BCFT attention across all conformal heads
    composite = np.zeros(len(indices))
    for h in heads:
        composite += bcft_head_attention(
            indices, n_docs, h['delta'], h['lambda'], h['C']
        )

    # Normalize
    if composite.max() > 0:
        comp_norm = composite / composite.max()
    else:
        comp_norm = np.ones(len(indices)) / len(indices)

    # Fit baseline + amplitude only (shape is fixed)
    A_mat = np.column_stack([np.ones(len(indices)), comp_norm])
    params, _, _, _ = np.linalg.lstsq(A_mat, acc, rcond=None)
    baseline, amplitude = params
    predicted = baseline + amplitude * comp_norm

    ss_res = np.sum((acc - predicted) ** 2)
    ss_tot = np.sum((acc - np.mean(acc)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

    rho, p = spearmanr(predicted, acc)

    print(f"  {n_docs} documents:")
    print(f"    R² = {R2:.4f}  Spearman ρ = {rho:+.4f} (p={p:.4f})")
    for i, (idx, m, pr, cn) in enumerate(
        zip(indices, acc, predicted, comp_norm)
    ):
        print(f"      doc {int(idx):>2d}: measured={m:5.1f}%  "
              f"predicted={pr:5.1f}%  shape={cn:.4f}")
    print()


# ========== FIT EFFECTIVE Δ TO COMPOSITE ==========
print("=" * 80)
print("  EFFECTIVE Δ AND λ OF THE COMPOSITE PROFILE")
print("=" * 80)
print()
print("  Fitting a single BCFT curve to the composite profile")
print("  to find the 'effective' parameters of the multi-layer stack")
print()

for n_docs in [20, 30]:
    entry = LIU[n_docs]
    indices = entry['indices'].astype(float)

    # Composite at dense positions for fitting
    all_pos = np.arange(n_docs, dtype=float)
    composite_dense = np.zeros(n_docs)
    for h in heads:
        composite_dense += bcft_head_attention(
            all_pos, n_docs, h['delta'], h['lambda'], h['C']
        )
    if composite_dense.max() > 0:
        composite_dense /= composite_dense.max()

    # Fit single BCFT curve to composite
    def loss(params):
        d, l = params
        if d < 0.01 or d > 2.0 or l < 0:
            return 1e12
        curve = bcft_single_head_curve(all_pos, n_docs, d, l)
        return np.sum((composite_dense - curve) ** 2)

    res = differential_evolution(loss, [(0.01, 2.0), (0.0, 100.0)],
                                 seed=42, maxiter=3000, tol=1e-14)
    d_eff, l_eff = res.x
    fit_curve = bcft_single_head_curve(all_pos, n_docs, d_eff, l_eff)
    ss_res = np.sum((composite_dense - fit_curve) ** 2)
    ss_tot = np.sum((composite_dense - np.mean(composite_dense)) ** 2)
    R2_fit = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

    # Valley of composite
    valley_idx = np.argmin(composite_dense)
    valley_frac = valley_idx / n_docs

    print(f"  {n_docs} documents:")
    print(f"    Effective Δ = {d_eff:.4f}  (vs single-head median 0.4921)")
    print(f"    Effective λ = {l_eff:.4f}  (vs single-head median 4.1144)")
    print(f"    Fit R² = {R2_fit:.6f}")
    print(f"    Valley at position {valley_idx} (fraction: {valley_frac:.3f})")
    print()


# ========== CONTRIBUTION WEIGHTING ==========
print("=" * 80)
print("  WHICH HEADS DOMINATE THE COMPOSITE?")
print("=" * 80)
print()

# For n_docs=20, compute each head's total contribution
n_docs = 20
indices = LIU[20]['indices'].astype(float)
contributions = []
for h in heads:
    attn = bcft_head_attention(indices, n_docs, h['delta'], h['lambda'], h['C'])
    contributions.append({
        'layer': h['layer'], 'head': h['head'],
        'delta': h['delta'], 'C': h['C'], 'lambda': h['lambda'],
        'total_attn': np.sum(attn),
        'attn_profile': attn,
    })

contributions.sort(key=lambda x: x['total_attn'], reverse=True)
total_all = sum(c['total_attn'] for c in contributions)

print(f"  Top 20 heads by contribution to composite (n_docs=20):")
print(f"  {'Layer':>5} {'Head':>5} {'Δ':>8} {'C':>10} {'λ':>8} "
      f"{'Contrib%':>9} {'Cum%':>7}")
print(f"  {'-'*60}")

cum = 0
for i, c in enumerate(contributions[:20]):
    pct = 100 * c['total_attn'] / total_all
    cum += pct
    print(f"  {c['layer']:>5d} {c['head']:>5d} {c['delta']:>8.4f} "
          f"{c['C']:>10.4f} {c['lambda']:>8.2f} {pct:>8.1f}% {cum:>6.1f}%")

print()
print(f"  Top 50 heads account for "
      f"{100*sum(c['total_attn'] for c in contributions[:50])/total_all:.1f}% "
      f"of total attention")
print(f"  Top 100 heads account for "
      f"{100*sum(c['total_attn'] for c in contributions[:100])/total_all:.1f}%")
print()

# What are the effective parameters of the dominant heads?
top_deltas = [c['delta'] for c in contributions[:50]]
top_lambdas = [c['lambda'] for c in contributions[:50]]
print(f"  Top 50 heads: median Δ = {np.median(top_deltas):.4f}, "
      f"median λ = {np.median(top_lambdas):.4f}")

# Composite from ONLY top N heads
for top_n in [10, 20, 50, 100]:
    comp_topn = np.zeros(len(indices))
    for c in contributions[:top_n]:
        comp_topn += c['attn_profile']
    if comp_topn.max() > 0:
        comp_topn /= comp_topn.max()

    A_mat = np.column_stack([np.ones(len(indices)), comp_topn])
    params, _, _, _ = np.linalg.lstsq(A_mat, LIU[20]['accuracy'], rcond=None)
    pred = params[0] + params[1] * comp_topn
    ss_res = np.sum((LIU[20]['accuracy'] - pred) ** 2)
    ss_tot = np.sum((LIU[20]['accuracy'] - np.mean(LIU[20]['accuracy'])) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
    rho, p = spearmanr(pred, LIU[20]['accuracy'])
    print(f"  Top {top_n:>3d} heads → R² = {R2:.4f}, ρ = {rho:+.4f} (p={p:.4f})")


# ========== LAYER-BY-LAYER COMPOSITION ==========
print()
print("=" * 80)
print("  LAYER-BY-LAYER COMPOSITION")
print("=" * 80)
print()
print("  How does the composite profile build up layer by layer?")
print()

n_docs = 20
indices = LIU[20]['indices'].astype(float)
acc_20 = LIU[20]['accuracy']

cumulative = np.zeros(len(indices))
print(f"  {'Layers':>12} {'R²':>8} {'ρ':>8} {'Eff_Δ':>8} {'Shape_summary':>30}")
print(f"  {'-'*70}")

for l_end in [5, 10, 15, 20, 25, 30, 35, 39]:
    layer_heads = [h for h in heads if h['layer'] <= l_end]
    comp_layer = np.zeros(len(indices))
    for h in layer_heads:
        comp_layer += bcft_head_attention(
            indices, n_docs, h['delta'], h['lambda'], h['C']
        )
    if comp_layer.max() > 0:
        cn = comp_layer / comp_layer.max()
    else:
        cn = np.ones(len(indices)) / len(indices)

    A_mat = np.column_stack([np.ones(len(indices)), cn])
    params, _, _, _ = np.linalg.lstsq(A_mat, acc_20, rcond=None)
    pred = params[0] + params[1] * cn
    ss_res = np.sum((acc_20 - pred) ** 2)
    ss_tot = np.sum((acc_20 - np.mean(acc_20)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
    rho, p = spearmanr(pred, acc_20)

    # Fit effective Δ to this subset
    all_pos = np.arange(n_docs, dtype=float)
    comp_dense = np.zeros(n_docs)
    for h in layer_heads:
        comp_dense += bcft_head_attention(
            all_pos, n_docs, h['delta'], h['lambda'], h['C']
        )
    if comp_dense.max() > 0:
        comp_dense /= comp_dense.max()

    def loss_eff(params):
        d, l = params
        if d < 0.01 or d > 2.0 or l < 0:
            return 1e12
        curve = bcft_single_head_curve(all_pos, n_docs, d, l)
        return np.sum((comp_dense - curve) ** 2)

    res_eff = differential_evolution(loss_eff, [(0.01, 2.0), (0.0, 100.0)],
                                     seed=42, maxiter=1000)
    d_eff = res_eff.x[0]

    shape = ", ".join(f"{v:.3f}" for v in cn)
    print(f"  {'0-'+str(l_end):>12} {R2:>8.4f} {rho:>+8.4f} "
          f"{d_eff:>8.4f}  [{shape}]")


# ========== THEORETICAL PREDICTION ==========
print()
print("=" * 80)
print("  SUM-OF-POWER-LAWS: THEORETICAL EFFECTIVE Δ")
print("=" * 80)
print()
print("  For a sum G(dx) = Σ C_i · dx^{-2Δ_i}, the effective exponent")
print("  at distance dx is: Δ_eff(dx) = -½ · d(log G)/d(log dx)")
print("  At large dx, Δ_eff → min(Δ_i) weighted by C_i")
print()

# Compute the composite power law at various distances (no boundary term)
dx_range = np.arange(3, 100, dtype=float)
G_sum = np.zeros_like(dx_range)
for h in heads:
    G_sum += h['C'] * dx_range ** (-2 * h['delta'])

# Local effective Δ
log_dx = np.log(dx_range)
log_G = np.log(G_sum)
# Numerical derivative
d_eff_local = np.zeros(len(dx_range) - 2)
for i in range(1, len(dx_range) - 1):
    d_eff_local[i-1] = -(log_G[i+1] - log_G[i-1]) / (
        2 * (log_dx[i+1] - log_dx[i-1])) / 2

print(f"  {'dx':>6} {'Δ_eff':>10}")
print(f"  {'-'*18}")
for dx_val in [5, 10, 15, 20, 30, 50, 80]:
    idx = np.argmin(np.abs(dx_range[1:-1] - dx_val))
    print(f"  {dx_val:>6d} {d_eff_local[idx]:>10.4f}")

print()
print(f"  At dx=20 (typical document separation in 20-doc context):")
idx20 = np.argmin(np.abs(dx_range[1:-1] - 20))
print(f"    Δ_eff = {d_eff_local[idx20]:.4f}")
print(f"    Free-fit Δ from Liu accuracy data = 0.1711")
print(f"    Single-head median Δ = 0.4921")
print()

# What would happen with only the low-Δ heads?
print("  Effect of filtering by Δ threshold:")
for d_thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.50]:
    filtered = [h for h in heads if h['delta'] < d_thresh]
    if len(filtered) < 2:
        continue
    G_filt = np.zeros_like(dx_range)
    for h in filtered:
        G_filt += h['C'] * dx_range ** (-2 * h['delta'])
    log_Gf = np.log(np.maximum(G_filt, 1e-30))
    idx20 = np.argmin(np.abs(dx_range[1:-1] - 20))
    d_eff_filt = -(log_Gf[idx20+2] - log_Gf[idx20]) / (
        2 * (log_dx[idx20+2] - log_dx[idx20])) / 2
    print(f"    Δ < {d_thresh:.2f}: {len(filtered):>4d} heads, "
          f"Δ_eff(dx=20) = {d_eff_filt:.4f}")


# ========== FINAL COMPARISON ==========
print()
print("=" * 80)
print("  SUMMARY: COMPOSITION RESOLVES THE GAP?")
print("=" * 80)
print()
print("  Single-head measurement:  median Δ = 0.4921, median λ = 4.11")
print("  Free fit to accuracy:     Δ = 0.12-0.17, λ = 14-50")
print(f"  Composite Δ_eff(dx=20):   {d_eff_local[np.argmin(np.abs(dx_range[1:-1] - 20))]:.4f}")
print()
print("  Multi-layer composition analysis:")
print("  - The composite profile is a sum of 1343 BCFT heads")
print("  - Dominated by high-C heads with varying Δ")
print("  - The effective Δ emerges from the mixture")
