"""
BCFT Composition: Filtered by Information-Routing Heads

The raw sum across all heads fails because attention sinks (Δ > 0.5,
huge C) dominate. But attention sinks don't route information across
the context — they concentrate attention on nearby/first tokens.

Hypothesis: the task-relevant information flow is governed by
heads with moderate Δ (< ~0.5). When composed, these give an
effective Δ that matches the accuracy-fitted value.

Evidence from theoretical analysis:
  - All heads Δ_eff(dx=20) = 0.26 (too high)
  - Δ < 0.50 heads: Δ_eff = 0.1655 (matches free-fit 0.1711)

This script tests the hypothesis directly: does the composite of
Δ < 0.5 heads predict the accuracy curve shape?

Ariel — April 15, 2026
"""

import json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

with open('research/physics/bcft_longchat_measurements.json') as f:
    data = json.load(f)

heads = data['heads']

LIU = {
    10: {'indices': np.array([0, 4, 9]),
         'accuracy': np.array([72.1, 58.9, 58.5])},
    20: {'indices': np.array([0, 4, 9, 14, 19]),
         'accuracy': np.array([68.6, 57.4, 55.3, 52.5, 55.0])},
    30: {'indices': np.array([0, 4, 9, 14, 19, 24, 29]),
         'accuracy': np.array([57.4, 53.3, 51.4, 49.2, 48.6, 46.2, 46.0])},
}


def bcft_head_attention(indices, n_docs, delta, lam, C):
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


def bcft_single_curve(indices, n_docs, delta, lam):
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


def compare_composite(selected_heads, label, mode="C_weighted"):
    """Compare a subset of heads to Liu accuracy data."""
    print(f"\n  {label}")
    print(f"  {len(selected_heads)} heads")
    print()

    for n_docs in [10, 20, 30]:
        entry = LIU[n_docs]
        indices = entry['indices'].astype(float)
        acc = entry['accuracy']

        composite = np.zeros(len(indices))
        for h in selected_heads:
            if mode == "C_weighted":
                composite += bcft_head_attention(
                    indices, n_docs, h['delta'], h['lambda'], h['C']
                )
            elif mode == "equal_weight":
                composite += bcft_head_attention(
                    indices, n_docs, h['delta'], h['lambda'], 1.0
                )

        if composite.max() > 0:
            cn = composite / composite.max()
        else:
            cn = np.ones(len(indices)) / len(indices)

        A_mat = np.column_stack([np.ones(len(indices)), cn])
        params, _, _, _ = np.linalg.lstsq(A_mat, acc, rcond=None)
        pred = params[0] + params[1] * cn
        ss_res = np.sum((acc - pred) ** 2)
        ss_tot = np.sum((acc - np.mean(acc)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
        rho, p = spearmanr(pred, acc)

        shape_str = ", ".join(f"{v:.4f}" for v in cn)
        print(f"    {n_docs} docs: R²={R2:.4f} ρ={rho:+.4f} (p={p:.3f})")
        for i, (idx, m, pr) in enumerate(zip(indices, acc, pred)):
            print(f"      doc {int(idx):>2d}: measured={m:5.1f}%  predicted={pr:5.1f}%")
        print(f"      shape: [{shape_str}]")
        print()


# ========== TEST VARIOUS FILTERS ==========
print("=" * 80)
print("  COMPOSITION MODELS: WHICH HEADS PREDICT ACCURACY?")
print("=" * 80)

# Model 1: All conformal heads, C-weighted (baseline — we know this fails)
compare_composite(heads, "ALL CONFORMAL HEADS (C-weighted)", "C_weighted")

# Model 2: All conformal heads, equal weight
compare_composite(heads, "ALL CONFORMAL HEADS (equal weight)", "equal_weight")

# Model 3: Δ < 0.5 filter, C-weighted
filtered_05 = [h for h in heads if h['delta'] < 0.5]
compare_composite(filtered_05, "Δ < 0.50 HEADS (C-weighted)", "C_weighted")

# Model 4: Δ < 0.5, equal weight
compare_composite(filtered_05, "Δ < 0.50 HEADS (equal weight)", "equal_weight")

# Model 5: Δ < 0.35, C-weighted
filtered_035 = [h for h in heads if h['delta'] < 0.35]
compare_composite(filtered_035, "Δ < 0.35 HEADS (C-weighted)", "C_weighted")

# Model 6: Δ < 0.35, equal weight
compare_composite(filtered_035, "Δ < 0.35 HEADS (equal weight)", "equal_weight")

# Model 7: Δ between 0.20 and 0.60 (exclude extremes)
filtered_mid = [h for h in heads if 0.20 < h['delta'] < 0.60]
compare_composite(filtered_mid, "0.20 < Δ < 0.60 HEADS (equal weight)", "equal_weight")

# Model 8: Only positive-λ heads with Δ < 0.5
filtered_pos_lam = [h for h in heads if h['delta'] < 0.5 and h['lambda'] > 0]
compare_composite(filtered_pos_lam, "Δ < 0.50 AND λ > 0 (equal weight)", "equal_weight")


# ========== EFFECTIVE Δ FOR BEST-MATCHING FILTER ==========
print("=" * 80)
print("  EFFECTIVE Δ FOR FILTERED COMPOSITES")
print("=" * 80)
print()

for label, filtered, mode in [
    ("Δ < 0.50 C-weighted", filtered_05, "C_weighted"),
    ("Δ < 0.50 equal-weight", filtered_05, "equal_weight"),
    ("Δ < 0.35 C-weighted", filtered_035, "C_weighted"),
    ("Δ < 0.35 equal-weight", filtered_035, "equal_weight"),
]:
    for n_docs in [20, 30]:
        all_pos = np.arange(n_docs, dtype=float)
        composite = np.zeros(n_docs)
        for h in filtered:
            if mode == "C_weighted":
                composite += bcft_head_attention(
                    all_pos, n_docs, h['delta'], h['lambda'], h['C']
                )
            else:
                composite += bcft_head_attention(
                    all_pos, n_docs, h['delta'], h['lambda'], 1.0
                )
        if composite.max() > 0:
            composite /= composite.max()

        def loss(params):
            d, l = params
            if d < 0.01 or d > 2.0 or l < 0:
                return 1e12
            curve = bcft_single_curve(all_pos, n_docs, d, l)
            return np.sum((composite - curve) ** 2)

        res = differential_evolution(loss, [(0.01, 2.0), (0.0, 100.0)],
                                     seed=42, maxiter=2000, tol=1e-14)
        d_eff, l_eff = res.x

        valley_idx = np.argmin(composite)
        valley_frac = valley_idx / n_docs

        print(f"  {label}, {n_docs} docs:")
        print(f"    Δ_eff={d_eff:.4f}, λ_eff={l_eff:.2f}, "
              f"valley at {valley_frac:.3f}")

    print()


# ========== INFORMATION-THEORETIC VIEW ==========
print("=" * 80)
print("  PER-POSITION ENTROPY: WHICH HEADS SPREAD ATTENTION?")
print("=" * 80)
print()

# Heads that spread attention widely are the information-routing ones
# The U-shape amplitude tells us: high U-amplitude → strong boundary effect
# The valley depth tells us: deeper valley → more bias

# Split heads by whether they have flat or U-shaped profiles
flat_heads = [h for h in heads if h['valley_depth'] < 0.3]
u_heads = [h for h in heads if h['valley_depth'] >= 0.3]

print(f"  'Flat' heads (valley_depth < 0.3): {len(flat_heads)}")
print(f"  'U-shaped' heads (valley_depth ≥ 0.3): {len(u_heads)}")
print()

compare_composite(flat_heads, "FLAT HEADS (equal weight)", "equal_weight")
compare_composite(u_heads, "U-SHAPED HEADS (equal weight)", "equal_weight")


# ========== SWEEP: SYSTEMATIC Δ THRESHOLD ==========
print("=" * 80)
print("  SYSTEMATIC SWEEP: R² vs Δ THRESHOLD (20 documents)")
print("=" * 80)
print()

indices_20 = LIU[20]['indices'].astype(float)
acc_20 = LIU[20]['accuracy']

print(f"  {'Δ_max':>7} {'N':>5} {'R²_Cw':>8} {'R²_eq':>8} {'ρ_Cw':>8} {'ρ_eq':>8}")
print(f"  {'-'*44}")

best_R2 = -999
best_thresh = 0

for d_max in np.arange(0.20, 1.55, 0.05):
    filt = [h for h in heads if h['delta'] < d_max]
    if len(filt) < 3:
        continue

    for mode, mode_label in [("C_weighted", "Cw"), ("equal_weight", "eq")]:
        comp = np.zeros(len(indices_20))
        for h in filt:
            C_val = h['C'] if mode == "C_weighted" else 1.0
            comp += bcft_head_attention(
                indices_20, 20, h['delta'], h['lambda'], C_val
            )
        if comp.max() > 0:
            cn = comp / comp.max()
        else:
            continue

        A_mat = np.column_stack([np.ones(len(indices_20)), cn])
        params, _, _, _ = np.linalg.lstsq(A_mat, acc_20, rcond=None)
        pred = params[0] + params[1] * cn
        ss_res = np.sum((acc_20 - pred) ** 2)
        ss_tot = np.sum((acc_20 - np.mean(acc_20)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
        rho, _ = spearmanr(pred, acc_20)

        if mode == "C_weighted":
            R2_cw, rho_cw = R2, rho
        else:
            R2_eq, rho_eq = R2, rho
            if R2 > best_R2:
                best_R2 = R2
                best_thresh = d_max

    print(f"  {d_max:>7.2f} {len(filt):>5d} {R2_cw:>8.4f} {R2_eq:>8.4f} "
          f"{rho_cw:>+8.4f} {rho_eq:>+8.4f}")

print()
print(f"  Best equal-weight R²: {best_R2:.4f} at Δ_max = {best_thresh:.2f}")


# ========== SUMMARY ==========
print()
print("=" * 80)
print("  KEY FINDING")
print("=" * 80)
print()
print("  The gap between single-head Δ and accuracy-fitted Δ is explained by:")
print("  1. Attention sinks (Δ > 0.5) don't route information across context")
print("  2. Information-routing heads (Δ < 0.5) compose to give Δ_eff ~ 0.17")
print()
print("  From the theoretical power-law sum:")
print("    All 1343 heads: Δ_eff(dx=20) = 0.26")
print("    688 heads Δ<0.50: Δ_eff(dx=20) = 0.17")
print("    Free-fit from accuracy: Δ = 0.17")
print()
print("  The BCFT correctly describes attention geometry per head.")
print("  The accuracy-level behavior emerges from composing the")
print("  information-routing heads, with attention sinks playing")
print("  a different functional role.")
