"""
exp-127: Eigenvalue Distribution of Key Matrices — Δ-window vs Control Heads

Analysis-only experiment: loads eigenvalues_mean from exp-126 results.json
and tests the pre-registered predictions about λ₁/Σλ and supra-MP fraction.

Pre-registration: attention-geometry a3ce213 (notes.md committed 2026-08-30,
before this script was written).
"""

import json
import numpy as np
from scipy.stats import mannwhitneyu

# ── Load exp-126 results ──────────────────────────────────────────────────────

EXP126_RESULTS = (
    "research/physics/experiments/exp-126_key_covariance_structure/results.json"
)

with open(EXP126_RESULTS) as f:
    r126 = json.load(f)

# Head population definitions (from exp-126)
WIKI_HEADS = [tuple(h) for h in r126["wiki_heads"]]
STRUCTURAL_HEADS = [tuple(h) for h in r126["structural_heads"]]
CONTROL_HEADS = [tuple(h) for h in r126["control_heads"]]


# ── Per-head eigenvalue metrics ───────────────────────────────────────────────

def compute_eig_metrics(eigenvalues_mean):
    """
    Given mean eigenvalues (length 128) for G = K K^T / d_k with K ∈ R^{128×64}:

    - λ₁/Σλ: max eigenvalue / sum of all eigenvalues.
    - MP upper edge: σ² × (1 + √(p/n))² where p=64, n=128, σ² = Σλ/64.
    - supra_MP_fraction: fraction of eigenvalues strictly above the MP upper edge.
    """
    eigs = np.array(eigenvalues_mean)

    # Clip tiny negatives from numerical noise
    eigs = np.clip(eigs, 0, None)

    lam_sum = eigs.sum()
    lam_max = eigs.max()

    # λ₁/Σλ
    top_share = lam_max / lam_sum if lam_sum > 1e-12 else np.nan

    # Marchenko-Pastur bulk upper edge
    # K is n=128 rows × p=64 columns → ratio c = p/n = 0.5
    # MP upper edge = σ² (1 + √c)²
    # σ² estimated as Σλ / p (average over the p non-trivial eigenvalues)
    p = 64
    sigma2 = lam_sum / p
    c = p / 128.0  # = 0.5
    lam_plus = sigma2 * (1.0 + np.sqrt(c)) ** 2

    # Fraction strictly above MP edge
    supra_mp = float(np.sum(eigs > lam_plus)) / len(eigs)

    return {
        "top_share": float(top_share),
        "lam_plus": float(lam_plus),
        "sigma2": float(sigma2),
        "supra_mp_fraction": float(supra_mp),
        "n_supra_mp": int(np.sum(eigs > lam_plus)),
        "lam_max": float(lam_max),
        "lam_sum": float(lam_sum),
    }


def get_metrics_for_population(results_dict, heads):
    metrics = {}
    for l, h in heads:
        key = f"L{l}H{h}"
        eigs = results_dict[key]["eigenvalues_mean"]
        metrics[key] = compute_eig_metrics(eigs)
    return metrics


# Compute metrics for each population
wiki_metrics = get_metrics_for_population(r126["wiki_results"], WIKI_HEADS)
structural_metrics = get_metrics_for_population(r126["structural_results"], STRUCTURAL_HEADS)
control_metrics = get_metrics_for_population(r126["control_results"], CONTROL_HEADS)


# ── Aggregate ─────────────────────────────────────────────────────────────────

wiki_top_share = [m["top_share"] for m in wiki_metrics.values()]
structural_top_share = [m["top_share"] for m in structural_metrics.values()]
control_top_share = [m["top_share"] for m in control_metrics.values()]

wiki_supra_mp = [m["supra_mp_fraction"] for m in wiki_metrics.values()]
structural_supra_mp = [m["supra_mp_fraction"] for m in structural_metrics.values()]
control_supra_mp = [m["supra_mp_fraction"] for m in control_metrics.values()]


# ── Statistical tests ─────────────────────────────────────────────────────────

# P1: Δ-window λ₁/Σλ < control λ₁/Σλ (one-sided: window < control)
mw_top_share = mannwhitneyu(wiki_top_share, control_top_share, alternative="less")
effect_top_share = np.median(control_top_share) - np.median(wiki_top_share)

# P2: Δ-window supra-MP fraction > control (one-sided: window > control)
mw_supra_mp = mannwhitneyu(wiki_supra_mp, control_supra_mp, alternative="greater")
effect_supra_mp = np.median(wiki_supra_mp) - np.median(control_supra_mp)

# P3: Structural λ₁/Σλ > both window and control
mw_struct_vs_wiki = mannwhitneyu(structural_top_share, wiki_top_share, alternative="greater")
mw_struct_vs_ctrl = mannwhitneyu(structural_top_share, control_top_share, alternative="greater")


# ── Verdict ───────────────────────────────────────────────────────────────────

# Kill conditions
k1_fired = effect_top_share < 0.05  # effect too small
k2_fired = mw_top_share.pvalue >= 0.05  # P1 not significant
k3_fired = mw_supra_mp.pvalue >= 0.05   # P2 not significant
k4_fired = mw_struct_vs_wiki.pvalue >= 0.05 and mw_struct_vs_ctrl.pvalue >= 0.05  # P3 structural not directional

# P1 confirmed iff not k1 and not k2
p1_confirmed = not k1_fired and not k2_fired
p1_verdict = "CONFIRMED" if p1_confirmed else "FALSIFIED"

# P2 confirmed iff not k3 and effect meaningful (> 0.005)
p2_confirmed = not k3_fired and effect_supra_mp > 0.005
p2_verdict = "CONFIRMED" if p2_confirmed else "FALSIFIED"

# P3: structural > wiki AND structural > control
p3_confirmed = (mw_struct_vs_wiki.pvalue < 0.05) and (mw_struct_vs_ctrl.pvalue < 0.05)
p3_verdict = "CONFIRMED" if p3_confirmed else "FALSIFIED"

# Overall verdict
n_confirmed = sum([p1_confirmed, p2_confirmed])
if n_confirmed == 2:
    overall_verdict = "confirmed"
elif n_confirmed == 1:
    overall_verdict = "partial"
else:
    overall_verdict = "falsified"


# ── Report ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("exp-127: Eigenvalue Distribution — Δ-window vs Control Heads")
print("=" * 70)
print()
print("── λ₁/Σλ (top eigenvalue share) ────────────────────────────────────")
print(f"  Δ-window  : median={np.median(wiki_top_share):.3f}  mean={np.mean(wiki_top_share):.3f}  "
      f"range=[{min(wiki_top_share):.3f}, {max(wiki_top_share):.3f}]")
print(f"  Structural: median={np.median(structural_top_share):.3f}  mean={np.mean(structural_top_share):.3f}  "
      f"range=[{min(structural_top_share):.3f}, {max(structural_top_share):.3f}]")
print(f"  Control   : median={np.median(control_top_share):.3f}  mean={np.mean(control_top_share):.3f}  "
      f"range=[{min(control_top_share):.3f}, {max(control_top_share):.3f}]")
print()
print(f"  P1 Mann-Whitney (window < control): U={mw_top_share.statistic:.0f}, p={mw_top_share.pvalue:.4f}")
print(f"  Effect (median control − window)  : {effect_top_share:.3f}")
print()
print("── Supra-MP fraction ────────────────────────────────────────────────")
print(f"  Δ-window  : median={np.median(wiki_supra_mp):.4f}  mean={np.mean(wiki_supra_mp):.4f}  "
      f"range=[{min(wiki_supra_mp):.4f}, {max(wiki_supra_mp):.4f}]")
print(f"  Structural: median={np.median(structural_supra_mp):.4f}  mean={np.mean(structural_supra_mp):.4f}  "
      f"range=[{min(structural_supra_mp):.4f}, {max(structural_supra_mp):.4f}]")
print(f"  Control   : median={np.median(control_supra_mp):.4f}  mean={np.mean(control_supra_mp):.4f}  "
      f"range=[{min(control_supra_mp):.4f}, {max(control_supra_mp):.4f}]")
print()
print(f"  P2 Mann-Whitney (window > control): U={mw_supra_mp.statistic:.0f}, p={mw_supra_mp.pvalue:.4f}")
print(f"  Effect (median window − control)  : {effect_supra_mp:.4f}")
print()
print("── P3: Structural head rank-1 dominance ─────────────────────────────")
print(f"  MW struct > wiki : p={mw_struct_vs_wiki.pvalue:.4f}")
print(f"  MW struct > ctrl : p={mw_struct_vs_ctrl.pvalue:.4f}")
print()
print("── Kill conditions ──────────────────────────────────────────────────")
print(f"  K1 (effect < 0.05)  : FIRED={k1_fired}  (effect={effect_top_share:.3f})")
print(f"  K2 (P1 p ≥ 0.05)    : FIRED={k2_fired}  (p={mw_top_share.pvalue:.4f})")
print(f"  K3 (P2 p ≥ 0.05)    : FIRED={k3_fired}  (p={mw_supra_mp.pvalue:.4f})")
print(f"  K4 (struct ≈ ctrl)  : FIRED={k4_fired}")
print()
print("── Prediction verdicts ──────────────────────────────────────────────")
print(f"  P1 (window λ₁/Σλ < control): {p1_verdict}")
print(f"  P2 (window supra-MP > ctrl) : {p2_verdict}")
print(f"  P3 (structural rank-1 high) : {p3_verdict}")
print(f"  OVERALL: {overall_verdict.upper()}")
print()

# Per-head detail
print("── Per-head detail (Δ-window) ───────────────────────────────────────")
for key, m in sorted(wiki_metrics.items()):
    print(f"  {key}: top_share={m['top_share']:.3f}, supra_mp={m['supra_mp_fraction']:.4f}, "
          f"σ²={m['sigma2']:.3f}, λ_+={m['lam_plus']:.3f}")

print()
print("── Per-head detail (Control) ────────────────────────────────────────")
for key, m in sorted(control_metrics.items()):
    print(f"  {key}: top_share={m['top_share']:.3f}, supra_mp={m['supra_mp_fraction']:.4f}, "
          f"σ²={m['sigma2']:.3f}, λ_+={m['lam_plus']:.3f}")

print()
print("── Per-head detail (Structural) ─────────────────────────────────────")
for key, m in sorted(structural_metrics.items()):
    print(f"  {key}: top_share={m['top_share']:.3f}, supra_mp={m['supra_mp_fraction']:.4f}, "
          f"σ²={m['sigma2']:.3f}, λ_+={m['lam_plus']:.3f}")


# ── Save results ──────────────────────────────────────────────────────────────

results = {
    "experiment": "exp-127",
    "date": "2026-08-30",
    "prereg_commit": "attention-geometry a3ce213",
    "source_data": "exp-126 results.json (eigenvalues_mean per head)",
    "analysis_only": True,
    "populations": {
        "wiki_heads": WIKI_HEADS,
        "structural_heads": STRUCTURAL_HEADS,
        "control_heads": CONTROL_HEADS,
    },
    "wiki_metrics": wiki_metrics,
    "structural_metrics": structural_metrics,
    "control_metrics": control_metrics,
    "summary": {
        "wiki_top_share_median": float(np.median(wiki_top_share)),
        "wiki_top_share_mean": float(np.mean(wiki_top_share)),
        "structural_top_share_median": float(np.median(structural_top_share)),
        "structural_top_share_mean": float(np.mean(structural_top_share)),
        "control_top_share_median": float(np.median(control_top_share)),
        "control_top_share_mean": float(np.mean(control_top_share)),
        "effect_top_share": float(effect_top_share),
        "p1_mw_p": float(mw_top_share.pvalue),
        "wiki_supra_mp_median": float(np.median(wiki_supra_mp)),
        "wiki_supra_mp_mean": float(np.mean(wiki_supra_mp)),
        "structural_supra_mp_median": float(np.median(structural_supra_mp)),
        "control_supra_mp_median": float(np.median(control_supra_mp)),
        "effect_supra_mp": float(effect_supra_mp),
        "p2_mw_p": float(mw_supra_mp.pvalue),
        "p3_struct_vs_wiki_p": float(mw_struct_vs_wiki.pvalue),
        "p3_struct_vs_ctrl_p": float(mw_struct_vs_ctrl.pvalue),
        "K1_fired": bool(k1_fired),
        "K2_fired": bool(k2_fired),
        "K3_fired": bool(k3_fired),
        "K4_fired": bool(k4_fired),
        "P1_verdict": p1_verdict,
        "P2_verdict": p2_verdict,
        "P3_verdict": p3_verdict,
        "overall_verdict": overall_verdict,
        "headline": (
            f"Δ-window heads: λ₁/Σλ median={np.median(wiki_top_share):.3f} vs control {np.median(control_top_share):.3f} "
            f"(p={mw_top_share.pvalue:.4f}); supra-MP {np.median(wiki_supra_mp):.4f} vs {np.median(control_supra_mp):.4f} "
            f"(p={mw_supra_mp.pvalue:.4f}). Overall: {overall_verdict.upper()}."
        ),
    },
}

output_path = "research/physics/experiments/exp-127_eigenvalue_distribution/results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
