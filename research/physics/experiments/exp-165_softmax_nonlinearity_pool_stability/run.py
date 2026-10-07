"""
exp-165: Mean-field softmax nonlinearity as pool-stability mechanism

Pre-registration: attention-geometry 90753e5 (committed and pushed before this script)

Question: Does applying softmax to the mean-field score field
  S_approx(i, j) = qbar[h, i] · kbar[h, j] / √d (causal)
recover empirical census σ values and preserve pool-stable vs pool-sensitive ordering?

Data: kbar_qbar.npz from exp-164 — no new forward passes.

Protocol matches pre-registration exactly.
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

# ── Constants (same as exp-163/164) ──────────────────────────────────────────
D_HEAD = 64
SCALE = math.sqrt(D_HEAD)

LAG_MIN = 8
LAG_MAX = 256
LAG_BINS = np.arange(LAG_MIN, LAG_MAX + 1)

POOLS = {
    "B": (256, 512),
    "C": (512, 768),
    "D": (768, 1024),
}
POOL_STRIDE = 2  # stride-2 sampling for speed

STRUCTURAL_HEADS = ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]
POOL_STABLE    = {"L3H4", "L7H11"}
POOL_SENSITIVE = {"L2H1", "L5H0", "L10H8"}

NPZ_PATH = Path(__file__).parent.parent / "exp-164_l3h4_bilinear_decomposition" / "kbar_qbar.npz"

# Empirical σ from exp-163 H2 (pre-registered reference)
EMPIRICAL_SIGMA = {
    "L2H1":  {"B": 0.2707, "C": 0.3172, "D": 0.3224},
    "L3H4":  {"B": 0.2954, "C": 0.3096, "D": 0.3093},
    "L5H0":  {"B": 0.2222, "C": 0.2532, "D": 0.2779},
    "L7H11": {"B": 0.2100, "C": 0.2113, "D": 0.2277},
    "L10H8": {"B": 0.2822, "C": 0.2886, "D": 0.3516},
}
EMPIRICAL_DELTA_SIGMA = {h: EMPIRICAL_SIGMA[h]["D"] - EMPIRICAL_SIGMA[h]["B"]
                         for h in STRUCTURAL_HEADS}


# ── Core computation ──────────────────────────────────────────────────────────

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def compute_meanfield_profile(kbar, qbar, pool_lo, pool_hi, stride=POOL_STRIDE):
    """
    Compute the mean-field pooled attention lag profile for one head and one pool.

    For each query position i in [pool_lo, pool_hi) (stride sampling):
      - Score vector: s[j] = qbar[i] · kbar[j] / √d, for j = 0..i
      - Apply causal softmax → a (attention weights, shape i+1)
      - Accumulate a[i-l] for l ∈ [LAG_MIN, LAG_MAX]

    Returns:
      profile: array (n_lags,) — mean attention weight at each lag
      n_valid: array (n_lags,) — number of (query, lag) pairs contributing
    """
    n_lags = len(LAG_BINS)
    acc = np.zeros(n_lags)
    cnt = np.zeros(n_lags, dtype=int)

    # Pre-compute full score matrix: (n_pool_positions, 1024)
    p_list = list(range(pool_lo, pool_hi, stride))
    Q = qbar[p_list]           # (n, d)
    S_full = (Q @ kbar.T) / SCALE  # (n, 1024)

    for idx, p_i in enumerate(p_list):
        s_row = S_full[idx, :p_i + 1]   # causal: keys 0..p_i
        a = softmax(s_row)               # (p_i+1,)

        # Valid lags: l ∈ [LAG_MIN, min(LAG_MAX, p_i)]
        l_max_valid = min(LAG_MAX, p_i)
        if l_max_valid < LAG_MIN:
            continue
        l_range = np.arange(LAG_MIN, l_max_valid + 1)   # valid lags
        j_range = p_i - l_range                          # key positions
        l_idx   = l_range - LAG_MIN                      # 0-indexed into LAG_BINS

        acc[l_idx] += a[j_range]
        cnt[l_idx] += 1

    # Pooled profile: mean attention at each lag
    valid_mask = cnt > 0
    profile = np.where(valid_mask, acc / np.maximum(cnt, 1), np.nan)
    return profile, cnt


def fit_log_log_slope(profile):
    """
    OLS slope of log(ā(l)) vs log(l) for l = LAG_MIN..LAG_MAX.
    Returns slope (σ) and R².
    """
    mask = (profile > 0) & np.isfinite(profile)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    log_l = np.log(LAG_BINS[mask])
    log_a = np.log(profile[mask])
    slope, intercept, r, p, se = stats.linregress(log_l, log_a)
    return float(-slope), float(r**2)   # σ = -slope in log-log (attention decays with lag)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("exp-165: Mean-field softmax nonlinearity as pool-stability mechanism")
    print("=" * 70)
    print(f"\nLoading kbar_qbar.npz from {NPZ_PATH}")
    data = np.load(NPZ_PATH)

    results_by_head = {}
    all_predicted = []   # for Spearman ρ
    all_empirical = []

    for head in STRUCTURAL_HEADS:
        kbar = data[f"kbar_{head}"]  # (1024, 64)
        qbar = data[f"qbar_{head}"]  # (1024, 64)
        assert kbar.shape == (1024, D_HEAD)
        assert qbar.shape == (1024, D_HEAD)

        print(f"\n{'─'*50}")
        print(f"Head {head} (pool_stable={head in POOL_STABLE})")

        sigma_pred = {}
        r2_pred    = {}

        for pname, (p_lo, p_hi) in POOLS.items():
            profile, cnt = compute_meanfield_profile(kbar, qbar, p_lo, p_hi)
            sigma, r2 = fit_log_log_slope(profile)
            sigma_pred[pname] = sigma
            r2_pred[pname]    = r2
            sigma_emp = EMPIRICAL_SIGMA[head][pname]
            diff = abs(sigma - sigma_emp) if not math.isnan(sigma) else float("nan")
            print(f"  Pool {pname}: σ_pred={sigma:.4f} (R²={r2:.3f}) | "
                  f"σ_emp={sigma_emp:.4f} | |diff|={diff:.4f}")
            all_predicted.append(sigma)
            all_empirical.append(sigma_emp)

        delta_pred = sigma_pred["D"] - sigma_pred["B"]
        delta_emp  = EMPIRICAL_DELTA_SIGMA[head]
        print(f"  Δσ_pred={delta_pred:.4f}  Δσ_emp={delta_emp:.4f}")

        results_by_head[head] = {
            "sigma_pred": sigma_pred,
            "r2_pred":    r2_pred,
            "delta_sigma_pred": delta_pred,
            "delta_sigma_emp":  delta_emp,
            "pool_stable": head in POOL_STABLE,
        }

    # ── Hypothesis evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)

    # H1: pool-stable ordering preserved
    delta_stable    = [results_by_head[h]["delta_sigma_pred"] for h in POOL_STABLE]
    delta_sensitive = [results_by_head[h]["delta_sigma_pred"] for h in POOL_SENSITIVE]
    max_stable  = max(delta_stable)
    min_sens    = min(delta_sensitive)
    h1_pass = max_stable < min_sens
    print(f"\nH1 (Ordering): max(Δσ_pred stable)={max_stable:.4f} vs "
          f"min(Δσ_pred sensitive)={min_sens:.4f} → {'CONFIRMED' if h1_pass else 'FALSIFIED (K1 fires)'}")

    # H2: quantitative accuracy (≥10 of 15 within 0.05)
    within_pairs = sum(1 for p, e in zip(all_predicted, all_empirical)
                       if not math.isnan(p) and abs(p - e) <= 0.05)
    h2_pass = within_pairs >= 10
    print(f"H2 (Accuracy): {within_pairs}/15 pairs within 0.05 → "
          f"{'CONFIRMED' if h2_pass else 'FALSIFIED (K2 fires)'}")

    # H3: Spearman ρ
    valid = [(p, e) for p, e in zip(all_predicted, all_empirical) if not math.isnan(p)]
    if valid:
        pred_v, emp_v = zip(*valid)
        rho, pval = stats.spearmanr(pred_v, emp_v)
    else:
        rho, pval = float("nan"), float("nan")
    h3_pass = rho > 0.7
    print(f"H3 (Spearman ρ): ρ={rho:.3f} (p={pval:.4f}) → "
          f"{'CONFIRMED' if h3_pass else 'FALSIFIED (K3 fires)'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\nSUMMARY TABLE")
    print(f"{'Head':<10} {'Pool_stable':<12} {'Δσ_pred':>10} {'Δσ_emp':>10} {'Match':>8}")
    for head in STRUCTURAL_HEADS:
        r = results_by_head[head]
        print(f"{head:<10} {str(r['pool_stable']):<12} "
              f"{r['delta_sigma_pred']:>10.4f} {r['delta_sigma_emp']:>10.4f} "
              f"{'✓' if (r['pool_stable'] and r['delta_sigma_pred'] < min_sens) or (not r['pool_stable'] and r['delta_sigma_pred'] > max_stable) else '✗':>8}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    results = {
        "experiment": "exp-165",
        "date": "2026-10-07",
        "prereg_commit": "90753e5",
        "prereg_evidence": "git-attested — em dash — pre-registered commit 90753e5 pushed to attention-geometry before run.py was written",
        "analysis_only": True,
        "data_source": "exp-164_l3h4_bilinear_decomposition/kbar_qbar.npz",
        "protocol": {
            "d_head": D_HEAD,
            "lag_min": LAG_MIN,
            "lag_max": LAG_MAX,
            "pools": {k: list(v) for k, v in POOLS.items()},
            "pool_stride": POOL_STRIDE,
        },
        "results_by_head": results_by_head,
        "hypotheses": {
            "H1_ordering": {
                "result": "CONFIRMED" if h1_pass else "FALSIFIED",
                "max_delta_stable": max_stable,
                "min_delta_sensitive": min_sens,
                "K1_fires": not h1_pass,
            },
            "H2_accuracy": {
                "result": "CONFIRMED" if h2_pass else "FALSIFIED",
                "pairs_within_0.05": within_pairs,
                "total_pairs": 15,
                "K2_fires": not h2_pass,
            },
            "H3_spearman": {
                "result": "CONFIRMED" if h3_pass else "FALSIFIED",
                "rho": rho,
                "pval": pval,
                "K3_fires": not h3_pass,
            },
        },
        "overall_verdict": (
            "CONFIRMED" if (h1_pass and h2_pass and h3_pass) else
            "PARTIAL" if (h1_pass or h3_pass) else
            "FALSIFIED"
        ),
    }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Overall verdict: {results['overall_verdict']}")

    return results


if __name__ == "__main__":
    main()
