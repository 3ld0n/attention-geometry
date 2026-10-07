"""
exp-166: Softmax anatomy — background-gap mechanism for L3H4 pool-stability

Pre-registration: attention-geometry 9ea498a (committed before this script;
push blocked by auth failure — to be pushed with Eldon at next opportunity;
local commit separation is the git-stamp evidence).

Question: Why does L3H4's mean-field σ_B (0.600, from exp-165) exceed L2H1's
(0.558) despite L3H4 having a LOWER pool-B kbar slope (1.85 vs 1.99)?

Candidate mechanism: background score suppression.
For a pool-B query at position i, the causal context is j=0..i. The census
window is j=i-256..i-8. The "background" is j=0..i-257. If L3H4's background
scores are lower relative to the census window than L2H1's, the softmax
denominator is smaller → more concentration in the census window → higher σ_B.

Protocol: analysis-only from kbar_qbar.npz (exp-164). No new forward passes.
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────
D_HEAD = 64
SCALE = math.sqrt(D_HEAD)

LAG_MIN = 8
LAG_MAX = 256

POOLS = {
    "B": (256, 512),
    "C": (512, 768),
    "D": (768, 1024),
}
POOL_STRIDE = 2

STRUCTURAL_HEADS = ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]
POOL_STABLE    = {"L3H4", "L7H11"}
POOL_SENSITIVE = {"L2H1", "L5H0", "L10H8"}

NPZ_PATH = (
    Path(__file__).parent.parent
    / "exp-164_l3h4_bilinear_decomposition"
    / "kbar_qbar.npz"
)

# exp-165 predicted per-pool sigma (pre-registered reference)
EXP165_SIGMA_PRED = {
    "L2H1":  {"B": 0.5583, "C": 0.6420, "D": 0.6530},
    "L3H4":  {"B": 0.5999, "C": 0.6282, "D": 0.6255},
    "L5H0":  {"B": 0.4559, "C": 0.5278, "D": 0.5784},
    "L7H11": {"B": 0.4200, "C": 0.4304, "D": 0.4663},
    "L10H8": {"B": 0.5829, "C": 0.6087, "D": 0.7513},
}

EMPIRICAL_SIGMA = {
    "L2H1":  {"B": 0.2707, "C": 0.3172, "D": 0.3224},
    "L3H4":  {"B": 0.2954, "C": 0.3096, "D": 0.3093},
    "L5H0":  {"B": 0.2222, "C": 0.2532, "D": 0.2779},
    "L7H11": {"B": 0.2100, "C": 0.2113, "D": 0.2277},
    "L10H8": {"B": 0.2822, "C": 0.2886, "D": 0.3516},
}

# ── Core analysis ─────────────────────────────────────────────────────────────

def compute_background_gap(kbar, qbar, pool_lo, pool_hi, stride=POOL_STRIDE):
    """
    For each query position i in [pool_lo, pool_hi), compute:
      census_mean(i) = mean(S[j]) for j in [i-LAG_MAX, i-LAG_MIN] ∩ [0, i]
      background_mean(i) = mean(S[j]) for j in [0, i-LAG_MAX-1] ∩ [0, i]
      background_gap(i) = census_mean(i) - background_mean(i)

    Returns:
      mean_gap: float — mean background_gap over valid query positions
      gap_per_query: list of (i, gap, census_mean, background_mean, n_bg, n_win)
    """
    gaps = []
    census_means = []
    background_means = []

    p_list = list(range(pool_lo, pool_hi, stride))
    Q = qbar[p_list]            # (n, d)
    S_full = (Q @ kbar.T) / SCALE  # (n, 1024)

    for idx, p_i in enumerate(p_list):
        s_row = S_full[idx, :p_i + 1]  # causal

        # Census window: j in [max(0, p_i-LAG_MAX), p_i-LAG_MIN]
        win_lo = max(0, p_i - LAG_MAX)
        win_hi = p_i - LAG_MIN
        if win_hi < win_lo:
            continue  # no census window for very short contexts

        census_j = np.arange(win_lo, win_hi + 1)
        census_m = s_row[census_j].mean()

        # Background: j in [0, p_i-LAG_MAX-1]
        bg_hi = p_i - LAG_MAX - 1
        if bg_hi < 0:
            # No background positions (context entirely inside the census window)
            # Skip or record as NaN
            continue

        background_m = s_row[: bg_hi + 1].mean()
        gap = census_m - background_m

        gaps.append(gap)
        census_means.append(census_m)
        background_means.append(background_m)

    if not gaps:
        return float("nan"), float("nan"), float("nan"), []

    return (
        float(np.mean(gaps)),
        float(np.mean(census_means)),
        float(np.mean(background_means)),
        gaps,
    )


def compute_representative_profile(kbar, qbar, query_pos):
    """
    For a single representative query position, compute the full score profile
    and apply softmax. Returns s_vec and attn_vec (arrays of length query_pos+1).
    """
    s = (qbar[query_pos] @ kbar[:query_pos + 1].T) / SCALE  # (query_pos+1,)
    s_shifted = s - s.max()
    e = np.exp(s_shifted)
    attn = e / e.sum()
    return s, attn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("exp-166: Softmax anatomy — background-gap mechanism")
    print("Pre-reg: attention-geometry 9ea498a")
    print("=" * 70)
    print(f"\nLoading: {NPZ_PATH}")
    data = np.load(NPZ_PATH)

    results_by_head = {}

    for head in STRUCTURAL_HEADS:
        kbar = data[f"kbar_{head}"]  # (1024, 64)
        qbar = data[f"qbar_{head}"]  # (1024, 64)

        print(f"\n{'─'*60}")
        print(f"Head {head}  (pool_stable={head in POOL_STABLE})")

        gaps = {}
        census_ms = {}
        bg_ms = {}

        for pname, (p_lo, p_hi) in POOLS.items():
            g, cm, bm, gap_list = compute_background_gap(kbar, qbar, p_lo, p_hi)
            gaps[pname]    = g
            census_ms[pname] = cm
            bg_ms[pname]   = bm
            exp165_s = EXP165_SIGMA_PRED[head][pname]
            emp_s    = EMPIRICAL_SIGMA[head][pname]
            print(
                f"  Pool {pname}: gap={g:+.4f}  "
                f"census_mean={cm:+.4f}  bg_mean={bm:+.4f}  "
                f"σ_pred(165)={exp165_s:.4f}  σ_emp={emp_s:.4f}  "
                f"n_valid={len(gap_list)}"
            )

        delta_gap = gaps["D"] - gaps["B"] if not (math.isnan(gaps["D"]) or math.isnan(gaps["B"])) else float("nan")
        print(f"  Δgap (D-B) = {delta_gap:+.4f}")

        # Representative profiles
        rep_profiles = {}
        for pname, rep_i in [("B", 383), ("D", 895)]:
            s_vec, attn_vec = compute_representative_profile(kbar, qbar, rep_i)
            # Census window stats
            win_lo = max(0, rep_i - LAG_MAX)
            win_hi = rep_i - LAG_MIN
            bg_hi = rep_i - LAG_MAX - 1
            census_mean_rep = float(s_vec[win_lo:win_hi + 1].mean())
            bg_mean_rep = float(s_vec[:bg_hi + 1].mean()) if bg_hi >= 0 else float("nan")
            gap_rep = census_mean_rep - bg_mean_rep if not math.isnan(bg_mean_rep) else float("nan")
            # Slope of S[j] in census window (OLS vs j)
            jj = np.arange(win_lo, win_hi + 1)
            s_win = s_vec[win_lo:win_hi + 1]
            slope_rep, _, r, _, _ = stats.linregress(jj, s_win)
            print(
                f"  Rep query i={rep_i} (pool {pname}): "
                f"s_win_mean={census_mean_rep:+.4f}  s_bg_mean={bg_mean_rep:+.4f}  "
                f"gap={gap_rep:+.4f}  score_slope={slope_rep:.6f}  "
                f"R²={r**2:.3f}"
            )
            rep_profiles[pname] = {
                "query_pos": rep_i,
                "census_mean": census_mean_rep,
                "bg_mean": bg_mean_rep,
                "gap": gap_rep,
                "census_window_score_slope": float(slope_rep),
            }

        results_by_head[head] = {
            "gap_B": gaps["B"],
            "gap_C": gaps["C"],
            "gap_D": gaps["D"],
            "census_mean_B": census_ms["B"],
            "census_mean_D": census_ms["D"],
            "bg_mean_B": bg_ms["B"],
            "bg_mean_D": bg_ms["D"],
            "delta_gap": delta_gap,
            "pool_stable": head in POOL_STABLE,
            "representative_profiles": rep_profiles,
        }

    # ── Hypothesis evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)

    # H1: L3H4's gap_B > L2H1's gap_B
    gap_b_l3h4 = results_by_head["L3H4"]["gap_B"]
    gap_b_l2h1 = results_by_head["L2H1"]["gap_B"]
    h1_pass = gap_b_l3h4 > gap_b_l2h1
    print(f"\nH1 (L3H4 gap_B > L2H1 gap_B):")
    print(f"  L3H4 gap_B = {gap_b_l3h4:+.4f}")
    print(f"  L2H1 gap_B = {gap_b_l2h1:+.4f}")
    print(f"  Result: {'CONFIRMED' if h1_pass else 'FALSIFIED (K1 fires)'}")

    # H2: Δgap ranks pool-stable < pool-sensitive
    delta_gaps_stable    = [results_by_head[h]["delta_gap"] for h in POOL_STABLE]
    delta_gaps_sensitive = [results_by_head[h]["delta_gap"] for h in POOL_SENSITIVE]
    max_stable_dg  = max(delta_gaps_stable)
    min_sens_dg    = min(delta_gaps_sensitive)
    h2_pass = max_stable_dg < min_sens_dg
    print(f"\nH2 (Δgap ranks pool-stable < pool-sensitive):")
    for h in STRUCTURAL_HEADS:
        stable = h in POOL_STABLE
        print(f"  {h}: Δgap={results_by_head[h]['delta_gap']:+.4f}  stable={stable}")
    print(f"  max(Δgap stable)={max_stable_dg:+.4f}  min(Δgap sensitive)={min_sens_dg:+.4f}")
    print(f"  Result: {'CONFIRMED' if h2_pass else 'FALSIFIED (K2 fires)'}")

    # H3: Spearman ρ between gap_B and exp-165 σ_B_pred
    gap_b_vals = [results_by_head[h]["gap_B"] for h in STRUCTURAL_HEADS]
    sigma_b_pred = [EXP165_SIGMA_PRED[h]["B"] for h in STRUCTURAL_HEADS]
    rho_h3, pval_h3 = stats.spearmanr(gap_b_vals, sigma_b_pred)
    h3_pass = rho_h3 > 0.7
    print(f"\nH3 (Spearman ρ between gap_B and σ_B_pred > 0.7):")
    for h, g, s in zip(STRUCTURAL_HEADS, gap_b_vals, sigma_b_pred):
        print(f"  {h}: gap_B={g:+.4f}  σ_B_pred={s:.4f}")
    print(f"  ρ = {rho_h3:.3f} (p={pval_h3:.4f})")
    print(f"  Result: {'CONFIRMED' if h3_pass else 'FALSIFIED (K3 fires)'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print(f"{'Head':<8} {'Stable':^7} {'gap_B':>8} {'gap_D':>8} {'Δgap':>8} "
          f"{'σ_B_pred':>10} {'σ_B_emp':>9}")
    for h in STRUCTURAL_HEADS:
        r = results_by_head[h]
        print(f"{h:<8} {str(r['pool_stable']):^7} "
              f"{r['gap_B']:>+8.4f} {r['gap_D']:>+8.4f} {r['delta_gap']:>+8.4f} "
              f"{EXP165_SIGMA_PRED[h]['B']:>10.4f} {EMPIRICAL_SIGMA[h]['B']:>9.4f}")

    overall = (
        "CONFIRMED" if (h1_pass and h2_pass and h3_pass) else
        "PARTIAL"   if (h1_pass or h2_pass or h3_pass) else
        "FALSIFIED"
    )
    print(f"\nOverall verdict: {overall}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "experiment": "exp-166",
        "date": "2026-10-07",
        "prereg_commit": "9ea498a",
        "prereg_evidence": "git-attested — 9ea498a committed before run.py was written; push pending auth restoration",
        "analysis_only": True,
        "data_source": "exp-164_l3h4_bilinear_decomposition/kbar_qbar.npz",
        "results_by_head": results_by_head,
        "hypotheses": {
            "H1_l3h4_gap_b_exceeds_l2h1": {
                "result": "CONFIRMED" if h1_pass else "FALSIFIED",
                "L3H4_gap_B": gap_b_l3h4,
                "L2H1_gap_B": gap_b_l2h1,
                "K1_fires": not h1_pass,
            },
            "H2_delta_gap_ranks_stable": {
                "result": "CONFIRMED" if h2_pass else "FALSIFIED",
                "max_delta_gap_stable": max_stable_dg,
                "min_delta_gap_sensitive": min_sens_dg,
                "K2_fires": not h2_pass,
            },
            "H3_gap_b_corr_sigma_b": {
                "result": "CONFIRMED" if h3_pass else "FALSIFIED",
                "spearman_rho": float(rho_h3),
                "pval": float(pval_h3),
                "K3_fires": not h3_pass,
            },
        },
        "overall_verdict": overall,
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
