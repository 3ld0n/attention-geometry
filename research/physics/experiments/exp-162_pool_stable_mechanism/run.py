"""exp-162 — Pool-stable head mechanism: absolute-key profile curvature.

Pre-registration: prereg.md in this folder, committed to attention-geometry at
4b01dab before this file existed.

Analysis-only from exp-112's scores_gpt2.npz. No new forward passes.

Ariel — 2026-09-29 evening MDT, Mission Valley.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"
EXP161 = HERE.parent / "exp-161_census_query_pool_test"

PREREG_COMMIT = "4b01dab"

STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
STRUCTURAL_LABELS = ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]

# Pool-sensitivity from exp-161 (Δσ across B/C/D pools)
EXP161_DELTA_SIGMA = {
    "L2H1": 0.0472,   # borderline
    "L3H4": 0.0138,   # stable
    "L5H0": 0.0520,   # sensitive
    "L7H11": 0.0150,  # stable
    "L10H8": 0.0643,  # sensitive
}

D_HEAD = 64


def ols_loglog_positive(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS log-log slope on positive y values only. Returns (sigma, R²)."""
    mask = y > 0
    if mask.sum() < 3:
        return float("nan"), float("nan")
    lx = np.log(x[mask].astype(float))
    ly = np.log(y[mask].astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    pred = X @ c
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    ss_res = float(((ly - pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    return float(-c[1]), float(r2)


def compute_abskey_profile(npz: np.lib.npyio.NpzFile, layer: int, head: int) -> np.ndarray:
    """Compute S_abskey[a] for a in [0, 511] for a given head.

    S_abskey[a] = m_q · dk[a] / sqrt(d_head)
    where m_q = mean(qbar[head], axis=0)
          dk[a] = kbar[head, a] - mean(kbar[head], axis=0)
    """
    key_q = f"qbar_random_L{layer}"
    key_k = f"kbar_random_L{layer}"
    qbar = npz[key_q].astype(np.float64)  # (12, 512, 64)
    kbar = npz[key_k].astype(np.float64)

    q = qbar[head]   # (512, 64)
    k = kbar[head]   # (512, 64)

    m_q = q.mean(axis=0)   # (64,)
    m_k = k.mean(axis=0)   # (64,)
    dk = k - m_k           # (512, 64)

    # S_abskey[a] = m_q · dk[a] / sqrt(d)
    s_abskey = (m_q @ dk.T) / np.sqrt(D_HEAD)   # (512,)
    return s_abskey


def main() -> None:
    print("exp-162 — Pool-stable head mechanism: absolute-key profile curvature")
    print(f"Pre-registration commit: {PREREG_COMMIT}")
    print()

    # Load data
    npz_path = EXP112 / "scores_gpt2.npz"
    print(f"Loading {npz_path}...", flush=True)
    npz = np.load(npz_path)
    positions = np.arange(512)   # absolute key positions 0..511

    results_per_head: dict[str, dict] = {}

    # ── Pre-registered analysis: two-half slope split ──────────────────────
    print("\n=== Pre-registered: two-half slope split ===")
    print(f"{'Head':<8} {'σ_full':<10} {'σ_half1[0-255]':<16} {'σ_half2[256-511]':<18} {'|Δσ_half|':<12} {'Δσ_exp161':<12}")
    print("-" * 80)

    abs_delta_sigma = {}
    for (l, h), label in zip(STRUCTURAL, STRUCTURAL_LABELS):
        s_abskey = compute_abskey_profile(npz, l, h)

        # Full range slope (positions 8..511, positive domain)
        sigma_full, r2_full = ols_loglog_positive(s_abskey[8:], positions[8:])

        # Half 1: positions 8..255
        half1_pos = positions[8:256]
        half1_val = s_abskey[8:256]
        sigma_h1, r2_h1 = ols_loglog_positive(half1_val, half1_pos)

        # Half 2: positions 256..511
        half2_pos = positions[256:512]
        half2_val = s_abskey[256:512]
        sigma_h2, r2_h2 = ols_loglog_positive(half2_val, half2_pos)

        delta_half = abs(sigma_h1 - sigma_h2) if not (np.isnan(sigma_h1) or np.isnan(sigma_h2)) else float("nan")
        abs_delta_sigma[label] = delta_half
        d_exp161 = EXP161_DELTA_SIGMA[label]

        print(f"{label:<8} {sigma_full:<10.4f} {sigma_h1:<16.4f} {sigma_h2:<18.4f} {delta_half:<12.4f} {d_exp161:<12.4f}")

        results_per_head[label] = {
            "sigma_full": float(sigma_full),
            "r2_full": float(r2_full),
            "sigma_half1_0_255": float(sigma_h1),
            "r2_half1": float(r2_h1),
            "sigma_half2_256_511": float(sigma_h2),
            "r2_half2": float(r2_h2),
            "abs_delta_sigma_halves": float(delta_half),
            "delta_sigma_exp161": float(d_exp161),
            "s_abskey_profile": {
                "positions_sampled": [0, 63, 127, 191, 255, 319, 383, 447, 511],
                "values": [float(s_abskey[p]) for p in [0, 63, 127, 191, 255, 319, 383, 447, 511]],
                "zero_crossing_approx": int(np.argmax(s_abskey < 0)) if np.any(s_abskey < 0) else None,
            }
        }

    print()

    # Pool-stable vs pool-sensitive classification
    pool_stable = ["L3H4", "L7H11"]
    pool_sensitive = ["L2H1", "L5H0", "L10H8"]

    mean_delta_stable = np.mean([abs_delta_sigma[h] for h in pool_stable])
    mean_delta_sensitive = np.mean([abs_delta_sigma[h] for h in pool_sensitive])

    print(f"Mean |Δσ_half| pool-stable   {pool_stable}: {mean_delta_stable:.4f}")
    print(f"Mean |Δσ_half| pool-sensitive {pool_sensitive}: {mean_delta_sensitive:.4f}")
    print(f"Ratio (stable/sensitive): {mean_delta_stable / mean_delta_sensitive:.3f}")

    # Spearman correlation with exp-161 pool-sensitivity
    ordered = STRUCTURAL_LABELS
    curvature = [abs_delta_sigma[h] for h in ordered]
    exp161_delta = [EXP161_DELTA_SIGMA[h] for h in ordered]
    rho, pval = spearmanr(curvature, exp161_delta)
    print(f"\nSpearman ρ(|Δσ_half|, Δσ_exp161): {rho:.4f}  (p={pval:.4f})")

    # ── Kill condition verdicts ─────────────────────────────────────────────
    print("\n=== Kill conditions ===")
    k1_fires = mean_delta_stable >= mean_delta_sensitive
    k2_fires = rho <= 0.0
    print(f"K1 (mechanism falsified — stable ≥ sensitive): {'FIRED' if k1_fires else 'did not fire'}")
    print(f"K2 (no Spearman signal — ρ ≤ 0): {'FIRED' if k2_fires else 'did not fire'}")

    h1_confirmed = (not k1_fires) and (rho > 0.5)
    h1_partial = (not k1_fires) and (rho > 0.0)
    if h1_confirmed:
        verdict_str = "H1 CONFIRMED (stable < sensitive AND ρ > 0.5)"
    elif h1_partial:
        verdict_str = "H1 PARTIAL (stable < sensitive but ρ ≤ 0.5)"
    else:
        verdict_str = "H1 NOT CONFIRMED"
    print(f"Verdict: {verdict_str}")

    # ── Additional characterization: full profile shape ─────────────────────
    print("\n=== Profile shape characterization (non-kill) ===")
    print(f"{'Head':<8} {'zero_crossing':<16} {'s[0]':<10} {'s[127]':<10} {'s[255]':<10} {'s[383]':<10} {'s[511]':<10}")
    print("-" * 70)
    for (l, h), label in zip(STRUCTURAL, STRUCTURAL_LABELS):
        s = compute_abskey_profile(npz, l, h)
        zc = results_per_head[label]["s_abskey_profile"]["zero_crossing_approx"]
        # Find first crossing from positive to negative
        crossings = []
        for i in range(len(s) - 1):
            if s[i] > 0 and s[i+1] <= 0:
                crossings.append(i+1)
            elif s[i] < 0 and s[i+1] >= 0:
                crossings.append(-(i+1))  # negative = zero→positive crossing
        first_pos_to_neg = min((c for c in crossings if c > 0), default=None)
        print(f"{label:<8} {str(first_pos_to_neg):<16} {s[0]:<10.4f} {s[127]:<10.4f} {s[255]:<10.4f} {s[383]:<10.4f} {s[511]:<10.4f}")
        results_per_head[label]["zero_crossing_pos_to_neg"] = first_pos_to_neg
        results_per_head[label]["profile_sampled"] = {
            str(p): float(s[p]) for p in [0, 31, 63, 95, 127, 159, 191, 223, 255,
                                           287, 319, 351, 383, 415, 447, 479, 511]
        }

    # ── Post-hoc analysis (labeled) ─────────────────────────────────────────
    # Exp-138 properties vs exp-161 pool-sensitivity Spearman rank correlation.
    # NOT pre-registered; labeled as post-hoc.
    print("\n=== Post-hoc (labeled): exp-138 properties vs pool-sensitivity ===")
    from scipy.stats import spearmanr as sp2

    props_138 = {}
    for (l, h), label in zip(STRUCTURAL, STRUCTURAL_LABELS):
        # Pull from exp-138 results
        s = compute_abskey_profile(npz, l, h)
        props_138[label] = {
            "sigma_abskey": float(
                [0.602, 0.660, 0.556, 0.470, 0.624][STRUCTURAL_LABELS.index(label)]
            ),
            "sigma_relative": float(
                [0.040, 0.050, 0.081, 0.041, 0.024][STRUCTURAL_LABELS.index(label)]
            ),
            "r2_relative": float(
                [0.039, 0.606, 0.900, 0.939, 0.627][STRUCTURAL_LABELS.index(label)]
            ),
            "ratio_abskey_to_full": float(
                [1.071, 1.082, 1.171, 1.096, 1.039][STRUCTURAL_LABELS.index(label)]
            ),
        }

    ordered = STRUCTURAL_LABELS
    for prop_name in ["sigma_abskey", "sigma_relative", "r2_relative", "ratio_abskey_to_full"]:
        vals = [props_138[h][prop_name] for h in ordered]
        exp161_vals = [EXP161_DELTA_SIGMA[h] for h in ordered]
        rho_ph, p_ph = sp2(vals, exp161_vals)
        print(f"  {prop_name:<28}: ρ={rho_ph:+.3f}  p={p_ph:.3f}")

    # Absolute-key profile at position 0 vs pool-sensitivity
    s_at_0 = [float(compute_abskey_profile(npz, l, h)[0]) for (l, h) in STRUCTURAL]
    rho_s0, p_s0 = sp2(s_at_0, [EXP161_DELTA_SIGMA[lab] for lab in STRUCTURAL_LABELS])
    print(f"  {'s_abskey[0]':<28}: ρ={rho_s0:+.3f}  p={p_s0:.3f}")
    print("  (All post-hoc; not pre-registered.)")

    # ── Save results ────────────────────────────────────────────────────────
    output = {
        "exp": "exp-162",
        "prereg_commit": PREREG_COMMIT,
        "prereg_evidence": "git-attested — commit 4b01dab pushed before run.py was written",
        "analysis_only": True,
        "data_source": str(EXP112 / "scores_gpt2.npz"),
        "heads": results_per_head,
        "aggregate": {
            "mean_abs_delta_sigma_pool_stable": float(mean_delta_stable),
            "mean_abs_delta_sigma_pool_sensitive": float(mean_delta_sensitive),
            "ratio_stable_over_sensitive": float(mean_delta_stable / mean_delta_sensitive),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
        },
        "kill_conditions": {
            "K1_fired": bool(k1_fires),
            "K2_fired": bool(k2_fires),
        },
        "verdict": verdict_str,
        "h1_confirmed": bool(h1_confirmed),
        "h1_partial": bool(h1_partial),
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
