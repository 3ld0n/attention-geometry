"""
exp-116 — Mixed score profiles: isolating q-side and k-side contributions to MF overshoot

Analysis-only. Loads saved arrays from exp-112 (scores_gpt2.npz) and
exp-113 (meanfield_gpt2.npz). No new forward passes.

Run from the repository root:
    python research/physics/experiments/exp-116_mixed_score_profiles/analyze_mixed.py
"""

import json
import math
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent.parent
EXP_DIR = Path(__file__).parent
EXP112 = REPO / "research/physics/experiments/exp-112_score_drift_decomposition"
EXP113 = REPO / "research/physics/experiments/exp-113_mean_field_reduction"

SCORES_NPZ = EXP112 / "scores_gpt2.npz"
MF_NPZ = EXP113 / "meanfield_gpt2.npz"
MF_RESULTS = EXP113 / "results_gpt2.json"

SEQ_LEN = 512
D_HEAD = 64
DEEP_LO = 256
LAG_LO = 8
LAG_HI = 256
LAGS = np.arange(LAG_LO, LAG_HI + 1)
QUERY_POOL = np.arange(DEEP_LO, SEQ_LEN)

STRUCTURAL = [("L2", 2, 1), ("L3", 3, 4), ("L5", 5, 0), ("L7", 7, 11), ("L10", 10, 8)]
SEMANTIC = [
    ("L4", 4, 10), ("L7", 7, 1), ("L8", 8, 2), ("L9", 9, 4), ("L9", 9, 6),
    ("L10", 10, 1), ("L10", 10, 2), ("L10", 10, 10),
    ("L11", 11, 0), ("L11", 11, 1), ("L11", 11, 2), ("L11", 11, 4),
    ("L11", 11, 5), ("L11", 11, 6), ("L11", 11, 7), ("L11", 11, 9),
]


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean()
    return float(np.dot(xm, y) / (np.dot(xm, xm) + 1e-30))


def compute_profiles(q_arr, k_arr, lags, query_pool):
    """
    Compute pooled lag profile: mean_i q_arr[i] · k_arr[i-dx] / sqrt(D_HEAD).
    q_arr: (seq_len, d_head), k_arr: (seq_len, d_head)
    Returns profile array (len(lags),).
    """
    profile = np.zeros(len(lags))
    for didx, dx in enumerate(lags):
        keys = query_pool - dx
        valid = keys >= 0
        if valid.sum() == 0:
            profile[didx] = np.nan
            continue
        q_v = q_arr[query_pool[valid]]   # (n_valid, d)
        k_v = k_arr[keys[valid]]          # (n_valid, d)
        profile[didx] = np.mean(np.sum(q_v * k_v, axis=-1)) / math.sqrt(D_HEAD)
    return profile


def analyze_pair(label, layer_str, head_idx, cond, scores_npz, mf_npz, sigma_pos_exp112, sigma_mf):
    layer_idx = int(layer_str[1:])

    # Load arrays
    qbar_h = scores_npz[f"qbar_{cond}_{layer_str}"][head_idx]   # (512, 64)
    kbar_h = scores_npz[f"kbar_{cond}_{layer_str}"][head_idx]   # (512, 64)
    q_mf   = mf_npz[f"q_mf_{cond}_{layer_str}H{head_idx}"]      # (512, 64)
    k_mf   = mf_npz[f"k_mf_{cond}_{layer_str}H{head_idx}"]      # (512, 64)

    # S_pos stored
    S_pos_stored = scores_npz[f"S_pos_{cond}"][layer_idx, head_idx, :]  # (249,)

    # Compute the four profiles
    S_mf_recon    = compute_profiles(q_mf,    k_mf,    LAGS, QUERY_POOL)
    S_mixed_k     = compute_profiles(q_mf,    kbar_h,  LAGS, QUERY_POOL)
    S_mixed_q     = compute_profiles(qbar_h,  k_mf,    LAGS, QUERY_POOL)
    S_pos_recon   = compute_profiles(qbar_h,  kbar_h,  LAGS, QUERY_POOL)

    # K1: verify S_pos_recon matches stored
    k1_max_diff = float(np.nanmax(np.abs(S_pos_recon - S_pos_stored)))
    k1_pass = k1_max_diff <= 1e-3

    # Verify the additive identity: S_residual = S_pos - S_mixed_k - S_mixed_q + S_mf
    S_residual = S_pos_stored - S_mixed_k - S_mixed_q + S_mf_recon

    # Slopes (negative OLS slope = sigma)
    log_dx = np.log(LAGS.astype(float))
    valid = ~np.isnan(S_mf_recon) & ~np.isnan(S_mixed_k) & ~np.isnan(S_mixed_q) & ~np.isnan(S_pos_stored)

    # Convention matching exp-112/113: sigma = -(OLS slope of S(dx) vs log(dx)) — LINEAR, not log-log.
    # This is the convention that exp-113 uses for sigma_mf and exp-112 for sigma_pos.
    # Under this convention, a uniform attenuation S_pos = f * S_mf (f < 1) gives
    # sigma_pos = f * sigma_mf < sigma_mf, explaining the "overshoot" as a linear-OLS effect.
    def linear_sigma(profile):
        lv = log_dx[valid]
        pv = profile[valid]
        return -ols_slope(lv, pv)  # -(slope of S vs log dx)

    sig_mf_recon   = linear_sigma(S_mf_recon)
    sig_mixed_k    = linear_sigma(S_mixed_k)
    sig_mixed_q    = linear_sigma(S_mixed_q)
    sig_pos_recon  = linear_sigma(S_pos_recon)
    sig_residual   = linear_sigma(S_residual)

    # Overshoot from exp-113
    overshoot = sigma_mf - sigma_pos_exp112

    # Correction decomposition (slope level)
    corr_k = sigma_mf - sig_mixed_k       # positive if S_mixed_k is shallower than S_mf
    corr_q = sigma_mf - sig_mixed_q       # positive if S_mixed_q is shallower
    corr_sum = corr_k + corr_q
    fraction_explained = corr_sum / (overshoot + 1e-30)

    # S_residual magnitude relative to S_mf
    res_rel = float(np.nanmean(np.abs(S_residual)) / (np.nanmean(np.abs(S_mf_recon)) + 1e-30))

    return {
        "label": label, "layer_idx": layer_idx, "head_idx": head_idx, "condition": cond,
        "k1_max_diff": k1_max_diff, "k1_pass": k1_pass,
        "sigma_mf_exp113": sigma_mf, "sigma_pos_exp112": sigma_pos_exp112,
        "overshoot": overshoot,
        "sig_mf_recon": sig_mf_recon,
        "sig_mixed_k": sig_mixed_k,
        "sig_mixed_q": sig_mixed_q,
        "sig_pos_recon": sig_pos_recon,
        "corr_k": corr_k, "corr_q": corr_q,
        "corr_sum": corr_sum,
        "fraction_explained": fraction_explained,
        "residual_rel_mag": res_rel,
        # P1: sig_mixed_k < sig_mf
        "p1_confirm": sig_mixed_k < sigma_mf,
        # P2: fraction >= 0.5 (both corrections together)
        "p2_confirm": fraction_explained >= 0.5,
        # Which side carries more?
        "k_side_dominant": abs(corr_k) > abs(corr_q),
    }


def main():
    print("Loading saved arrays...")
    scores_npz = np.load(SCORES_NPZ)
    mf_npz = np.load(MF_NPZ)
    mf_results = json.load(open(MF_RESULTS))

    def get_sigmas(cond_key, label):
        for section in ["registered", "exploratory"]:
            d = mf_results["conditions"].get(cond_key, {}).get(section, {})
            if label in d:
                return d[label]["sigma_pos_exp112"], d[label]["sigma_mf"]
        return None, None

    structural_results = []
    print("\n--- Structural heads (random tokens) ---")
    for (ls, li, hi) in STRUCTURAL:
        label = f"L{li}H{hi}"
        sp, sm = get_sigmas("random", label)
        if sp is None:
            print(f"  {label}: missing"); continue
        res = analyze_pair(label, ls, hi, "random", scores_npz, mf_npz, sp, sm)
        structural_results.append(res)
        print(f"  {label}: K1={'PASS' if res['k1_pass'] else 'FAIL'}  "
              f"sig_mixed_k={res['sig_mixed_k']:.4f}  sig_mixed_q={res['sig_mixed_q']:.4f}  "
              f"sig_mf={res['sig_mf_recon']:.4f}  sig_pos={res['sig_pos_recon']:.4f}  "
              f"overshoot={res['overshoot']:.3f}  corr_k={res['corr_k']:.4f}  corr_q={res['corr_q']:.4f}  "
              f"frac={res['fraction_explained']:.3f}  res_rel={res['residual_rel_mag']:.4f}")

    semantic_results = []
    print("\n--- Semantic heads (WikiText) ---")
    for (ls, li, hi) in SEMANTIC:
        label = f"L{li}H{hi}"
        sp, sm = get_sigmas("wikitext", label)
        if sp is None:
            sp, sm = get_sigmas("random", label)
        if sp is None:
            print(f"  {label}: missing"); continue
        res = analyze_pair(label, ls, hi, "wikitext", scores_npz, mf_npz, sp, sm)
        semantic_results.append(res)
        print(f"  {label}: K1={'PASS' if res['k1_pass'] else 'FAIL'}  "
              f"sig_mk={res['sig_mixed_k']:.4f}  sig_mq={res['sig_mixed_q']:.4f}  "
              f"frac={res['fraction_explained']:.3f}")

    # Verdicts
    print("\n=== VERDICTS ===")
    k1_fails = [r["label"] for r in structural_results if not r["k1_pass"]]
    print(f"K1: {'PASS' if not k1_fails else f'FAIL {k1_fails}'}")

    p1_confirm = [r for r in structural_results if r["p1_confirm"]]
    p1_dead = [r for r in structural_results if not r["p1_confirm"]]
    p1_v = "CONFIRMED" if len(p1_confirm) >= 4 else ("DEAD" if len(p1_dead) >= 3 else "AMBIGUOUS")
    print(f"P1 (sig_mixed_k < sig_mf): {len(p1_confirm)}/5 → {p1_v}")

    p2_confirm = [r for r in structural_results if r["p2_confirm"]]
    p2_v = "CONFIRMED" if len(p2_confirm) >= 4 else "DEAD" if (5 - len(p2_confirm)) >= 3 else "AMBIGUOUS"
    print(f"P2 (fraction >= 0.5): {len(p2_confirm)}/5 → {p2_v}")

    p3_confirm = [r for r in semantic_results if r["p1_confirm"]]
    p3_v = "CONFIRMED" if len(p3_confirm) >= 10 else ("DEAD" if (len(semantic_results) - len(p3_confirm)) >= 10 else "AMBIGUOUS")
    print(f"P3 (semantic sig_mixed_k < sig_mf): {len(p3_confirm)}/{len(semantic_results)} → {p3_v}")

    print("\n--- Side dominance ---")
    k_dom = [r for r in structural_results if r["k_side_dominant"]]
    print(f"  Key-side dominant (|corr_k| > |corr_q|): {len(k_dom)}/5")
    for r in structural_results:
        print(f"    {r['label']}: corr_k={r['corr_k']:.4f}  corr_q={r['corr_q']:.4f}  dominant={'k' if r['k_side_dominant'] else 'q'}")

    output = {
        "protocol": {"seq_len": SEQ_LEN, "d_head": D_HEAD, "deep_lo": DEEP_LO,
                     "lag_lo": LAG_LO, "lag_hi": LAG_HI},
        "gates": {"K1": {"fail_labels": k1_fails, "pass": not k1_fails}},
        "verdicts": {
            "P1": {"n_confirm": len(p1_confirm), "n": 5, "verdict": p1_v,
                   "confirm_labels": [r["label"] for r in p1_confirm]},
            "P2": {"n_confirm": len(p2_confirm), "n": 5, "verdict": p2_v},
            "P3": {"n_confirm": len(p3_confirm), "n": len(semantic_results), "verdict": p3_v},
        },
        "structural_per_head": structural_results,
        "semantic_per_head": semantic_results,
    }
    out_path = EXP_DIR / "results_gpt2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save mixed profiles
    npz_data = {}
    for r in structural_results + semantic_results:
        ls = f"L{r['layer_idx']}"
        hi = r['head_idx']
        cond = r['condition']
        label = r['label']
        qbar_h = scores_npz[f"qbar_{cond}_{ls}"][hi]
        kbar_h = scores_npz[f"kbar_{cond}_{ls}"][hi]
        q_mf = mf_npz[f"q_mf_{cond}_{ls}H{hi}"]
        k_mf = mf_npz[f"k_mf_{cond}_{ls}H{hi}"]
        npz_data[f"S_mixed_k_{cond}_{label}"] = compute_profiles(q_mf, kbar_h, LAGS, QUERY_POOL).astype(np.float32)
        npz_data[f"S_mixed_q_{cond}_{label}"] = compute_profiles(qbar_h, k_mf, LAGS, QUERY_POOL).astype(np.float32)
        npz_data[f"S_mf_{cond}_{label}"] = compute_profiles(q_mf, k_mf, LAGS, QUERY_POOL).astype(np.float32)
    np.savez(EXP_DIR / "mixed_profiles_gpt2.npz", **npz_data)
    print(f"Mixed profiles saved to {EXP_DIR / 'mixed_profiles_gpt2.npz'}")


if __name__ == "__main__":
    main()
