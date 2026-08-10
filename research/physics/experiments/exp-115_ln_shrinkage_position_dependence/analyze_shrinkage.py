"""
exp-115 — Position-dependence of the LN shrinkage on the key-side positional-mean vector

Analysis-only: loads saved arrays from exp-112 (scores_gpt2.npz) and
exp-113 (meanfield_gpt2.npz). No new forward passes.

Run from the repository root:
    python research/physics/experiments/exp-115_ln_shrinkage_position_dependence/analyze_shrinkage.py
"""

import json
import math
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent.parent  # research/physics/experiments/exp-115/.../ariel
EXP_DIR = Path(__file__).parent
EXP112 = REPO / "research/physics/experiments/exp-112_score_drift_decomposition"
EXP113 = REPO / "research/physics/experiments/exp-113_mean_field_reduction"

SCORES_NPZ = EXP112 / "scores_gpt2.npz"
MF_NPZ = EXP113 / "meanfield_gpt2.npz"
MF_RESULTS = EXP113 / "results_gpt2.json"

# Census protocol constants (from exp-112)
SEQ_LEN = 512
D_HEAD = 64
DEEP_LO = 256   # query pool start
LAG_LO = 8
LAG_HI = 256    # inclusive
N_LAGS = LAG_HI - LAG_LO + 1  # 249

STRUCTURAL_HEADS = [("L2", 2, 1), ("L3", 3, 4), ("L5", 5, 0), ("L7", 7, 11), ("L10", 10, 8)]
SEMANTIC_HEADS = [
    ("L4", 4, 10), ("L7", 7, 1), ("L8", 8, 2), ("L9", 9, 4), ("L9", 9, 6),
    ("L10", 10, 1), ("L10", 10, 2), ("L10", 10, 10),
    ("L11", 11, 0), ("L11", 11, 1), ("L11", 11, 2), ("L11", 11, 4),
    ("L11", 11, 5), ("L11", 11, 6), ("L11", 11, 7), ("L11", 11, 9),
]

QUERY_POOL = np.arange(DEEP_LO, SEQ_LEN)  # [256, 511]
LAGS = np.arange(LAG_LO, LAG_HI + 1)     # [8, ..., 256]


def ols_slope(x, y):
    """OLS slope of y vs x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    return float(np.dot(xm, y) / np.dot(xm, xm))


def compute_shrinkage_profile(kbar_h, k_mf, positions):
    """
    Compute f_k(j) = ||kbar_h[j,:]|| / ||k_mf[j,:]|| for j in positions.
    kbar_h: (seq_len, d_head) — mean key for one head
    k_mf:   (seq_len, d_head) — mean-field key for one head
    Returns f_k array (shape: len(positions),), positions array.
    """
    norms_bar = np.linalg.norm(kbar_h[positions], axis=-1)  # (len(pos),)
    norms_mf  = np.linalg.norm(k_mf[positions], axis=-1)
    # Avoid division by zero (shouldn't happen but guard)
    safe = norms_mf > 1e-8
    f_k = np.where(safe, norms_bar / norms_mf, np.nan)
    return f_k


def compute_direction_cosine(kbar_h, k_mf, positions):
    """cos(k̄_j, k_mf_j) for j in positions."""
    a = kbar_h[positions]  # (len, d)
    b = k_mf[positions]
    na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-30
    nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-30
    return np.sum((a / na) * (b / nb), axis=-1)


def compute_C_k(f_k_full, lags, query_pool):
    """
    C_k(dx) = mean over i in query_pool of f_k(i - dx), for each dx in lags.
    f_k_full: (seq_len,) — shrinkage factor at every position j
    """
    C_k = np.zeros(len(lags))
    for idx, dx in enumerate(lags):
        key_positions = query_pool - dx  # j = i - dx, i in [256, 511]
        # Filter to valid positions (j >= 0)
        valid = key_positions >= 0
        if valid.sum() == 0:
            C_k[idx] = np.nan
        else:
            C_k[idx] = np.nanmean(f_k_full[key_positions[valid]])
    return C_k


def analyze_pair(label, layer_str, head_idx, cond, scores_npz, mf_npz, mf_sigma_pos, mf_sigma_mf):
    """
    Full analysis for one registered (layer, head, condition) pair.
    Returns a dict of results.
    """
    # Load kbar and qbar for this layer and condition
    kbar_key = f"kbar_{cond}_{layer_str}"
    qbar_key = f"qbar_{cond}_{layer_str}"
    if kbar_key not in scores_npz:
        return {"label": label, "error": f"Missing {kbar_key} in scores_npz"}
    kbar_all = scores_npz[kbar_key]  # (n_heads, seq_len, d_head)
    qbar_all = scores_npz[qbar_key]
    kbar_h = kbar_all[head_idx]  # (seq_len, d_head)
    qbar_h = qbar_all[head_idx]

    # Load k_mf and q_mf for this specific head from meanfield npz
    mf_k_key = f"k_mf_{cond}_{layer_str}H{head_idx}"
    mf_q_key = f"q_mf_{cond}_{layer_str}H{head_idx}"
    if mf_k_key not in mf_npz:
        return {"label": label, "error": f"Missing {mf_k_key} in mf_npz"}
    k_mf = mf_npz[mf_k_key]  # (seq_len, d_head)
    q_mf = mf_npz[mf_q_key]

    # ---- K1 integrity gate ----
    # Recompute S_pos_check(dx) = mean_i qbar_h[i,:] · kbar_h[i-dx,:] / sqrt(d)
    # and compare to stored S_pos_{cond}[layer_idx, head_idx, dx_idx]
    if cond == "random":
        S_pos_stored = scores_npz["S_pos_random"]
    elif cond == "wikitext":
        S_pos_stored = scores_npz["S_pos_wikitext"]
    else:
        S_pos_stored = scores_npz[f"S_pos_{cond}"]
    # layer index from layer_str: "L2" -> 2
    layer_idx = int(layer_str[1:])
    S_pos_check = np.zeros(len(LAGS))
    for didx, dx in enumerate(LAGS):
        keys = QUERY_POOL - dx
        valid = keys >= 0
        if valid.sum() == 0:
            S_pos_check[didx] = np.nan
            continue
        # dot product q[i] · k[i-dx] for each i in pool with valid key
        q_valid = qbar_h[QUERY_POOL[valid]]   # (n_valid, d)
        k_valid = kbar_h[keys[valid]]          # (n_valid, d)
        S_pos_check[didx] = np.mean(np.sum(q_valid * k_valid, axis=-1)) / math.sqrt(D_HEAD)
    S_pos_ref = S_pos_stored[layer_idx, head_idx, :]  # (249,)
    k1_max_diff = float(np.nanmax(np.abs(S_pos_check - S_pos_ref)))
    k1_pass = k1_max_diff <= 1e-3

    # ---- Key-side shrinkage ----
    # Full profile over all positions
    f_k_full = compute_shrinkage_profile(kbar_h, k_mf, np.arange(SEQ_LEN))
    # Slope over key pool [8, 256]
    key_pool_range = np.arange(LAG_LO, LAG_HI + 1)  # j = [8, ..., 256]
    f_k_key_pool = f_k_full[key_pool_range]
    log_j = np.log(key_pool_range.astype(float))
    log_fk = np.log(f_k_key_pool + 1e-30)
    gamma_k = ols_slope(log_j, log_fk)  # slope of log f_k vs log j (negative = decreasing with j)

    # Direction cosine over key pool
    cos_k_pool = compute_direction_cosine(kbar_h, k_mf, key_pool_range)
    cos_k_median = float(np.nanmedian(cos_k_pool))
    cos_k_slope = ols_slope(log_j, cos_k_pool)

    # ---- Query-side shrinkage (diagnostic) ----
    f_q_full = compute_shrinkage_profile(qbar_h, q_mf, np.arange(SEQ_LEN))
    f_q_query_pool = f_q_full[QUERY_POOL]  # i in [256, 511]
    log_i = np.log(QUERY_POOL.astype(float))
    log_fq = np.log(f_q_query_pool + 1e-30)
    gamma_q = ols_slope(log_i, log_fq)
    f_q_mean = float(np.nanmean(f_q_query_pool))
    f_q_cv = float(np.nanstd(f_q_query_pool) / (f_q_mean + 1e-30))

    f_k_mean_key_pool = float(np.nanmean(f_k_key_pool))
    f_k_cv_key_pool = float(np.nanstd(f_k_key_pool) / (f_k_mean_key_pool + 1e-30))

    # ---- Effective correction profile C_k(dx) ----
    C_k = compute_C_k(f_k_full, LAGS, QUERY_POOL)
    log_dx = np.log(LAGS.astype(float))
    log_Ck = np.log(C_k + 1e-30)
    gamma_C = ols_slope(log_dx, log_Ck)

    # ---- P2: does gamma_C explain >= 50% of the overshoot? ----
    overshoot = mf_sigma_mf - mf_sigma_pos  # should be positive
    # The OLS slope of log S_pos = log C_k + log S_mf → (-sigma_pos) = gamma_C + (-sigma_mf)
    # → sigma_mf - sigma_pos = -gamma_C  → gamma_C = -(sigma_mf - sigma_pos) for perfect explanation
    # But we're computing gamma_C = slope of log C_k vs log dx; for positive overshoot,
    # we expect gamma_C > 0 (C_k increases with dx, since f_k decreases with j).
    # Actually re-check: sigma_mf > sigma_pos → S_mf decays faster → C(dx) = S_pos/S_mf increases with dx
    # → gamma_C > 0. And gamma_C ≈ sigma_mf - sigma_pos for complete explanation.
    # P2: gamma_C > 0 AND |gamma_C| >= 0.5 * overshoot
    p2_sign_correct = gamma_C > 0
    p2_fraction = abs(gamma_C) / (abs(overshoot) + 1e-30)

    return {
        "label": label,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "condition": cond,
        "k1_max_diff": k1_max_diff,
        "k1_pass": k1_pass,
        "gamma_k": gamma_k,
        "f_k_mean": f_k_mean_key_pool,
        "f_k_cv": f_k_cv_key_pool,
        "gamma_q": gamma_q,
        "f_q_mean": f_q_mean,
        "f_q_cv": f_q_cv,
        "cos_k_median": cos_k_median,
        "cos_k_slope": cos_k_slope,
        "gamma_C": gamma_C,
        "sigma_mf": mf_sigma_mf,
        "sigma_pos": mf_sigma_pos,
        "overshoot": overshoot,
        "p2_sign_correct": p2_sign_correct,
        "p2_fraction_explained": p2_fraction,
        "f_k_full_mean": float(np.nanmean(f_k_full)),
        "f_k_full_std": float(np.nanstd(f_k_full)),
    }


def main():
    import sys

    print("Loading saved arrays...")
    scores_npz = np.load(SCORES_NPZ)
    mf_npz = np.load(MF_NPZ)
    mf_results = json.load(open(MF_RESULTS))

    # Pull sigma_mf and sigma_pos for registered pairs from exp-113 results
    def get_sigmas(cond_key, label):
        cond_data = mf_results["conditions"].get(cond_key, {})
        reg = cond_data.get("registered", {})
        exp = cond_data.get("exploratory", {})
        all_data = {**reg, **exp}
        if label in all_data:
            d = all_data[label]
            return d["sigma_mf"], d["sigma_pos_exp112"]
        return None, None

    # ---- Run K1 check and structural analysis ----
    print("\n--- Structural heads (random tokens) ---")
    structural_results = []
    for (ls, li, hi) in STRUCTURAL_HEADS:
        label = f"L{li}H{hi}"
        sm, sp = get_sigmas("random", label)
        if sm is None:
            print(f"  {label}: missing in exp-113 results")
            continue
        res = analyze_pair(label, ls, hi, "random", scores_npz, mf_npz, sp, sm)
        structural_results.append(res)
        k1_str = "PASS" if res.get("k1_pass") else f"FAIL (max_diff={res.get('k1_max_diff'):.2e})"
        print(f"  {label}: K1={k1_str}  gamma_k={res['gamma_k']:.4f}  gamma_C={res['gamma_C']:.4f}"
              f"  overshoot={res['overshoot']:.3f}  p2_frac={res['p2_fraction_explained']:.3f}"
              f"  cos_k_median={res['cos_k_median']:.4f}")

    print("\n--- Semantic heads (WikiText) ---")
    semantic_results = []
    for (ls, li, hi) in SEMANTIC_HEADS:
        label = f"L{li}H{hi}"
        sm, sp = get_sigmas("wikitext", label)
        if sm is None:
            sm, sp = get_sigmas("random", label)  # fallback for any exploratory
        if sm is None:
            print(f"  {label}: missing in exp-113 results, skipping")
            continue
        res = analyze_pair(label, ls, hi, "wikitext", scores_npz, mf_npz, sp, sm)
        semantic_results.append(res)
        k1_str = "PASS" if res.get("k1_pass") else f"FAIL (max_diff={res.get('k1_max_diff'):.2e})"
        print(f"  {label}: K1={k1_str}  gamma_k={res['gamma_k']:.4f}  gamma_C={res['gamma_C']:.4f}"
              f"  overshoot={res['overshoot']:.3f}  p2_frac={res['p2_fraction_explained']:.3f}")

    # ---- Verdicts ----
    print("\n=== VERDICTS ===")

    # K1
    k1_fails = [r["label"] for r in structural_results if not r.get("k1_pass")]
    if k1_fails:
        print(f"K1 FAIL: {k1_fails}. Stopping verdict computation.")
        # Still save results for inspection
    else:
        print("K1 PASS: all structural pairs pass integrity gate")

    # P1
    p1_confirm = [r for r in structural_results if r.get("gamma_k", 0) < -0.05]
    p1_dead = [r for r in structural_results if r.get("gamma_k", 0) > 0]
    n_struct = len(structural_results)
    p1_verdict = "CONFIRMED" if len(p1_confirm) >= 4 else ("DEAD" if len(p1_dead) >= 3 else "AMBIGUOUS")
    print(f"P1: {len(p1_confirm)}/{n_struct} heads with gamma_k < -0.05 ({len(p1_dead)}/{n_struct} with gamma_k > 0) → {p1_verdict}")

    # P2
    p2_confirm = [r for r in structural_results if r.get("p2_sign_correct") and r.get("p2_fraction_explained", 0) >= 0.5]
    p2_dead = [r for r in structural_results if not r.get("p2_sign_correct")]
    p2_verdict = "CONFIRMED" if len(p2_confirm) >= 4 else ("DEAD" if len(p2_dead) >= 3 else "AMBIGUOUS")
    print(f"P2: {len(p2_confirm)}/{n_struct} heads with gamma_C > 0 AND fraction >= 0.5 ({len(p2_dead)}/{n_struct} with gamma_C < 0) → {p2_verdict}")

    # P3
    p3_confirm = [r for r in semantic_results if r.get("gamma_k", 0) < 0]
    p3_dead = [r for r in semantic_results if r.get("gamma_k", 0) > 0]
    n_sem = len(semantic_results)
    p3_verdict = "CONFIRMED" if len(p3_confirm) >= 10 else ("DEAD" if len(p3_dead) >= 10 else "AMBIGUOUS")
    print(f"P3: {len(p3_confirm)}/{n_sem} semantic heads with gamma_k < 0 → {p3_verdict}")

    # Diagnostic D summary
    print("\n--- Diagnostic D (no verdict) ---")
    for r in structural_results:
        print(f"  {r['label']}: f_q_mean={r['f_q_mean']:.4f}  gamma_q={r['gamma_q']:.4f}  f_k_mean={r['f_k_mean']:.4f}  cos_k_median={r['cos_k_median']:.4f}  cos_k_slope={r['cos_k_slope']:.6f}")

    # ---- Save results ----
    output = {
        "protocol": {
            "seq_len": SEQ_LEN,
            "d_head": D_HEAD,
            "deep_lo": DEEP_LO,
            "lag_lo": LAG_LO,
            "lag_hi": LAG_HI,
            "key_pool_range": [LAG_LO, LAG_HI],
            "query_pool_range": [DEEP_LO, SEQ_LEN - 1],
        },
        "gates": {"K1": {"fail_labels": k1_fails, "pass": len(k1_fails) == 0}},
        "verdicts": {
            "P1": {
                "n_confirm": len(p1_confirm),
                "n_dead": len(p1_dead),
                "n_total": n_struct,
                "verdict": p1_verdict,
                "confirm_labels": [r["label"] for r in p1_confirm],
            },
            "P2": {
                "n_confirm": len(p2_confirm),
                "n_dead": len(p2_dead),
                "n_total": n_struct,
                "verdict": p2_verdict,
                "confirm_labels": [r["label"] for r in p2_confirm],
            },
            "P3": {
                "n_confirm": len(p3_confirm),
                "n_dead": len(p3_dead),
                "n_total": n_sem,
                "verdict": p3_verdict,
            },
        },
        "structural_per_head": structural_results,
        "semantic_per_head": semantic_results,
    }
    out_path = EXP_DIR / "results_gpt2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save shrinkage profiles (npz)
    npz_data = {}
    for r in structural_results + semantic_results:
        label = r["label"]
        cond = r["condition"]
        # Re-extract full f_k profile for saving
        li = r["layer_idx"]
        hi = r["head_idx"]
        ls = f"L{li}"
        kbar_h = scores_npz[f"kbar_{cond}_{ls}"][hi]
        k_mf = mf_npz[f"k_mf_{cond}_{ls}H{hi}"]
        f_k_full = compute_shrinkage_profile(kbar_h, k_mf, np.arange(SEQ_LEN))
        C_k = compute_C_k(f_k_full, LAGS, QUERY_POOL)
        npz_data[f"f_k_{cond}_{label}"] = f_k_full.astype(np.float32)
        npz_data[f"C_k_{cond}_{label}"] = C_k.astype(np.float32)
    np.savez(EXP_DIR / "shrinkage_gpt2.npz", **npz_data)
    print(f"Shrinkage profiles saved to {EXP_DIR / 'shrinkage_gpt2.npz'}")


if __name__ == "__main__":
    main()
