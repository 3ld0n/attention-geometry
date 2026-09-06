"""
exp-134: W_proj column structure — position-correlation amplification mechanism
Pre-registration: 807f4fc (pushed 2026-09-06, before this script written)

Protocol: random-token census, 50 sequences, length 512, seed=42, mean-first.
Identical census to exp-131/132/133 (SEQ_LEN=512, N_SEQS=50, SEED=42).

The question: why does W_proj (c_proj, GPT-2 MLP block 0) amplify σ from
~0.121 (h_gelu input) to ~0.313 (mlp_out output)?

Three tests registered:
  P1: σ_d distribution is non-uniform — top-20% output channels carry ≥50% of total |σ_d|
  P2: W_proj rows for top-k channels align more with position-correlated h_gelu directions
  P3: h_gelu position-correlated structure is low-dimensional (R ≤ 100 for 50% variance)

Extra (not pre-registered, exploratory): null test — does random 768-d projection
of h_gelu produce σ ≈ 0.313? If yes, the amplification is a dimensionality effect.

Ariel — 2026-09-06.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"

spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
exp112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp112)

pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW = exp112.WINDOW       # lags 8..256, 249 elements
DEEP_LO = exp112.DEEP_LO     # 256 — pool from position 256 onward
FIT_LO = exp112.FIT_LO       # 8
FIT_HI = exp112.FIT_HI       # 256

SEQ_LEN = 512
N_SEQS = 50
SEED = 42
VOCAB_SIZE = 50257
TOP_K_FRAC = 0.20  # P1: top-k = 20% of 768 = 154 channels
PREREG_COMMIT = "807f4fc"


def sigma_from_mean(mean_arr: np.ndarray) -> tuple[float, float, float]:
    """σ (power-law exponent), R², C[8] from mean-first mean array.
    mean_arr: (SEQ_LEN, D) mean over sequences per position.
    Reproduces exp-132 sigma_meanfirst."""
    norms = np.linalg.norm(mean_arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = mean_arr / norms  # (SEQ_LEN, D) — row-normalized
    C = M @ M.T           # (SEQ_LEN, SEQ_LEN) — pairwise cosines
    profile = pooled_window_profile(C)  # (249,) mean at each lag in WINDOW
    sigma = -ols_slope(profile, WINDOW)
    lx = np.log(WINDOW.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    y_pred = X @ np.linalg.lstsq(X, profile, rcond=None)[0]
    ss_tot = float(((profile - profile.mean())**2).sum())
    ss_res = float(((profile - y_pred)**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(sigma), float(r2), float(profile[0])


def channel_profiles(mean_arr: np.ndarray) -> np.ndarray:
    """Compute per-channel contribution to the aggregate correlation profile.
    mean_arr: (SEQ_LEN, D)
    Returns profiles: (D, len(WINDOW)) — contribution of each channel d to
    the aggregate profile (using joint L2 normalization of position vectors).
    Verified: sum over d of profiles[d, :] == pooled_window_profile(M @ M.T)
    """
    norms = np.linalg.norm(mean_arr, axis=-1, keepdims=True)  # (SEQ_LEN, 1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = mean_arr / norms  # (SEQ_LEN, D) — jointly L2-normalized per position
    D = M.shape[1]
    profiles = np.zeros((D, len(WINDOW)))
    for w, dx in enumerate(WINDOW):
        # Query positions: max(DEEP_LO, dx) .. SEQ_LEN-1
        # Key positions: q - dx (>= 0 by construction since q >= dx)
        q_lo = max(DEEP_LO, dx)
        if q_lo >= SEQ_LEN:
            continue
        q = np.arange(q_lo, SEQ_LEN)  # (n_pairs,)
        k = q - dx                    # (n_pairs,) key positions
        products = M[q, :] * M[k, :]  # (n_pairs, D)
        profiles[:, w] = products.mean(axis=0)  # (D,)
    return profiles  # (D, len(WINDOW))


def sigma_per_channel(profiles: np.ndarray) -> np.ndarray:
    """Fit σ_d = -slope of log(profile_d) vs log(lag) for each channel d.
    profiles: (D, len(WINDOW))
    Returns sigma_d: (D,)
    """
    lags = WINDOW.astype(float)
    D = profiles.shape[0]
    sigma_d = np.zeros(D)
    for d in range(D):
        p = profiles[d]
        if p.max() > 1e-10:
            sigma_d[d] = -ols_slope(p, lags)
        else:
            sigma_d[d] = 0.0
    return sigma_d


def main():
    print(f"exp-134: W_proj column structure")
    print(f"Pre-registration: {PREREG_COMMIT}\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32,
        attn_implementation="eager").to(device).eval()

    # Extract W_proj (c_proj) from MLP block 0
    # GPT-2 Conv1D stores weights as (in_features, out_features)
    # c_proj: (3072, 768)
    W_proj_raw = model.transformer.h[0].mlp.c_proj.weight.detach().float().cpu().numpy()
    print(f"W_proj (c_proj) raw shape: {W_proj_raw.shape}")
    # In GPT-2 Conv1D: output = input @ weight + bias
    # So weight is (in_features, out_features) = (3072, 768)
    # W_proj[k, d] maps h_gelu[:, k] to mlp_out[:, d]
    # "row d" of the standard linear convention = column d of W_proj_raw = W_proj_raw[:, d]
    W_proj = W_proj_raw  # (3072, 768) — column d gives the input weights for output dim d

    print(f"Effective W_proj shape used: (3072, 768) — column d = weights for output channel d")

    rng = np.random.RandomState(SEED)
    seqs_np = rng.randint(0, VOCAB_SIZE, size=(N_SEQS, SEQ_LEN))
    seqs = torch.tensor(seqs_np, dtype=torch.long, device=device)

    # Accumulators
    acc_h05  = np.zeros((SEQ_LEN, 768),  dtype=np.float64)
    acc_hgelu= np.zeros((SEQ_LEN, 3072), dtype=np.float64)
    acc_mlp0 = np.zeros((SEQ_LEN, 768),  dtype=np.float64)
    acc_h0   = np.zeros((SEQ_LEN, 768),  dtype=np.float64)

    _buf: dict[str, np.ndarray] = {}

    def hook_h0_pre(module, args):
        _buf["h0"] = args[0].detach().float().cpu().numpy()[0]

    def hook_mlp_pre(module, inp):
        # inp[0] is h^(0.5) — residual stream before MLP
        _buf["h05"] = inp[0].detach().float().cpu().numpy()[0]

    def hook_hgelu(module, inp, out):
        # c_proj's pre_hook gives inp[0] = h_gelu (post-GeLU, pre-projection)
        _buf["hgelu"] = inp[0].detach().float().cpu().numpy()[0]

    def hook_mlp_out(module, inp, out):
        _buf["mlp0"] = out.detach().float().cpu().numpy()[0]

    handles = []
    handles.append(model.transformer.h[0].register_forward_pre_hook(hook_h0_pre))
    handles.append(model.transformer.h[0].mlp.register_forward_pre_hook(
        lambda m, inp: hook_mlp_pre(m, inp)))
    handles.append(model.transformer.h[0].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: hook_hgelu(m, inp, None)))
    handles.append(model.transformer.h[0].mlp.register_forward_hook(hook_mlp_out))

    print(f"Census: {N_SEQS} seqs × {SEQ_LEN} tokens, random, seed={SEED}")
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_SEQS}...")
            _buf.clear()
            model(seq.unsqueeze(0))
            acc_h0    += _buf["h0"].astype(np.float64) / N_SEQS
            acc_h05   += _buf["h05"].astype(np.float64) / N_SEQS
            acc_hgelu += _buf["hgelu"].astype(np.float64) / N_SEQS
            acc_mlp0  += _buf["mlp0"].astype(np.float64) / N_SEQS

    for h in handles:
        h.remove()

    print("\n=== Verification: reproduce exp-132/133 σ values ===")
    sigma_h0,    r2_h0,    c8_h0    = sigma_from_mean(acc_h0)
    sigma_h05,   r2_h05,   c8_h05   = sigma_from_mean(acc_h05)
    sigma_hgelu, r2_hgelu, c8_hgelu = sigma_from_mean(acc_hgelu)
    sigma_mlp0,  r2_mlp0,  c8_mlp0  = sigma_from_mean(acc_mlp0)

    for label, s, r2, c8, expect in [
        ("h^(0)        [embedding]      ", sigma_h0,    r2_h0,    c8_h0,    0.403),
        ("h^(0.5)      [MLP input]      ", sigma_h05,   r2_h05,   c8_h05,   0.144),
        ("h_gelu       [post-GeLU]      ", sigma_hgelu, r2_hgelu, c8_hgelu, 0.121),
        ("mlp_out^(0)  [MLP write]      ", sigma_mlp0,  r2_mlp0,  c8_mlp0,  0.313),
    ]:
        match = "✓" if abs(s - expect) < 0.05 else "✗"
        print(f"  {label}  σ={s:+.4f}  R²={r2:.4f}  C[8]={c8:.3f}  (expect ~{expect}) {match}")

    print()

    # --- P3: h_gelu position-correlated dimensionality via PCA ---
    print("=== P3: h_gelu position-correlated dimensionality ===")
    # Use deep positions only (>= DEEP_LO) for the PCA — same window as σ
    h_gelu_deep = acc_hgelu[DEEP_LO:, :]  # (256, 3072) — deep positions
    h_gelu_c = h_gelu_deep - h_gelu_deep.mean(axis=0, keepdims=True)
    # SVD of (256 × 3072) — at most 256 non-trivial singular values
    U, S, Vt = np.linalg.svd(h_gelu_c, full_matrices=False)  # S: (256,)
    total_var = float((S**2).sum())
    cumvar = np.cumsum(S**2) / total_var

    r_50 = int(np.searchsorted(cumvar, 0.50)) + 1
    r_80 = int(np.searchsorted(cumvar, 0.80)) + 1
    r_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    var_at_r100 = float(cumvar[min(99, len(cumvar)-1)])

    print(f"  Components for 50% position-variance: {r_50}  (threshold ≤ 100)")
    print(f"  Components for 80% position-variance: {r_80}")
    print(f"  Components for 95% position-variance: {r_95}")
    print(f"  Variance at R=100: {var_at_r100*100:.1f}%")
    p3_verdict = "CONFIRMED" if r_50 <= 100 else "FALSIFIED"
    k3_fired = var_at_r100 < 0.20
    print(f"  P3 verdict: {p3_verdict}")

    # Top position-correlated directions of h_gelu (from SVD right singular vectors)
    V_pos = Vt[:r_50, :]  # (r_50, 3072)

    print()

    # --- P1 + P2: Per-output-channel σ distribution ---
    print("=== P1 / P2: Per-channel σ_d distribution and W_proj alignment ===")
    print("  Computing per-channel correlation profiles (768 channels × 249 lags)...")
    profiles_mlp = channel_profiles(acc_mlp0)  # (768, 249)

    # Sanity check: does sum over d reproduce the aggregate profile?
    profile_sum = profiles_mlp.sum(axis=0)  # (249,)
    sigma_sum, r2_sum, _ = sigma_from_mean.__wrapped__(profile_sum) if hasattr(sigma_from_mean, '__wrapped__') else (None, None, None)
    # Direct check: compute aggregate profile the normal way
    norms_mlp = np.linalg.norm(acc_mlp0, axis=-1, keepdims=True)
    norms_mlp = np.where(norms_mlp < 1e-10, 1.0, norms_mlp)
    M_mlp = acc_mlp0 / norms_mlp
    C_mlp = M_mlp @ M_mlp.T
    agg_profile = pooled_window_profile(C_mlp)  # (249,)
    max_diff = float(np.abs(profile_sum - agg_profile).max())
    print(f"  Sanity check: max |Σ_d profile_d - agg_profile| = {max_diff:.2e}  (expect ~0)")

    print("  Fitting σ_d for each channel...")
    sigma_d = sigma_per_channel(profiles_mlp)  # (768,)

    abs_sigma_d = np.abs(sigma_d)
    k_20pct = int(np.ceil(TOP_K_FRAC * 768))   # 154 channels

    sorted_idx = np.argsort(abs_sigma_d)[::-1]
    top_k_idx = sorted_idx[:k_20pct]
    bot_k_idx = sorted_idx[-k_20pct:]

    total_abs_sigma = float(abs_sigma_d.sum())
    top_k_share = float(abs_sigma_d[top_k_idx].sum() / total_abs_sigma) if total_abs_sigma > 0 else 0
    max_sigma = float(abs_sigma_d.max())
    mean_sigma = float(abs_sigma_d.mean())
    ratio_max_mean = max_sigma / mean_sigma if mean_sigma > 0 else 0

    print(f"\n  σ_d distribution (768 output channels):")
    print(f"    Max |σ_d|: {max_sigma:.4f},  Mean |σ_d|: {mean_sigma:.4f},  Ratio: {ratio_max_mean:.1f}")
    print(f"    Top-{k_20pct} channels (20%) carry: {top_k_share*100:.1f}% of total |σ_d|")
    print(f"    Channels above 5× mean: {int((abs_sigma_d > 5 * mean_sigma).sum())}")
    print(f"    Channels with σ_d > 0.2: {int((abs_sigma_d > 0.2).sum())}")
    print(f"    Channels with σ_d > 0.3: {int((abs_sigma_d > 0.3).sum())}")

    p1_verdict = "CONFIRMED" if top_k_share >= 0.50 else "FALSIFIED"
    k1_fired = ratio_max_mean <= 5.0 and top_k_share < 0.50
    print(f"  P1 verdict: {p1_verdict}  (top-20% carry {top_k_share*100:.1f}%, threshold 50%)")

    # Percentile distribution
    pct_labels = [10, 25, 50, 75, 90, 95, 99]
    pct_vals = {str(p): float(np.percentile(abs_sigma_d, p)) for p in pct_labels}
    print(f"  Percentiles: " + "  ".join(f"p{p}={pct_vals[str(p)]:.4f}" for p in pct_labels))

    # Cumulative share at various thresholds
    cum_share = {}
    for k_thr in [10, 20, 50, 100, 154, 200, 400]:
        if k_thr <= 768:
            cum_share[str(k_thr)] = float(abs_sigma_d[sorted_idx[:k_thr]].sum() / total_abs_sigma)

    # Top-10 channels
    top10 = [{"channel": int(sorted_idx[i]), "sigma_d": float(abs_sigma_d[sorted_idx[i]])}
             for i in range(10)]
    print(f"  Top-10 channels: {[(t['channel'], round(t['sigma_d'], 4)) for t in top10]}")

    # --- P2: W_proj column alignment with position-correlated h_gelu directions ---
    print("\n  P2: W_proj column alignment with position-correlated h_gelu directions")
    # W_proj is (3072, 768) — column d is the input weight vector for output d
    # Normalize each column
    W_cols = W_proj[:, :]  # (3072, 768)
    col_norms = np.linalg.norm(W_cols, axis=0)  # (768,)
    W_cols_n = W_cols / (col_norms[None, :] + 1e-10)  # (3072, 768) — normalized

    # V_pos: (r_50, 3072) — position-correlated directions in h_gelu space
    # Projection of each W_proj column onto V_pos subspace:
    proj = V_pos @ W_cols_n  # (r_50, 768)
    align_score = float((proj**2).sum(axis=0).mean())         # mean squared proj norm, all cols
    align_top_k = float((proj[:, top_k_idx]**2).sum(axis=0).mean())  # top-k channels
    align_bot_k = float((proj[:, bot_k_idx]**2).sum(axis=0).mean())  # bot-k channels
    align_random = r_50 / 3072.0  # expected for random unit vector

    print(f"    Subspace dim: {r_50}  (50% position-variance of h_gelu)")
    print(f"    Expected alignment (random): {align_random:.6f}")
    print(f"    Mean alignment (all 768 cols): {align_score:.6f}")
    print(f"    Mean alignment (top-{k_20pct} high-σ cols): {align_top_k:.6f}")
    print(f"    Mean alignment (bot-{k_20pct} low-σ cols): {align_bot_k:.6f}")
    print(f"    Top-k/random ratio: {align_top_k / align_random:.2f}")
    print(f"    Top-k > bot-k: {align_top_k > align_bot_k}")

    p2_verdict = "CONFIRMED" if align_top_k > align_bot_k else "FALSIFIED"
    k2_fired = not (align_top_k > align_bot_k)
    print(f"  P2 verdict: {p2_verdict}")

    # --- Exploratory: null test — random 768-d projection of h_gelu ---
    print("\n=== Exploratory: null test — random 768-d projection ===")
    print("  (not pre-registered; tests whether amplification is dimensionality effect)")
    rng2 = np.random.RandomState(SEED + 1)
    W_null = rng2.randn(3072, 768).astype(np.float32)
    # Normalize columns
    W_null /= np.linalg.norm(W_null, axis=0, keepdims=True) + 1e-10
    # Compute h_null = acc_hgelu @ W_null
    h_null = acc_hgelu @ W_null  # (SEQ_LEN, 768)
    sigma_null, r2_null, _ = sigma_from_mean(h_null)
    print(f"  σ(h_gelu @ W_random) = {sigma_null:.4f}  (if ≈ σ_mlp = {sigma_mlp0:.4f}, amplification is dimensionality)")
    print(f"  σ(h_gelu)            = {sigma_hgelu:.4f}  (baseline)")
    null_test_conclusion = (
        "amplification is dimensionality effect (no learned column structure needed)"
        if abs(sigma_null - sigma_mlp0) < 0.03 else
        "amplification exceeds random projection — W_proj has learned column structure"
        if sigma_mlp0 > sigma_null + 0.03 else
        "ambiguous"
    )
    print(f"  Conclusion: {null_test_conclusion}")

    # --- Exploratory: σ of h_gelu projected onto top singular directions of W_proj ---
    print("\n=== Exploratory: σ of h_gelu projected onto W_proj SVD directions ===")
    U_w, S_w, Vt_w = np.linalg.svd(W_proj, full_matrices=False)
    # W_proj (3072, 768): output = h_gelu @ W_proj
    # SVD: W_proj = U_w @ diag(S_w) @ Vt_w
    # U_w: (3072, 768) — input singular vectors (directions in h_gelu space)
    # Vt_w: (768, 768) — output singular vectors (directions in mlp_out space)
    # Top-k projection of h_gelu onto top-k input singular directions:
    svd_sigmas = {}
    for k_svd in [1, 2, 5, 10, 20, 50, 100, 200, 400, 768]:
        if k_svd <= U_w.shape[1]:
            h_svd = acc_hgelu @ U_w[:, :k_svd]  # (SEQ_LEN, k_svd)
            s_svd, r2_svd, _ = sigma_from_mean(h_svd)
            svd_sigmas[k_svd] = {"sigma": s_svd, "r2": r2_svd}
            print(f"  σ(h_gelu @ top-{k_svd:3d} W_proj input SV directions) = {s_svd:.4f}  R²={r2_svd:.4f}")

    # Also: project h_gelu onto BOTTOM singular directions
    print()
    for k_svd in [5, 10, 50]:
        h_svd = acc_hgelu @ U_w[:, -k_svd:]  # bottom-k
        s_svd, r2_svd, _ = sigma_from_mean(h_svd)
        print(f"  σ(h_gelu @ bot-{k_svd:3d} W_proj input SV directions) = {s_svd:.4f}  R²={r2_svd:.4f}")

    # Overall verdict
    print("\n=== Summary ===")
    if p1_verdict == "CONFIRMED" and p2_verdict == "CONFIRMED" and p3_verdict == "CONFIRMED":
        overall = "confirmed"
    elif k1_fired or k2_fired:
        overall = "falsified"
    else:
        overall = "partial" if (p1_verdict == "CONFIRMED" or p2_verdict == "CONFIRMED") else "inconclusive"

    headline = (f"exp-134 W_proj column structure: {overall.upper()}. "
                f"P1={p1_verdict} (top-20% carry {top_k_share*100:.1f}% of σ_d); "
                f"P2={p2_verdict} (top-k align {align_top_k:.4f} vs bot-k {align_bot_k:.4f}); "
                f"P3={p3_verdict} (h_gelu 50%var in {r_50} components). "
                f"Null test: σ(random proj)={sigma_null:.4f} vs σ(W_proj)={sigma_mlp0:.4f}.")
    print(f"  P1: {p1_verdict}")
    print(f"  P2: {p2_verdict}")
    print(f"  P3: {p3_verdict}")
    print(f"  Overall: {overall.upper()}")
    print(f"\n  {headline}")

    results = {
        "experiment": "exp-134",
        "date": "2026-09-06",
        "model": "gpt2",
        "protocol": f"random-token census, {N_SEQS} sequences, length {SEQ_LEN}, seed={SEED}, mean-first",
        "prereg_commit": PREREG_COMMIT,
        "n_seqs": N_SEQS, "seq_len": SEQ_LEN, "seed": SEED,

        "verification": {
            "sigma_h0":     sigma_h0,    "r2_h0":     r2_h0,
            "sigma_h05":    sigma_h05,   "r2_h05":    r2_h05,
            "sigma_hgelu":  sigma_hgelu, "r2_hgelu":  r2_hgelu,
            "sigma_mlp0":   sigma_mlp0,  "r2_mlp0":   r2_mlp0,
            "exp132_expected": {"h0": 0.4033, "h05": 0.1441, "hgelu": 0.1209, "mlp0": 0.3125},
        },

        "P3_hgelu_dimensionality": {
            "r_50pct": r_50, "r_80pct": r_80, "r_95pct": r_95,
            "var_at_100": var_at_r100,
            "singular_values_top10": S[:10].tolist(),
            "verdict": p3_verdict, "K3_fired": k3_fired,
        },

        "P1_channel_concentration": {
            "k_20pct": k_20pct,
            "top_k_share": top_k_share,
            "max_sigma_d": max_sigma,
            "mean_sigma_d": mean_sigma,
            "ratio_max_mean": ratio_max_mean,
            "channels_above_5x_mean": int((abs_sigma_d > 5 * mean_sigma).sum()),
            "channels_above_0p2": int((abs_sigma_d > 0.2).sum()),
            "channels_above_0p3": int((abs_sigma_d > 0.3).sum()),
            "percentiles": pct_vals,
            "cumulative_share": cum_share,
            "top10_channels": top10,
            "verdict": p1_verdict, "K1_fired": k1_fired,
        },

        "P2_alignment": {
            "subspace_dim_r50": r_50,
            "expected_random": align_random,
            "mean_all_cols": align_score,
            "mean_top_k_cols": align_top_k,
            "mean_bot_k_cols": align_bot_k,
            "top_k_over_random_ratio": align_top_k / align_random,
            "top_k_gt_bot_k": bool(align_top_k > align_bot_k),
            "verdict": p2_verdict, "K2_fired": k2_fired,
        },

        "exploratory_null_test": {
            "sigma_random_proj": sigma_null,
            "r2_random_proj": r2_null,
            "sigma_hgelu_baseline": sigma_hgelu,
            "sigma_mlp0_actual": sigma_mlp0,
            "conclusion": null_test_conclusion,
        },

        "exploratory_svd_analysis": {
            "description": "σ of h_gelu projected onto top-k W_proj input singular directions",
            "results": {str(k): v for k, v in svd_sigmas.items()},
            "finding": (f"Top-5 W_proj input SVDs give σ={svd_sigmas.get(5, {}).get('sigma', 0):.4f} "
                        f"vs random projection {sigma_null:.4f} and full output {sigma_mlp0:.4f}"),
        },

        "overall_verdict": overall,
        "headline": headline,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
