"""
exp-046 — Sign anomaly investigation via eigenvalue spectral density (GPT-2)

Pre-stated hypotheses (see notes.md for full context):

  H1 (GPT-2 sign anomaly):
    ρ(λ_proxy, valley_depth) < 0 for GPT-2 conformal heads.
    The sign anomaly (first found in Pythia-2.8B and GPT-Neo-2.7B, April 17)
    extends to GPT-2 (learned PE). λ_proxy measures boundary enhancement
    as G(shallow_query, Δx) / G(deep_query, Δx) - 1.

  H2 (Effective rank ordering):
    Conformal heads have higher effective rank in W_QK = (W_Q^T W_K + W_K^T W_Q)/2
    than non-conformal heads. Conformal power-law structure requires spanning
    many positional modes; a single-mode (low-rank) head cannot produce
    the full conformal decay pattern.

  H3 (Mediation chain — candidate mechanism for sign anomaly):
    ρ(λ_proxy, eff_rank) < 0  AND  ρ(eff_rank, valley_depth) > 0.
    Together: high λ → low effective rank (boundary-mode dominated) →
    shallower valley. This would explain the sign anomaly through geometry,
    not a failure of the physics.

  H4 (GOE level spacing):
    Conformal heads have GOE-like level spacing in W_QK eigenvalues
    (mean Oganesyan-Huse r-ratio ≈ 0.536). Non-conformal heads show
    Poisson-like spacing (r-ratio ≈ 0.386). Tests SYK chaotic dynamics
    in the weight-space geometry.

  H0 (null):
    W_QK eigenvalues uncorrelated with attention geometry — negative result.

Physical background:
  - For each attention head (layer l, head h), the attention score is
    score_ij = q_i^T k_j / sqrt(d_k) = x_i^T (W_Q_h W_K_h^T) x_j / sqrt(d_k)
  - The head's intrinsic QK geometry is encoded in W_QK_h = W_Q_h^T W_K_h ∈ R^{d_k×d_k}
  - Its eigenvalue spectrum reveals: how many independent modes (eff_rank),
    level spacing statistics (GOE/Poisson → chaotic/integrable), and whether
    the head is dominated by a few boundary modes vs spanning the full space.
  - For GPT-2: d_model=768, n_heads=12, d_k=64, n_layers=12 → 144 heads, 64 eigenvalues each.

Compare to:
  exp-007:  GPT-2, 44/144 conformal, Δ_med = 0.2493 (SYK confirmed)
  exp-045:  GPT-2 DFT spectral function — structural finite-support bias;
            this experiment uses weight-matrix eigenvalues (no DFT, no bias)
  April 17 BCFT: sign anomaly on Pythia-2.8B (18/21 negative layers) and
            GPT-Neo-2.7B (24/31 negative layers)
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from transformers import GPT2LMHeadModel

# ── constants (same as exp-007 for position-space analysis) ───────────────────
SEQ_LEN      = 256
N_INPUTS     = 50
MAX_DX       = 56
MIN_POS      = 32
FIT_LOW      = 3
FIT_HIGH     = 50
R2_THRESHOLD = 0.90
RNG_SEED     = 42
SYK_NEAR_TOL = 0.05

# GPT-2 architecture
D_K      = 64    # d_model / n_heads = 768 / 12
D_MODEL  = 768
N_LAYERS = 12
N_HEADS  = 12

# Valley proxy: deep query positions only
VALLEY_Q_MIN = SEQ_LEN // 2   # x_q >= 128

# Boundary proxy λ_proxy: compare shallow vs deep query positions
SHALLOW_Q_LOW  = MIN_POS         # x_q ∈ [32, 64)
SHALLOW_Q_HIGH = MIN_POS * 2
DEEP_Q_LOW     = 128             # x_q ∈ [128, 200)
DEEP_Q_HIGH    = 200
PROXY_DX_LOW   = 5              # Δx ∈ [5, 15) — intermediate lags
PROXY_DX_HIGH  = 15

# Level spacing: GOE / Poisson reference values (Oganesyan-Huse r-ratio)
GOE_R_RATIO     = 0.536
POISSON_R_RATIO = 0.386

OUT_DIR      = Path("research/physics/experiments/exp-046_sign_anomaly_eigenvalue")
RESULTS_FILE = OUT_DIR / "results.json"


# ── 1. Eigenvalue analysis (input-independent) ─────────────────────────────────

def extract_wqk_eigendata(model: GPT2LMHeadModel) -> list[dict]:
    """
    For each (layer, head): compute eigenvalues of the symmetrized QK product
      M = (W_Q^T W_K + W_K^T W_Q) / 2  ∈ R^{d_k × d_k}

    In GPT-2 HuggingFace (Conv1D), c_attn.weight has shape (d_model, 3*d_model).
    Output = input @ weight + bias, so the three projections are:
      Q_h = x @ weight[:, h*d_k:(h+1)*d_k]
      K_h = x @ weight[:, d_model+h*d_k:d_model+(h+1)*d_k]
    Hence W_Q_h = weight[:, h*dk:(h+1)*dk]  ∈ R^{768×64}
          W_K_h = weight[:, dm+h*dk:dm+(h+1)*dk] ∈ R^{768×64}
    W_QK = W_Q_h.T @ W_K_h ∈ R^{64×64}
    """
    results = []
    for l in range(N_LAYERS):
        W_full = model.transformer.h[l].attn.c_attn.weight.detach().float().numpy()
        # shape: (768, 2304)
        for h in range(N_HEADS):
            W_Q_h = W_full[:, h * D_K          : (h + 1) * D_K]
            W_K_h = W_full[:, D_MODEL + h * D_K : D_MODEL + (h + 1) * D_K]

            W_QK = W_Q_h.T @ W_K_h           # (64, 64)
            M    = (W_QK + W_QK.T) / 2.0     # symmetrize

            # Eigenvalues (real, sorted ascending since M symmetric)
            eigvals = np.linalg.eigvalsh(M)  # ascending

            # Effective rank via Shannon entropy of |eigenvalue| distribution
            abs_ev = np.abs(eigvals)
            total  = abs_ev.sum()
            if total > 1e-12:
                p       = abs_ev / total
                p_nz    = p[p > 1e-14]
                entropy = float(-np.sum(p_nz * np.log(p_nz)))
                eff_rank = float(np.exp(entropy))
            else:
                eff_rank = float(D_K)

            # Oganesyan-Huse r-ratio: scale-free level spacing statistic
            # r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})  where s_i = λ_{i+1} - λ_i
            spacings = np.diff(eigvals)   # sorted → spacings >= 0
            eps = 1e-30
            if len(spacings) >= 3:
                s_lo = spacings[:-1]
                s_hi = spacings[1:]
                r_vals = np.minimum(s_lo, s_hi) / (np.maximum(s_lo, s_hi) + eps)
                r_ratio = float(np.mean(r_vals))
            else:
                r_ratio = float("nan")

            results.append({
                "layer":    l,
                "head":     h,
                "eigvals":  eigvals.tolist(),       # full 64-element spectrum
                "eff_rank": round(eff_rank, 4),
                "r_ratio":  round(r_ratio, 6) if not math.isnan(r_ratio) else None,
            })

    return results


# ── 2. Lag profile and power-law fit (exp-007 method) ─────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    """Mean attention weight G_>(r) as a function of lag r."""
    n = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), n):
            j = i - dx
            if 0 <= j < n:
                A[dx]      += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A.astype(np.float32)


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray,
                  low: int, high: int) -> tuple[float, float]:
    """Fit G(r) ~ r^{-2Δ}, return (Δ, R²)."""
    mask = (dx_arr >= low) & (dx_arr < high) & (y_arr > 1e-20)
    if mask.sum() < 5:
        return float("nan"), 0.0
    log_x = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask].astype(float))
    A_mat = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    pred   = A_mat @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2    = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    delta = float(-coeffs[1] / 2.0)
    return delta, r2


# ── 3. Position-resolved accumulator ──────────────────────────────────────────

def empty_accum() -> dict:
    return {
        # valley: for x_q >= VALLEY_Q_MIN, thirds of past context
        "valley_start_sum": 0.0, "valley_start_cnt": 0,
        "valley_mid_sum":   0.0, "valley_mid_cnt":   0,
        "valley_end_sum":   0.0, "valley_end_cnt":   0,
        # boundary proxy: shallow vs deep query positions, fixed Δx range
        "proxy_shallow_sum": 0.0, "proxy_shallow_cnt": 0,
        "proxy_deep_sum":    0.0, "proxy_deep_cnt":    0,
    }


def accumulate_position_stats(attn_head: np.ndarray, accum: dict) -> None:
    """
    From one head's full causal attention matrix, accumulate valley and
    boundary-proxy statistics in-place.

    Valley: for x_q in [VALLEY_Q_MIN, n), split x_k in [0, x_q) into thirds:
      start = [0, x_q//4), mid = [x_q//4, 3*x_q//4), end = [3*x_q//4, x_q)
    valley_depth = 1 - G_mid / max(G_start, G_end)

    Boundary proxy: compare mean attention at shallow (x_q in [32,64), same Δx)
    to deep (x_q in [128,200), same Δx) — measures boundary enhancement.
    """
    n = attn_head.shape[0]

    # Valley accumulation
    for xq in range(VALLEY_Q_MIN, n):
        e1 = max(1, xq // 4)
        e2 = max(e1 + 1, 3 * xq // 4)
        e3 = xq   # attend to positions 0..xq-1

        if e1 > 0 and e2 > e1 and e3 > e2:
            accum["valley_start_sum"] += float(attn_head[xq, 0:e1].sum())
            accum["valley_start_cnt"] += e1
            accum["valley_mid_sum"]   += float(attn_head[xq, e1:e2].sum())
            accum["valley_mid_cnt"]   += (e2 - e1)
            accum["valley_end_sum"]   += float(attn_head[xq, e2:e3].sum())
            accum["valley_end_cnt"]   += (e3 - e2)

    # Boundary proxy: shallow positions
    for xq in range(SHALLOW_Q_LOW, min(SHALLOW_Q_HIGH, n)):
        for dx in range(PROXY_DX_LOW, min(PROXY_DX_HIGH, xq)):
            xk = xq - dx
            if 0 <= xk < n:
                accum["proxy_shallow_sum"] += float(attn_head[xq, xk])
                accum["proxy_shallow_cnt"] += 1

    # Boundary proxy: deep positions
    for xq in range(DEEP_Q_LOW, min(DEEP_Q_HIGH, n)):
        for dx in range(PROXY_DX_LOW, min(PROXY_DX_HIGH, xq)):
            xk = xq - dx
            if 0 <= xk < n:
                accum["proxy_deep_sum"] += float(attn_head[xq, xk])
                accum["proxy_deep_cnt"] += 1


def compute_valley_and_lambda(accum: dict) -> tuple[float, float, dict]:
    """Compute valley_depth and λ_proxy from accumulator."""
    cnt_s = accum["valley_start_cnt"]
    cnt_m = accum["valley_mid_cnt"]
    cnt_e = accum["valley_end_cnt"]

    if cnt_s > 0 and cnt_m > 0 and cnt_e > 0:
        g_start = accum["valley_start_sum"] / cnt_s
        g_mid   = accum["valley_mid_sum"]   / cnt_m
        g_end   = accum["valley_end_sum"]   / cnt_e
        denom   = max(g_start, g_end)
        valley  = float(1.0 - g_mid / denom) if denom > 1e-12 else float("nan")
        start_dominates = (g_start >= g_end)
    else:
        g_start = g_mid = g_end = float("nan")
        valley  = float("nan")
        start_dominates = None

    cnt_sh = accum["proxy_shallow_cnt"]
    cnt_dp = accum["proxy_deep_cnt"]
    if cnt_sh > 0 and cnt_dp > 0:
        g_sh = accum["proxy_shallow_sum"] / cnt_sh
        g_dp = accum["proxy_deep_sum"]    / cnt_dp
        lam  = float(g_sh / g_dp - 1.0) if g_dp > 1e-15 else float("nan")
    else:
        g_sh = g_dp = lam = float("nan")

    details = {
        "g_start": g_start, "g_mid": g_mid, "g_end": g_end,
        "g_shallow": g_sh,  "g_deep": g_dp,
        "start_dominates": start_dominates,
    }
    return valley, lam, details


# ── 4. Spearman correlation helper ─────────────────────────────────────────────

def spearman(x_list: list, y_list: list) -> tuple[float, float, int]:
    """Spearman ρ on pairs where both values are finite."""
    pairs = [
        (xi, yi) for xi, yi in zip(x_list, y_list)
        if xi is not None and yi is not None
        and math.isfinite(xi) and math.isfinite(yi)
    ]
    if len(pairs) < 4:
        return float("nan"), float("nan"), len(pairs)
    xv, yv = zip(*pairs)
    rho, p  = scipy_stats.spearmanr(xv, yv)
    return float(rho), float(p), len(pairs)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("exp-046: Sign anomaly investigation via eigenvalue spectral density (GPT-2)")
    print(f"  Protocol: N_INPUTS={N_INPUTS}, SEQ_LEN={SEQ_LEN}, RNG_SEED={RNG_SEED}")
    print(f"  Power-law fit: Δx∈[{FIT_LOW},{FIT_HIGH}), R²>{R2_THRESHOLD}")
    print(f"  Valley: x_q≥{VALLEY_Q_MIN}, boundary proxy: Δx∈[{PROXY_DX_LOW},{PROXY_DX_HIGH})")

    # ── load model ──────────────────────────────────────────────────────────
    print("Loading GPT-2 (eager)...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    vocab_size = model.config.vocab_size
    total_heads = N_LAYERS * N_HEADS  # 144

    # ── eigenvalue analysis (input-independent) ─────────────────────────────
    print("Computing W_QK eigenvalues for all 144 heads (input-independent)...", flush=True)
    eigen_data = extract_wqk_eigendata(model)
    eigen_map  = {(e["layer"], e["head"]): e for e in eigen_data}
    print("  Done.", flush=True)

    # ── forward passes: collect lag profiles + position-resolved stats ───────
    rng       = np.random.default_rng(RNG_SEED)
    dx_arr    = np.arange(MAX_DX)
    lag_sum   = {l: {h: np.zeros(MAX_DX) for h in range(N_HEADS)} for l in range(N_LAYERS)}
    pos_accum = {l: {h: empty_accum()    for h in range(N_HEADS)} for l in range(N_LAYERS)}

    print(f"Running {N_INPUTS} forward passes...", flush=True)
    for inp_idx in range(N_INPUTS):
        token_ids = torch.tensor(
            rng.integers(0, vocab_size, size=(1, SEQ_LEN)), dtype=torch.long
        )
        with torch.no_grad():
            outputs = model(token_ids, output_attentions=True)

        for l in range(N_LAYERS):
            attn = outputs.attentions[l][0].float().numpy()  # (n_heads, seq, seq)
            for h in range(N_HEADS):
                lag_sum[l][h] += compute_lag_profile(attn[h], MIN_POS, MAX_DX)
                accumulate_position_stats(attn[h], pos_accum[l][h])

        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS} done", flush=True)

    # Average lag profiles
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            lag_sum[l][h] /= N_INPUTS

    # ── per-head analysis ────────────────────────────────────────────────────
    print("Computing per-head results...", flush=True)
    per_head: list[dict] = []

    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            G     = lag_sum[l][h]
            delta, r2 = fit_power_law(dx_arr, G, FIT_LOW, FIT_HIGH)

            conformal = (r2 >= R2_THRESHOLD and not math.isnan(delta) and delta > 0)
            syk_near  = (conformal and abs(delta - 0.25) <= SYK_NEAR_TOL)

            valley, lam, det = compute_valley_and_lambda(pos_accum[l][h])

            eig = eigen_map[(l, h)]

            def _r(v: float, n: int = 6) -> float | None:
                return round(v, n) if (v is not None and math.isfinite(v)) else None

            per_head.append({
                "layer":         l,
                "head":          h,
                "delta_pos":     _r(delta, 6),
                "r2_pos":        round(r2, 6),
                "conformal":     conformal,
                "syk_near":      syk_near,
                "valley_depth":  _r(valley, 6),
                "lambda_proxy":  _r(lam, 6),
                "eff_rank":      eig["eff_rank"],
                "r_ratio":       eig["r_ratio"],
                "g_start":       _r(det["g_start"], 7),
                "g_mid":         _r(det["g_mid"],   7),
                "g_end":         _r(det["g_end"],   7),
                "g_shallow":     _r(det["g_shallow"],7),
                "g_deep":        _r(det["g_deep"],   7),
                "start_dominates": det["start_dominates"],
            })

    # ── subset lists ─────────────────────────────────────────────────────────
    conformal_heads = [r for r in per_head if r["conformal"]]
    non_conf_heads  = [r for r in per_head if not r["conformal"]]
    syk_heads       = [r for r in per_head if r["syk_near"]]

    # ── H1: ρ(λ_proxy, valley_depth) for conformal heads ────────────────────
    rho_lam_val, p_lam_val, n_lam_val = spearman(
        [r["lambda_proxy"] for r in conformal_heads],
        [r["valley_depth"] for r in conformal_heads],
    )

    # ── H2: effective rank conformal vs non-conformal ────────────────────────
    conf_rank  = [r["eff_rank"] for r in conformal_heads if r["eff_rank"] is not None]
    ncnf_rank  = [r["eff_rank"] for r in non_conf_heads  if r["eff_rank"] is not None]
    mean_conf_rank = statistics.mean(conf_rank) if conf_rank else float("nan")
    mean_ncnf_rank = statistics.mean(ncnf_rank) if ncnf_rank else float("nan")

    # ── H3: mediation chain ──────────────────────────────────────────────────
    rho_lam_rank, p_lam_rank, n_lam_rank = spearman(
        [r["lambda_proxy"] for r in conformal_heads],
        [r["eff_rank"]     for r in conformal_heads],
    )
    rho_rank_val, p_rank_val, n_rank_val = spearman(
        [r["eff_rank"]     for r in conformal_heads],
        [r["valley_depth"] for r in conformal_heads],
    )
    mediation_holds = (
        not math.isnan(rho_lam_rank) and rho_lam_rank < 0
        and not math.isnan(rho_rank_val) and rho_rank_val > 0
    )

    # ── H4: r-ratio GOE test ─────────────────────────────────────────────────
    conf_r  = [r["r_ratio"] for r in conformal_heads if r["r_ratio"] is not None]
    ncnf_r  = [r["r_ratio"] for r in non_conf_heads  if r["r_ratio"] is not None]
    mean_conf_r = statistics.mean(conf_r) if conf_r else float("nan")
    mean_ncnf_r = statistics.mean(ncnf_r) if ncnf_r else float("nan")
    goe_conf  = (not math.isnan(mean_conf_r)
                 and abs(mean_conf_r - GOE_R_RATIO) < abs(mean_conf_r - POISSON_R_RATIO))

    # ── valley geometry: start vs end dominance ──────────────────────────────
    n_start_dom = sum(1 for r in conformal_heads if r["start_dominates"] is True)
    n_end_dom   = sum(1 for r in conformal_heads if r["start_dominates"] is False)

    # ── Δ_pos and eff_rank overall stats ─────────────────────────────────────
    rho_delta_val, p_delta_val, n_delta_val = spearman(
        [r["delta_pos"]   for r in conformal_heads],
        [r["valley_depth"] for r in conformal_heads],
    )
    rho_delta_rank, _, _ = spearman(
        [r["delta_pos"] for r in conformal_heads],
        [r["eff_rank"]  for r in conformal_heads],
    )

    # ── print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"=== RESULTS === ({len(conformal_heads)}/{total_heads} conformal, {len(syk_heads)} SYK-near)")
    print(f"{'='*60}")

    print(f"\nH1: Sign anomaly in GPT-2?")
    print(f"  ρ(λ_proxy, valley_depth) = {rho_lam_val:+.4f}  (p={p_lam_val:.2e}, n={n_lam_val})")
    if math.isnan(rho_lam_val):
        print(f"  INCONCLUSIVE: insufficient valid pairs")
    elif rho_lam_val < -0.1:
        print(f"  NEGATIVE → sign anomaly CONFIRMED in GPT-2 ✓")
    elif rho_lam_val > 0.1:
        print(f"  POSITIVE → sign anomaly NOT present in GPT-2 ✗")
    else:
        print(f"  NEAR ZERO → no strong signal in either direction")

    print(f"\nH2: Effective rank ordering?")
    print(f"  Mean eff_rank: conformal={mean_conf_rank:.2f}, non-conformal={mean_ncnf_rank:.2f}")
    if not math.isnan(mean_conf_rank) and not math.isnan(mean_ncnf_rank):
        if mean_conf_rank > mean_ncnf_rank:
            print(f"  Conformal > non-conformal ✓ (Δrank = {mean_conf_rank - mean_ncnf_rank:.2f})")
        else:
            print(f"  Conformal ≤ non-conformal ✗ (Δrank = {mean_conf_rank - mean_ncnf_rank:.2f})")
    print(f"  ρ(Δ_pos, eff_rank) = {rho_delta_rank:+.4f} (among conformal heads)")

    print(f"\nH3: Mediation chain (high λ → low rank → shallow valley)?")
    print(f"  ρ(λ_proxy, eff_rank)      = {rho_lam_rank:+.4f}  (p={p_lam_rank:.2e}, n={n_lam_rank})")
    print(f"  ρ(eff_rank, valley_depth) = {rho_rank_val:+.4f}  (p={p_rank_val:.2e}, n={n_rank_val})")
    if mediation_holds:
        print(f"  Mediation chain HOLDS ✓ (high λ → low rank → shallow valley)")
    else:
        print(f"  Mediation chain does NOT hold ✗")

    print(f"\nH4: Level spacing GOE test?")
    print(f"  r-ratio: all={statistics.mean([r['r_ratio'] for r in per_head if r['r_ratio']]):.4f}, "
          f"conformal={mean_conf_r:.4f}, non-conformal={mean_ncnf_r:.4f}")
    print(f"  GOE reference={GOE_R_RATIO}  Poisson reference={POISSON_R_RATIO}")
    print(f"  Conformal heads: {'closer to GOE ✓' if goe_conf else 'closer to Poisson'}")

    print(f"\nValley geometry (conformal heads, n={len(conformal_heads)}):")
    print(f"  start_dominates: {n_start_dom}  |  end_dominates: {n_end_dom}")
    print(f"  Note: max(start, end) in valley denominator — "
          f"{'start dominates majority' if n_start_dom > n_end_dom else 'end (recency) dominates majority'}")

    print(f"\nAnchor check (should reproduce exp-007):")
    conf_deltas = [r["delta_pos"] for r in conformal_heads if r["delta_pos"] is not None]
    if conf_deltas:
        d_med = statistics.median(conf_deltas)
        print(f"  Δ_pos median (conformal) = {d_med:.4f}  (exp-007: 0.2493)")
    print(f"  ρ(Δ_pos, valley_depth)   = {rho_delta_val:+.4f}  (p={p_delta_val:.2e})")
    print(f"  [exp-007 correlation ρ ~ 0.94 expected for Pythia-70m; GPT-2 may differ]")

    # ── save ─────────────────────────────────────────────────────────────────
    result = {
        "experiment": "exp-046",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H1": "ρ(λ_proxy, valley_depth) < 0 for conformal heads → sign anomaly in GPT-2",
            "H2": "Conformal heads have higher eff_rank in W_QK than non-conformal",
            "H3": "ρ(λ_proxy, eff_rank) < 0 AND ρ(eff_rank, valley) > 0 → mediation chain",
            "H4": "Conformal heads: GOE r-ratio ≈ 0.536; non-conformal: Poisson ≈ 0.386",
            "H0": "W_QK eigenvalues uncorrelated with attention geometry (null)",
        },
        "protocol": {
            "model": "gpt2",
            "attn_implementation": "eager",
            "seq_len": SEQ_LEN,
            "n_inputs": N_INPUTS,
            "rng_seed": RNG_SEED,
            "min_pos": MIN_POS,
            "max_dx": MAX_DX,
            "fit_range": [FIT_LOW, FIT_HIGH],
            "r2_threshold": R2_THRESHOLD,
            "valley_q_min": VALLEY_Q_MIN,
            "boundary_proxy": {
                "shallow_qpos_range": [SHALLOW_Q_LOW, SHALLOW_Q_HIGH],
                "deep_qpos_range": [DEEP_Q_LOW, DEEP_Q_HIGH],
                "dx_range": [PROXY_DX_LOW, PROXY_DX_HIGH],
                "definition": "G(shallow,Δx) / G(deep,Δx) - 1",
                "note": (
                    "Proxy for BCFT boundary parameter λ: measures ratio of "
                    "boundary-regime attention to bulk-regime attention at same Δx."
                ),
            },
            "eigenvalue": {
                "matrix": "W_QK_sym = (W_Q^T W_K + W_K^T W_Q) / 2  ∈ R^{64×64}",
                "eff_rank_def": "exp(Shannon entropy of |eigenvalue|/sum|eigenvalue|)",
                "r_ratio_def": "Oganesyan-Huse r-ratio on sorted eigenvalue spacings",
                "goe_reference": GOE_R_RATIO,
                "poisson_reference": POISSON_R_RATIO,
            },
        },
        "syk_predictions": {
            "delta": 0.25,
            "note": "SYK q=4 in D=1 predicts Δ=1/4; measured 0.2493 in exp-007",
        },
        "summary": {
            "total_heads": total_heads,
            "conformal_heads": len(conformal_heads),
            "syk_near_heads": len(syk_heads),
            # H1
            "H1_rho_lambda_valley": round(rho_lam_val, 4)  if math.isfinite(rho_lam_val)  else None,
            "H1_p":                 float(p_lam_val)        if math.isfinite(p_lam_val)    else None,
            "H1_n": n_lam_val,
            "H1_verdict": (
                "sign_anomaly_confirmed" if (math.isfinite(rho_lam_val) and rho_lam_val < -0.1)
                else "no_sign_anomaly" if (math.isfinite(rho_lam_val) and rho_lam_val > 0.1)
                else "inconclusive"
            ),
            # H2
            "H2_mean_eff_rank_conformal":     round(mean_conf_rank, 3) if math.isfinite(mean_conf_rank) else None,
            "H2_mean_eff_rank_non_conformal":  round(mean_ncnf_rank, 3) if math.isfinite(mean_ncnf_rank) else None,
            "H2_conformal_higher_rank":        (mean_conf_rank > mean_ncnf_rank) if (math.isfinite(mean_conf_rank) and math.isfinite(mean_ncnf_rank)) else None,
            "H2_rho_delta_rank":               round(rho_delta_rank, 4) if math.isfinite(rho_delta_rank) else None,
            # H3
            "H3_rho_lambda_rank":  round(rho_lam_rank, 4) if math.isfinite(rho_lam_rank) else None,
            "H3_p_lambda_rank":    float(p_lam_rank)      if math.isfinite(p_lam_rank)   else None,
            "H3_rho_rank_valley":  round(rho_rank_val, 4) if math.isfinite(rho_rank_val) else None,
            "H3_p_rank_valley":    float(p_rank_val)      if math.isfinite(p_rank_val)   else None,
            "H3_mediation_holds":  mediation_holds,
            # H4
            "H4_r_ratio_conformal":     round(mean_conf_r, 4) if math.isfinite(mean_conf_r) else None,
            "H4_r_ratio_non_conformal": round(mean_ncnf_r, 4) if math.isfinite(mean_ncnf_r) else None,
            "H4_conformal_closer_to_goe": goe_conf,
            # Valley geometry
            "valley_start_dominates_count": n_start_dom,
            "valley_end_dominates_count":   n_end_dom,
            # Anchor
            "conformal_delta_pos_median": round(statistics.median(conf_deltas), 6) if conf_deltas else None,
            "rho_delta_pos_valley":       round(rho_delta_val, 4) if math.isfinite(rho_delta_val) else None,
        },
        "per_head": per_head,
        "reference": {
            "exp007": {"conformal_heads": 44, "total": 144, "delta_median": 0.2493, "syk_near": 44},
            "exp045": {"pearson_r_pos_freq": 0.940, "note": "DFT approach, structural bias found"},
            "april17_sign_anomaly": {
                "pythia_2_8b_negative_layers": "18/21",
                "gptneo_2_7b_negative_layers": "24/31",
                "note": "ρ(λ_BCFT, valley) negative — BCFT 2D fit, Pythia/GPT-Neo on Modal A100",
            },
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
