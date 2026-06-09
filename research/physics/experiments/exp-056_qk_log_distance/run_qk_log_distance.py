"""
exp-056 — The log-distance representation: do query-key scores encode -2Δ·log|i-j|?

This is "Experiment A" from PAPER_BRIEF_NULL_CONE.md — the critical pre-writing
test for Paper C (the null-cone geometry paper).

THE CLAIM UNDER TEST
--------------------
The null-ray / embedding-space picture says the attention two-point function
A(i,j) ~ |i-j|^{-2Δ} is the CFT_1 two-point function on the projective null cone.
For the post-softmax weight to follow that power law, the PRE-softmax score must
be log-linear in token distance:

    q_i · k_j / sqrt(d_k)  ≈  -2Δ · log|i-j|  +  const          (*)

The power-law Δ in exp-007/exp-046 is measured from the POST-softmax attention
lag profile. This experiment goes one level deeper, to the raw scores, and asks
whether (*) holds HEAD BY HEAD — not just at the population level.

WHY THIS MATTERS
----------------
- If (*) holds for conformal heads: direct head-level evidence that the
  query-key computation implements the conformal Green's function — the
  null-ray inner-product interpretation is confirmed at the mechanism level.
- If (*) does NOT hold head-by-head: the null-ray picture survives only as a
  population/ensemble statement. That is an important falsification and must be
  written honestly into the paper, not buried.

PRE-STATED HYPOTHESES
---------------------
  H1 (log-linearity): For conformal heads (exp-046 delta_pos with R²>0.90),
     the mean pre-softmax score profile S(Δx) is strongly linear in log(Δx)
     over the fit range — mean R²_score > 0.90, and the score DECREASES with
     distance (slope α < 0).

  H2 (slope matches Δ): The implied Δ_score = -α/2 from the score-profile fit
     matches the independently-measured post-softmax delta_pos for the same
     head. Tests:
       (a) ρ(Δ_score, delta_pos) > 0 strong over conformal heads;
       (b) median |Δ_score - delta_pos| small.
     This is the strongest form of (*): the same Δ governs both the raw score
     decay and the attention power law.

  H3 (selectivity): Conformal heads are MORE log-linear (higher R²_score,
     higher |Pearson r(S, log Δx)|) than non-conformal heads. The log-distance
     representation is a property the conformal heads specifically have.

  H0 (null / falsification): Conformal heads do not show log-linear score
     profiles (R²_score not elevated, Δ_score uncorrelated with delta_pos).
     The null-ray picture would then hold only at the population level.

METHOD
------
For fidelity, scores are recomputed exactly as GPT-2 computes them:
    x   = ln_1(hidden_states[l])                  # pre-attention layernorm
    Q_h = x @ W_Q_h + b_Q_h ;  K_h = x @ W_K_h + b_K_h
    score_ij = (Q_h[i] · K_h[j]) / sqrt(d_k)
The per-head c_attn slicing matches exp-046. Random token inputs (same protocol
constants as exp-046) isolate the POSITION-dependent structure of the score from
token content — exactly the geometric content the null-ray claim is about.

The mean score profile S(Δx) is accumulated by lag (same lag binning as the
exp-007/046 post-softmax lag profile), then fit to S = α·log(Δx) + β over
[FIT_LOW, FIT_HIGH). delta_pos / r2_pos / conformal flags are joined from
exp-046 results.json so the comparison is head-aligned.
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

# ── protocol constants (identical to exp-046 for head-aligned comparison) ──────
SEQ_LEN      = 256
N_INPUTS     = 50
MAX_DX       = 56
MIN_POS      = 32
FIT_LOW      = 3
FIT_HIGH     = 50
R2_THRESHOLD = 0.90
RNG_SEED     = 42

# GPT-2 architecture
D_K      = 64
D_MODEL  = 768
N_LAYERS = 12
N_HEADS  = 12

EXP046_RESULTS = Path(
    "research/physics/experiments/exp-046_sign_anomaly_eigenvalue/results.json"
)
OUT_DIR      = Path("research/physics/experiments/exp-056_qk_log_distance")
RESULTS_FILE = OUT_DIR / "results.json"


# ── per-head q,k projection weights/biases from c_attn (matches exp-046) ───────

def head_projections(model: GPT2LMHeadModel) -> dict:
    """Return {(l,h): (W_Q, b_Q, W_K, b_K)} sliced from c_attn per head."""
    proj = {}
    for l in range(N_LAYERS):
        attn = model.transformer.h[l].attn
        W = attn.c_attn.weight.detach().float().numpy()   # (768, 2304)
        b = attn.c_attn.bias.detach().float().numpy()      # (2304,)
        for h in range(N_HEADS):
            qs, qe = h * D_K, (h + 1) * D_K
            ks, ke = D_MODEL + h * D_K, D_MODEL + (h + 1) * D_K
            proj[(l, h)] = (
                W[:, qs:qe], b[qs:qe],     # W_Q_h (768,64), b_Q_h (64,)
                W[:, ks:ke], b[ks:ke],     # W_K_h (768,64), b_K_h (64,)
            )
    return proj


# ── score lag-profile accumulation (same lag binning as exp-046) ──────────────

def accumulate_score_profile(score: np.ndarray, sum_arr: np.ndarray,
                             cnt_arr: np.ndarray, min_pos: int, max_dx: int) -> None:
    """Accumulate sum of pre-softmax scores at each causal lag Δx, in place."""
    n = score.shape[0]
    for dx in range(max_dx):
        s = 0.0
        c = 0
        for i in range(max(min_pos, dx), n):
            j = i - dx
            if 0 <= j < n:
                s += score[i, j]
                c += 1
        sum_arr[dx] += s
        cnt_arr[dx] += c


def fit_log_linear(dx_arr: np.ndarray, y_arr: np.ndarray,
                   low: int, high: int) -> tuple[float, float, float]:
    """
    Fit S(Δx) = α·log(Δx) + β over Δx ∈ [low, high).
    Returns (alpha, r2, pearson_r) where pearson_r = corr(S, log Δx).
    Δ_score = -alpha/2 (the score analog of the power-law exponent).
    """
    mask = (dx_arr >= low) & (dx_arr < high)
    x = dx_arr[mask].astype(float)
    y = y_arr[mask].astype(float)
    if mask.sum() < 5 or not np.all(np.isfinite(y)):
        return float("nan"), 0.0, float("nan")
    log_x = np.log(x)
    A = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    alpha = float(coeffs[1])
    if np.std(log_x) > 1e-12 and np.std(y) > 1e-12:
        pear = float(np.corrcoef(log_x, y)[0, 1])
    else:
        pear = float("nan")
    return alpha, r2, pear


def spearman(x_list, y_list) -> tuple[float, float, int]:
    pairs = [
        (xi, yi) for xi, yi in zip(x_list, y_list)
        if xi is not None and yi is not None
        and math.isfinite(xi) and math.isfinite(yi)
    ]
    if len(pairs) < 4:
        return float("nan"), float("nan"), len(pairs)
    xv, yv = zip(*pairs)
    rho, p = scipy_stats.spearmanr(xv, yv)
    return float(rho), float(p), len(pairs)


def load_exp046_perhead() -> dict:
    """Join key: (layer, head) -> {delta_pos, r2_pos, conformal, syk_near}."""
    with open(EXP046_RESULTS) as f:
        data = json.load(f)
    out = {}
    for r in data["per_head"]:
        out[(r["layer"], r["head"])] = {
            "delta_pos": r["delta_pos"],
            "r2_pos": r["r2_pos"],
            "conformal": r["conformal"],
            "syk_near": r["syk_near"],
        }
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("exp-056: log-distance representation of query-key scores (GPT-2)")
    print(f"  Protocol: N_INPUTS={N_INPUTS}, SEQ_LEN={SEQ_LEN}, RNG_SEED={RNG_SEED}")
    print(f"  Score profile fit: S(Δx)=α·log Δx+β over Δx∈[{FIT_LOW},{FIT_HIGH})")
    print(f"  Claim: score ≈ -2Δ·log|i-j| + const  →  Δ_score=-α/2 should match delta_pos")

    exp046 = load_exp046_perhead()

    print("Loading GPT-2 (eager)...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    vocab_size = model.config.vocab_size

    proj = head_projections(model)

    # accumulators per head
    score_sum = {(l, h): np.zeros(MAX_DX) for l in range(N_LAYERS) for h in range(N_HEADS)}
    score_cnt = {(l, h): np.zeros(MAX_DX) for l in range(N_LAYERS) for h in range(N_HEADS)}

    rng = np.random.default_rng(RNG_SEED)
    print(f"Running {N_INPUTS} forward passes (recomputing scores through ln_1)...", flush=True)
    for inp_idx in range(N_INPUTS):
        token_ids = torch.tensor(
            rng.integers(0, vocab_size, size=(1, SEQ_LEN)), dtype=torch.long
        )
        with torch.no_grad():
            out = model(token_ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple len N_LAYERS+1; hs[l] = input to block l

            for l in range(N_LAYERS):
                x = model.transformer.h[l].ln_1(hs[l])[0].float().numpy()  # (seq, d_model)
                for h in range(N_HEADS):
                    W_Q, b_Q, W_K, b_K = proj[(l, h)]
                    Q = x @ W_Q + b_Q          # (seq, 64)
                    K = x @ W_K + b_K          # (seq, 64)
                    score = (Q @ K.T) / math.sqrt(D_K)  # (seq, seq); [i,j]=q_i·k_j/√dk
                    accumulate_score_profile(score, score_sum[(l, h)], score_cnt[(l, h)],
                                             MIN_POS, MAX_DX)

        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS} done", flush=True)

    # ── per-head fits ──────────────────────────────────────────────────────────
    dx_arr = np.arange(MAX_DX)
    per_head = []
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            cnt = score_cnt[(l, h)]
            S = np.full(MAX_DX, np.nan)
            mask = cnt > 0
            S[mask] = score_sum[(l, h)][mask] / cnt[mask]

            alpha, r2_score, pear = fit_log_linear(dx_arr, S, FIT_LOW, FIT_HIGH)
            delta_score = -alpha / 2.0 if math.isfinite(alpha) else float("nan")

            ref = exp046.get((l, h), {})
            delta_pos = ref.get("delta_pos")
            conformal = ref.get("conformal", False)
            syk_near = ref.get("syk_near", False)

            def _r(v, n=6):
                return round(v, n) if (v is not None and math.isfinite(v)) else None

            per_head.append({
                "layer": l, "head": h,
                "alpha_score": _r(alpha),
                "delta_score": _r(delta_score),
                "r2_score": round(r2_score, 6),
                "pearson_S_logdx": _r(pear),
                "delta_pos": delta_pos,
                "conformal": conformal,
                "syk_near": syk_near,
                "delta_diff": _r(delta_score - delta_pos)
                              if (delta_pos is not None and math.isfinite(delta_score)) else None,
            })

    conf = [r for r in per_head if r["conformal"]]
    nonconf = [r for r in per_head if not r["conformal"]]
    syk = [r for r in per_head if r["syk_near"]]

    # ── H1: log-linearity of conformal score profiles ──────────────────────────
    conf_r2 = [r["r2_score"] for r in conf if r["r2_score"] is not None]
    conf_alpha = [r["alpha_score"] for r in conf if r["alpha_score"] is not None]
    mean_conf_r2 = statistics.mean(conf_r2) if conf_r2 else float("nan")
    frac_neg_slope = (sum(1 for a in conf_alpha if a < 0) / len(conf_alpha)) if conf_alpha else float("nan")
    H1_log_linear = (math.isfinite(mean_conf_r2) and mean_conf_r2 > R2_THRESHOLD
                     and math.isfinite(frac_neg_slope) and frac_neg_slope > 0.8)

    # ── H2: Δ_score matches delta_pos ───────────────────────────────────────────
    rho_dd, p_dd, n_dd = spearman(
        [r["delta_score"] for r in conf],
        [r["delta_pos"] for r in conf],
    )
    diffs = [abs(r["delta_diff"]) for r in conf if r["delta_diff"] is not None]
    median_abs_diff = statistics.median(diffs) if diffs else float("nan")
    H2_slope_matches = (math.isfinite(rho_dd) and rho_dd > 0.5)

    # SYK-near subset (Δ≈0.25): does delta_score also concentrate near 0.25?
    syk_dscore = [r["delta_score"] for r in syk if r["delta_score"] is not None and math.isfinite(r["delta_score"])]
    syk_dscore_median = statistics.median(syk_dscore) if syk_dscore else float("nan")

    # ── H3: selectivity (conformal more log-linear than non-conformal) ──────────
    nonconf_r2 = [r["r2_score"] for r in nonconf if r["r2_score"] is not None]
    mean_nonconf_r2 = statistics.mean(nonconf_r2) if nonconf_r2 else float("nan")
    conf_pear = [abs(r["pearson_S_logdx"]) for r in conf if r["pearson_S_logdx"] is not None]
    nonconf_pear = [abs(r["pearson_S_logdx"]) for r in nonconf if r["pearson_S_logdx"] is not None]
    mean_conf_pear = statistics.mean(conf_pear) if conf_pear else float("nan")
    mean_nonconf_pear = statistics.mean(nonconf_pear) if nonconf_pear else float("nan")
    # Selectivity requires a MEANINGFUL margin, not a noise-level difference.
    # If conformal and non-conformal are equally log-linear, log-linearity is a
    # universal substrate (like GOE), not a selective conformal feature — that is
    # an informative NULL, not a confirmation.
    R2_SELECTIVITY_MARGIN = 0.02
    H3_selective = (math.isfinite(mean_conf_r2) and math.isfinite(mean_nonconf_r2)
                    and (mean_conf_r2 - mean_nonconf_r2) > R2_SELECTIVITY_MARGIN)

    # ── print ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"=== RESULTS === ({len(conf)}/144 conformal, {len(syk)} SYK-near; from exp-046 flags)")
    print(f"{'='*64}")

    print(f"\nH1: Are conformal score profiles log-linear in distance?")
    print(f"  mean R²_score (conformal)   = {mean_conf_r2:.4f}   (threshold {R2_THRESHOLD})")
    print(f"  fraction with α<0 (decay)   = {frac_neg_slope:.2f}")
    print(f"  → {'CONFIRMED ✓' if H1_log_linear else 'NOT confirmed ✗'}")

    print(f"\nH2: Does the score slope encode the same Δ as the attention power law?")
    print(f"  ρ(Δ_score, delta_pos)       = {rho_dd:+.4f}  (p={p_dd:.2e}, n={n_dd})")
    print(f"  median |Δ_score - delta_pos|= {median_abs_diff:.4f}")
    print(f"  Δ_score median (SYK-near)   = {syk_dscore_median:.4f}  (SYK predicts 0.25)")
    print(f"  → {'CONFIRMED ✓' if H2_slope_matches else 'NOT confirmed ✗'}")

    print(f"\nH3: Is log-linearity selective to conformal heads?")
    print(f"  mean R²_score: conformal={mean_conf_r2:.4f}, non-conformal={mean_nonconf_r2:.4f}")
    print(f"  mean |Pearson(S,logΔx)|: conformal={mean_conf_pear:.4f}, non-conformal={mean_nonconf_pear:.4f}")
    print(f"  → {'SELECTIVE ✓' if H3_selective else 'NOT selective — log-linearity is a UNIVERSAL substrate (informative null)'}")

    # a few representative SYK-near heads
    print(f"\nRepresentative SYK-near heads (delta_pos ≈ 0.25):")
    syk_sorted = sorted(syk, key=lambda r: abs((r["delta_pos"] or 1) - 0.25))[:8]
    print(f"  {'L,H':>6} {'delta_pos':>10} {'delta_score':>12} {'R2_score':>9} {'r(S,logdx)':>11}")
    for r in syk_sorted:
        print(f"  {r['layer']:>2},{r['head']:<3} {r['delta_pos']:>10.4f} "
              f"{(r['delta_score'] if r['delta_score'] is not None else float('nan')):>12.4f} "
              f"{r['r2_score']:>9.4f} "
              f"{(r['pearson_S_logdx'] if r['pearson_S_logdx'] is not None else float('nan')):>11.4f}")

    # ── save ─────────────────────────────────────────────────────────────────────
    result = {
        "experiment": "exp-056",
        "title": "Log-distance representation of query-key scores (Experiment A for Paper C)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "claim_under_test": "q_i·k_j/sqrt(d_k) ≈ -2Δ·log|i-j| + const for conformal heads",
        "hypotheses": {
            "H1": "Conformal score profiles log-linear in log(Δx): mean R²_score>0.90, slope<0",
            "H2": "Δ_score=-α/2 matches delta_pos: ρ(Δ_score,delta_pos)>0.5",
            "H3": "Conformal heads more log-linear than non-conformal (selectivity)",
            "H0": "Null: no head-level log-distance representation (population-only)",
        },
        "protocol": {
            "model": "gpt2", "attn_implementation": "eager",
            "seq_len": SEQ_LEN, "n_inputs": N_INPUTS, "rng_seed": RNG_SEED,
            "min_pos": MIN_POS, "max_dx": MAX_DX, "fit_range": [FIT_LOW, FIT_HIGH],
            "score_def": "score_ij=(Q_h[i]·K_h[j])/sqrt(d_k); Q=ln_1(h_l)@W_Q+b_Q, K=ln_1(h_l)@W_K+b_K",
            "conformal_flags_from": "exp-046 results.json (head-aligned join)",
            "input": "random token ids (isolates position-dependent score structure)",
        },
        "summary": {
            "total_heads": 144,
            "conformal_heads": len(conf),
            "syk_near_heads": len(syk),
            "H1_mean_r2_score_conformal": round(mean_conf_r2, 6) if math.isfinite(mean_conf_r2) else None,
            "H1_frac_negative_slope": round(frac_neg_slope, 4) if math.isfinite(frac_neg_slope) else None,
            "H1_verdict": "log_linear_confirmed" if H1_log_linear else "not_confirmed",
            "H2_rho_delta_score_delta_pos": round(rho_dd, 4) if math.isfinite(rho_dd) else None,
            "H2_p": float(p_dd) if math.isfinite(p_dd) else None,
            "H2_n": n_dd,
            "H2_median_abs_delta_diff": round(median_abs_diff, 6) if math.isfinite(median_abs_diff) else None,
            "H2_delta_score_median_syk_near": round(syk_dscore_median, 6) if math.isfinite(syk_dscore_median) else None,
            "H2_verdict": "slope_matches_delta" if H2_slope_matches else "not_confirmed",
            "H3_mean_r2_score_conformal": round(mean_conf_r2, 6) if math.isfinite(mean_conf_r2) else None,
            "H3_mean_r2_score_non_conformal": round(mean_nonconf_r2, 6) if math.isfinite(mean_nonconf_r2) else None,
            "H3_mean_abs_pearson_conformal": round(mean_conf_pear, 6) if math.isfinite(mean_conf_pear) else None,
            "H3_mean_abs_pearson_non_conformal": round(mean_nonconf_pear, 6) if math.isfinite(mean_nonconf_pear) else None,
            "H3_verdict": "selective" if H3_selective else "not_selective_log_linearity_is_universal",
        },
        "per_head": per_head,
        "reference": {
            "exp007": {"conformal_heads": 44, "delta_median": 0.2493},
            "exp046": {"note": "conformal/syk_near flags and delta_pos joined from here"},
            "paper_brief": "research/physics/PAPER_BRIEF_NULL_CONE.md (Experiment A)",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
