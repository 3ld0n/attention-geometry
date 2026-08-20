"""exp-123 — Theory-of-A Level 3: Layer-Norm-Corrected Quantitative Closure.

Pre-registration committed to attention-geometry at c4cac26 before this script
was written or run. Analysis-only: GPT-2 small weights, no forward passes.

Corrects exp-122 by applying each block's layer norm (ln_1) to
h̄^(0)[i] = emb_mean + wpe[i] before W_V projection, then convolving with
the analytic causal conformal attention kernel ā(dx) ~ dx^{-2Δ} (Δ=0.249).

Tests whether the LN-corrected d_model σ_out at L2H1 improves over
exp-122's 0.214 toward σ_delta = 0.249 (exp-117).

Ariel — August 20, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

# --- constants matching the census protocol (exp-107/110/112/122) ---
SEQ_LEN = 512
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256
WINDOW = np.arange(FIT_LO, FIT_HI + 1)
NW = len(WINDOW)
DELTA = 0.249  # census exponent from exp-107/118
LN_EPS = 1e-5  # GPT-2's default layer_norm_epsilon

STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]


# --- utilities ---------------------------------------------------------------

def ols_slope(y: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[1])


def ols_r2(y: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ c
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def pooled_window_profile(mat: np.ndarray) -> np.ndarray:
    """mat: (L, L) — compute lag-averaged profile at lags 8..256,
    averaging over query positions i >= DEEP_LO = 256."""
    out = np.empty(NW)
    for w, dx in enumerate(WINDOW):
        diag = np.diagonal(mat, offset=-dx)  # mat[i, i-dx] for i=dx..511
        k_lo = max(DEEP_LO, dx) - dx        # first entry with i >= 256
        out[w] = diag[k_lo:].mean()
    return out


def build_causal_power_law_attention(seq_len: int, delta: float) -> np.ndarray:
    """A[i, j] = (i-j)^{-2Δ} for j < i (causal), normalized per row.
    Row 0 and any row with no causal predecessors are all-zero."""
    A = np.zeros((seq_len, seq_len), dtype=np.float64)
    for i in range(1, seq_len):
        lags = np.arange(1, i + 1, dtype=np.float64)
        weights = lags ** (-2.0 * delta)
        weights /= weights.sum()
        A[i, :i] = weights[::-1]
    return A


def apply_ln(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply layer norm to each row of x.
    x: (N, D), gamma/beta: (D,) -> output: (N, D)"""
    mu = x.mean(axis=1, keepdims=True)
    diff = x - mu
    sigma = np.sqrt((diff ** 2).mean(axis=1, keepdims=True) + LN_EPS)
    return gamma[None, :] * (diff / sigma) + beta[None, :]


def main() -> None:
    print("exp-123: layer-norm-corrected Level-3 quantitative closure", flush=True)
    print("Pre-registration commit: c4cac26 (attention-geometry, pushed before this script was written)\n",
          flush=True)

    # --- load GPT-2 small weights (cpu only; no forward pass needed) -----------
    device = "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()

    n_head = cfg.num_attention_heads     # 12
    d_model = cfg.hidden_size            # 768
    d_head = d_model // n_head           # 64

    # --- extract embeddings --------------------------------------------------
    wpe = model.transformer.wpe.weight.detach().cpu().numpy().astype(np.float64)
    pos_emb = wpe[:SEQ_LEN, :]  # (512, 768)
    wte = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float64)
    emb_mean = wte.mean(axis=0)  # (768,) — mean token embedding

    print(f"pos_emb shape: {pos_emb.shape}  norm[0]={np.linalg.norm(pos_emb[0]):.4f}", flush=True)
    print(f"emb_mean norm: {np.linalg.norm(emb_mean):.4f}", flush=True)

    # h̄^(0)[i] = emb_mean + wpe[i], shape (512, 768)
    h0 = pos_emb + emb_mean[None, :]  # broadcast emb_mean over positions
    print(f"h̄^(0) mean norm: {np.linalg.norm(h0, axis=1).mean():.4f}", flush=True)

    # Reference: bare pos_emb (no LN, no emb_mean) — same as exp-122 baseline
    C_pos_raw = pooled_window_profile(pos_emb @ pos_emb.T)
    sigma_pos_raw = -ols_slope(C_pos_raw, WINDOW)
    r2_pos_raw = ols_r2(C_pos_raw, WINDOW)
    print(f"\nBare pos_emb: σ={sigma_pos_raw:.3f}  R²={r2_pos_raw:.3f}  "
          f"C(8)={C_pos_raw[0]:.1f}  C(256)={C_pos_raw[-1]:.1f}", flush=True)

    # --- build analytic causal conformal attention matrix ----------------------
    print(f"\nBuilding causal power-law attention matrix (Δ={DELTA}, seq={SEQ_LEN})...",
          flush=True)
    A = build_causal_power_law_attention(SEQ_LEN, DELTA)
    row_sums = A.sum(axis=1)
    print(f"  Row-sum check: rows 1..5 sums = {row_sums[1:6].round(6)}", flush=True)

    # --- per structural head: LN-corrected pos_emb propagation ----------------
    print("\n=== Per-head LN-corrected propagation ===", flush=True)
    head_results = {}

    for (layer, head) in STRUCTURAL:
        name = f"L{layer}H{head}"

        # Apply layer norm from this block — each head uses its own block's LN
        ln_gamma = (model.transformer.h[layer].ln_1.weight
                    .detach().cpu().numpy().astype(np.float64))
        ln_beta = (model.transformer.h[layer].ln_1.bias
                   .detach().cpu().numpy().astype(np.float64))
        h0_ln = apply_ln(h0, ln_gamma, ln_beta)  # (512, 768)

        # Extract W_V for this head
        c_attn_w = (model.transformer.h[layer].attn.c_attn.weight
                    .detach().cpu().numpy().astype(np.float64))
        W_V_full = c_attn_w[:, 2 * d_model: 3 * d_model]       # (768, 768)
        W_V_h = W_V_full[:, head * d_head: (head + 1) * d_head] # (768, 64)

        # Add value bias
        c_attn_b = (model.transformer.h[layer].attn.c_attn.bias
                    .detach().cpu().numpy().astype(np.float64))
        b_V_h = c_attn_b[2 * d_model: 3 * d_model][head * d_head: (head + 1) * d_head]

        # Project LN-corrected h̄^(0) through W_V_h
        v = h0_ln @ W_V_h + b_V_h[None, :]  # (512, 64)

        # W_O projection
        c_proj_w = (model.transformer.h[layer].attn.c_proj.weight
                    .detach().cpu().numpy().astype(np.float64))
        W_O_h = c_proj_w[:, head * d_head: (head + 1) * d_head]  # (768, 64)

        # --- Primary: d_head space ---
        out_dh = A @ v  # (512, 64)
        norms_dh = np.linalg.norm(out_dh, axis=1, keepdims=True)
        safe_mask = (norms_dh[:, 0] > 1e-12)
        out_norm_dh = np.zeros_like(out_dh)
        out_norm_dh[safe_mask] = out_dh[safe_mask] / norms_dh[safe_mask]
        corr_dh = out_norm_dh @ out_norm_dh.T
        C_dh = pooled_window_profile(corr_dh)
        sigma_dh = -ols_slope(C_dh, WINDOW)
        r2_dh = ols_r2(C_dh, WINDOW)

        # --- Secondary: d_model space ---
        v_out = v @ W_O_h.T   # (512, 768)
        out_dm = A @ v_out    # (512, 768)
        norms_dm = np.linalg.norm(out_dm, axis=1, keepdims=True)
        out_norm_dm = np.zeros_like(out_dm)
        out_norm_dm[safe_mask] = out_dm[safe_mask] / norms_dm[safe_mask]
        corr_dm = out_norm_dm @ out_norm_dm.T
        C_dm = pooled_window_profile(corr_dm)
        sigma_dm = -ols_slope(C_dm, WINDOW)
        r2_dm = ols_r2(C_dm, WINDOW)

        # P1/P2 verdicts (on d_model result, the primary comparison to exp-122)
        exp122_baseline = {"L2H1": 0.214, "L3H4": 0.175, "L5H0": 0.203, "L7H11": 0.241, "L10H8": 0.180}
        improved = sigma_dm > exp122_baseline[name]

        head_results[name] = {
            "d_head_space": {
                "sigma_out": float(sigma_dh),
                "R2": float(r2_dh),
                "C_at_dx8": float(C_dh[0]),
                "C_at_dx256": float(C_dh[-1]),
            },
            "d_model_space": {
                "sigma_out": float(sigma_dm),
                "R2": float(r2_dm),
                "C_at_dx8": float(C_dm[0]),
                "C_at_dx256": float(C_dm[-1]),
            },
            "exp122_baseline_dmodel": float(exp122_baseline[name]),
            "improved_over_exp122": bool(improved),
        }

        print(f"  {name}: d_head σ={sigma_dh:.4f} R²={r2_dh:.3f}  "
              f"d_model σ={sigma_dm:.4f} R²={r2_dm:.3f}  "
              f"exp122_baseline={exp122_baseline[name]:.3f}  "
              f"improved={'YES' if improved else 'NO'}",
              flush=True)

    # --- registered prediction verdicts ---
    l2h1_dm = head_results["L2H1"]["d_model_space"]["sigma_out"]
    l2h1_dh = head_results["L2H1"]["d_head_space"]["sigma_out"]
    l2h1_r2 = head_results["L2H1"]["d_head_space"]["R2"]

    P1_ok = l2h1_dm > 0.22    # directional improvement over exp-122's 0.214
    P2_ok = (0.22 <= l2h1_dm <= 0.28)   # quantitative match window
    P3_kill = l2h1_dm <= 0.20  # no improvement — route cannot close the gap

    # Count heads improved
    n_improved = sum(v["improved_over_exp122"] for v in head_results.values())

    if P3_kill:
        overall = "P3_KILL: LN correction does not improve d_model result"
        verdict = "falsified"
    elif P2_ok:
        overall = "P2_CONFIRMED: LN-corrected d_model σ_out falls in quantitative match window"
        verdict = "confirmed"
    elif P1_ok:
        overall = "P1_CONFIRMED: directional improvement, but outside tight quantitative window"
        verdict = "partial"
    else:
        overall = "INCONCLUSIVE"
        verdict = "inconclusive"

    print(f"\n=== Registered verdicts ===", flush=True)
    print(f"  P1 (d_model L2H1 > 0.22): {'OK' if P1_ok else 'FAIL'}  (σ={l2h1_dm:.4f})", flush=True)
    print(f"  P2 (d_model L2H1 ∈ [0.22, 0.28]): {'OK' if P2_ok else 'FAIL'}  (σ={l2h1_dm:.4f})", flush=True)
    print(f"  P3 kill (d_model L2H1 ≤ 0.20): {'FIRED' if P3_kill else 'not fired'}", flush=True)
    print(f"  Heads improved over exp-122: {n_improved}/5", flush=True)
    print(f"  Overall: {overall}", flush=True)

    # --- save results ---------------------------------------------------------
    results = {
        "exp": "exp-123",
        "date": "2026-08-20",
        "prereg_commit": "c4cac26",
        "prereg_evidence": "git-attested — committed to attention-geometry at c4cac26 before run script was written",
        "model": "gpt2",
        "method": (
            "analysis-only; LN(emb_mean + pos_emb) projected through layer's W_V_h, "
            "then convolved with analytic causal conformal kernel A[i,j]=(i-j)^{-2Δ} Δ=0.249"
        ),
        "delta": DELTA,
        "seq_len": SEQ_LEN,
        "fit_lags": [FIT_LO, FIT_HI],
        "correction_vs_exp122": "applies LN(emb_mean + pos_emb) using each block's ln_1 instead of bare pos_emb",
        "pos_emb_reference": {
            "sigma_bare_pos_emb": float(sigma_pos_raw),
            "R2_bare_pos_emb": float(r2_pos_raw),
        },
        "structural_heads": head_results,
        "registered_verdicts": {
            "P1": {
                "criterion": "sigma_out (d_model) for L2H1 > 0.22",
                "L2H1_sigma_dmodel": float(l2h1_dm),
                "ok": bool(P1_ok),
            },
            "P2": {
                "criterion": "sigma_out (d_model) for L2H1 in [0.22, 0.28]",
                "L2H1_sigma_dmodel": float(l2h1_dm),
                "ok": bool(P2_ok),
            },
            "P3_kill": {
                "criterion": "sigma_out (d_model) for L2H1 <= 0.20",
                "fired": bool(P3_kill),
            },
            "n_improved": n_improved,
            "overall": overall,
        },
        "exp122_comparison": {
            "L2H1_exp122_dmodel": 0.214,
            "L2H1_exp123_dmodel": float(l2h1_dm),
            "L2H1_sigma_delta": 0.249,
            "gap_closed": float(l2h1_dm) > 0.214,
        },
        "status": "complete",
        "verdict": verdict,
    }

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
