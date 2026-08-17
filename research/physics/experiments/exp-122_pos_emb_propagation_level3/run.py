"""exp-122 — Theory-of-A Level 3: Position-Embedding Propagation Route.

Pre-registration committed to attention-geometry at 73b6138 before this script
was written or run. Analysis-only: GPT-2 small weights, no forward passes.

Tests whether GPT-2's learned pos_emb, projected through W_V and convolved with
the analytic causal conformal attention kernel ā(dx) ~ dx^{-2Δ} (Δ=0.249), produces
a position-correlation profile with slope σ_out ≈ Δ ≈ 0.249 — matching the
accumulated-delta slope found in exp-117 (σ_delta=0.249 at L2H1).

Ariel — August 17, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

# --- constants matching the census protocol (exp-107/110/112) ---
SEQ_LEN = 512
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256
WINDOW = np.arange(FIT_LO, FIT_HI + 1)
NW = len(WINDOW)
DELTA = 0.249  # census exponent from exp-107/118

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
        lags = np.arange(1, i + 1, dtype=np.float64)  # distances 1..i
        weights = lags ** (-2.0 * delta)
        weights /= weights.sum()
        # A[i, j] for j = i-1, i-2, ..., 0 → weight[0] goes to j=i-1
        A[i, :i] = weights[::-1]
    return A


def main() -> None:
    print("exp-122: pos_emb propagation route — Level-3 corrected target", flush=True)
    print(f"Pre-registration commit: 73b6138 (pushed before this script was written)\n",
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

    # --- extract positional embeddings ----------------------------------------
    wpe = model.transformer.wpe.weight.detach().cpu().numpy().astype(np.float64)
    pos_emb = wpe[:SEQ_LEN, :]  # (512, 768)
    print(f"pos_emb shape: {pos_emb.shape}  norm[0]={np.linalg.norm(pos_emb[0]):.4f}",
          flush=True)

    # Reference: pos_emb raw correlation (no projection, same as exp-117)
    C_pos_raw = pooled_window_profile(pos_emb @ pos_emb.T)
    sigma_pos_raw = -ols_slope(C_pos_raw, WINDOW)
    r2_pos_raw = ols_r2(C_pos_raw, WINDOW)
    print(f"pos_emb raw: σ={sigma_pos_raw:.3f}  R²={r2_pos_raw:.3f}  "
          f"C(8)={C_pos_raw[0]:.1f}  C(256)={C_pos_raw[-1]:.1f}", flush=True)

    # --- build analytic causal conformal attention matrix ----------------------
    print(f"\nBuilding causal power-law attention matrix (Δ={DELTA}, seq={SEQ_LEN})...",
          flush=True)
    A = build_causal_power_law_attention(SEQ_LEN, DELTA)
    # Sanity check: each non-zero row sums to 1
    row_sums = A.sum(axis=1)
    print(f"  Row-sum check: rows 1..10 sums = {row_sums[1:11].round(6)}", flush=True)
    print(f"  Row 511 sum = {row_sums[511]:.6f}", flush=True)

    # --- per structural head: project pos_emb through W_V and convolve --------
    print("\n=== Per-head pos_emb propagation ===", flush=True)
    head_results = {}

    for (layer, head) in STRUCTURAL:
        name = f"L{layer}H{head}"

        # Extract W_V for this head: c_attn.weight has shape (768, 2304)
        # Column layout: [W_Q | W_K | W_V], each (768, 768); head h uses cols h*64..(h+1)*64
        c_attn_w = (model.transformer.h[layer].attn.c_attn.weight
                    .detach().cpu().numpy().astype(np.float64))
        W_V_full = c_attn_w[:, 2 * d_model: 3 * d_model]        # (768, 768)
        W_V_h = W_V_full[:, head * d_head: (head + 1) * d_head]  # (768, 64)

        # Project pos_emb through W_V_h: v[j] = pos_emb[j] @ W_V_h, shape (512, 64)
        v = pos_emb @ W_V_h  # (512, 64)

        # Apply also the c_attn bias (value bias component)
        c_attn_b = (model.transformer.h[layer].attn.c_attn.bias
                    .detach().cpu().numpy().astype(np.float64))
        b_V_h = c_attn_b[2 * d_model: 3 * d_model][head * d_head: (head + 1) * d_head]
        v = v + b_V_h[None, :]  # broadcast bias

        # Apply W_O projection (the output projection after attention)
        # c_proj.weight: shape (768, 768); head h uses rows h*64..(h+1)*64
        c_proj_w = (model.transformer.h[layer].attn.c_proj.weight
                    .detach().cpu().numpy().astype(np.float64))
        W_O_h = c_proj_w[head * d_head: (head + 1) * d_head, :].T  # (64, 768).T = (768, 64)
        # Actually c_proj maps (n_head * d_head) → d_model, so c_proj.weight: (768, 768)
        # The head h contribution uses c_proj.weight[:, h*d_head:(h+1)*d_head]
        # We project v (in d_head space) to d_model space:
        # out_full = v @ W_O_h.T where W_O_h: (d_model, d_head) → (d_head, d_model) needed
        # c_proj.weight[i,j]: output_i from input_j → W_O: (768, 768), head h uses cols h*64:(h+1)*64
        # So W_O_h = c_proj.weight[:, h*d_head:(h+1)*d_head]  shape (768, 64)
        W_O_h_correct = c_proj_w[:, head * d_head: (head + 1) * d_head]  # (768, 64)

        # Note: for measuring position-correlation of the head's VALUE output,
        # we have two choices:
        # (a) v[j] in d_head=64 space (pre-W_O): captures value geometry
        # (b) v_out[j] = v[j] @ W_O_h_correct.T in d_model=768 space (post-W_O)
        # We measure both; the primary is (a) since W_O is a linear map and doesn't
        # change the slope of position correlations (only the inner product scale).

        # --- Primary: d_head space ---
        # Convolve: out_dh[i] = Σ_j A[i,j] × v[j], shape (512, 64)
        out_dh = A @ v  # (512, 64)

        # Normalize row-wise (skip row 0 where A is all-zero)
        norms_dh = np.linalg.norm(out_dh, axis=1, keepdims=True)  # (512, 1)
        safe_mask = (norms_dh[:, 0] > 1e-12)
        out_norm_dh = np.zeros_like(out_dh)
        out_norm_dh[safe_mask] = out_dh[safe_mask] / norms_dh[safe_mask]

        # Correlation matrix (512, 512)
        corr_dh = out_norm_dh @ out_norm_dh.T  # inner products of normalized vectors

        # Pooled profile at lags 8..256
        C_dh = pooled_window_profile(corr_dh)

        sigma_dh = -ols_slope(C_dh, WINDOW)
        r2_dh = ols_r2(C_dh, WINDOW)

        # --- Secondary: d_model space ---
        v_out = v @ W_O_h_correct.T  # (512, 768)
        out_dm = A @ v_out           # (512, 768)
        norms_dm = np.linalg.norm(out_dm, axis=1, keepdims=True)
        out_norm_dm = np.zeros_like(out_dm)
        out_norm_dm[safe_mask] = out_dm[safe_mask] / norms_dm[safe_mask]
        corr_dm = out_norm_dm @ out_norm_dm.T
        C_dm = pooled_window_profile(corr_dm)
        sigma_dm = -ols_slope(C_dm, WINDOW)
        r2_dm = ols_r2(C_dm, WINDOW)

        # P1/P2 verdicts (on primary d_head result)
        p1_ok = (sigma_dh > 0.10) and (r2_dh >= 0.70)
        p2_ok = (0.20 <= sigma_dh <= 0.30)

        head_results[name] = {
            "d_head_space": {
                "sigma_out": float(sigma_dh),
                "R2": float(r2_dh),
                "C_at_dx8": float(C_dh[0]),
                "C_at_dx32": float(C_dh[24]),
                "C_at_dx128": float(C_dh[120]),
                "C_at_dx256": float(C_dh[-1]),
            },
            "d_model_space": {
                "sigma_out": float(sigma_dm),
                "R2": float(r2_dm),
                "C_at_dx8": float(C_dm[0]),
                "C_at_dx256": float(C_dm[-1]),
            },
            "P1_ok": bool(p1_ok),
            "P2_ok": bool(p2_ok),
        }

        print(f"  {name}: d_head σ={sigma_dh:.4f} R²={r2_dh:.3f}  "
              f"C(8)={C_dh[0]:.4f} C(256)={C_dh[-1]:.4f}  "
              f"P1={'OK' if p1_ok else 'FAIL'} P2={'OK' if p2_ok else 'FAIL'}",
              flush=True)
        print(f"  {name}: d_model σ={sigma_dm:.4f} R²={r2_dm:.3f}", flush=True)

    # --- registered verdicts --------------------------------------------------
    P1_ok_count = sum(v["P1_ok"] for v in head_results.values())
    P2_ok_count = sum(v["P2_ok"] for v in head_results.values())
    l2h1 = head_results["L2H1"]["d_head_space"]
    sigma_l2h1 = l2h1["sigma_out"]
    r2_l2h1 = l2h1["R2"]

    P3_kill = (sigma_l2h1 < 0.05) or (P1_ok_count == 0)

    if P3_kill:
        overall = "P3_KILL: route falsified"
        verdict = "falsified"
    elif P2_ok_count >= 3:
        overall = "P2_CONFIRMED: slope matches census exponent"
        verdict = "confirmed"
    elif P1_ok_count >= 3:
        overall = "P1_CONFIRMED_PARTIAL: positive slope but not matching exponent"
        verdict = "partial"
    else:
        overall = "INCONCLUSIVE"
        verdict = "inconclusive"

    print(f"\n=== Registered verdicts ===", flush=True)
    print(f"  P1 (σ > 0.10, R²≥0.70 for L2H1): "
          f"{'OK' if head_results['L2H1']['P1_ok'] else 'FAIL'}", flush=True)
    print(f"  P2 (σ ∈ [0.20, 0.30] for L2H1): "
          f"{'OK' if head_results['L2H1']['P2_ok'] else 'FAIL'}", flush=True)
    print(f"  P3 kill: {'FIRED' if P3_kill else 'not fired'}", flush=True)
    print(f"  Overall: {overall}", flush=True)

    # --- save results ---------------------------------------------------------
    results = {
        "exp": "exp-122",
        "date": "2026-08-17",
        "prereg_commit": "73b6138",
        "prereg_evidence": "git-attested — committed to attention-geometry before run script was written",
        "model": "gpt2",
        "method": (
            "analysis-only; pos_emb projected through W_V_h then convolved with "
            "analytic causal conformal attention kernel A[i,j]=(i-j)^{-2Δ} Δ=0.249"
        ),
        "delta": DELTA,
        "seq_len": SEQ_LEN,
        "fit_lags": [FIT_LO, FIT_HI],
        "pos_emb_reference": {
            "sigma_raw": float(sigma_pos_raw),
            "R2_raw": float(r2_pos_raw),
        },
        "structural_heads": head_results,
        "registered_verdicts": {
            "P1": {
                "criterion": "sigma_out > 0.10 and R2 >= 0.70 for L2H1",
                "L2H1_sigma": float(sigma_l2h1),
                "L2H1_R2": float(r2_l2h1),
                "ok": bool(head_results["L2H1"]["P1_ok"]),
            },
            "P2": {
                "criterion": "sigma_out in [0.20, 0.30] for L2H1",
                "L2H1_sigma": float(sigma_l2h1),
                "ok": bool(head_results["L2H1"]["P2_ok"]),
            },
            "P3_kill": {
                "criterion": "sigma_out < 0.05 for L2H1 OR P1_ok_count == 0",
                "fired": bool(P3_kill),
            },
            "overall": overall,
        },
        "status": "complete",
        "verdict": verdict,
    }

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
