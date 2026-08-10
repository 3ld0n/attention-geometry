"""exp-117 — Theory-of-A Level 3: mean-field score from embedding-layer geometry.

Pre-registration committed to attention-geometry at 8a22802 before this script
was written or run. Analysis-only: no forward passes needed beyond weight access.

Tests whether S_mf(dx) at layer 0 — computed from embedding weights alone
(emb_mean + wpe projected through block-0 W_Q/W_K) — has the same power-law
slope sigma_mf as the layer-2+ mean-field profiles (exp-113).

Ariel — August 10, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"
EXP113 = HERE.parent / "exp-113_mean_field_reduction"

sys.path.insert(0, str(EXP112))
from measure_scores import pooled_window_profile, ols_slope, WINDOW  # noqa: E402

# Structural heads identified from exp-107/109 (random-token SYK-window population)
STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]

SEQ_LEN = 512
FIT_LO, FIT_HI = 8, 256


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Apply layer norm: (x - mean) / std * weight + bias. x: (N, D)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def power_law_r2(y: np.ndarray, lags: np.ndarray) -> float:
    """R^2 of linear OLS fit of y vs log(lags)."""
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ c
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    print("exp-117: Theory-of-A Level 3 — embedding-layer derivation", flush=True)
    print(f"Pre-registration commit: 8a22802 (pushed before this run)\n", flush=True)

    # Load GPT-2
    device = "cpu"  # weight access only, no forward passes
    cfg = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()

    n_head = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_head = d_model // n_head
    scaling = d_head ** -0.5

    # Extract embedding matrices (numpy, float64 for precision)
    wte = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float64)
    wpe = model.transformer.wpe.weight.detach().cpu().numpy().astype(np.float64)
    print(f"wte shape: {wte.shape}  wpe shape: {wpe.shape}", flush=True)

    # emb_mean = mean over ALL vocabulary embeddings (random-token protocol = uniform)
    emb_mean = wte.mean(axis=0)  # (768,)
    print(f"emb_mean norm: {np.linalg.norm(emb_mean):.4f}", flush=True)
    print(f"wpe[0] norm:   {np.linalg.norm(wpe[0]):.4f}", flush=True)

    # h̄^(0)_i = emb_mean + wpe[i] for i = 0..511
    hbar0 = emb_mean[None, :] + wpe[:SEQ_LEN, :]  # (512, 768)
    print(f"hbar0 shape: {hbar0.shape}", flush=True)

    # Block-0 layer norm parameters
    block0 = model.transformer.h[0]
    ln1_w = block0.ln_1.weight.detach().cpu().numpy().astype(np.float64)
    ln1_b = block0.ln_1.bias.detach().cpu().numpy().astype(np.float64)
    ln1_eps = block0.ln_1.eps

    # Apply LN to h̄^(0)
    hbar0_ln = layer_norm(hbar0, ln1_w, ln1_b, eps=ln1_eps)  # (512, 768)
    print(f"hbar0_ln norms: min={np.linalg.norm(hbar0_ln, axis=-1).min():.4f}, "
          f"max={np.linalg.norm(hbar0_ln, axis=-1).max():.4f}", flush=True)

    # Block-0 W_Q, W_K (from c_attn: (768, 768*3) -> first 768 cols = W_QKV)
    # GPT-2 c_attn combines Q, K, V: weight is (768, 2304), bias is (2304,)
    c_attn_w = block0.attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)
    c_attn_b = block0.attn.c_attn.bias.detach().cpu().numpy().astype(np.float64)
    # Columns: [W_Q | W_K | W_V] each of size d_model = 768
    W_Q = c_attn_w[:, :d_model]           # (768, 768)
    W_K = c_attn_w[:, d_model:2*d_model]  # (768, 768)
    b_Q = c_attn_b[:d_model]
    b_K = c_attn_b[d_model:2*d_model]

    # Compute all queries and keys from mean-field input at layer 0
    Q_all = hbar0_ln @ W_Q + b_Q  # (512, 768)
    K_all = hbar0_ln @ W_K + b_K  # (512, 768)

    # Reshape per head: (n_head, 512, d_head)
    Q_heads = Q_all.reshape(SEQ_LEN, n_head, d_head).transpose(1, 0, 2)
    K_heads = K_all.reshape(SEQ_LEN, n_head, d_head).transpose(1, 0, 2)

    # --- Load exp-113 sigma_mf for structural heads ---
    npz113 = np.load(EXP113 / "meanfield_gpt2.npz")
    r113 = json.loads((EXP113 / "results_gpt2.json").read_text())

    print("\n=== Layer-0 mean-field score profiles (structural heads) ===", flush=True)
    results_heads = {}
    for (rl, rh) in STRUCTURAL:
        q_h = Q_heads[rh]  # (512, d_head)
        k_h = K_heads[rh]  # (512, d_head)
        score_mat = (q_h @ k_h.T) * scaling  # (512, 512)
        S_mf0 = pooled_window_profile(score_mat)  # (249,)
        sigma_mf0 = -ols_slope(S_mf0, WINDOW)
        r2_0 = power_law_r2(S_mf0, WINDOW)

        # Compare to exp-113 sigma_mf
        s113 = r113["conditions"]["random"]["registered"][f"L{rl}H{rh}"]
        sigma_mf_exp113 = s113["sigma_mf"]
        ratio = sigma_mf0 / sigma_mf_exp113

        results_heads[f"L{rl}H{rh}"] = {
            "sigma_mf_layer0": float(sigma_mf0),
            "R2_layer0": float(r2_0),
            "sigma_mf_exp113": float(sigma_mf_exp113),
            "ratio_layer0_to_exp113": float(ratio),
            "within_30pct": bool(0.70 <= ratio <= 1.30),
            "P1_ok": bool(r2_0 >= 0.85),
        }
        print(f"  L{rl}H{rh}: sigma_mf(L0)={sigma_mf0:+.4f}  R2={r2_0:.4f}  "
              f"exp113={sigma_mf_exp113:.4f}  ratio={ratio:.3f}  "
              f"P1={'OK' if r2_0 >= 0.85 else 'FAIL'}  "
              f"P2={'OK' if 0.70 <= ratio <= 1.30 else 'FAIL'}", flush=True)

    # --- P3: positional embedding raw correlation ---
    print("\n=== Positional embedding raw correlation (no projection) ===", flush=True)
    pos_mat = (wpe[:SEQ_LEN] @ wpe[:SEQ_LEN].T)  # (512, 512)
    C_pos = pooled_window_profile(pos_mat)  # (249,) at lags 8..256
    C_pos_monotone = all(C_pos[i] >= C_pos[i+1] for i in range(len(C_pos)-1))
    n_violations = sum(C_pos[i] < C_pos[i+1] for i in range(len(C_pos)-1))
    sigma_pos_raw = -ols_slope(C_pos, WINDOW)
    r2_pos = power_law_r2(C_pos, WINDOW)
    print(f"  C_pos slope (sigma): {sigma_pos_raw:.4f}  R2: {r2_pos:.4f}", flush=True)
    print(f"  Monotone: {C_pos_monotone}  Non-monotone steps: {n_violations}", flush=True)
    print(f"  C_pos[dx=8]={C_pos[0]:.1f}  C_pos[dx=32]={C_pos[24]:.1f}  "
          f"C_pos[dx=128]={C_pos[120]:.1f}  C_pos[dx=256]={C_pos[-1]:.1f}", flush=True)
    P3_ok = (C_pos_monotone or n_violations <= 5)  # small wobble allowed

    # --- Decomposition: constant vs positional contributions ---
    print("\n=== Score decomposition at layer 0 ===", flush=True)
    decomp_heads = {}
    for (rl, rh) in STRUCTURAL:
        q_h = Q_heads[rh]
        k_h = K_heads[rh]

        # Constant component = LN(emb_mean) projected (same for all positions,
        # approximated as mean of q_h over positions = q̄)
        q_const = q_h.mean(axis=0)  # (d_head,)
        k_const = k_h.mean(axis=0)

        # Positional component = q_h - q_const
        q_pos = q_h - q_const[None, :]
        k_pos = k_h - k_const[None, :]

        # Three terms of score matrix
        S_cc = (q_const @ k_const) * scaling  # scalar (constant × constant)
        S_cp = pooled_window_profile((q_const[None, :].repeat(SEQ_LEN, axis=0) @
                                      k_pos.T) * scaling)  # (249,)
        S_pc = pooled_window_profile((q_pos @
                                      k_const[None, :].repeat(SEQ_LEN, axis=0).T) * scaling)  # (249,)
        S_pp = pooled_window_profile((q_pos @ k_pos.T) * scaling)  # (249,)
        S_total = pooled_window_profile(((q_h @ k_h.T) * scaling))

        # The dx-varying part is S_pp (pos × pos); S_cp and S_pc also vary with i−dx
        # Check: what fraction of S_total slope comes from S_pp?
        sig_total = -ols_slope(S_total, WINDOW)
        sig_pp = -ols_slope(S_pp, WINDOW)
        sig_cp = -ols_slope(S_cp, WINDOW)
        sig_pc = -ols_slope(S_pc, WINDOW)

        decomp_heads[f"L{rl}H{rh}"] = {
            "sigma_total": float(sig_total),
            "sigma_pp": float(sig_pp),
            "sigma_cp": float(sig_cp),
            "sigma_pc": float(sig_pc),
            "S_cc_scalar": float(S_cc),
        }
        print(f"  L{rl}H{rh}: sigma_total={sig_total:.4f}  "
              f"sigma_pp={sig_pp:.4f}  sigma_cp={sig_cp:.4f}  sigma_pc={sig_pc:.4f}",
              flush=True)

    # --- Registered verdict ---
    P1_n_ok = sum(v["P1_ok"] for v in results_heads.values())
    P2_n_ok = sum(v["within_30pct"] for v in results_heads.values())
    P1_verdict = "CONFIRMED" if P1_n_ok >= 4 else "DEAD" if P1_n_ok <= 2 else "AMBIGUOUS"
    P2_verdict = "CONFIRMED" if P2_n_ok >= 4 else "DEAD" if P2_n_ok <= 2 else "AMBIGUOUS"
    P3_verdict = "CONFIRMED" if P3_ok else "DEAD"

    print(f"\n=== Registered verdicts ===", flush=True)
    print(f"  P1 (layer-0 power-law): {P1_n_ok}/5 OK -> {P1_verdict}", flush=True)
    print(f"  P2 (slope consistency): {P2_n_ok}/5 OK -> {P2_verdict}", flush=True)
    print(f"  P3 (C_pos monotone): {P3_verdict}", flush=True)

    # --- Save results ---
    results = {
        "exp": "exp-117",
        "date": "2026-08-10",
        "prereg_commit": "8a22802",
        "model": "gpt2",
        "method": "embedding-layer weights only, no forward passes",
        "structural_heads": results_heads,
        "positional_correlation": {
            "sigma": float(sigma_pos_raw),
            "R2": float(r2_pos),
            "monotone": bool(C_pos_monotone),
            "n_violations": int(n_violations),
            "values_at_key_lags": {
                "dx8": float(C_pos[0]),
                "dx32": float(C_pos[24]),
                "dx128": float(C_pos[120]),
                "dx256": float(C_pos[-1]),
            },
        },
        "decomposition": decomp_heads,
        "registered_verdicts": {
            "P1": {"n_ok": P1_n_ok, "threshold": 4, "verdict": P1_verdict},
            "P2": {"n_ok": P2_n_ok, "threshold": 4, "verdict": P2_verdict},
            "P3": {"verdict": P3_verdict, "n_violations": int(n_violations)},
        },
        "overall_verdict": (
            "LEVEL3_CONFIRMED"
            if P1_verdict == "CONFIRMED" and P2_verdict == "CONFIRMED"
            else "LEVEL3_PARTIAL"
            if P1_verdict != "DEAD" and P2_verdict != "DEAD"
            else "LEVEL3_DEAD"
        ),
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    np.savez_compressed(HERE / "smf_layer0.npz",
                        C_pos=C_pos,
                        **{f"S_mf_L0_L{rl}H{rh}": pooled_window_profile(
                               (Q_heads[rh] @ K_heads[rh].T) * scaling)
                           for (rl, rh) in STRUCTURAL})

    print(f"\nOverall: {results['overall_verdict']}", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
