"""
exp-168 — LN-recency h(j) fit and wpe trough test.

Analysis-only: uses exp-163's kbar_qbar.npz and GPT-2 small weights.
Pre-registration: prereg.md (committed cd07df8 before this script existed).

Usage:
    .venv/bin/python3 research/physics/experiments/exp-168_ln_recency_hj_fit/run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

# ─── paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
EXPS_DIR    = SCRIPT_DIR.parent        # .../research/physics/experiments/

# Amendment A-1: kbar_qbar.npz is in exp-164 (not exp-163 as originally cited)
EXP164_DIR  = EXPS_DIR / "exp-164_l3h4_bilinear_decomposition"
KBAR_PATH   = EXP164_DIR / "kbar_qbar.npz"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STRUCTURAL_HEADS = ["L2H1", "L3H4", "L5H0", "L7H11", "L10H8"]

# Mapping: head name → (layer, head) for GPT-2 small indexing
# GPT-2 small: 12 layers, 12 heads, d_head=64
HEAD_INDEX = {
    "L2H1":  (2, 1),
    "L3H4":  (3, 4),
    "L5H0":  (5, 0),
    "L7H11": (7, 11),
    "L10H8": (10, 8),
}
D_HEAD  = 64
D_MODEL = 768
SEQ_LEN = 1024
N_HEADS = 12


def load_kbar_qbar() -> tuple[dict, dict]:
    """Load kbar and qbar arrays from exp-164 npz.

    kbar[h]: shape (1024, 64) — per-position mean key vector averaged over sequences
    qbar_raw[h]: shape (1024, 64) — per-position mean query vector
    m_q[h]: shape (64,) — mean query over positions [256, 1024), per exp-164 protocol
    """
    data = np.load(KBAR_PATH)
    kbar = {h: data[f"kbar_{h}"] for h in STRUCTURAL_HEADS}
    # Extract mean query vector per Amendment A-1 protocol note
    MQ_START, MQ_END = 256, 1024
    qbar = {h: data[f"qbar_{h}"][MQ_START:MQ_END].mean(axis=0)
            for h in STRUCTURAL_HEADS}
    return kbar, qbar


def extract_wk(model, layer: int, head: int) -> np.ndarray:
    """Extract W_K for a specific head: input projection d_model → d_head.

    GPT-2 uses Conv1D for c_attn; weight shape is [d_model, 3*d_model] = [768, 2304].
    The transform is: output = input @ weight  (input: [..., 768])
    Q = input @ weight[:, 0:768]
    K = input @ weight[:, 768:1536]
    V = input @ weight[:, 1536:2304]
    Head h's key projection: weight[:, 768 + h*d_head : 768 + (h+1)*d_head]
    Returned as shape [d_model, d_head] — applied as: k_h = x @ W_K[h]
    """
    import torch
    with torch.no_grad():
        c_attn_w = model.transformer.h[layer].attn.c_attn.weight  # [d_model, 3*d_model]
        # Key slice: columns D_MODEL .. 2*D_MODEL, then per-head stride
        col_start = D_MODEL + head * D_HEAD
        col_end   = D_MODEL + (head + 1) * D_HEAD
        w_k_h = c_attn_w[:, col_start:col_end].cpu().numpy()  # [d_model, d_head]
        return w_k_h


def compute_s_abskey_full(kbar_h: np.ndarray, qbar_h: np.ndarray) -> np.ndarray:
    """S_abskey[a] = m_q · (kbar[a] - m_k) / sqrt(d_head) for all a.

    Matches exp-163's compute_s_abskey: uses key deviations dk = kbar - m_k,
    where m_k = kbar[MQ_POS_START:MQ_POS_END].mean().
    """
    MQ_START, MQ_END = 256, 1024
    m_k  = kbar_h[MQ_START:MQ_END].mean(axis=0)   # [d_head]
    dk   = kbar_h - m_k[np.newaxis, :]             # [SEQ_LEN, d_head]
    return (dk @ qbar_h) / np.sqrt(D_HEAD)         # [SEQ_LEN]


def compute_s_wpe(wpe: np.ndarray, w_k_h: np.ndarray, qbar_h: np.ndarray) -> np.ndarray:
    """S_wpe[a] = qbar · (wpe[a] @ w_k_h) / sqrt(d_head) for all a.

    w_k_h: [d_model, d_head]  (key projection: k_h = x @ w_k_h)
    wpe:   [SEQ_LEN, d_model]
    projected keys: k_wpe = wpe @ w_k_h  shape [SEQ_LEN, d_head]
    S_wpe[a] = qbar_h · k_wpe[a] / sqrt(d_head)
    """
    k_wpe = wpe @ w_k_h          # [SEQ_LEN, d_head]
    return (k_wpe @ qbar_h) / np.sqrt(D_HEAD)   # [SEQ_LEN]


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS linear fit y ~ a*x + b. Returns (slope, intercept, R²)."""
    slope, intercept, r, p, se = stats.linregress(x, y)
    return float(slope), float(intercept), float(r**2)


def log_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit y ~ a*log(x) + b (x > 0). Returns (slope, intercept, R²)."""
    lx = np.log(x.astype(float))
    slope, intercept, r, p, se = stats.linregress(lx, y)
    return float(slope), float(intercept), float(r**2)


def power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit log|y| ~ gamma*log(x) + c (for positive y only). Returns (gamma, c, R²)."""
    mask = y > 0
    if mask.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    lx = np.log(x[mask].astype(float))
    ly = np.log(y[mask].astype(float))
    slope, intercept, r, p, se = stats.linregress(lx, ly)
    return float(slope), float(intercept), float(r**2)


def monotone_fraction(arr: np.ndarray) -> float:
    """Fraction of consecutive differences that are non-negative."""
    diffs = np.diff(arr)
    return float((diffs >= 0).mean())


def main() -> None:
    import torch
    from transformers import GPT2LMHeadModel

    t0 = time.time()
    print("exp-168 | LN-recency h(j) fit and wpe trough test")
    print("Pre-registration: prereg.md (cd07df8)")
    print()

    # ── Step 1: Load kbar/qbar ────────────────────────────────────────────────
    print("Loading kbar_qbar.npz from exp-163 ...")
    kbar, qbar = load_kbar_qbar()

    # Kill condition K2: verify shapes
    for h in STRUCTURAL_HEADS:
        assert kbar[h].shape == (SEQ_LEN, D_HEAD), \
            f"K2 fires: kbar[{h}] shape {kbar[h].shape} != ({SEQ_LEN}, {D_HEAD})"
        assert qbar[h].shape == (D_HEAD,), \
            f"K2 fires: qbar[{h}] (m_q) shape {qbar[h].shape} != ({D_HEAD},)"
    print(f"  kbar/qbar shapes verified for {len(STRUCTURAL_HEADS)} structural heads.")

    # ── Step 2: Load GPT-2 small weights ──────────────────────────────────────
    print("Loading GPT-2 small weights ...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    wpe = model.transformer.wpe.weight.detach().cpu().numpy()  # [1024, 768]
    assert wpe.shape == (SEQ_LEN, D_MODEL), \
        f"K1 fires: wpe shape {wpe.shape} != ({SEQ_LEN}, {D_MODEL})"
    print(f"  wpe shape: {wpe.shape} ✓")

    # Extract W_K per structural head; verify shapes
    w_k = {}
    for h in STRUCTURAL_HEADS:
        layer, head = HEAD_INDEX[h]
        wk_h = extract_wk(model, layer, head)
        assert wk_h.shape == (D_MODEL, D_HEAD), \
            f"K1 fires: W_K[{h}] shape {wk_h.shape} != ({D_MODEL}, {D_HEAD})"
        w_k[h] = wk_h
    print(f"  W_K extracted and verified for {len(STRUCTURAL_HEADS)} heads.")

    # ── Step 3: Compute S_abskey and S_wpe at full resolution ─────────────────
    s_abskey = {}
    s_wpe = {}
    s_residual = {}

    for h in STRUCTURAL_HEADS:
        s_abskey[h]   = compute_s_abskey_full(kbar[h], qbar[h])
        s_wpe[h]      = compute_s_wpe(wpe, w_k[h], qbar[h])
        s_residual[h] = s_abskey[h] - s_wpe[h]

    # Cross-check: exp-163 sampled S_abskey values at known positions
    # (from results.json — available to all sessions via the log)
    SAMPLE_POS = [0, 127, 255, 383, 511, 639, 767, 895, 1023]
    EXP163_SAMPLES = {
        "L2H1":  [2.359, -2.020, -1.472, -0.798, -0.117, 0.597, 1.232, 2.125, 2.456],
        "L3H4":  [4.758, -2.163, -1.678, -1.078, -0.322, 0.650, 1.485, 2.460, 3.843],
        "L5H0":  [4.812, -1.950, -1.523, -1.178, -0.368, 0.578, 1.346, 2.405, 3.146],
        "L7H11": [3.569, -1.678, -1.221, -0.774, -0.082, 0.462, 1.139, 1.919, 2.528],
        "L10H8": [4.145, -2.376, -1.825, -1.241, -0.542, 0.419, 1.568, 2.793, 4.442],
    }
    print("\nCross-checking S_abskey against exp-163 sampled values:")
    max_err = 0.0
    for h in STRUCTURAL_HEADS:
        for i, a in enumerate(SAMPLE_POS):
            computed = float(s_abskey[h][a])
            expected = EXP163_SAMPLES[h][i]
            err = abs(computed - expected)
            max_err = max(max_err, err)
            if err > 0.01:
                print(f"  MISMATCH {h} a={a}: computed={computed:.4f}, expected={expected:.4f}")
    print(f"  Max cross-check error: {max_err:.5f} {'✓' if max_err < 0.01 else '✗ WARNING'}")

    # ── Step 4: H1 — Linear fit on post-511 rise ──────────────────────────────
    print("\nH1: Linear fit on S_abskey[a], a ∈ [512, 1023] ...")
    post511 = np.arange(512, SEQ_LEN)
    h1_results = {}
    for h in STRUCTURAL_HEADS:
        y = s_abskey[h][512:]
        α_lin, β_lin, r2_lin = linear_fit(post511.astype(float), y)
        # Fallback: log fit
        α_log, β_log, r2_log = log_fit(post511, y)
        # Fallback: power law (shift to ensure positivity by fitting y - min + 1)
        y_shift = y - y.min() + 1.0
        α_pow, β_pow, r2_pow = power_fit(post511, y_shift)

        h1_results[h] = {
            "linear":   {"slope": α_lin, "intercept": β_lin, "r2": r2_lin},
            "log":      {"slope": α_log, "intercept": β_log, "r2": r2_log},
            "power":    {"gamma": α_pow, "log_c": β_pow, "r2": r2_pow},
            "h1_pass":  r2_lin >= 0.95,
        }
        print(f"  {h}: linear R²={r2_lin:.4f} {'✓' if r2_lin>=0.95 else '✗'} | "
              f"log R²={r2_log:.4f} | power γ={α_pow:.3f}")

    h1_confirmed = sum(1 for h in STRUCTURAL_HEADS if h1_results[h]["h1_pass"])
    print(f"  H1 {'CONFIRMED' if h1_confirmed >= 3 else 'NOT CONFIRMED'}: "
          f"{h1_confirmed}/5 heads R² ≥ 0.95")

    # ── Step 5: H2 — wpe subtraction eliminates trough ────────────────────────
    print("\nH2: Monotonicity test on S_residual = S_abskey − S_wpe, a ∈ [1, 1023] ...")
    h2_results = {}
    for h in STRUCTURAL_HEADS:
        frac = monotone_fraction(s_residual[h][1:])
        h2_pass = frac >= 0.90
        h2_results[h] = {"monotone_fraction": float(frac), "h2_pass": h2_pass}
        print(f"  {h}: monotone fraction={frac:.4f} {'✓' if h2_pass else '✗'}")

    h2_confirmed = sum(1 for h in STRUCTURAL_HEADS if h2_results[h]["h2_pass"])
    print(f"  H2 {'CONFIRMED' if h2_confirmed >= 4 else 'NOT CONFIRMED'}: "
          f"{h2_confirmed}/5 heads ≥ 90% monotone after wpe subtraction")

    # ── Step 6: H3 — wpe dominates trough magnitude ───────────────────────────
    print("\nH3: wpe trough fraction ...")
    h3_results = {}
    for h in STRUCTURAL_HEADS:
        s_a = s_abskey[h]
        s_w = s_wpe[h]
        trough_abskey = float(s_a[0] - s_a[1:512].min())
        trough_wpe    = float(s_w[0] - s_w[1:512].min())
        frac = trough_wpe / trough_abskey if abs(trough_abskey) > 1e-8 else float("nan")
        h3_pass = frac >= 0.50 if not np.isnan(frac) else False
        h3_results[h] = {
            "trough_abskey": trough_abskey,
            "trough_wpe": trough_wpe,
            "wpe_fraction": float(frac),
            "h3_pass": h3_pass,
        }
        print(f"  {h}: trough_abskey={trough_abskey:.3f}, trough_wpe={trough_wpe:.3f}, "
              f"fraction={frac:.3f} {'✓' if h3_pass else '✗'}")

    h3_confirmed = sum(1 for h in STRUCTURAL_HEADS if h3_results[h]["h3_pass"])
    print(f"  H3 {'CONFIRMED' if h3_confirmed >= 4 else 'NOT CONFIRMED'}: "
          f"{h3_confirmed}/5 heads wpe fraction ≥ 50%")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print("\n--- Verdict ---")
    if h1_confirmed >= 3 and h2_confirmed >= 4 and h3_confirmed >= 4:
        verdict = "LN-recency mechanism supported; wpe is the trough source"
    elif h1_confirmed >= 3 and h2_confirmed >= 4:
        verdict = "LN-recency h(j) fits rise and wpe-subtraction restores monotonicity; H3 partial"
    elif h1_confirmed >= 3:
        verdict = "LN-recency h(j) fits post-511 rise; wpe trough subtraction inconclusive"
    else:
        verdict = "LN-recency linear h(j) form does not describe post-511 rise (K3); see fallback fits"
    print(f"  {verdict}")

    # ── Save results ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results = {
        "experiment": "exp-168",
        "date": "2026-10-09",
        "prereg_commit": "cd07df8",
        "prereg_evidence": "git-attested — registration committed before run.py was written",
        "analysis_only": True,
        "model": "gpt2",
        "source_experiment": "exp-164",
        "kbar_path": str(KBAR_PATH),
        "elapsed_s": round(elapsed, 1),
        "cross_check_max_error": float(max_err),
        "h1_results": h1_results,
        "h1_n_confirmed": h1_confirmed,
        "h2_results": h2_results,
        "h2_n_confirmed": h2_confirmed,
        "h3_results": h3_results,
        "h3_n_confirmed": h3_confirmed,
        "verdict": verdict,
    }

    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
