"""exp-146 — Random-token census on GPT-2 medium.

Frozen protocol: N_INPUTS=50, SEQ_LEN=512, SEED=42, R²≥0.90 per-head OLS fit.
Identifies the structural population (Δ ∈ [0.20, 0.30]) distinct from the
WikiText-native Δ-window population (exp-118, 59 heads).

Pre-registration: prereg.md in this folder (commit 8aa31dd, pushed before this
file was written).

Ariel — September 18, 2026, ~12:35 AM MDT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

# ── Frozen protocol parameters ───────────────────────────────────────────────
MODEL_ID = "openai-community/gpt2-medium"
N_INPUTS = 50
SEQ_LEN = 512
SEED = 42
R2_MIN = 0.90
DELTA_MIN = 0.05
FIT_LO, FIT_HI = 8, 256
DEEP_LO = 256

# Δ-window definition for "structural" heads (SYK-q4 prediction)
DELTA_WINDOW_LO = 0.20
DELTA_WINDOW_HI = 0.30

# WikiText-native Δ-window heads from exp-118 (59 heads in GPT-2 medium)
# Format: (layer, head)
# Source: exp-118 results.json, verified at session start 2026-09-18.
# Concentrated in deep layers (L19–L23), consistent with exp-118's 80% deep-fraction.
# These are the heads exp-145 used as intervention targets (erroneously — wrong population).
# Listed for Jaccard comparison in P2.
WIKITEXT_NATIVE = {
    (0, 6),
    (1, 12),
    (3, 12),
    (7, 5),
    (7, 15),
    (8, 5),
    (8, 11),
    (8, 13),
    (9, 7),
    (9, 13),
    (11, 10),
    (11, 14),
    (12, 4),
    (13, 15),
    (14, 11),
    (14, 15),
    (15, 0),
    (15, 11),
    (16, 9),
    (16, 13),
    (16, 15),
    (17, 3),
    (17, 4),
    (18, 9),
    (18, 10),
    (18, 12),
    (18, 13),
    (19, 2),
    (19, 7),
    (19, 8),
    (19, 11),
    (19, 13),
    (20, 3),
    (20, 6),
    (20, 7),
    (20, 8),
    (20, 9),
    (20, 10),
    (20, 12),
    (20, 13),
    (21, 10),
    (22, 1),
    (22, 4),
    (22, 5),
    (22, 7),
    (22, 8),
    (22, 9),
    (22, 12),
    (22, 13),
    (22, 14),
    (22, 15),
    (23, 0),
    (23, 1),
    (23, 2),
    (23, 3),
    (23, 6),
    (23, 10),
    (23, 12),
    (23, 14),
}
assert len(WIKITEXT_NATIVE) == 59, f"Expected 59 WikiText-native heads, got {len(WIKITEXT_NATIVE)}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def lag_profile(att: np.ndarray) -> np.ndarray:
    """att: (n_heads, L, L) causal attention → per-head A(dx) for dx in [0, L-1].
    
    A(dx) = mean of att[h, i, i-dx] over all i ≥ max(DEEP_LO, dx).
    This is the frozen protocol used in the published replication kit.
    """
    n_heads, L, _ = att.shape
    prof = np.zeros((n_heads, L))
    for dx in range(1, L):
        # diagonal offset=-dx: att[h, i, i-dx]
        diag = np.diagonal(att, offset=-dx, axis1=-2, axis2=-1)  # shape (n_heads, L-dx)
        k_lo = max(DEEP_LO, dx) - dx  # first valid query index within the diagonal
        if k_lo < diag.shape[-1]:
            prof[:, dx] = diag[:, k_lo:].mean(axis=-1)
    return prof


def ols_fit(profile: np.ndarray):
    """OLS fit of log A vs log dx for dx in [FIT_LO, FIT_HI].
    
    Returns (delta, r2): delta = -slope/2 (conformal exponent), r2 = OLS R².
    """
    lags = np.arange(FIT_LO, FIT_HI + 1)
    y = profile[FIT_LO:FIT_HI + 1]
    ok = y > 1e-15
    if ok.sum() < 10:
        return float("nan"), 0.0
    log_lags = np.log(lags[ok].astype(float))
    log_y = np.log(y[ok])
    X = np.column_stack([np.ones(ok.sum()), log_lags])
    coef, residuals, rank, sv = np.linalg.lstsq(X, log_y, rcond=None)
    slope = coef[1]
    y_pred = X @ coef
    ss_res = ((log_y - y_pred) ** 2).sum()
    ss_tot = ((log_y - log_y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    delta = -slope / 2.0
    return float(delta), float(r2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"exp-146 — Random-token census: {MODEL_ID}", flush=True)
    print(f"Protocol: N_INPUTS={N_INPUTS}, SEQ_LEN={SEQ_LEN}, SEED={SEED}, "
          f"R2_MIN={R2_MIN}, FIT=[{FIT_LO},{FIT_HI}], DEEP_LO={DEEP_LO}", flush=True)

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    # Load model in fp32 with eager attention (required for output_attentions=True;
    # SDPA and FlashAttention backends do not return attention weights)
    print("Loading model...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    model.to(device)
    print(f"Loaded. n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}", flush=True)
    n_layer = config.n_layer
    n_head = config.n_head

    # Generate random-token input sequences
    rng = torch.Generator()
    rng.manual_seed(SEED)
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (N_INPUTS, SEQ_LEN), generator=rng)
    print(f"Input shape: {input_ids.shape}, vocab_size={vocab_size}", flush=True)

    # Accumulate attention profiles across all inputs
    # profiles: (n_layer, n_head, SEQ_LEN) — sum before averaging
    profiles_sum = np.zeros((n_layer, n_head, SEQ_LEN), dtype=np.float64)
    profiles_count = np.zeros((n_layer, n_head, SEQ_LEN), dtype=np.int64)

    print("Running forward passes...", flush=True)
    with torch.no_grad():
        for i, ids in enumerate(input_ids):
            if (i + 1) % 10 == 0:
                print(f"  Input {i+1}/{N_INPUTS}", flush=True)
            ids = ids.unsqueeze(0).to(device)
            outputs = model(ids, output_attentions=True)
            for layer_idx, att_layer in enumerate(outputs.attentions):
                # att_layer: (1, n_head, SEQ_LEN, SEQ_LEN) — fp32
                att_np = att_layer.squeeze(0).cpu().float().numpy()  # (n_head, L, L)
                prof = lag_profile(att_np)  # (n_head, SEQ_LEN)
                # For each head and dx: only add where profile > 0 (DEEP_LO gate)
                mask = prof > 0
                profiles_sum[layer_idx] += prof
                profiles_count[layer_idx] += mask.astype(np.int64)

    print("Forward passes complete. Fitting...", flush=True)

    # Average across inputs (use sum/N_INPUTS; count tracks actual data points)
    profiles_mean = profiles_sum / N_INPUTS  # (n_layer, n_head, SEQ_LEN)

    # Fit each head
    results = []
    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            profile = profiles_mean[layer_idx, head_idx]
            delta, r2 = ols_fit(profile)
            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "delta": delta,
                "r2": r2,
                "conformal": bool(r2 >= R2_MIN and not np.isnan(delta) and delta >= DELTA_MIN),
                "structural": bool(r2 >= R2_MIN and not np.isnan(delta)
                                   and DELTA_WINDOW_LO <= delta <= DELTA_WINDOW_HI),
                "wikitext_native": bool((layer_idx, head_idx) in WIKITEXT_NATIVE),
            })

    # Summary statistics
    conformal_heads = [(r["layer"], r["head"]) for r in results if r["conformal"]]
    structural_heads = [(r["layer"], r["head"]) for r in results if r["structural"]]
    structural_set = set(structural_heads)
    wikitext_set = WIKITEXT_NATIVE

    n_conformal = len(conformal_heads)
    n_structural = len(structural_heads)

    # Jaccard overlap between structural and WikiText-native
    intersection = structural_set & wikitext_set
    union = structural_set | wikitext_set
    jaccard = len(intersection) / len(union) if union else 0.0

    # Δ statistics for structural heads
    structural_deltas = [r["delta"] for r in results if r["structural"]]
    delta_median = float(np.median(structural_deltas)) if structural_deltas else float("nan")
    delta_min_val = float(np.min(structural_deltas)) if structural_deltas else float("nan")
    delta_max_val = float(np.max(structural_deltas)) if structural_deltas else float("nan")

    # Prediction verdicts
    P1 = n_structural >= 1
    P2 = jaccard < 0.8
    P3 = DELTA_WINDOW_LO <= delta_median <= DELTA_WINDOW_HI if not np.isnan(delta_median) else False
    P4 = n_structural > 5  # GPT-2 small had 5 structural heads

    K1 = n_conformal == 0
    K2 = jaccard == 1.0 and n_structural > 0
    K3 = n_structural == 0 and n_conformal > 0

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Total heads: {n_layer * n_head}", flush=True)
    print(f"Conformal heads (R²≥{R2_MIN}, Δ≥{DELTA_MIN}): {n_conformal}", flush=True)
    print(f"Structural heads (R²≥{R2_MIN}, Δ∈[{DELTA_WINDOW_LO},{DELTA_WINDOW_HI}]): {n_structural}", flush=True)
    print(f"WikiText-native heads (exp-118): {len(WIKITEXT_NATIVE)}", flush=True)
    print(f"Overlap |structural ∩ wiki|: {len(intersection)}", flush=True)
    print(f"Jaccard(structural, wiki): {jaccard:.3f}", flush=True)
    print(f"Structural Δ: median={delta_median:.4f}, min={delta_min_val:.4f}, max={delta_max_val:.4f}", flush=True)
    print(f"GPT-2 small structural count: 5", flush=True)
    print(flush=True)
    print("Structural heads (layer, head, Δ, R²):", flush=True)
    for r in sorted(results, key=lambda x: (not x["structural"], x["layer"], x["head"])):
        if r["structural"]:
            wiki = "wiki" if r["wikitext_native"] else "new"
            print(f"  L{r['layer']}H{r['head']:2d}  Δ={r['delta']:.4f}  R²={r['r2']:.4f}  [{wiki}]", flush=True)
    print(flush=True)
    print("Prediction verdicts:", flush=True)
    print(f"  P1 (n_structural ≥ 1):       {'FIRED' if P1 else 'DEAD'}", flush=True)
    print(f"  P2 (Jaccard < 0.8):          {'FIRED' if P2 else 'DEAD'}", flush=True)
    print(f"  P3 (Δ_med ∈ window):         {'FIRED' if P3 else 'DEAD'}", flush=True)
    print(f"  P4 (n_structural > 5):       {'FIRED' if P4 else 'DEAD'}", flush=True)
    print(flush=True)
    print("Kill conditions:", flush=True)
    print(f"  K1 (no conformal heads):     {'FIRED' if K1 else 'not fired'}", flush=True)
    print(f"  K2 (populations identical):  {'FIRED' if K2 else 'not fired'}", flush=True)
    print(f"  K3 (0 structural, some conf):{'FIRED' if K3 else 'not fired'}", flush=True)

    # Save results
    out = {
        "experiment": "exp-146",
        "model": MODEL_ID,
        "protocol": {
            "N_INPUTS": N_INPUTS,
            "SEQ_LEN": SEQ_LEN,
            "SEED": SEED,
            "R2_MIN": R2_MIN,
            "DELTA_MIN": DELTA_MIN,
            "FIT_LO": FIT_LO,
            "FIT_HI": FIT_HI,
            "DEEP_LO": DEEP_LO,
            "DELTA_WINDOW_LO": DELTA_WINDOW_LO,
            "DELTA_WINDOW_HI": DELTA_WINDOW_HI,
        },
        "summary": {
            "n_total_heads": n_layer * n_head,
            "n_conformal": n_conformal,
            "n_structural": n_structural,
            "n_wikitext_native": len(WIKITEXT_NATIVE),
            "n_overlap": len(intersection),
            "jaccard": jaccard,
            "structural_delta_median": delta_median,
            "structural_delta_min": delta_min_val,
            "structural_delta_max": delta_max_val,
            "gpt2_small_n_structural": 5,
        },
        "predictions": {
            "P1_structural_exists": bool(P1),
            "P2_populations_distinct": bool(P2),
            "P3_delta_in_window": bool(P3),
            "P4_larger_than_small": bool(P4),
        },
        "kills": {
            "K1_no_conformal": bool(K1),
            "K2_populations_identical": bool(K2),
            "K3_no_structural_some_conformal": bool(K3),
        },
        "structural_heads": [
            {
                "layer": r["layer"],
                "head": r["head"],
                "delta": r["delta"],
                "r2": r["r2"],
                "wikitext_native": r["wikitext_native"],
            }
            for r in results if r["structural"]
        ],
        "all_heads": results,
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
