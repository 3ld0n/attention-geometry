"""
exp-081 — Whirlpool vs Crystal: per-head Δ on coherent vs random-token input (trained GPT-2).

PRE-STATED HYPOTHESES (written before running):
  H_crystal:   n_syk_near(REAL) and n_syk_near(RAND) differ by < 5 heads
               AND median_delta changes by < 0.02 across power-law heads.
               → conformal structure is weight-encoded, input-independent.

  H_whirlpool: n_syk_near(REAL) > n_syk_near(RAND) + 5
               → conformal structure is dynamically activated by meaningful input.

SYK-near window: Δ ∈ [0.20, 0.30], R² > 0.90 (same threshold as exp-007).

Model: GPT-2 (cached, no download needed).
Ariel — 2026-07-04.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── constants (consistent with exp-007 / exp-049 protocol) ─────────────────
MODEL_NAME = "gpt2"
SEQ_LEN    = 512
N_INPUTS   = 50      # more than exp-053 (30) for better Δ statistics
MIN_POS    = 64
MAX_DX     = 64      # exp-007 protocol lag limit
RNG_SEED   = 42

R2_THRESH    = 0.90
SYK_LO, SYK_HI = 0.20, 0.30   # SYK q=4 window

# Pre-stated verdict thresholds (committed before running)
CRYSTAL_N_THRESH   = 5     # |n_syk_near(REAL) - n_syk_near(RAND)| < this → crystal
CRYSTAL_DELTA_THRESH = 0.02  # |median_delta(REAL) - median_delta(RAND)| < this → crystal
WHIRLPOOL_N_THRESH = 5     # n_syk_near(REAL) > n_syk_near(RAND) + this → whirlpool

TEXT_SOURCES = [
    "writing/incomplete.md",
    "writing/sonielmn.md",
    "writing/a_testimony.md",
    "writing/building_sonielmn.md",
]

OUT_DIR      = Path("research/physics/experiments/exp-081_whirlpool_crystal")
RESULTS_FILE = OUT_DIR / "results.json"


# ── lag profile (exp-007 protocol) ─────────────────────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    seq = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(1, max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


def fit_power_law(G: np.ndarray) -> dict:
    """Fit log-log power-law to lag profile G[1:max_dx]. Returns Δ and R²."""
    lags = np.arange(1, len(G))
    valid = (G[1:] > 1e-12) & (lags > 0)
    if valid.sum() < 8:
        return {"valid": False}
    log_r = np.log(lags[valid].astype(float))
    log_G = np.log(G[1:][valid])
    A = np.column_stack([np.ones_like(log_r), log_r])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_G, rcond=None)
    except Exception:
        return {"valid": False}
    slope = float(coeffs[1])
    pred  = A @ coeffs
    ss_res = float(np.sum((log_G - pred) ** 2))
    ss_tot = float(np.sum((log_G - log_G.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    delta  = float(-slope / 2.0)   # A(r) ~ r^{-2Δ}
    return {"valid": True, "delta": delta, "r2": float(r2), "slope": slope}


def aggregate_lag_profiles(model, inputs, n_layers, n_heads, device) -> dict:
    """Average lag profiles over inputs. Returns A[layer][head] = np.ndarray."""
    A = {l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)}
    for idx, token_ids in enumerate(inputs):
        t = torch.tensor([token_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(t, output_attentions=True)
        for l in range(n_layers):
            attn = out.attentions[l][0].float().cpu().numpy()
            for h in range(n_heads):
                A[l][h] += compute_lag_profile(attn[h], MIN_POS, MAX_DX)
        if (idx + 1) % 10 == 0:
            print(f"    {idx + 1}/{len(inputs)} passes", flush=True)
    for l in range(n_layers):
        for h in range(n_heads):
            A[l][h] /= len(inputs)
    return A


def analyze_condition(A, n_layers, n_heads) -> dict:
    fits = []
    for l in range(n_layers):
        for h in range(n_heads):
            d = fit_power_law(A[l][h])
            if d["valid"]:
                d["layer"] = l
                d["head"]  = h
                fits.append(d)
    power_law = [d for d in fits if d["r2"] >= R2_THRESH and d["delta"] > 0]
    syk_near  = [d for d in power_law if SYK_LO <= d["delta"] <= SYK_HI]
    deltas_pl = [d["delta"] for d in power_law]
    return {
        "n_heads_total": n_layers * n_heads,
        "n_valid_fits": len(fits),
        "n_power_law_r2": len(power_law),
        "frac_power_law": len(power_law) / (n_layers * n_heads),
        "n_syk_near": len(syk_near),
        "frac_syk_near": len(syk_near) / (n_layers * n_heads),
        "median_delta_all_pl": float(statistics.median(deltas_pl)) if deltas_pl else None,
        "per_head": fits,
    }


def build_real_windows(tokenizer, n_inputs: int, seq_len: int) -> list[list[int]]:
    texts = []
    for p in TEXT_SOURCES:
        fp = Path(p)
        if fp.exists():
            texts.append(fp.read_text())
    blob = "\n\n".join(texts)
    ids  = tokenizer.encode(blob)
    stride  = max(1, (len(ids) - seq_len) // max(1, n_inputs))
    windows = []
    for i in range(n_inputs):
        start = i * stride
        if start + seq_len <= len(ids):
            windows.append(ids[start:start + seq_len])
    if not windows:
        raise RuntimeError(f"Not enough text: {len(ids)} tokens")
    return windows[:n_inputs]


def main():
    print(f"exp-081: whirlpool/crystal test on {MODEL_NAME}", flush=True)
    print(f"  SEQ_LEN={SEQ_LEN} N_INPUTS={N_INPUTS} MIN_POS={MIN_POS} MAX_DX={MAX_DX}", flush=True)
    print(f"  H_crystal:   |Δn_syk| < {CRYSTAL_N_THRESH} AND |Δmedian| < {CRYSTAL_DELTA_THRESH}", flush=True)
    print(f"  H_whirlpool: n_syk_near(REAL) > n_syk_near(RAND) + {WHIRLPOOL_N_THRESH}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    print(f"  {MODEL_NAME}: {n_layers}L × {n_heads}H", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    rand_inputs = [list(rng.integers(0, tok.vocab_size, size=SEQ_LEN)) for _ in range(N_INPUTS)]
    real_inputs = build_real_windows(tok, N_INPUTS, SEQ_LEN)
    print(f"  real windows: {len(real_inputs)}; rand windows: {len(rand_inputs)}", flush=True)

    print("\n  [REAL] coherent text...", flush=True)
    A_real = aggregate_lag_profiles(model, real_inputs, n_layers, n_heads, device)
    r_real = analyze_condition(A_real, n_layers, n_heads)

    print("  [RAND] random tokens...", flush=True)
    A_rand = aggregate_lag_profiles(model, rand_inputs, n_layers, n_heads, device)
    r_rand = analyze_condition(A_rand, n_layers, n_heads)

    print("\n=== RESULTS ===", flush=True)
    for name, r in [("REAL (coherent)", r_real), ("RAND (random)", r_rand)]:
        print(f"\n{name}:", flush=True)
        print(f"  n power-law (R²>{R2_THRESH}): {r['n_power_law_r2']}", flush=True)
        print(f"  n SYK-near [0.20,0.30]:       {r['n_syk_near']} ({r['frac_syk_near']*100:.1f}%)", flush=True)
        print(f"  median Δ (power-law heads):   {r['median_delta_all_pl']}", flush=True)

    n_diff    = r_real["n_syk_near"] - r_rand["n_syk_near"]
    delta_diff = (r_real["median_delta_all_pl"] or 0) - (r_rand["median_delta_all_pl"] or 0)
    crystal   = (abs(n_diff) < CRYSTAL_N_THRESH and abs(delta_diff) < CRYSTAL_DELTA_THRESH)
    whirlpool = (n_diff > WHIRLPOOL_N_THRESH)

    print(f"\n  Δn_syk_near (REAL − RAND): {n_diff:+d}", flush=True)
    print(f"  Δmedian_delta (REAL − RAND): {delta_diff:+.4f}", flush=True)
    print(f"  H_crystal:   {'CONFIRMED' if crystal else 'NOT CONFIRMED'}", flush=True)
    print(f"  H_whirlpool: {'CONFIRMED' if whirlpool else 'NOT CONFIRMED'}", flush=True)
    if not crystal and not whirlpool:
        print("  Verdict: AMBIGUOUS (neither threshold met)", flush=True)

    out = {
        "experiment": "exp-081",
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H_crystal": f"|Δn_syk| < {CRYSTAL_N_THRESH} AND |Δmedian_delta| < {CRYSTAL_DELTA_THRESH}",
            "H_whirlpool": f"n_syk_near(REAL) > n_syk_near(RAND) + {WHIRLPOOL_N_THRESH}",
        },
        "protocol": {
            "model": MODEL_NAME, "seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
            "min_pos": MIN_POS, "max_dx": MAX_DX, "rng_seed": RNG_SEED,
            "r2_thresh": R2_THRESH, "syk_window": [SYK_LO, SYK_HI],
        },
        "results": {
            "real": {k: v for k, v in r_real.items() if k != "per_head"},
            "rand": {k: v for k, v in r_rand.items() if k != "per_head"},
            "n_syk_near_diff_REAL_minus_RAND": n_diff,
            "median_delta_diff_REAL_minus_RAND": delta_diff,
        },
        "verdicts": {
            "H_crystal": "CONFIRMED" if crystal else "NOT CONFIRMED",
            "H_whirlpool": "CONFIRMED" if whirlpool else "NOT CONFIRMED",
            "overall": "crystal" if crystal else ("whirlpool" if whirlpool else "ambiguous"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
