"""
exp-079 — Periodic component: untrained-vs-trained GPT-2 control.

Follow-up to exp-053. Question: does the ~3.5-lag periodic component require
trained positional embeddings, or is it structural (Gaussian-init artifact)?

PRE-STATED HYPOTHESES (written before running):
  H_PE:     Untrained GPT-2 shows < 3% significant periodic heads.
              Period does NOT cluster near 3.5 lags.
  H_struct: Untrained GPT-2 shows >= 5% significant heads (same as trained).

PROTOCOL: Identical to exp-053 RAND condition, but model is random-init
(no pretrained weights). GPT2Config() default init.

Ariel — 2026-07-04, ~4 AM MDT.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# ── constants (identical to exp-053) ────────────────────────────────────────────
SEQ_LEN   = 512
N_INPUTS  = 30
MIN_POS   = 64
MAX_DX    = 192
RNG_SEED  = 42

K_LOW       = 2
K_HIGH      = 60
PEAK_THRESH = 0.30
APERIODIC_R2_MIN = 0.80

OUT_DIR      = Path("research/physics/experiments/exp-079_periodic_untrained_control")
RESULTS_FILE = OUT_DIR / "results.json"

# ── thresholds for verdict (committed before running) ────────────────────────────
H_PE_THRESHOLD     = 0.03   # < 3% sig heads → H_PE (trained PE required)
H_STRUCT_THRESHOLD = 0.05   # >= 5% sig heads → H_struct (structural)


# ── lag profile (verbatim from exp-053) ─────────────────────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    seq = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


# ── periodic / aperiodic decomposition (verbatim from exp-053) ──────────────────

def decompose(G_pos: np.ndarray, k_low: int, k_high: int) -> dict:
    profile = G_pos[1:].astype(np.float64)
    profile = np.clip(profile, 1e-12, None)
    spectrum = np.abs(np.fft.rfft(profile))
    L_fft = len(profile)
    k_arr = np.arange(len(spectrum))
    fit_mask = (k_arr >= k_low) & (k_arr < min(k_high, len(spectrum))) & (spectrum > 1e-30)
    if fit_mask.sum() < 6:
        return {"valid": False}
    log_k = np.log10(k_arr[fit_mask].astype(float))
    log_s = np.log10(spectrum[fit_mask])
    A_mat = np.column_stack([np.ones_like(log_k), log_k])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_s, rcond=None)
    pred = A_mat @ coeffs
    resid = log_s - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_s - log_s.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    slope = float(coeffs[1])
    delta_aperiodic = float(-(slope + 1.0) / 2.0) if slope < 0 else float("nan")
    peak_idx = int(np.argmax(resid))
    peak_height = float(resid[peak_idx])
    peak_k = int(k_arr[fit_mask][peak_idx])
    peak_period = float(L_fft / peak_k) if peak_k > 0 else float("nan")
    return {
        "valid": True,
        "aperiodic_slope": slope,
        "aperiodic_r2": float(r2),
        "delta_aperiodic": delta_aperiodic,
        "periodicity_index": peak_height,
        "peak_k": peak_k,
        "peak_period": peak_period,
        "significant": bool(peak_height > PEAK_THRESH),
    }


def aggregate(model, inputs, n_layers, n_heads) -> dict[int, dict[int, np.ndarray]]:
    A = {l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)}
    for idx, token_ids in enumerate(inputs):
        t = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            out = model(t, output_attentions=True)
        for l in range(n_layers):
            attn = out.attentions[l][0].float().numpy()
            for h in range(n_heads):
                A[l][h] += compute_lag_profile(attn[h], MIN_POS, MAX_DX)
        if (idx + 1) % 10 == 0:
            print(f"    {idx + 1}/{len(inputs)} passes", flush=True)
    for l in range(n_layers):
        for h in range(n_heads):
            A[l][h] /= len(inputs)
    return A


def analyze_condition(A, n_layers, n_heads) -> dict:
    per_head = []
    for l in range(n_layers):
        for h in range(n_heads):
            d = decompose(A[l][h], K_LOW, K_HIGH)
            if d.get("valid"):
                d["layer"] = l
                d["head"] = h
                per_head.append(d)
    idx_all = [d["periodicity_index"] for d in per_head]
    clean = [d for d in per_head if d["aperiodic_r2"] >= APERIODIC_R2_MIN]
    idx_clean = [d["periodicity_index"] for d in clean]
    n_sig = sum(d["significant"] for d in per_head)
    periods_sig = [d["peak_period"] for d in per_head if d["significant"]]
    slopes = [d["aperiodic_slope"] for d in per_head]
    return {
        "n_heads": len(per_head),
        "n_clean_1f": len(clean),
        "median_periodicity_index": float(statistics.median(idx_all)) if idx_all else None,
        "median_periodicity_index_clean": float(statistics.median(idx_clean)) if idx_clean else None,
        "frac_significant": n_sig / len(per_head) if per_head else None,
        "n_significant": n_sig,
        "median_aperiodic_slope": float(statistics.median(slopes)) if slopes else None,
        "median_peak_period_significant": float(statistics.median(periods_sig)) if periods_sig else None,
        "per_head": per_head,
    }


def main():
    print("exp-079: untrained GPT-2 periodic control", flush=True)
    print(f"  SEQ_LEN={SEQ_LEN} N_INPUTS={N_INPUTS} MIN_POS={MIN_POS} MAX_DX={MAX_DX}", flush=True)
    print(f"  H_PE threshold: < {H_PE_THRESHOLD*100:.0f}% sig heads", flush=True)
    print(f"  H_struct threshold: >= {H_STRUCT_THRESHOLD*100:.0f}% sig heads", flush=True)

    # Random-init GPT-2 — same architecture, no pretrained weights
    # Use eager attention so output_attentions=True works (queue note: config._attn_implementation)
    torch.manual_seed(RNG_SEED)
    config = GPT2Config()
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    model.eval()
    n_layers, n_heads = model.config.n_layer, model.config.n_head
    vocab = model.config.vocab_size
    print(f"  Untrained GPT-2: {n_layers}L × {n_heads}H, vocab={vocab}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    rand_inputs = [list(rng.integers(0, vocab, size=SEQ_LEN)) for _ in range(N_INPUTS)]

    print("  [UNTRAINED-RAND] aggregating...", flush=True)
    A_rand = aggregate(model, rand_inputs, n_layers, n_heads)
    res = analyze_condition(A_rand, n_layers, n_heads)

    print("\n=== RESULTS ===", flush=True)
    print(f"  heads analyzed:               {res['n_heads']}", flush=True)
    print(f"  clean-1/f heads:              {res['n_clean_1f']}", flush=True)
    print(f"  median periodicity index:     {res['median_periodicity_index']:.4f}", flush=True)
    print(f"  heads w/ significant peak:    {res['n_significant']} ({res['frac_significant']*100:.1f}%)", flush=True)
    print(f"  median aperiodic slope:       {res['median_aperiodic_slope']:.4f}", flush=True)
    print(f"  median peak period (sig):     {res['median_peak_period_significant']}", flush=True)

    # Comparison with exp-053 RAND baseline
    TRAINED_RAND_FRAC = 10 / 144  # 6.9% from exp-053
    TRAINED_RAND_PERIOD = 4.03

    print("\n=== VERDICT ===", flush=True)
    frac = res["frac_significant"] or 0.0
    print(f"  Untrained frac_sig = {frac*100:.1f}% vs trained {TRAINED_RAND_FRAC*100:.1f}%", flush=True)
    if frac < H_PE_THRESHOLD:
        verdict = "H_PE CONFIRMED — periodic component requires trained positional embeddings"
    elif frac >= H_STRUCT_THRESHOLD:
        verdict = "H_struct CONFIRMED — periodic component is structural (persists without training)"
    else:
        verdict = f"AMBIGUOUS — {frac*100:.1f}% significant heads (between {H_PE_THRESHOLD*100:.0f}% and {H_STRUCT_THRESHOLD*100:.0f}%)"
    print(f"  {verdict}", flush=True)

    if res["median_peak_period_significant"] is not None:
        period_shift = abs(res["median_peak_period_significant"] - TRAINED_RAND_PERIOD)
        print(f"  Period shift vs trained: untrained={res['median_peak_period_significant']:.2f} "
              f"vs trained={TRAINED_RAND_PERIOD:.2f} (Δ={period_shift:.2f} lags)", flush=True)

    out = {
        "experiment": "exp-079",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": {
            "H_PE": f"< {H_PE_THRESHOLD*100:.0f}% sig heads if trained PE required",
            "H_struct": f">= {H_STRUCT_THRESHOLD*100:.0f}% sig heads if structural",
        },
        "baseline_exp053_rand": {
            "frac_significant": TRAINED_RAND_FRAC,
            "n_significant": 10,
            "median_peak_period": TRAINED_RAND_PERIOD,
        },
        "protocol": {
            "model": "gpt2-random-init",
            "seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
            "min_pos": MIN_POS, "max_dx": MAX_DX, "rng_seed": RNG_SEED,
            "k_fit": [K_LOW, K_HIGH], "peak_thresh": PEAK_THRESH,
        },
        "untrained_rand": res,
        "verdict": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
