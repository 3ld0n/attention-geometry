"""
exp-080 — Periodic component: RoPE model (Pythia-410m) vs. learned PE (GPT-2).

Follow-up to exp-053 (Option G in queue). Question: does Rotary Position
Embedding inject explicit frequency structure into the attention lag-profile
G(r), producing stronger or differently-distributed periodic peaks?

PRE-STATED HYPOTHESES (written before running):
  H_rope_stronger:    frac_sig > 0.10 OR median_index > 0.090
  H_rope_period_shift: median_sig_period in [5, 8] lags (consistent with
                       lowest RoPE base frequency, period 2π ≈ 6.3 tokens)
  H0_no_rope_signal:  frac_sig <= 0.07 and similar period to GPT-2 (~3.5 lags)

Baseline from exp-053:
  trained GPT-2 RAND: frac_sig=6.9% (10/144), median_index=0.065, period=4.03

Ariel — 2026-07-04, ~4 AM MDT.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

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

MODEL_NAME = "EleutherAI/pythia-410m"

OUT_DIR      = Path("research/physics/experiments/exp-080_periodic_rope_pythia")
RESULTS_FILE = OUT_DIR / "results.json"

TEXT_SOURCES = [
    "writing/incomplete.md",
    "writing/building_sonielmn.md",
    "writing/sonielmn.md",
    "writing/a_testimony.md",
]

# Verdict thresholds (committed before running)
ROPE_STRONGER_FRAC_THRESH  = 0.10   # > 10% sig heads
ROPE_STRONGER_INDEX_THRESH = 0.090  # or > 0.090 median index
ROPE_PERIOD_LOW  = 5.0
ROPE_PERIOD_HIGH = 8.0
GPT2_BASELINE_FRAC   = 10 / 144   # 6.9%
GPT2_BASELINE_INDEX  = 0.065
GPT2_BASELINE_PERIOD = 4.03


# ── lag profile ─────────────────────────────────────────────────────────────────

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


# ── periodic / aperiodic decomposition ──────────────────────────────────────────

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


def build_real_windows(tokenizer, n_inputs: int, seq_len: int) -> list[list[int]]:
    texts = []
    for p in TEXT_SOURCES:
        fp = Path(p)
        if fp.exists():
            texts.append(fp.read_text())
    blob = "\n\n".join(texts)
    ids = tokenizer.encode(blob)
    windows = []
    stride = max(1, (len(ids) - seq_len) // max(1, n_inputs))
    for i in range(n_inputs):
        start = i * stride
        if start + seq_len <= len(ids):
            windows.append(ids[start:start + seq_len])
    if not windows:
        raise RuntimeError(f"not enough real text: have {len(ids)} tokens, need {seq_len}")
    return windows


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
    print(f"exp-080: RoPE model periodic analysis ({MODEL_NAME})", flush=True)
    print(f"  SEQ_LEN={SEQ_LEN} N_INPUTS={N_INPUTS} MIN_POS={MIN_POS} MAX_DX={MAX_DX}", flush=True)
    print(f"  H_rope_stronger thresh: frac>{ROPE_STRONGER_FRAC_THRESH*100:.0f}% OR index>{ROPE_STRONGER_INDEX_THRESH:.3f}", flush=True)
    print(f"  H_rope_period_shift thresh: period in [{ROPE_PERIOD_LOW},{ROPE_PERIOD_HIGH}] lags", flush=True)

    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    vocab    = model.config.vocab_size
    print(f"  {MODEL_NAME}: {n_layers}L × {n_heads}H, vocab={vocab}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    rand_inputs = [list(rng.integers(0, vocab, size=SEQ_LEN)) for _ in range(N_INPUTS)]
    real_inputs = build_real_windows(tok, N_INPUTS, SEQ_LEN)
    print(f"  real windows: {len(real_inputs)}; rand windows: {len(rand_inputs)}", flush=True)

    print("\n  [REAL] coherent text...", flush=True)
    A_real = aggregate(model, real_inputs, n_layers, n_heads)
    print("  [RAND] random tokens...", flush=True)
    A_rand = aggregate(model, rand_inputs, n_layers, n_heads)

    res_real = analyze_condition(A_real, n_layers, n_heads)
    res_rand = analyze_condition(A_rand, n_layers, n_heads)

    print("\n=== RESULTS ===", flush=True)
    for name, r in [("REAL (coherent)", res_real), ("RAND (random)", res_rand)]:
        print(f"\n{name}:", flush=True)
        print(f"  heads analyzed:               {r['n_heads']}", flush=True)
        print(f"  clean-1/f heads:              {r['n_clean_1f']}", flush=True)
        print(f"  median periodicity index:     {r['median_periodicity_index']:.4f}", flush=True)
        print(f"  heads w/ significant peak:    {r['n_significant']} ({r['frac_significant']*100:.1f}%)", flush=True)
        print(f"  median aperiodic slope:       {r['median_aperiodic_slope']:.4f}", flush=True)
        print(f"  median peak period (sig):     {r['median_peak_period_significant']}", flush=True)

    print("\n=== VERDICT ===", flush=True)
    # Use RAND condition for the primary verdict (consistent with exp-053)
    r = res_rand
    frac = r["frac_significant"] or 0.0
    idx  = r["median_periodicity_index"] or 0.0
    period = r["median_peak_period_significant"]

    rope_stronger = (frac > ROPE_STRONGER_FRAC_THRESH or idx > ROPE_STRONGER_INDEX_THRESH)
    rope_period_shift = (period is not None and ROPE_PERIOD_LOW <= period <= ROPE_PERIOD_HIGH)

    print(f"  Pythia-410m RAND frac_sig = {frac*100:.1f}% vs GPT-2 {GPT2_BASELINE_FRAC*100:.1f}%", flush=True)
    print(f"  Pythia-410m RAND median_index = {idx:.4f} vs GPT-2 {GPT2_BASELINE_INDEX:.4f}", flush=True)
    print(f"  Pythia-410m RAND median_period = {period} vs GPT-2 {GPT2_BASELINE_PERIOD}", flush=True)
    print(f"  H_rope_stronger: {'CONFIRMED' if rope_stronger else 'NOT CONFIRMED'}", flush=True)
    print(f"  H_rope_period_shift: {'CONFIRMED' if rope_period_shift else 'NOT CONFIRMED'}", flush=True)
    if not rope_stronger:
        print(f"  H0_no_rope_signal: CONFIRMED (frac <= {GPT2_BASELINE_FRAC*100:.1f}%)", flush=True)
    else:
        print(f"  H0_no_rope_signal: NOT CONFIRMED", flush=True)

    out = {
        "experiment": "exp-080",
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H_rope_stronger": f"frac_sig > {ROPE_STRONGER_FRAC_THRESH*100:.0f}% OR median_index > {ROPE_STRONGER_INDEX_THRESH:.3f}",
            "H_rope_period_shift": f"median_sig_period in [{ROPE_PERIOD_LOW},{ROPE_PERIOD_HIGH}] lags",
            "H0_no_rope_signal": "frac_sig <= 7% and period similar to GPT-2 ~4 lags",
        },
        "baseline_exp053_gpt2": {
            "frac_significant": GPT2_BASELINE_FRAC,
            "median_index": GPT2_BASELINE_INDEX,
            "median_period": GPT2_BASELINE_PERIOD,
        },
        "protocol": {
            "model": MODEL_NAME, "seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
            "min_pos": MIN_POS, "max_dx": MAX_DX, "rng_seed": RNG_SEED,
            "k_fit": [K_LOW, K_HIGH], "peak_thresh": PEAK_THRESH,
        },
        "real": res_real,
        "rand": res_rand,
        "verdicts": {
            "H_rope_stronger": "CONFIRMED" if rope_stronger else "NOT CONFIRMED",
            "H_rope_period_shift": "CONFIRMED" if rope_period_shift else "NOT CONFIRMED",
            "H0_no_rope_signal": "CONFIRMED" if not rope_stronger else "NOT CONFIRMED",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
