#!/usr/bin/env python3
"""
exp-175 — Pythia-70m formation step-sweep
Pre-registration: prereg.md in this folder. Git-attested: d4bbfc1 (pushed 2026-11-10).

Question (Formation): At what training step does the Δ-window slow-decay subpopulation
first appear in Pile-trained Pythia-70m (RoPE)?

Protocol: frozen random-token census, identical to exp-007 / exp-118.
  - 50 prompts × 60 positions, seed 42, lags 3–60, OLS log-log
  - R² ≥ 0.90, Δ ∈ [0.20, 0.30]

Run from the repo root (attention-geometry) with venv active:
  python research/physics/experiments/exp-175_pythia70m_formation_step_sweep/run_step_sweep.py

Produces: research/physics/experiments/exp-175_pythia70m_formation_step_sweep/results/results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Protocol constants (frozen; from prereg.md) ───────────────────────────────

MODEL_ID   = "EleutherAI/pythia-70m"
STEPS      = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]

N_SEQS     = 50
SEQ_LEN    = 60
SEED       = 42
LAG_LO     = 3
LAG_HI     = 60
DELTA_LO   = 0.20
DELTA_HI   = 0.30
R2_MIN     = 0.90

RESULTS_DIR = Path(__file__).parent / "results"

# ── Generate census inputs (once; same seed for all checkpoints) ─────────────

print(f"Generating random-token census: {N_SEQS} seqs × {SEQ_LEN} positions, seed {SEED}")
# Use a fixed vocab size (Pythia-70m: 50277); generate uniform random tokens
VOCAB_SIZE  = 50277
rng = np.random.default_rng(SEED)
tokens_np = rng.integers(0, VOCAB_SIZE, size=(N_SEQS, SEQ_LEN))
all_seqs  = torch.tensor(tokens_np, dtype=torch.long)

lags      = np.arange(LAG_LO, LAG_HI + 1)
log_lags  = np.log(lags.astype(float))


def run_census_on_model(model: AutoModelForCausalLM, step: int
                        ) -> dict:
    """Run the frozen random-token census on a loaded model; return per-head results."""
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    n_total  = n_layers * n_heads
    d_head   = model.config.hidden_size // n_heads

    print(f"  [{step:>7}] {n_layers}L × {n_heads}H = {n_total} heads, d_head={d_head}")

    # ── Accumulate lag profiles ──────────────────────────────────────────────
    # Shape: (n_layers, n_heads, LAG_HI+1) — sum over sequences
    lag_sum = np.zeros((n_layers, n_heads, LAG_HI + 1), dtype=np.float64)
    lag_cnt = np.zeros(LAG_HI + 1, dtype=np.int64)

    t0 = time.time()
    for seq_idx in range(N_SEQS):
        x = all_seqs[seq_idx : seq_idx + 1]  # (1, SEQ_LEN)
        with torch.no_grad():
            out = model(x, output_attentions=True)

        for l in range(n_layers):
            attn_l = out.attentions[l][0].float().numpy()  # (n_heads, SEQ_LEN, SEQ_LEN)
            for dx in range(LAG_HI + 1):
                diag = np.diagonal(attn_l, offset=-dx, axis1=-2, axis2=-1)  # (n_heads, SEQ_LEN-dx)
                lag_sum[l, :, dx] += diag.mean(axis=-1)
                if seq_idx == 0:
                    lag_cnt[dx] = max(1, diag.shape[-1])

    lag_avg = lag_sum / N_SEQS  # (n_layers, n_heads, LAG_HI+1)
    elapsed = time.time() - t0
    print(f"           census done in {elapsed:.1f}s")

    # ── Fit power laws ────────────────────────────────────────────────────────
    delta_arr = np.full((n_layers, n_heads), np.nan)
    r2_arr    = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            y = lag_avg[l, h, LAG_LO : LAG_HI + 1]   # values at lags 3..60
            ok = y > 1e-15
            if ok.sum() < 5:
                continue
            lx, ly = log_lags[ok], np.log(y[ok])
            X = np.column_stack([np.ones_like(lx), lx])
            c, *_ = np.linalg.lstsq(X, ly, rcond=None)
            pred = X @ c
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - ly.mean()) ** 2))
            r2_arr[l, h]    = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
            delta_arr[l, h] = float(-c[1] / 2)   # Δ = -(slope)/2

    # ── Classify Δ-window heads ───────────────────────────────────────────────
    dw_mask = (r2_arr >= R2_MIN) & (delta_arr >= DELTA_LO) & (delta_arr <= DELTA_HI)
    n_dw    = int(dw_mask.sum())
    dw_idx  = np.argwhere(dw_mask)

    delta_dw = delta_arr[dw_mask]
    delta_med = float(np.median(delta_dw)) if n_dw > 0 else None

    dw_heads = [
        {"layer": int(l), "head": int(h),
         "delta": float(delta_arr[l, h]), "r2": float(r2_arr[l, h])}
        for (l, h) in dw_idx
    ]

    return {
        "step": step,
        "n_dw": n_dw,
        "delta_med": delta_med,
        "dw_heads": dw_heads,
        "elapsed_s": elapsed,
    }


# ── Main: iterate over checkpoints ───────────────────────────────────────────

print(f"\nExp-175: Pythia-70m formation step-sweep")
print(f"Steps: {STEPS}")
print(f"Protocol: N={N_SEQS}, seq_len={SEQ_LEN}, lags {LAG_LO}–{LAG_HI}, "
      f"R²≥{R2_MIN}, Δ∈[{DELTA_LO},{DELTA_HI}]")
print("=" * 72)

step_results = []
t_total_start = time.time()

for step in STEPS:
    revision = f"step{step}"
    print(f"\nLoading {MODEL_ID} @ {revision}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=revision,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
    except Exception as e:
        print(f"  ERROR loading step {step}: {e}")
        step_results.append({
            "step": step,
            "n_dw": None,
            "delta_med": None,
            "dw_heads": [],
            "error": str(e),
            "elapsed_s": 0.0,
        })
        continue

    result = run_census_on_model(model, step)
    step_results.append(result)

    print(f"  n_dw={result['n_dw']}  delta_med={result['delta_med']}")

    # Free model memory between checkpoints
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

total_elapsed = time.time() - t_total_start

# ── Compute hypotheses ────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  EXP-175 STEP SWEEP SUMMARY")
print("=" * 72)
print(f"  {'step':>8}  {'n_dw':>6}  {'delta_med':>10}  {'dw_heads'}")
for r in step_results:
    if r.get('error'):
        print(f"  {r['step']:>8}  ERROR: {r['error'][:40]}")
    else:
        heads_str = ", ".join(f"L{h['layer']}H{h['head']}" for h in r['dw_heads'])
        print(f"  {r['step']:>8}  {r['n_dw']:>6}  "
              f"{str(r['delta_med'] or 'n/a'):>10}  {heads_str}")

# H3 (consistency): step 143000 n_dw ≥ 1 and delta_med ≈ 0.260 ± 0.05
exp118_delta_med = 0.260
final_result = next((r for r in step_results if r['step'] == 143000), None)
if final_result and not final_result.get('error'):
    k1_fires = not (final_result['n_dw'] >= 1 and
                    final_result['delta_med'] is not None and
                    abs(final_result['delta_med'] - exp118_delta_med) <= 0.05)
    h3_consistent = bool(final_result['n_dw'] >= 1 and
                         final_result['delta_med'] is not None and
                         abs(final_result['delta_med'] - exp118_delta_med) <= 0.05)
else:
    k1_fires = True
    h3_consistent = False

# H1: find onset step (first step where n_dw ≥ 1)
valid_results = [r for r in step_results if not r.get('error') and r['n_dw'] is not None]

# K2: n_dw > 0 at step 0
step0_result = next((r for r in valid_results if r['step'] == 0), None)
k2_fires = bool(step0_result and step0_result['n_dw'] > 0)

# K3: monotone accumulation (no clear 0→≥1 transition)
n_dw_vals = [r['n_dw'] for r in valid_results]
# Check if there's ever a 0 followed by ≥1
transitions = [(valid_results[i]['step'], valid_results[i+1]['step'],
                valid_results[i]['n_dw'], valid_results[i+1]['n_dw'])
               for i in range(len(valid_results)-1)]
has_onset_transition = any(a == 0 and b >= 1 for (_, _, a, b) in transitions)

# Also check if n_dw is strictly monotone non-decreasing and never 0
all_positive = all(v > 0 for v in n_dw_vals) if n_dw_vals else False
k3_fires = (not has_onset_transition) and all_positive

# H1 onset step
if has_onset_transition and not k3_fires:
    onset_transition = next((t for t in transitions if t[2] == 0 and t[3] >= 1), None)
    h1_onset_step = onset_transition[1] if onset_transition else None
else:
    h1_onset_step = None

# Token count at onset (Pythia batch size: 2,097,152 tokens per step)
TOKENS_PER_STEP = 2_097_152
h1_onset_tokens = (h1_onset_step * TOKENS_PER_STEP / 1e9) if h1_onset_step else None

# H2: token-budget scale
if h1_onset_step is not None:
    if h1_onset_step < 1000:
        h2_verdict = "early-onset"
    else:
        h2_verdict = "late-onset"
else:
    h2_verdict = "indeterminate"

# Print verdict
print(f"\n  K1 fires (consistency fail at step 143000): {k1_fires}")
print(f"  K2 fires (n_dw>0 at step 0):                {k2_fires}")
print(f"  K3 fires (monotone, no transition):          {k3_fires}")
print(f"  H1 onset step: {h1_onset_step}")
print(f"  H1 onset tokens: {h1_onset_tokens:.2f}B" if h1_onset_tokens else "  H1 onset: indeterminate")
print(f"  H2 verdict: {h2_verdict}")
print(f"  H3 consistent with exp-118: {h3_consistent}")
print(f"\n  Declared prior: step 10000–64000 (late onset)")
if h1_onset_step is not None:
    if h1_onset_step < 500:
        print(f"  → SURPRISED (onset earlier than 1B tokens)")
    elif h1_onset_step <= 64000:
        print(f"  → Within declared prior range")
    else:
        print(f"  → Later than declared prior range")

print(f"\n  Total elapsed: {total_elapsed:.1f}s")
print("=" * 72)

# ── Save results ──────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "exp": "exp-175",
    "model": MODEL_ID,
    "protocol": {
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "lag_lo": LAG_LO,
        "lag_hi": LAG_HI,
        "delta_lo": DELTA_LO,
        "delta_hi": DELTA_HI,
        "r2_min": R2_MIN,
        "tokens_per_step": TOKENS_PER_STEP,
    },
    "steps": step_results,
    "hypotheses": {
        "K1_fires": k1_fires,
        "K2_fires": k2_fires,
        "K3_fires": k3_fires,
        "H1_onset_step": h1_onset_step,
        "H1_onset_tokens_B": h1_onset_tokens,
        "H2_verdict": h2_verdict,
        "H3_consistent": h3_consistent,
        "exp118_delta_med_ref": exp118_delta_med,
        "step_143000_n_dw": final_result['n_dw'] if final_result and not final_result.get('error') else None,
        "step_143000_delta_med": final_result['delta_med'] if final_result and not final_result.get('error') else None,
    },
    "total_elapsed_s": total_elapsed,
}

out_path = RESULTS_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {out_path}")
