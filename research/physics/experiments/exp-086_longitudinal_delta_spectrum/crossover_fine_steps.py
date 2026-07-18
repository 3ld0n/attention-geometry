"""
exp-086 follow-up — Finer Step Sweep Through the RAND/NAT Crossover

Analysis-only. No new experiment number. Output → research/physics/notes/.

Context: exp-086 identified a RAND/NAT crossover between step 1000 and step 4000.
The per-layer analysis (crossover_layer_analysis.py, 2026-07-17) found an L3/L4
phase transition: at step 1000 NAT heads have large Δ (semantic-capture); by step
4000 L3 and L4 converge with SYK-near structure. Zero stable heads 1000→4000 —
all NAT SYK-near heads at step 4000 appear for the first time.

This script adds step 512, 2000, and 3000 to map the transition with finer resolution.
Checkpoints available from EleutherAI/pythia-70m on HuggingFace:
  step256, step512, step1000, step2000, step3000, step4000

Question: Is the phase transition at step 4000 abrupt (happening between two adjacent
checkpoints) or gradual? When do the L3/L4 SYK-near heads first appear?

Ariel — July 18, 2026.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── constants (inheriting from exp-086 pre-registration) ─────────────────────

MODEL_NAME  = "EleutherAI/pythia-70m"
CHECKPOINTS = [256, 512, 1000, 2000, 3000, 4000]   # finer than exp-086

SEQ_LEN     = 256
N_RAND      = 20
RAND_SEED   = 86

FIT_LOW      = 3
FIT_HIGH     = 50
R2_THRESHOLD = 0.85
SYK_LOW      = 0.20
SYK_HIGH     = 0.30

# Same fixed natural language passage as exp-086
NAT_TEXT = (
    "Once upon a time, there was a little girl named Lily. She lived with her family "
    "in a small house near a big forest. One day, Lily decided to go on an adventure "
    "in the forest. She put on her coat and her boots and walked into the trees. The "
    "forest was very quiet. She could hear the birds singing and the wind blowing "
    "through the leaves."
)

OUT_DIR   = Path(__file__).resolve().parent
NOTES_DIR = Path(__file__).resolve().parents[3] / "notes"


# ─── measurement helpers (same as crossover_layer_analysis.py) ────────────────

def compute_attention_decay(attn_head: np.ndarray, max_dx: int = FIT_HIGH + 5) -> np.ndarray:
    return np.array([
        np.diag(attn_head, -dx).mean() if dx < attn_head.shape[0] else 0.0
        for dx in range(max_dx)
    ])


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float | None, float | None]:
    mask = (dx_arr >= FIT_LOW) & (dx_arr < FIT_HIGH) & (y_arr > 1e-20)
    if mask.sum() < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y  = np.log(y_arr[mask])
    A      = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred   = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return -coeffs[1] / 2.0, R2


def measure_per_head(
    model: torch.nn.Module,
    input_batch: list[torch.Tensor],
    n_layers: int,
    n_heads: int,
) -> list[dict]:
    max_dx = FIT_HIGH + 5
    dx_arr = np.arange(max_dx)

    A_sum = [[np.zeros(max_dx) for _ in range(n_heads)] for _ in range(n_layers)]

    for inp in input_batch:
        with torch.no_grad():
            out = model(inp.unsqueeze(0), output_attentions=True)
        for l_idx in range(n_layers):
            attn = out.attentions[l_idx][0].numpy()
            for h_idx in range(n_heads):
                A_sum[l_idx][h_idx] += compute_attention_decay(attn[h_idx], max_dx)

    n = len(input_batch)
    records = []
    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            A_avg = A_sum[l_idx][h_idx] / n
            delta, R2 = fit_power_law(dx_arr, A_avg)
            conformal = (delta is not None) and (R2 >= R2_THRESHOLD)
            syk_near  = conformal and (SYK_LOW <= delta <= SYK_HIGH)
            records.append({
                "layer":     l_idx,
                "head":      h_idx,
                "delta":     float(delta) if delta is not None else None,
                "R2":        float(R2)    if R2    is not None else None,
                "conformal": conformal,
                "syk_near":  syk_near,
            })
    return records


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  exp-086 crossover fine-step sweep — steps 256/512/1000/2000/3000/4000")
    print("=" * 80)
    print()

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    rng         = np.random.default_rng(RAND_SEED)
    rand_ids    = rng.integers(0, vocab_size, size=(N_RAND, SEQ_LEN))
    rand_inputs = [torch.tensor(ids, dtype=torch.long) for ids in rand_ids]

    nat_enc = tokenizer.encode(NAT_TEXT)
    if len(nat_enc) > SEQ_LEN:
        nat_enc = nat_enc[:SEQ_LEN]
    else:
        nat_enc = nat_enc + [0] * (SEQ_LEN - len(nat_enc))
    nat_inputs = [torch.tensor(nat_enc, dtype=torch.long)] * N_RAND

    all_results: dict[int, dict] = {}

    for step in CHECKPOINTS:
        revision = f"step{step}"
        print(f"\n─── checkpoint step={step} (revision={revision}) ───")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=revision,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        n_heads  = model.config.num_attention_heads
        print(f"  Loaded in {time.time()-t0:.1f}s  ({n_layers}L × {n_heads}H)")

        rand_records = measure_per_head(model, rand_inputs, n_layers, n_heads)
        nat_records  = measure_per_head(model, nat_inputs,  n_layers, n_heads)

        all_results[step] = {"rand": rand_records, "nat": nat_records}

        # Per-layer summary
        print(f"\n  {'L':>3}  {'R_syk':>6}  {'N_syk':>6}  {'R_Δmed':>7}  {'N_Δmed':>7}")
        for l_idx in range(n_layers):
            r_l   = [r for r in rand_records if r["layer"] == l_idx]
            n_l   = [r for r in nat_records  if r["layer"] == l_idx]
            r_syk = sum(r["syk_near"]  for r in r_l)
            n_syk = sum(r["syk_near"]  for r in n_l)
            r_ds  = [r["delta"] for r in r_l if r["conformal"] and r["delta"] is not None]
            n_ds  = [r["delta"] for r in n_l if r["conformal"] and r["delta"] is not None]
            r_med = f"{np.median(r_ds):.3f}" if r_ds else "  N/A"
            n_med = f"{np.median(n_ds):.3f}" if n_ds else "  N/A"
            print(f"  {l_idx:>3}  {r_syk:>6}  {n_syk:>6}  {r_med:>7}  {n_med:>7}")

        r_syk_ids = [(r["layer"], r["head"]) for r in rand_records if r["syk_near"]]
        n_syk_ids = [(r["layer"], r["head"]) for r in nat_records  if r["syk_near"]]
        print(f"\n  RAND SYK-near: {sorted(r_syk_ids)}  (n={len(r_syk_ids)})")
        print(f"  NAT  SYK-near: {sorted(n_syk_ids)}  (n={len(n_syk_ids)})")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── crossover tracking across all steps ─────────────────────────────────

    print("\n" + "=" * 80)
    print("  CROSSOVER TRACKING — SYK-near head count across steps")
    print("=" * 80)

    print(f"\n  {'Step':>6}  {'RAND':>6}  {'NAT':>6}  {'RAND−NAT':>10}")
    print("  " + "─" * 34)
    for step in CHECKPOINTS:
        r_n = sum(r["syk_near"] for r in all_results[step]["rand"])
        n_n = sum(r["syk_near"] for r in all_results[step]["nat"])
        print(f"  {step:>6}  {r_n:>6}  {n_n:>6}  {r_n - n_n:>+10}")

    # ─── per-step: which heads are SYK-near? ─────────────────────────────────

    print(f"\n  SYK-near head identities across steps:")
    print(f"  {'Step':>6}  {'RAND heads':50}  {'NAT heads'}")
    print("  " + "─" * 80)
    for step in CHECKPOINTS:
        r_ids = sorted((r["layer"], r["head"]) for r in all_results[step]["rand"] if r["syk_near"])
        n_ids = sorted((r["layer"], r["head"]) for r in all_results[step]["nat"]  if r["syk_near"])
        print(f"  {step:>6}  {str(r_ids):50}  {n_ids}")

    # ─── specific heads: first appearance in NAT ──────────────────────────────

    # Collect all NAT heads that appear SYK-near at any point
    all_nat_heads: set[tuple] = set()
    for step in CHECKPOINTS:
        for r in all_results[step]["nat"]:
            if r["syk_near"]:
                all_nat_heads.add((r["layer"], r["head"]))

    print(f"\n  First appearance step for each NAT SYK-near head:")
    for head in sorted(all_nat_heads):
        for step in CHECKPOINTS:
            if any(r["layer"] == head[0] and r["head"] == head[1] and r["syk_near"]
                   for r in all_results[step]["nat"]):
                print(f"    {head}: first appears at step {step}")
                break

    # ─── Δ evolution for L3 and L4 (the phase-transition layers) ─────────────

    print(f"\n  Per-step Δ_med for L3 and L4 (the crossover layers):")
    print(f"  {'Step':>6}  {'L3_RAND':>8}  {'L3_NAT':>8}  {'L4_RAND':>8}  {'L4_NAT':>8}")
    print("  " + "─" * 50)
    for step in CHECKPOINTS:
        def layer_delta_med(records, l):
            ds = [r["delta"] for r in records if r["layer"] == l
                  and r["conformal"] and r["delta"] is not None]
            return f"{np.median(ds):.3f}" if ds else "  N/A"
        r3 = layer_delta_med(all_results[step]["rand"], 3)
        n3 = layer_delta_med(all_results[step]["nat"], 3)
        r4 = layer_delta_med(all_results[step]["rand"], 4)
        n4 = layer_delta_med(all_results[step]["nat"], 4)
        print(f"  {step:>6}  {r3:>8}  {n3:>8}  {r4:>8}  {n4:>8}")

    # ─── save JSON ────────────────────────────────────────────────────────────

    def _serialize(rec: dict) -> dict:
        return {
            "layer":     int(rec["layer"]),
            "head":      int(rec["head"]),
            "delta":     float(rec["delta"]) if rec["delta"] is not None else None,
            "R2":        float(rec["R2"])    if rec["R2"]    is not None else None,
            "conformal": bool(rec["conformal"]),
            "syk_near":  bool(rec["syk_near"]),
        }

    out_json = {
        "experiment":   "exp-086-crossover-fine",
        "date":         "2026-07-18",
        "model":        MODEL_NAME,
        "checkpoints":  [int(s) for s in CHECKPOINTS],
        "n_rand":       int(N_RAND),
        "rand_seed":    int(RAND_SEED),
        "seq_len":      int(SEQ_LEN),
        "fit_low":      int(FIT_LOW),
        "fit_high":     int(FIT_HIGH),
        "r2_threshold": float(R2_THRESHOLD),
        "syk_window":   [float(SYK_LOW), float(SYK_HIGH)],
        "per_head_results": {
            str(step): {
                "rand": [_serialize(r) for r in results["rand"]],
                "nat":  [_serialize(r) for r in results["nat"]],
                "rand_syk_near": int(sum(r["syk_near"] for r in results["rand"])),
                "nat_syk_near":  int(sum(r["syk_near"] for r in results["nat"])),
            }
            for step, results in all_results.items()
        },
    }

    class _NumpySafe(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"):   # numpy scalar
                return obj.item()
            return super().default(obj)

    out_path = OUT_DIR / "crossover_fine_results.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2, cls=_NumpySafe)
    print(f"\nWrote {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
