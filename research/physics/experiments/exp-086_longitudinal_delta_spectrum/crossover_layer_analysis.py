"""
exp-086 follow-up — Per-Layer Δ Profiles at the Crossover (steps 1000 vs 4000)

Analysis-only. No new experiment number. Output → research/physics/notes/.

Question: At step 1000 RAND dominates (8 vs 1 SYK-near); at step 4000 NAT surpasses
(5 RAND vs 8 NAT). Which layers/heads drive this crossover? Do specific layers acquire
SYK-near structure on NAT but not RAND between steps 1000 and 4000?

Protocol: same fixed inputs as exp-086 (RAND: N=20, seed=86; NAT: same passage).
Checkpoints: step256 (pre-crossover control), step1000, step4000.
Output: per-head Δ, R², SYK-near flag for each (step × condition).
Note: written to research/physics/notes/2026-07-17_crossover_layer_analysis.md.

Ariel — July 17, 2026.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── constants (inheriting from exp-086 pre-registration) ─────────────────────

MODEL_NAME   = "EleutherAI/pythia-70m"
CHECKPOINTS  = [256, 1000, 4000]   # pre-crossover, crossover start, crossover end
SEQ_LEN      = 256
N_RAND       = 20
RAND_SEED    = 86

FIT_LOW      = 3
FIT_HIGH     = 50
R2_THRESHOLD = 0.85
SYK_LOW      = 0.20
SYK_HIGH     = 0.30

NAT_TEXT = (
    "Once upon a time, there was a little girl named Lily. She lived with her family "
    "in a small house near a big forest. One day, Lily decided to go on an adventure "
    "in the forest. She put on her coat and her boots and walked into the trees. The "
    "forest was very quiet. She could hear the birds singing and the wind blowing "
    "through the leaves."
)

OUT_DIR = Path(__file__).resolve().parent
NOTES_DIR = Path(__file__).resolve().parents[3] / "notes"


# ─── measurement helpers (same as exp-086) ────────────────────────────────────

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
    """
    Returns a list of dicts, one per (layer, head), with keys:
      layer, head, delta, R2, syk_near, conformal
    """
    max_dx = FIT_HIGH + 5
    dx_arr = np.arange(max_dx)

    # Accumulate attention profiles
    A_sum = [[np.zeros(max_dx) for _ in range(n_heads)] for _ in range(n_layers)]

    for inp in input_batch:
        with torch.no_grad():
            out = model(inp.unsqueeze(0), output_attentions=True)
        for l_idx in range(n_layers):
            attn = out.attentions[l_idx][0].numpy()  # (n_heads, seq, seq)
            for h_idx in range(n_heads):
                A_sum[l_idx][h_idx] += compute_attention_decay(attn[h_idx], max_dx)

    n = len(input_batch)
    records = []
    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            A_avg = A_sum[l_idx][h_idx] / n
            delta, R2 = fit_power_law(dx_arr, A_avg)
            conformal  = (delta is not None) and (R2 >= R2_THRESHOLD)
            syk_near   = conformal and (SYK_LOW <= delta <= SYK_HIGH)
            records.append({
                "layer":     l_idx,
                "head":      h_idx,
                "delta":     float(delta) if delta is not None else None,
                "R2":        float(R2) if R2 is not None else None,
                "conformal": conformal,
                "syk_near":  syk_near,
            })
    return records


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  exp-086 crossover analysis — per-layer Δ at steps 256, 1000, 4000")
    print("=" * 80)
    print()

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    rng       = np.random.default_rng(RAND_SEED)
    rand_ids  = rng.integers(0, vocab_size, size=(N_RAND, SEQ_LEN))
    rand_inputs = [torch.tensor(ids, dtype=torch.long) for ids in rand_ids]

    nat_enc = tokenizer.encode(NAT_TEXT)
    if len(nat_enc) > SEQ_LEN:
        nat_enc = nat_enc[:SEQ_LEN]
    else:
        nat_enc = nat_enc + [0] * (SEQ_LEN - len(nat_enc))
    nat_inputs = [torch.tensor(nat_enc, dtype=torch.long)] * N_RAND

    all_results = {}   # step → {"rand": [...], "nat": [...]}

    for step in CHECKPOINTS:
        revision = f"step{step}"
        print(f"\n─── checkpoint step={step} ───")
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

        t1 = time.time()
        rand_records = measure_per_head(model, rand_inputs, n_layers, n_heads)
        print(f"  RAND measured in {time.time()-t1:.1f}s")

        t2 = time.time()
        nat_records  = measure_per_head(model, nat_inputs,  n_layers, n_heads)
        print(f"  NAT  measured in {time.time()-t2:.1f}s")

        all_results[step] = {"rand": rand_records, "nat": nat_records}

        # Quick summary per layer
        print(f"\n  Per-layer SYK-near count  (L=layer, R=RAND, N=NAT):")
        print(f"  {'L':>3}  {'R_syk':>6}  {'N_syk':>6}  {'R_conf':>7}  {'N_conf':>7}  "
              f"{'R_Δmed':>7}  {'N_Δmed':>7}")
        for l_idx in range(n_layers):
            r_l = [r for r in rand_records if r["layer"] == l_idx]
            n_l = [r for r in nat_records  if r["layer"] == l_idx]
            r_syk  = sum(r["syk_near"]  for r in r_l)
            n_syk  = sum(r["syk_near"]  for r in n_l)
            r_conf = sum(r["conformal"] for r in r_l)
            n_conf = sum(r["conformal"] for r in n_l)
            r_ds   = [r["delta"] for r in r_l if r["conformal"] and r["delta"] is not None]
            n_ds   = [r["delta"] for r in n_l if r["conformal"] and r["delta"] is not None]
            r_med  = f"{np.median(r_ds):.3f}" if r_ds else "  N/A"
            n_med  = f"{np.median(n_ds):.3f}" if n_ds else "  N/A"
            print(f"  {l_idx:>3}  {r_syk:>6}  {n_syk:>6}  {r_conf:>7}  {n_conf:>7}  "
                  f"{r_med:>7}  {n_med:>7}")

        # SYK-near head identities
        r_syk_ids = [(r["layer"], r["head"]) for r in rand_records if r["syk_near"]]
        n_syk_ids = [(r["layer"], r["head"]) for r in nat_records  if r["syk_near"]]
        print(f"\n  RAND SYK-near heads: {r_syk_ids}")
        print(f"  NAT  SYK-near heads: {n_syk_ids}")
        only_rand = set(r_syk_ids) - set(n_syk_ids)
        only_nat  = set(n_syk_ids) - set(r_syk_ids)
        shared    = set(r_syk_ids) & set(n_syk_ids)
        print(f"  Only RAND: {sorted(only_rand)}")
        print(f"  Only NAT:  {sorted(only_nat)}")
        print(f"  Shared:    {sorted(shared)}")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── crossover analysis ──────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("  CROSSOVER ANALYSIS: step1000 → step4000")
    print("=" * 80)

    for cond, cond_label in [("rand", "RAND"), ("nat", "NAT")]:
        print(f"\n  {cond_label}:")
        ids_1000 = {(r["layer"], r["head"])
                    for r in all_results[1000][cond] if r["syk_near"]}
        ids_4000 = {(r["layer"], r["head"])
                    for r in all_results[4000][cond] if r["syk_near"]}
        ids_256  = {(r["layer"], r["head"])
                    for r in all_results[256][cond]  if r["syk_near"]}
        gained   = ids_4000 - ids_1000
        lost     = ids_1000 - ids_4000
        stable   = ids_1000 & ids_4000
        new_256  = ids_1000 - ids_256
        print(f"    step256:  n={len(ids_256)}  heads={sorted(ids_256)}")
        print(f"    step1000: n={len(ids_1000)}  heads={sorted(ids_1000)}")
        print(f"    step4000: n={len(ids_4000)}  heads={sorted(ids_4000)}")
        print(f"    Gained 1000→4000:  {sorted(gained)}")
        print(f"    Lost   1000→4000:  {sorted(lost)}")
        print(f"    Stable 1000→4000:  {sorted(stable)}")

    # ─── save JSON ────────────────────────────────────────────────────────────

    def _serialize(rec: dict) -> dict:
        """Convert numpy/Python booleans and None-safe floats for JSON."""
        return {
            "layer":     int(rec["layer"]),
            "head":      int(rec["head"]),
            "delta":     float(rec["delta"]) if rec["delta"] is not None else None,
            "R2":        float(rec["R2"])    if rec["R2"]    is not None else None,
            "conformal": bool(rec["conformal"]),
            "syk_near":  bool(rec["syk_near"]),
        }

    out_json = {
        "experiment": "exp-086-crossover",
        "date": "2026-07-17",
        "model": MODEL_NAME,
        "checkpoints": CHECKPOINTS,
        "n_rand": N_RAND,
        "rand_seed": RAND_SEED,
        "seq_len": SEQ_LEN,
        "per_head_results": {
            str(step): {
                "rand": [_serialize(r) for r in results["rand"]],
                "nat":  [_serialize(r) for r in results["nat"]],
            }
            for step, results in all_results.items()
        },
    }

    out_path = OUT_DIR / "crossover_results.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
