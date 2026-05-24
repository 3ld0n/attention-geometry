"""
GPT-2 family depth convergence — learned positional embedding control.

Same per-head conformal protocol as exp-007 (gpt2_per_head_analysis.py).
Tests whether median Δ on R²>0.90 power-law heads stays at SYK q=4 (Δ≈0.25)
as depth increases: small (12L), medium (24L), large (36L).

Pre-registered prediction (queue May 19): Medium and Large should match
Δ≈0.25±0.01 tighter than RoPE Pythia at comparable depth if learned
embeddings select the SYK q=4 fixed point.

Ariel — May 24, 2026 (physics room)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
R2_THRESHOLD = 0.90
SYK_Q4_DELTA = 0.25
SYK_NEAR_TOL = 0.05


def compute_head_attention_decay(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    seq = attn_head.shape[0]
    A = np.zeros(max_dx)
    counts = np.zeros(max_dx)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray, cutoff_low: int = 3, cutoff_high: int | None = None):
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    exponent = -coeffs[1]
    return exponent / 2, R2


def run_model(model_id: str, n_inputs: int) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading {model_id}...")
    model = GPT2LMHeadModel.from_pretrained(model_id)
    model.eval()

    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    vocab_size = model.config.vocab_size
    dx_arr = np.arange(MAX_DX)

    A_heads = {l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)}

    print(f"Processing {n_inputs} inputs (seq_len={SEQ_LEN}, {n_layers}L, {n_heads}H)...")
    for inp_idx in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)
        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{n_inputs} done")

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= n_inputs

    all_results = []
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            delta, R2_pl = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            all_results.append({"layer": l + 1, "head": h, "delta": delta, "R2_pl": R2_pl})

    good = [
        r
        for r in all_results
        if r["delta"] is not None and r["R2_pl"] is not None and r["R2_pl"] > R2_THRESHOLD
    ]
    deltas = np.array([r["delta"] for r in good])
    syk_near = [r for r in good if abs(r["delta"] - SYK_Q4_DELTA) <= SYK_NEAR_TOL]

    layer_medians = {}
    for l in range(1, n_layers + 1):
        ld = [
            r["delta"]
            for r in good
            if r["layer"] == l
        ]
        if ld:
            layer_medians[str(l)] = float(np.median(ld))

    total_heads = n_layers * n_heads
    summary = {
        "model_id": model_id,
        "n_layer": n_layers,
        "n_head": n_heads,
        "total_heads": total_heads,
        "n_inputs": n_inputs,
        "protocol": "exp-007 per-head power-law, random tokens, R2>0.90",
        "delta_median": float(np.median(deltas)) if len(deltas) else None,
        "delta_mean": float(np.mean(deltas)) if len(deltas) else None,
        "delta_std": float(np.std(deltas)) if len(deltas) else None,
        "heads_R2_gt_90": len(good),
        "heads_syk_near_q4": len(syk_near),
        "fraction_syk_near": len(syk_near) / total_heads if total_heads else 0,
        "layer_median_delta": layer_medians,
        "syk_q4_prediction": SYK_Q4_DELTA,
        "deviation_from_syk": (
            float(np.median(deltas) - SYK_Q4_DELTA) if len(deltas) else None
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id, e.g. gpt2, gpt2-medium, gpt2-large",
    )
    parser.add_argument("--n-inputs", type=int, default=N_INPUTS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: results.json in script dir)",
    )
    args = parser.parse_args()

    summary = run_model(args.model, args.n_inputs)
    summary["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path = args.out or Path(__file__).parent / "results.json"
    # If multiple models, caller should pass distinct --out
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    if "models" not in existing:
        existing = {"models": {}, "protocol": summary["protocol"]}
    existing["models"][args.model] = summary
    existing["last_updated_utc"] = summary["timestamp_utc"]
    out_path.write_text(json.dumps(existing, indent=2))

    print()
    print("=" * 60)
    print(f"  {args.model}: {summary['n_layer']}L × {summary['n_head']}H")
    print(f"  Heads R²>{R2_THRESHOLD}: {summary['heads_R2_gt_90']}/{summary['total_heads']}")
    print(f"  Median Δ = {summary['delta_median']:.4f}" if summary["delta_median"] else "  No PL heads")
    print(f"  |Δ−0.25| = {abs(summary['deviation_from_syk']):.4f}" if summary["deviation_from_syk"] is not None else "")
    print(f"  SYK-near (|Δ−0.25|≤{SYK_NEAR_TOL}): {summary['heads_syk_near_q4']}/{summary['total_heads']}")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
