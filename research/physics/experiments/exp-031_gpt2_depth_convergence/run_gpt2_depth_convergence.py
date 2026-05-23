"""
GPT-2 Medium/Large depth convergence — learned-embedding control (exp-031).

Same per-head protocol as exp-007 (gpt2_per_head_analysis.py):
  random inputs, SEQ_LEN=256, N_INPUTS=50, FIT_LOW=3, FIT_HIGH=50,
  R²>0.90 power-law heads → median Δ.

Prediction: learned embeddings → SYK q=4, median Δ ≈ 0.25 ± 0.01 at 24L and 36L.
"""

from __future__ import annotations

import argparse
import json
import sys
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
SYK_DELTA = 0.25

MODELS = {
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
}


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


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray, cutoff_low: int, cutoff_high: int):
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    exponent = -coeffs[1]
    return exponent, exponent / 2, R2


def analyze_model(model_id: str, short_name: str) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n{'=' * 72}\n  {short_name} ({model_id})\n{'=' * 72}")
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(
        model_id,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    vocab_size = model.config.vocab_size
    dx_arr = np.arange(MAX_DX)

    A_heads = {
        layer: {h: np.zeros(MAX_DX) for h in range(n_heads)} for layer in range(n_layers)
    }

    print(f"  layers={n_layers} heads/layer={n_heads} total_heads={n_layers * n_heads}")
    print(f"  Processing {N_INPUTS} random inputs (seq_len={SEQ_LEN})...")

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        if outputs.attentions is None:
            raise RuntimeError(
                f"{model_id}: attentions not returned — need eager attention implementation"
            )

        for layer in range(n_layers):
            attn = outputs.attentions[layer][0].cpu().numpy()
            for h in range(n_heads):
                A_heads[layer][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)

        if (inp_idx + 1) % 10 == 0:
            print(f"    {inp_idx + 1}/{N_INPUTS}")

    for layer in range(n_layers):
        for h in range(n_heads):
            A_heads[layer][h] /= N_INPUTS

    all_results = []
    for layer in range(n_layers):
        for h in range(n_heads):
            A = A_heads[layer][h]
            _, delta, r2_pl = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            all_results.append(
                {
                    "layer": layer + 1,
                    "head": h,
                    "delta": None if delta is None else float(delta),
                    "R2_pl": None if r2_pl is None else float(r2_pl),
                }
            )

    good = [
        r
        for r in all_results
        if r["delta"] is not None
        and r["R2_pl"] is not None
        and r["R2_pl"] > R2_THRESHOLD
    ]
    deltas = np.array([r["delta"] for r in good])
    syk_near = [r for r in good if abs(r["delta"] - SYK_DELTA) <= 0.06]

    layer_medians = {}
    for layer in range(1, n_layers + 1):
        ld = [
            r["delta"]
            for r in good
            if r["layer"] == layer
        ]
        if ld:
            layer_medians[str(layer)] = {
                "n": len(ld),
                "median_delta": float(np.median(ld)),
                "mean_delta": float(np.mean(ld)),
            }

    summary = {
        "short_name": short_name,
        "model_id": model_id,
        "n_layers": n_layers,
        "n_heads_per_layer": n_heads,
        "total_heads": n_layers * n_heads,
        "protocol": {
            "source": "exp-007 gpt2_per_head_analysis.py",
            "SEQ_LEN": SEQ_LEN,
            "N_INPUTS": N_INPUTS,
            "FIT_LOW": FIT_LOW,
            "FIT_HIGH": FIT_HIGH,
            "R2_threshold": R2_THRESHOLD,
        },
        "heads_R2_gt_threshold": len(good),
        "heads_syk_near_q4": len(syk_near),
        "delta_median": float(np.median(deltas)) if len(deltas) else None,
        "delta_mean": float(np.mean(deltas)) if len(deltas) else None,
        "delta_std": float(np.std(deltas)) if len(deltas) else None,
        "delta_min": float(np.min(deltas)) if len(deltas) else None,
        "delta_max": float(np.max(deltas)) if len(deltas) else None,
        "syk_prediction": SYK_DELTA,
        "delta_minus_syk": (
            float(np.median(deltas) - SYK_DELTA) if len(deltas) else None
        ),
        "within_0.01_of_syk": (
            abs(float(np.median(deltas)) - SYK_DELTA) <= 0.01 if len(deltas) else None
        ),
        "within_0.05_of_syk": (
            abs(float(np.median(deltas)) - SYK_DELTA) <= 0.05 if len(deltas) else None
        ),
        "per_layer": layer_medians,
        "per_head": all_results,
    }

    print(f"\n  Heads R²>{R2_THRESHOLD}: {len(good)}/{n_layers * n_heads}")
    if len(deltas):
        print(f"  Median Δ = {summary['delta_median']:.4f}  (SYK q=4 → {SYK_DELTA})")
        print(f"  |median − 0.25| = {abs(summary['delta_median'] - SYK_DELTA):.4f}")
        print(f"  SYK-near (|Δ−0.25|≤0.06): {len(syk_near)} heads")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
    )
    args = parser.parse_args()
    names = list(MODELS.keys()) if "all" in args.models else args.models

    results = {
        "experiment": "exp-031",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": "Learned positional embeddings converge to SYK q=4 Δ≈0.25 with depth (GPT-2 Medium 24L, Large 36L).",
        "reference": "exp-007 gpt2 small median Δ=0.2493",
        "models": {},
    }

    for name in names:
        results["models"][name] = analyze_model(MODELS[name], name)

    out_dir = Path(__file__).resolve().parent
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {results_path}")

    ts = results["timestamp"]
    archive = Path(__file__).resolve().parents[2] / "results"
    archive.mkdir(parents=True, exist_ok=True)
    archive_path = archive / f"exp-031_gpt2_depth_convergence_{ts}.json"
    archive_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {archive_path}")

    # Quick verdict
    for name, s in results["models"].items():
        med = s.get("delta_median")
        if med is None:
            print(f"  {name}: INCONCLUSIVE (no R²>{R2_THRESHOLD} heads)")
        elif abs(med - SYK_DELTA) <= 0.01:
            print(f"  {name}: CONFIRMS SYK (median Δ={med:.4f})")
        elif abs(med - SYK_DELTA) <= 0.05:
            print(f"  {name}: NEAR SYK (median Δ={med:.4f})")
        else:
            print(f"  {name}: DEVIATES (median Δ={med:.4f})")


if __name__ == "__main__":
    main()
