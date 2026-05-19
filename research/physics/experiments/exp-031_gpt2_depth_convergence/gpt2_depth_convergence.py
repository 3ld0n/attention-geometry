"""
GPT-2 family depth convergence — learned positional embeddings (exp-031)

Hypothesis (pre-registered): With learned position embeddings, deeper GPT-2
variants converge to SYK q=4 fixed point Δ ≈ 0.25 ± 0.01, tighter than RoPE
Pythia at comparable depth. GPT-2 small (12L) already gives Δ_med = 0.2493
(exp-007).

Protocol matches exp-007: per-head attention decay A(Δx), log-log fit,
report median Δ among heads with R² > 0.90 on fit window [3, 50).

Usage:
  .venv/bin/python3 gpt2_depth_convergence.py --model gpt2-medium
  .venv/bin/python3 gpt2_depth_convergence.py --model gpt2-large
  .venv/bin/python3 gpt2_depth_convergence.py --model gpt2

Ariel — May 19, 2026 (physics room)
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

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.models import RESEARCH_MODELS  # noqa: E402

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
R2_THRESHOLD = 0.90
SYK_Q4 = 0.25

MODEL_ALIASES = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
}


def compute_head_attention_decay(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    seq = attn_head.shape[0]
    a = np.zeros(max_dx)
    counts = np.zeros(max_dx)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                a[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    a[mask] /= counts[mask]
    return a


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray, cutoff_low: int, cutoff_high: int):
    mask = (
        (dx_arr >= cutoff_low)
        & (dx_arr < cutoff_high)
        & (np.abs(y_arr) > 1e-20)
    )
    if np.sum(mask) < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    design = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
    pred = design @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    exponent = -coeffs[1]
    return exponent / 2, r2


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_model(model_key: str, hf_id: str | None = None) -> dict:
    hf_id = hf_id or MODEL_ALIASES[model_key]
    meta = RESEARCH_MODELS.get(model_key, {})

    torch.manual_seed(42)
    np.random.seed(42)
    device = _device()

    print(f"Loading {hf_id} on {device}...")
    model = GPT2LMHeadModel.from_pretrained(hf_id)
    model.eval()
    model.to(device)

    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    vocab_size = model.config.vocab_size
    dx_arr = np.arange(MAX_DX)

    a_heads = {
        layer: {head: np.zeros(MAX_DX) for head in range(n_heads)}
        for layer in range(n_layers)
    }

    print(f"Processing {N_INPUTS} random inputs (seq_len={SEQ_LEN})...")
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), device=device)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for layer in range(n_layers):
            attn = outputs.attentions[layer][0].numpy()
            for head in range(n_heads):
                a_heads[layer][head] += compute_head_attention_decay(
                    attn[head], MIN_POS, MAX_DX
                )
        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS}")

    for layer in range(n_layers):
        for head in range(n_heads):
            a_heads[layer][head] /= N_INPUTS

    all_results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            delta, r2 = fit_power_law(
                dx_arr, a_heads[layer][head], FIT_LOW, FIT_HIGH
            )
            all_results.append(
                {
                    "layer": layer + 1,
                    "head": head,
                    "delta": float(delta) if delta is not None else None,
                    "r2_pl": float(r2) if r2 is not None else None,
                }
            )

    good = [
        r
        for r in all_results
        if r["delta"] is not None
        and r["r2_pl"] is not None
        and r["r2_pl"] > R2_THRESHOLD
    ]
    deltas = np.array([r["delta"] for r in good])
    syk_near = [
        r
        for r in good
        if abs(r["delta"] - SYK_Q4) <= 0.06
    ]

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
        "model_key": model_key,
        "hf_id": hf_id,
        "pos_enc": meta.get("pos_enc", "learned"),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads": n_layers * n_heads,
        "heads_r2_gt_threshold": len(good),
        "heads_syk_near_q4": len(syk_near),
        "median_delta": float(np.median(deltas)) if len(deltas) else None,
        "mean_delta": float(np.mean(deltas)) if len(deltas) else None,
        "std_delta": float(np.std(deltas)) if len(deltas) else None,
        "min_delta": float(np.min(deltas)) if len(deltas) else None,
        "max_delta": float(np.max(deltas)) if len(deltas) else None,
        "syk_q4_prediction": SYK_Q4,
        "delta_minus_syk": (
            float(np.median(deltas) - SYK_Q4) if len(deltas) else None
        ),
        "protocol": {
            "seq_len": SEQ_LEN,
            "n_inputs": N_INPUTS,
            "fit_window": [FIT_LOW, FIT_HIGH],
            "r2_threshold": R2_THRESHOLD,
            "matches_exp": "exp-007",
            "device": str(device),
        },
        "layer_medians_r2_gt_threshold": layer_medians,
        "per_head": all_results,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="GPT-2 depth convergence (exp-031)")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_ALIASES.keys()),
        help="Model alias",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: results_<model>.json in experiment dir)",
    )
    args = parser.parse_args()

    out = args.out or Path(__file__).parent / f"results_{args.model.replace('-', '_')}.json"
    summary = run_model(args.model)

    print()
    print("=" * 60)
    print(f"  {args.model}: {summary['n_layers']} layers, {summary['total_heads']} heads")
    print(f"  Heads R² > {R2_THRESHOLD}: {summary['heads_r2_gt_threshold']}")
    print(f"  Median Δ = {summary['median_delta']:.4f}" if summary["median_delta"] else "  No good heads")
    print(f"  |Δ − 0.25| = {abs(summary['delta_minus_syk']):.4f}" if summary["delta_minus_syk"] is not None else "")
    print("=" * 60)

    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
