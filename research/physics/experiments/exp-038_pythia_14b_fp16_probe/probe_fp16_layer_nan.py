#!/usr/bin/env python3
"""
exp-038 — Per-layer attention NaN fraction: fp16 vs fp32 on Pythia-1.4b (MPS).

Pre-stated (exp-037 follow-up): if fp16 MPS causes BCFT truncation the same way as
410m, first all-NaN layer should be BCFT JSON last layer + 1 = 20 + 1 = 21.

One forward pass, seq_len=512, random tokens — same as exp-037 / BCFT protocol.

Ariel — May 28, 2026
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

OUT = Path(__file__).resolve().parent / "results.json"
MODEL_NAME = "EleutherAI/pythia-1.4b"
SEQ_LEN = 512
BCFT_JSON_LAST_LAYER = 20  # exp-037 coverage audit


def layer_nan_stats(attentions, n_layers: int) -> list[dict]:
    rows = []
    for l in range(n_layers):
        a = attentions[l][0].cpu().float().numpy()
        if a.ndim == 4:
            a = a[0]
        nan_frac = float(np.isnan(a).mean())
        finite_frac = float(np.isfinite(a).mean())
        rows.append(
            {
                "layer": l,
                "nan_fraction": nan_frac,
                "finite_fraction": finite_frac,
                "all_nan": nan_frac >= 0.999,
            }
        )
    return rows


def probe(dtype: torch.dtype, device: torch.device) -> dict:
    load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    vocab = model.config.vocab_size

    input_ids = torch.randint(0, vocab, (1, SEQ_LEN), device=device)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    rows = layer_nan_stats(outputs.attentions, n_layers)
    first_all_nan = next((r["layer"] for r in rows if r["all_nan"]), None)
    del model, outputs
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "dtype": str(dtype),
        "device": str(device),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "first_all_nan_layer": first_all_nan,
        "per_layer": rows,
    }


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS required for this probe (local M5)")

    device = torch.device("mps")
    print(f"Probing {MODEL_NAME} on {device}...", flush=True)

    fp16 = probe(torch.float16, device)
    fp32 = probe(torch.float32, device)

    first_nan = fp16["first_all_nan_layer"]
    expected = BCFT_JSON_LAST_LAYER + 1

    out = {
        "experiment": "exp-038_pythia_14b_fp16_probe",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "hypothesis": (
            f"fp16 MPS attention goes all-NaN at layer L*; L* should align with "
            f"BCFT JSON last valid layer ({BCFT_JSON_LAST_LAYER}) + 1 = {expected}."
        ),
        "fp16": fp16,
        "fp32": fp32,
        "alignment": {
            "bcft_json_last_layer_with_heads": BCFT_JSON_LAST_LAYER,
            "expected_first_all_nan_layer": expected,
            "fp16_first_all_nan_layer": first_nan,
            "layers_match": first_nan == expected if first_nan is not None else None,
        },
        "verdict": "confirmed" if first_nan == expected else "partial",
    }

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["alignment"], indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
