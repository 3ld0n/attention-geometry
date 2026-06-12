"""
exp-062 — Frozen-protocol measurement (Phase 1.1, pre-registered).

Pre-registration of record: notes/2026-06-11_corpus_statistics_preregistration.md §5.
Census protocol (exp-007 lineage): 50 random-token sequences, L = 512, fp32
with dtype + NaN assertions at extraction, lag profile over queries
i >= max(256, dx), OLS log A vs log dx on lags [8, 256]. Conformal head:
R² >= 0.90 and Δ >= 0.05 (exp-060 selection). SYK-near: |Δ - 0.25| <= 0.05.

Random tokens are uniform over the MODEL'S TRAINING ALPHABET (prereg §5.1):
the 256-id alphabet for synthetic-corpus models (alphabet.json), the full
50304 vocab for C-NAT models. Seed 42. Per-input profiles saved gzipped
(standing constraint §2.3).

Usage:
  python measure.py <checkpoint_dir> <out_tag> [--alphabet alphabet.json | --full-vocab]
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

OUT = Path(__file__).resolve().parent
RESULTS_DIR = OUT / "measurements"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 512
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256
N_INPUTS = 50
RNG_SEED = 42
R2_THRESHOLD = 0.90
DELTA_MIN = 0.05
SYK_WINDOW = 0.05


def lag_profile(att: np.ndarray, deep_lo: int) -> np.ndarray:
    """att: (n_heads, L, L). Lag profile averaged over queries q >= max(deep_lo, dx).
    Identical to exp-058 census implementation."""
    n_heads, L, _ = att.shape
    prof = np.zeros((n_heads, L), dtype=np.float64)
    for dx in range(L):
        diag = np.diagonal(att, offset=-dx, axis1=-2, axis2=-1)
        k_lo = max(deep_lo, dx) - dx
        if k_lo < diag.shape[-1]:
            prof[:, dx] = diag[:, k_lo:].mean(axis=-1)
    return prof


def fit_power_law(profile: np.ndarray, lo: int, hi: int) -> dict:
    lags = np.arange(lo, hi + 1)
    y = profile[lo: hi + 1]
    valid = y > 1e-15
    if valid.sum() < 5:
        return {"delta": None, "r2": None, "n_points": int(valid.sum())}
    lx, ly = np.log(lags[valid].astype(float)), np.log(y[valid])
    A = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return {"delta": float(-coef[1] / 2), "r2": float(r2), "n_points": int(valid.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_tag")
    ap.add_argument("--alphabet", default=str(OUT / "alphabet.json"))
    ap.add_argument("--full-vocab", action="store_true")
    args = ap.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model = GPTNeoXForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32,
        attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    if args.full_vocab:
        draw_pool = np.arange(model.config.vocab_size)
    else:
        draw_pool = np.array(json.loads(Path(args.alphabet).read_text())["ids"])

    rng = np.random.default_rng(RNG_SEED)
    mean_prof = np.zeros((n_layers, n_heads, SEQ_LEN), dtype=np.float64)
    per_input = []
    for inp in range(N_INPUTS):
        ids = rng.choice(draw_pool, size=SEQ_LEN, replace=True)
        x = torch.tensor(ids[None, :], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x, output_attentions=True)
        this = np.zeros((n_layers, n_heads, SEQ_LEN), dtype=np.float32)
        for l in range(n_layers):
            a_t = out.attentions[l]
            assert a_t.dtype == torch.float32, f"layer {l}: dtype {a_t.dtype} != fp32"
            a = a_t[0].float().cpu().numpy()
            assert not np.isnan(a).any(), f"layer {l}: NaN in attention"
            prof = lag_profile(a, DEEP_LO)
            mean_prof[l] += prof
            this[l] = prof.astype(np.float32)
        per_input.append(this)
        del out
    mean_prof /= N_INPUTS

    heads, deltas_conformal = [], []
    for l in range(n_layers):
        for h in range(n_heads):
            fit = fit_power_law(mean_prof[l, h], FIT_LO, FIT_HI)
            rec = {"layer": l, "head": h, **fit}
            rec["conformal"] = bool(
                fit["delta"] is not None and fit["r2"] is not None
                and fit["r2"] >= R2_THRESHOLD and fit["delta"] >= DELTA_MIN)
            rec["syk_near"] = bool(
                rec["conformal"] and abs(fit["delta"] - 0.25) <= SYK_WINDOW)
            heads.append(rec)
            if rec["conformal"]:
                deltas_conformal.append(fit["delta"])

    summary = {
        "checkpoint": str(args.checkpoint), "tag": args.out_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": {"seq_len": SEQ_LEN, "deep_lo": DEEP_LO,
                     "fit": [FIT_LO, FIT_HI], "n_inputs": N_INPUTS,
                     "seed": RNG_SEED, "alphabet_size": int(len(draw_pool))},
        "n_heads_total": n_layers * n_heads,
        "n_conformal": len(deltas_conformal),
        "n_syk_near": sum(h["syk_near"] for h in heads),
        "delta_median_conformal": (float(np.median(deltas_conformal))
                                   if deltas_conformal else None),
        "forms": len(deltas_conformal) >= 10,
        "heads": heads,
    }
    out_json = RESULTS_DIR / f"{args.out_tag}.json"
    out_json.write_text(json.dumps(summary, indent=1))
    with gzip.open(RESULTS_DIR / f"{args.out_tag}_per_input.json.gz", "wt") as f:
        json.dump([p.tolist() for p in per_input], f)
    print(f"{args.out_tag}: conformal {summary['n_conformal']}/{summary['n_heads_total']}, "
          f"SYK-near {summary['n_syk_near']}, "
          f"median Δ {summary['delta_median_conformal']}")


if __name__ == "__main__":
    main()
