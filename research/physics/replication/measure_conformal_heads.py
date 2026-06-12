"""
Two-point protocol for attention exponents — standalone replication script.

Measures, for every attention head of a HuggingFace causal LM, the decay of the
lag-averaged attention profile A(dx) over token distance, fits
A(dx) ~ dx^(-2*Delta) by OLS in log-log, and reports the per-head exponent
Delta and fit R^2. The claim under test (see README): in trained softmax
models a sub-population of heads is power-law ("conformal") with the median
Delta of the high-R^2 subset near 0.25; re-initialized weights give zero
conformal heads.

Protocol (frozen; identical to the published census):
  - 50 random-token sequences, length 512, fixed seed
  - fp32 attention, verified at extraction (fp16 underflows corrupt deep layers)
  - lag profile averaged over queries i >= max(256, dx)
  - OLS on log A vs log dx over lags [8, 256]
  - conformal: R^2 >= 0.90 and Delta >= 0.05

Usage:
  python measure_conformal_heads.py gpt2
  python measure_conformal_heads.py EleutherAI/pythia-410m --device cuda
  python measure_conformal_heads.py gpt2 --randomized   # control: fresh init

Runtime: minutes for <1B models on a single GPU or Apple Silicon; a 7B census
runs in ~15 minutes. No training, no labels, no datasets.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

SEQ_LEN = 512
N_INPUTS = 50
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256
SEED = 42
R2_MIN, DELTA_MIN = 0.90, 0.05


def lag_profile(att: np.ndarray) -> np.ndarray:
    """att: (n_heads, L, L) -> per-head profile A(dx), queries i >= max(DEEP_LO, dx)."""
    n_heads, L, _ = att.shape
    prof = np.zeros((n_heads, L))
    for dx in range(L):
        diag = np.diagonal(att, offset=-dx, axis1=-2, axis2=-1)
        k_lo = max(DEEP_LO, dx) - dx
        if k_lo < diag.shape[-1]:
            prof[:, dx] = diag[:, k_lo:].mean(axis=-1)
    return prof


def fit_head(profile: np.ndarray):
    lags = np.arange(FIT_LO, FIT_HI + 1)
    y = profile[FIT_LO:FIT_HI + 1]
    ok = y > 1e-15
    if ok.sum() < 5:
        return None, None
    lx, ly = np.log(lags[ok].astype(float)), np.log(y[ok])
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ c
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-30 else 0.0
    return float(-c[1] / 2), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--randomized", action="store_true",
                    help="control: same architecture, fresh random init")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu"))

    if args.randomized:
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        cfg._attn_implementation = "eager"
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        model = model.to(torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="eager",
            trust_remote_code=True)
    model = model.to(device).eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    n_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head

    rng = np.random.default_rng(SEED)
    mean_prof = np.zeros((n_layers, n_heads, SEQ_LEN))
    for inp in range(N_INPUTS):
        ids = torch.tensor(rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)),
                           dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        for l in range(n_layers):
            a = out.attentions[l]
            assert a.dtype == torch.float32, (
                f"layer {l}: attention dtype {a.dtype} != float32 — "
                "fp16/bf16 underflow corrupts the tail; load in fp32")
            a = a[0].cpu().numpy()
            assert not np.isnan(a).any(), f"layer {l}: NaN attention"
            mean_prof[l] += lag_profile(a)
        del out
        if (inp + 1) % 10 == 0:
            print(f"  forward {inp + 1}/{N_INPUTS}", flush=True)
    mean_prof /= N_INPUTS

    heads, deltas = [], []
    for l in range(n_layers):
        for h in range(n_heads):
            d, r2 = fit_head(mean_prof[l, h])
            conf = bool(d is not None and r2 >= R2_MIN and d >= DELTA_MIN)
            heads.append({"layer": l, "head": h, "delta": d, "r2": r2,
                          "conformal": conf})
            if conf:
                deltas.append(d)

    summary = {
        "model": args.model, "randomized_control": args.randomized,
        "n_heads": n_layers * n_heads, "n_conformal": len(deltas),
        "delta_median_conformal": float(np.median(deltas)) if deltas else None,
        "n_syk_near": int(sum(abs(d - 0.25) <= 0.05 for d in deltas)),
        "protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS, "seed": SEED,
                     "fit_lags": [FIT_LO, FIT_HI], "queries": f"i>={DEEP_LO}",
                     "conformal": f"R2>={R2_MIN} and Delta>={DELTA_MIN}"},
        "heads": heads,
    }
    out_path = args.out or (args.model.replace("/", "--")
                            + ("_randomized" if args.randomized else "") + ".json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{args.model}{' (randomized)' if args.randomized else ''}: "
          f"{len(deltas)}/{n_layers * n_heads} conformal heads, "
          f"median Delta = {summary['delta_median_conformal']}, "
          f"SYK-near (|Delta-0.25|<=0.05): {summary['n_syk_near']}\n-> {out_path}")


if __name__ == "__main__":
    main()
