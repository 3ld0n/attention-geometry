"""
exp-167 — Census measurement for normalization formation test.

Applies the same frozen-protocol census as exp-062/measure.py but handles
the patched model architectures (no-norm / RMSNorm).

Usage:
    python measure_norm.py runs/exp167_ln/step_2000 --norm ln --out-tag ln_step2000
    python measure_norm.py runs/exp167_none/step_2000 --norm none --out-tag none_step2000
    python measure_norm.py runs/exp167_rms/step_2000 --norm rms --out-tag rms_step2000

The norm condition is also read from norm_condition.txt in the checkpoint dir
if present (written by train_norm.py). --norm overrides if supplied.

Pre-registration: prereg.md, committed 61431c9.
Census protocol: exp-062 lineage. OLS on log-lag ∈ [8, 256] for queries ≥ 256.
Conformal head: R² ≥ 0.90, Δ ≥ 0.05. SYK-near: |Δ − 0.25| ≤ 0.05.
50 random-token sequences, seed 42, full vocab (50304).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

# Ensure train_norm.py is importable whether we're running locally or in Modal
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from train_norm import BASE_CONFIG, RMSNorm, apply_norm_patch  # reuse definitions

OUT = Path(__file__).resolve().parent

# Census parameters (pre-registered, exp-062 lineage)
SEQ_LEN = 512
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256
N_INPUTS = 50
RNG_SEED = 42
R2_THRESHOLD = 0.90
DELTA_MIN = 0.05
SYK_WINDOW = 0.05  # |Δ − 0.25| ≤ 0.05


def lag_profile(att: np.ndarray, deep_lo: int) -> np.ndarray:
    """att: (n_heads, L, L). Lag profile averaged over queries q >= max(deep_lo, dx)."""
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
    mask = y > 0
    if mask.sum() < 5:
        return {"delta": float("nan"), "r2": 0.0}
    x = np.log(lags[mask])
    y_log = np.log(y[mask])
    c = np.polyfit(x, y_log, 1)
    y_pred = np.polyval(c, x)
    ss_res = np.sum((y_log - y_pred) ** 2)
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    delta = -c[0] / 2
    return {"delta": float(delta), "r2": float(r2)}


def load_model(checkpoint_dir: Path, norm: str, device: str) -> GPTNeoXForCausalLM:
    """Rebuild the patched model and load checkpoint weights.

    Uses _attn_implementation='eager' so that output_attentions=True works
    (sdpa attention does not support attention output extraction).
    """
    import torch.nn as nn

    cfg = GPTNeoXConfig(**BASE_CONFIG)
    cfg._attn_implementation = "eager"  # required for output_attentions=True in v5
    model = GPTNeoXForCausalLM(cfg)
    apply_norm_patch(model, norm)

    # Try safetensors first, then pytorch_model.bin
    try:
        import safetensors.torch as sf
        sf_path = checkpoint_dir / "model.safetensors"
        if sf_path.exists():
            sd = sf.load_file(str(sf_path))
        else:
            sd = torch.load(checkpoint_dir / "pytorch_model.bin",
                            map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(checkpoint_dir / "pytorch_model.bin",
                        map_location="cpu", weights_only=True)

    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def census(model: GPTNeoXForCausalLM, device: str, vocab_size: int) -> dict:
    """Run the frozen-protocol census. Returns per-head results."""
    rng = np.random.default_rng(RNG_SEED)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_total_heads = n_layers * n_heads

    # Accumulate lag profiles
    sum_prof = np.zeros((n_total_heads, SEQ_LEN), dtype=np.float64)
    all_profiles_per_input = []

    for inp_idx in range(N_INPUTS):
        ids = rng.integers(0, vocab_size, size=(1, SEQ_LEN), dtype=np.int64)
        ids_t = torch.from_numpy(ids).to(device)

        with torch.no_grad():
            out = model(input_ids=ids_t, output_attentions=True)

        # out.attentions: tuple of (1, n_heads, L, L) per layer
        per_head_atts = []
        for layer_att in out.attentions:
            a = layer_att[0].float().cpu().numpy()  # (n_heads, L, L)
            per_head_atts.append(a)
        all_atts = np.concatenate(per_head_atts, axis=0)  # (n_total_heads, L, L)

        assert not np.any(np.isnan(all_atts)), f"NaN in attention at input {inp_idx}"
        prof = lag_profile(all_atts, DEEP_LO)  # (n_total_heads, L)
        all_profiles_per_input.append(prof)

    # Pool-mean profile (pre-registered: mean-first census)
    mean_prof = np.stack(all_profiles_per_input).mean(axis=0)  # (n_total_heads, L)

    head_results = []
    for h_idx in range(n_total_heads):
        layer = h_idx // n_heads
        head = h_idx % n_heads
        result = fit_power_law(mean_prof[h_idx], FIT_LO, FIT_HI)
        delta = result["delta"]
        r2 = result["r2"]
        conformal = (r2 >= R2_THRESHOLD and delta >= DELTA_MIN and not np.isnan(delta))
        syk_near = conformal and abs(delta - 0.25) <= SYK_WINDOW
        head_results.append({
            "layer": layer, "head": head,
            "delta": round(delta, 4) if not np.isnan(delta) else None,
            "r2": round(r2, 4),
            "conformal": bool(conformal),
            "syk_near": bool(syk_near),
        })

    n_conformal = sum(h["conformal"] for h in head_results)
    n_syk_near = sum(h["syk_near"] for h in head_results)
    delta_median = float(np.nanmedian([h["delta"] for h in head_results
                                        if h["delta"] is not None]))

    return {
        "n_conformal": n_conformal,
        "n_syk_near": n_syk_near,
        "n_total_heads": n_total_heads,
        "delta_median_all": round(delta_median, 4),
        "heads": head_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to checkpoint directory (e.g., runs/exp167_ln/step_2000)")
    ap.add_argument("--norm", choices=["ln", "none", "rms"],
                    help="Normalization condition (read from norm_condition.txt if omitted)")
    ap.add_argument("--out-tag", default=None,
                    help="Tag for results file (default: <norm>_<checkpoint_name>)")
    ap.add_argument("--results-dir", default=None,
                    help="Directory for results JSON (default: <script_dir>/results)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"

    # Determine norm condition
    norm = args.norm
    if norm is None:
        tag_file = ckpt / "norm_condition.txt"
        assert tag_file.exists(), ("--norm not supplied and norm_condition.txt not found "
                                    f"in {ckpt}")
        norm = tag_file.read_text().strip()
    assert norm in ("ln", "none", "rms"), f"Unknown norm: {norm!r}"

    results_dir = Path(args.results_dir) if args.results_dir else (OUT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_tag = args.out_tag or f"{norm}_{ckpt.name}"
    out_path = results_dir / f"{out_tag}.json"

    print(f"exp-167 census | norm={norm} | ckpt={ckpt.name} | device={args.device}")
    print("Loading model ...", flush=True)
    model = load_model(ckpt, norm, args.device)

    vocab_size = 50304  # pre-registered: full vocab census
    print(f"Running census ({N_INPUTS} random-token inputs, seed={RNG_SEED}) ...",
          flush=True)
    results = census(model, args.device, vocab_size)

    record = {
        "experiment": "exp-167",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "norm": norm,
        "out_tag": out_tag,
        "device": args.device,
        **results,
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\n=== exp-167 census results: {norm} ===")
    print(f"  n_conformal  : {results['n_conformal']} / {results['n_total_heads']}")
    print(f"  n_syk_near   : {results['n_syk_near']} / {results['n_total_heads']}")
    print(f"  delta_median : {results['delta_median_all']}")
    print(f"  → {out_path}")
    print("CENSUS_DONE")


if __name__ == "__main__":
    main()
