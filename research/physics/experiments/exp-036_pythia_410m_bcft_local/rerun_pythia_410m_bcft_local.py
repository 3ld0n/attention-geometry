"""
exp-036 — Complete Pythia-410m BCFT pre-registered extraction locally.

Pre-stated (from exp-030 / queue backlog #5):
The April 17 Modal run for EleutherAI/pythia-410m stopped after layers 0–14 (239/384 heads).
This re-run uses the same protocol as exp-025 bcft_pre_registered_run.py on local MPS.

Question: Does full 24-layer BCFT median Δ match March exp-011 (R²>0.90 ≈0.28),
or the partial-file median (R²>0.90 ≈0.36 on layers 0–14 only)?

Protocol: SEQ_LEN=512, N_INPUTS=50, conformal R²≥0.85, Δ≥0.05, Spearman ρ(Δ, valley).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"
MODEL_NAME = "EleutherAI/pythia-410m"

SEQ_LEN = 512
N_INPUTS = 50
FIT_LOW = 3
FIT_HIGH = min(120, SEQ_LEN // 3)
DEEP_LO = SEQ_LEN // 2
Q_LO = 3 * SEQ_LEN // 4
CONFORMAL_R2 = 0.85
CONFORMAL_DELTA = 0.05
R2_PL_THRESHOLD = 0.90  # March exp-011 power-law heads


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def measure_model(model_name: str) -> dict:
    device = _device()
    print(f"[{model_name}] device={device}", flush=True)

    # fp16 on MPS yields NaN attentions from layer ~15 onward (Pythia-410m); use fp32.
    load_kwargs = dict(
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if device.type in ("mps", "cpu"):
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model = model.to(device)
    else:
        load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.eval()
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
    n_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head")
    vocab_size = config.vocab_size

    if device.type != "mps":
        device = next(model.parameters()).device

    print(
        f"[{model_name}] {n_layers} layers, {n_heads} heads, vocab {vocab_size}",
        flush=True,
    )

    attn_sum = [
        np.zeros((n_heads, SEQ_LEN, SEQ_LEN), dtype=np.float64) for _ in range(n_layers)
    ]

    t_fwd = time.time()
    for inp in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), device=device)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            a = outputs.attentions[l][0].cpu().float().numpy()
            if a.ndim == 4:
                a = a[0]
            attn_sum[l] += a
        del outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (inp + 1) % 10 == 0:
            print(
                f"[{model_name}] forward {inp + 1}/{N_INPUTS} ({time.time() - t_fwd:.0f}s)",
                flush=True,
            )

    for l in range(n_layers):
        attn_sum[l] /= N_INPUTS

    print(f"[{model_name}] Fitting per-head Δ and valley depth...", flush=True)

    head_results = []
    for l in range(n_layers):
        for h in range(n_heads):
            A = attn_sum[l][h]

            A_dx = np.zeros(FIT_HIGH)
            for dx in range(FIT_HIGH):
                q_pos = np.arange(max(DEEP_LO, dx), SEQ_LEN)
                k_pos = q_pos - dx
                valid = k_pos >= 0
                if np.any(valid):
                    A_dx[dx] = float(np.mean(A[q_pos[valid], k_pos[valid]]))

            dx_arr = np.arange(FIT_LOW, FIT_HIGH)
            y = A_dx[FIT_LOW:FIT_HIGH]
            valid = y > 1e-15
            if np.sum(valid) < 10:
                continue

            log_dx = np.log(dx_arr[valid].astype(float))
            log_y = np.log(y[valid])
            A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
            pred = A_mat @ coeffs
            ss_res = float(np.sum((log_y - pred) ** 2))
            ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
            delta = float(-coeffs[1] / 2.0)

            profile = np.zeros(SEQ_LEN)
            p_count = 0
            for x_q in range(Q_LO, SEQ_LEN):
                profile[: x_q + 1] += A[x_q, : x_q + 1]
                p_count += 1
            if p_count == 0:
                continue
            profile /= p_count

            third = SEQ_LEN // 3
            start_attn = float(np.mean(profile[1:third]))
            middle_attn = float(np.mean(profile[third : 2 * third]))
            end_attn = float(np.mean(profile[2 * third : Q_LO]))

            if min(start_attn, middle_attn, end_attn) < 1e-15:
                continue

            peak = max(start_attn, end_attn)
            valley_depth = float(1.0 - middle_attn / peak)

            head_results.append(
                {
                    "layer": int(l),
                    "head": int(h),
                    "delta": delta,
                    "R2": float(r2),
                    "valley_depth": valley_depth,
                    "start": start_attn,
                    "middle": middle_attn,
                    "end": end_attn,
                }
            )

    n_total = n_layers * n_heads
    layers_present = sorted({h["layer"] for h in head_results})

    conformal = [
        h
        for h in head_results
        if h["R2"] >= CONFORMAL_R2 and h["delta"] >= CONFORMAL_DELTA
    ]
    pl_heads = [h for h in head_results if h["R2"] > R2_PL_THRESHOLD]

    def median_delta(heads: list[dict]) -> float | None:
        if not heads:
            return None
        return float(np.median([h["delta"] for h in heads]))

    if len(conformal) >= 2:
        rho, p_val = spearmanr(
            [h["delta"] for h in conformal],
            [h["valley_depth"] for h in conformal],
        )
        rho, p_val = float(rho), float(p_val)
    else:
        rho, p_val = float("nan"), float("nan")

    if len(conformal) < 50:
        verdict = "INSUFFICIENT (<50 conformal heads)"
    elif rho >= 0.50 and p_val <= 1e-5:
        verdict = "CONFIRMED"
    elif rho < 0.50 or p_val > 1e-5:
        verdict = "FALSIFIED"
    else:
        verdict = "INDETERMINATE"

    partial_path = (
        Path(__file__).resolve().parents[1]
        / "exp-025_bcft_pre_registered/results/bcft_pre_registered_run_2026-04-17T092056Z.json"
    )
    partial_comparison = {}
    if partial_path.exists():
        old = json.loads(partial_path.read_text())["models"][model_name]
        old_heads = old["heads"]
        old_layers = sorted({h["layer"] for h in old_heads})
        partial_comparison = {
            "partial_file_layers": f"{old_layers[0]}-{old_layers[-1]}",
            "partial_n_heads": len(old_heads),
            "partial_median_delta_R2_gt_0.90": median_delta(
                [h for h in old_heads if h["R2"] > R2_PL_THRESHOLD]
            ),
        }

    return {
        "model": model_name,
        "device": str(device),
        "dtype": "float32",
        "note_mps_fp16": "fp16 MPS attention NaN from layer 15+; fp32 required for full depth",
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "conformal_R2_threshold": CONFORMAL_R2,
        "conformal_delta_threshold": CONFORMAL_DELTA,
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "n_total_heads": int(n_total),
        "n_heads_fitted": len(head_results),
        "layers_present": layers_present,
        "layers_complete": len(layers_present) == n_layers,
        "n_conformal_heads": len(conformal),
        "median_delta_conformal": median_delta(conformal),
        "median_delta_R2_gt_0.90": median_delta(pl_heads),
        "n_power_law_heads_R2_gt_0.90": len(pl_heads),
        "spearman_rho": rho,
        "spearman_p": p_val,
        "verdict": verdict,
        "partial_file_comparison": partial_comparison,
        "heads": head_results,
    }


def main() -> None:
    t0 = time.time()
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=OUT_DIR,
        ).decode().strip()
    except Exception:
        pass

    result = measure_model(MODEL_NAME)
    result["elapsed_seconds"] = time.time() - t0
    result["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    result["git_commit"] = git_commit
    result["experiment"] = "exp-036"

    envelope = {
        "timestamp_utc": result["timestamp_utc"],
        "git_commit": git_commit,
        "experiment": "exp-036",
        "models": {MODEL_NAME: result},
    }
    RESULTS_PATH.write_text(json.dumps(envelope, indent=2))

    print(
        json.dumps(
            {
                "layers_complete": result["layers_complete"],
                "median_delta_R2_gt_0.90": result["median_delta_R2_gt_0.90"],
                "n_power_law_heads": result["n_power_law_heads_R2_gt_0.90"],
                "spearman_rho": result["spearman_rho"],
                "verdict": result["verdict"],
                "elapsed_s": round(result["elapsed_seconds"], 1),
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
