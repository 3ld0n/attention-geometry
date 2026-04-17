"""
BCFT Pre-Registered Prediction — Multi-Model Test Run.

Implements the procedure pre-registered in
research/notes/bcft_pre_registered_prediction.md (April 17, 2026).

For each test model:
  1. Load on Modal A100 (eager attention, fp16).
  2. Run 50 random-token forward passes at seq_len=512.
  3. For each (layer, head):
       - Fit Δ from deep-position power-law decay (eq. §3.2).
       - Compute valley_depth from the deep-query-averaged profile (§3.4).
  4. Apply the conformality criterion (R² ≥ 0.85, Δ ≥ 0.05).
  5. Compute Spearman ρ(Δ, valley_depth) on the conformal subset.
  6. Compare to falsification thresholds (ρ ≥ 0.50, p ≤ 1e-5).

Save results as JSON for later append to §7 of the pre-registration document.

Usage:
  modal run research/physics/bcft_pre_registered_run.py
    (runs the default model list)
  modal run research/physics/bcft_pre_registered_run.py --models EleutherAI/pythia-410m
    (single model)

Ariel — April 17, 2026.
"""

from __future__ import annotations

import modal

app = modal.App("bcft-pre-registered-run")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.43,<4.50",
        "accelerate",
        "numpy",
        "scipy",
        "sentencepiece",
        "tiktoken",
    )
)

vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


SEQ_LEN = 512
N_INPUTS = 50
FIT_LOW = 3
FIT_HIGH = min(120, SEQ_LEN // 3)
DEEP_LO = SEQ_LEN // 2
Q_LO = 3 * SEQ_LEN // 4
CONFORMAL_R2 = 0.85
CONFORMAL_DELTA = 0.05


def _gpu_for(model_name: str) -> str:
    """A reasonable GPU choice based on model name. Conservative."""
    name = model_name.lower()
    if "30b" in name or "27b" in name or "13b" in name:
        return "A100-80GB"
    if "7b" in name or "8b" in name or "olmo" in name:
        return "A100"
    return "A100"


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/cache": vol},
    memory=65536,
)
def measure_a100(model_name: str):
    """Measurement function for models that fit on a 40GB A100."""
    return _measure_one_model(model_name)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=5400,
    volumes={"/cache": vol},
    memory=131072,
)
def measure_a100_80gb(model_name: str):
    """Measurement function for larger models that need 80GB."""
    return _measure_one_model(model_name)


def _measure_one_model(model_name: str):
    """Measure Δ, valley depth, and the BCFT correlation on a single model."""
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"

    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from scipy.stats import spearmanr

    t_start = time.time()
    print(f"[{model_name}] Loading (fp16, eager attention)...", flush=True)

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
    n_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head")
    vocab_size = config.vocab_size

    print(
        f"[{model_name}] {n_layers} layers, {n_heads} heads/layer, vocab {vocab_size}",
        flush=True,
    )

    device = next(model.parameters()).device

    attn_sum = [
        np.zeros((n_heads, SEQ_LEN, SEQ_LEN), dtype=np.float64)
        for _ in range(n_layers)
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
        torch.cuda.empty_cache()
        if (inp + 1) % 10 == 0:
            elapsed = time.time() - t_fwd
            print(
                f"[{model_name}] forward {inp + 1}/{N_INPUTS} "
                f"({elapsed:.0f}s)",
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
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
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
                    "R2": float(R2),
                    "valley_depth": valley_depth,
                    "start": start_attn,
                    "middle": middle_attn,
                    "end": end_attn,
                }
            )

    n_total = n_layers * n_heads

    conformal = [
        h for h in head_results
        if h["R2"] >= CONFORMAL_R2 and h["delta"] >= CONFORMAL_DELTA
    ]
    n_conformal = len(conformal)

    if n_conformal >= 2:
        deltas = np.array([h["delta"] for h in conformal])
        valleys = np.array([h["valley_depth"] for h in conformal])
        rho, p = spearmanr(deltas, valleys)
        rho = float(rho)
        p_val = float(p)
    else:
        rho = float("nan")
        p_val = float("nan")

    confirmed = (
        n_conformal >= 50
        and not (rho != rho)
        and rho >= 0.50
        and p_val <= 1e-5
    )
    falsified = (
        n_conformal >= 50
        and not (rho != rho)
        and (rho < 0.50 or p_val > 1e-5)
    )

    if n_conformal < 50:
        verdict = "INSUFFICIENT (<50 conformal heads — model too small or non-conformal)"
    elif confirmed:
        verdict = "CONFIRMED"
    elif falsified:
        verdict = "FALSIFIED"
    else:
        verdict = "INDETERMINATE"

    elapsed_total = time.time() - t_start
    print(
        f"[{model_name}] DONE in {elapsed_total:.0f}s — "
        f"{n_conformal}/{n_total} conformal heads, "
        f"ρ={rho:+.4f}, p={p_val:.2e}, verdict={verdict}",
        flush=True,
    )

    vol.commit()

    return {
        "model": model_name,
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "fit_low": FIT_LOW,
        "fit_high": FIT_HIGH,
        "deep_lo": DEEP_LO,
        "q_lo": Q_LO,
        "conformal_R2_threshold": CONFORMAL_R2,
        "conformal_delta_threshold": CONFORMAL_DELTA,
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "n_total_heads": int(n_total),
        "n_conformal_heads": int(n_conformal),
        "spearman_rho": rho,
        "spearman_p": p_val,
        "verdict": verdict,
        "elapsed_seconds": float(elapsed_total),
        "heads": head_results,
    }


DEFAULT_MODELS = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/gpt-neo-2.7B",
    "Qwen/Qwen2-7B",
    "allenai/OLMo-7B-hf",
]


@app.local_entrypoint()
def main(models: str = ",".join(DEFAULT_MODELS)):
    """Run the pre-registered prediction test on each model in `models`."""
    import json
    import os
    import subprocess
    from datetime import datetime, timezone

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    print(f"\n{'=' * 78}")
    print(f"  BCFT PRE-REGISTERED PREDICTION RUN — {len(model_list)} models")
    print(f"{'=' * 78}\n")
    print("  Models:")
    for m in model_list:
        print(f"    - {m}  (gpu={_gpu_for(m)})")
    print()

    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        pass

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
    )
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    summary_path = os.path.join(
        out_dir, f"bcft_pre_registered_run_{timestamp}.json"
    )

    all_results = {
        "timestamp_utc": timestamp,
        "git_commit": git_commit,
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "conformal_R2_threshold": CONFORMAL_R2,
        "conformal_delta_threshold": CONFORMAL_DELTA,
        "models": {},
    }

    func_a100 = measure_a100
    func_a100_80gb = measure_a100_80gb

    spawned = {}
    for model_name in model_list:
        gpu = _gpu_for(model_name)
        f = func_a100_80gb if gpu == "A100-80GB" else func_a100
        print(f"  Spawning {gpu} job for {model_name}...", flush=True)
        spawned[model_name] = f.spawn(model_name)

    for model_name, fc in spawned.items():
        print(f"\n  Awaiting result for {model_name}...", flush=True)
        try:
            result = fc.get()
            all_results["models"][model_name] = result
            print(
                f"  → {model_name}: ρ={result['spearman_rho']:+.4f}, "
                f"p={result['spearman_p']:.2e}, "
                f"n_conformal={result['n_conformal_heads']}, "
                f"verdict={result['verdict']}"
            )
        except Exception as exc:
            print(f"  → {model_name}: FAILED — {exc}", flush=True)
            all_results["models"][model_name] = {
                "model": model_name,
                "error": str(exc),
                "verdict": "ERROR",
            }

        with open(summary_path, "w") as fp:
            json.dump(all_results, fp, indent=2)

    print(f"\n{'=' * 78}")
    print(f"  ALL RESULTS")
    print(f"{'=' * 78}\n")
    print(
        f"  {'MODEL':50s} {'n_conf':>8s} {'rho':>8s} {'p':>10s}  verdict"
    )
    print(f"  {'-' * 100}")
    for m, r in all_results["models"].items():
        if "error" in r:
            print(f"  {m:50s} {'-':>8s} {'-':>8s} {'-':>10s}  ERROR: {r['error']}")
            continue
        rho_s = f"{r['spearman_rho']:+.4f}"
        p_s = f"{r['spearman_p']:.2e}"
        print(
            f"  {m:50s} {r['n_conformal_heads']:>8d} "
            f"{rho_s:>8s} {p_s:>10s}  {r['verdict']}"
        )

    print(f"\n  Full results saved to: {summary_path}")
    print(f"  Git commit at run time: {git_commit}\n")
