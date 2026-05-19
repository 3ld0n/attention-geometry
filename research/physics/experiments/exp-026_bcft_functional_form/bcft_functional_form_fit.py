"""BCFT functional-form fit on full position-resolved attention.

The pre-registered run (bcft_pre_registered_run.py) fits only the bare
power law α(Δx) ≈ C · Δx^(-2Δ), averaged over deep query positions. This
script fits the full BCFT two-point function

    α(x_q, x_k) ≈ C · Δx^(-2Δ) · (1 + λ · η^Δ),    η = Δx² / (4 · x_q · x_k)

per head, using the 2D position-resolved data α(x_q, x_k) rather than the
1D Δx-averaged profile.

Diagnostic question: does the BCFT form, with the boundary parameter λ,
recover the framework's Δ ↔ valley_depth correlation in Pythia-2.8B's
late layers — where the bare power law fits poorly?

For each head, we save: Δ_PL, R²_PL, Δ_BCFT, λ_BCFT, R²_BCFT, plus
the same `valley_depth` measure used in the pre-registration.

Usage:
  modal run research/physics/bcft_functional_form_fit.py
    (default: Pythia-2.8B + GPT-Neo-2.7B)
  modal run research/physics/bcft_functional_form_fit.py --models <comma-list>

Ariel — April 17, 2026.
"""

from __future__ import annotations

import modal

app = modal.App("bcft-functional-form-fit")

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
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def measure_a100(model_name: str):
    return _measure_one_model(model_name)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=5400,
    volumes={"/cache": vol},
    memory=131072,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def measure_a100_80gb(model_name: str):
    return _measure_one_model(model_name)


def _measure_one_model(model_name: str):
    """Run forward passes, then per-head fit PL and BCFT forms on 2D data."""
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"

    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from scipy.optimize import minimize
    from scipy.stats import spearmanr

    t_start = time.time()
    print(f"[{model_name}] Loading (fp16, eager attention)...", flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if hf_token:
        load_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    config = model.config
    n_layers = (
        getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
    )
    n_heads = (
        getattr(config, "num_attention_heads", None) or getattr(config, "n_head")
    )
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

    print(
        f"[{model_name}] Fitting per-head PL and BCFT forms (2D)...",
        flush=True,
    )

    head_results = []

    for l in range(n_layers):
        for h in range(n_heads):
            A = attn_sum[l][h]

            # 1D Δx-averaged profile (same as pre-registration) ----------
            A_dx = np.zeros(FIT_HIGH)
            for dx in range(FIT_HIGH):
                q_pos = np.arange(max(DEEP_LO, dx), SEQ_LEN)
                k_pos = q_pos - dx
                valid = k_pos >= 0
                if np.any(valid):
                    A_dx[dx] = float(np.mean(A[q_pos[valid], k_pos[valid]]))

            dx_arr_1d = np.arange(FIT_LOW, FIT_HIGH)
            y_1d = A_dx[FIT_LOW:FIT_HIGH]
            valid_1d = y_1d > 1e-15
            if np.sum(valid_1d) < 10:
                continue

            log_dx_1d = np.log(dx_arr_1d[valid_1d].astype(float))
            log_y_1d = np.log(y_1d[valid_1d])
            A_mat = np.column_stack([np.ones_like(log_dx_1d), log_dx_1d])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y_1d, rcond=None)
            pred = A_mat @ coeffs
            ss_res = float(np.sum((log_y_1d - pred) ** 2))
            ss_tot = float(np.sum((log_y_1d - np.mean(log_y_1d)) ** 2))
            R2_1d = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
            delta_1d = float(-coeffs[1] / 2.0)

            # 2D position-resolved data for the BCFT fit -----------------
            all_dx = []
            all_xq = []
            all_g = []
            for x_q in range(FIT_LOW * 4, SEQ_LEN):
                upper = min(FIT_HIGH, x_q)
                for dx in range(FIT_LOW, upper):
                    x_k = x_q - dx
                    if x_k <= 0:
                        continue
                    g_val = float(A[x_q, x_k])
                    if g_val < 1e-15:
                        continue
                    all_dx.append(float(dx))
                    all_xq.append(float(x_q))
                    all_g.append(g_val)
            all_dx = np.array(all_dx)
            all_xq = np.array(all_xq)
            all_g = np.array(all_g)

            if len(all_g) < 30:
                continue

            ss_tot_2d = float(np.sum((all_g - np.mean(all_g)) ** 2))
            if ss_tot_2d < 1e-30:
                continue

            # Bare power-law fit on 2D data
            def pl_loss(params):
                C, delta = params
                if C <= 0 or delta <= 0:
                    return 1e12
                pred = C * np.power(all_dx, -2.0 * delta)
                return float(np.sum((all_g - pred) ** 2))

            C_init = float(np.mean(all_g * np.power(all_dx, 2.0 * delta_1d)))
            delta_init = max(0.05, delta_1d)
            res_pl = minimize(
                pl_loss, [C_init, delta_init],
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12},
            )
            C_pl = abs(float(res_pl.x[0]))
            delta_pl = abs(float(res_pl.x[1]))
            pred_pl = C_pl * np.power(all_dx, -2.0 * delta_pl)
            R2_pl = 1.0 - float(np.sum((all_g - pred_pl) ** 2)) / ss_tot_2d

            # BCFT fit on 2D data
            xk_arr = all_xq - all_dx
            eta = (all_dx ** 2) / (4.0 * all_xq * xk_arr)

            def bcft_loss(params):
                C, delta, lam = params
                if C <= 0 or delta <= 0:
                    return 1e12
                correction = 1.0 + lam * np.power(eta, delta)
                if np.any(correction <= 0):
                    return 1e12
                pred = C * np.power(all_dx, -2.0 * delta) * correction
                return float(np.sum((all_g - pred) ** 2))

            res_bcft = minimize(
                bcft_loss, [C_pl, delta_pl, 0.0],
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-12},
            )
            C_bcft = abs(float(res_bcft.x[0]))
            delta_bcft = abs(float(res_bcft.x[1]))
            lambda_bcft = float(res_bcft.x[2])
            correction = 1.0 + lambda_bcft * np.power(eta, delta_bcft)
            pred_bcft = C_bcft * np.power(all_dx, -2.0 * delta_bcft) * correction
            R2_bcft = 1.0 - float(np.sum((all_g - pred_bcft) ** 2)) / ss_tot_2d

            # Valley depth — same definition as pre-registration ---------
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
                    "delta_1d": delta_1d,
                    "R2_1d": float(R2_1d),
                    "delta_PL": delta_pl,
                    "R2_PL": float(R2_pl),
                    "delta_BCFT": delta_bcft,
                    "lambda_BCFT": lambda_bcft,
                    "R2_BCFT": float(R2_bcft),
                    "n_points_2d": int(len(all_g)),
                    "valley_depth": valley_depth,
                    "start": start_attn,
                    "middle": middle_attn,
                    "end": end_attn,
                }
            )

    n_total = n_layers * n_heads

    # Use the SAME conformality criterion as the pre-registration: the
    # 1D Δx-averaged power-law fit. (Doing it any other way would
    # change the head sample and confound the comparison.)
    conformal = [
        h for h in head_results
        if h["R2_1d"] >= CONFORMAL_R2 and h["delta_1d"] >= CONFORMAL_DELTA
    ]
    n_conformal = len(conformal)

    if n_conformal >= 2:
        deltas_pl = np.array([h["delta_PL"] for h in conformal])
        deltas_bcft = np.array([h["delta_BCFT"] for h in conformal])
        deltas_1d = np.array([h["delta_1d"] for h in conformal])
        valleys = np.array([h["valley_depth"] for h in conformal])
        rho_pl, p_pl = spearmanr(deltas_pl, valleys)
        rho_bcft, p_bcft = spearmanr(deltas_bcft, valleys)
        rho_1d, p_1d = spearmanr(deltas_1d, valleys)
    else:
        rho_pl = p_pl = rho_bcft = p_bcft = rho_1d = p_1d = float("nan")

    elapsed_total = time.time() - t_start
    print(
        f"[{model_name}] DONE in {elapsed_total:.0f}s — "
        f"{n_conformal}/{n_total} conformal heads",
        flush=True,
    )
    print(
        f"[{model_name}]   ρ(Δ_1d,   valley) = {rho_1d:+.4f}  p={p_1d:.2e}",
        flush=True,
    )
    print(
        f"[{model_name}]   ρ(Δ_PL,   valley) = {rho_pl:+.4f}  p={p_pl:.2e}",
        flush=True,
    )
    print(
        f"[{model_name}]   ρ(Δ_BCFT, valley) = {rho_bcft:+.4f}  p={p_bcft:.2e}",
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
        "spearman_rho_1d": float(rho_1d),
        "spearman_p_1d": float(p_1d),
        "spearman_rho_PL": float(rho_pl),
        "spearman_p_PL": float(p_pl),
        "spearman_rho_BCFT": float(rho_bcft),
        "spearman_p_BCFT": float(p_bcft),
        "elapsed_seconds": float(elapsed_total),
        "heads": head_results,
    }


DEFAULT_MODELS = [
    "EleutherAI/pythia-2.8b",
    "EleutherAI/gpt-neo-2.7B",
]


@app.local_entrypoint()
def main(models: str = ",".join(DEFAULT_MODELS)):
    """Run the BCFT functional-form fit on each model in `models`."""
    import json
    import os
    import subprocess
    from datetime import datetime, timezone

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    print(f"\n{'=' * 78}")
    print(f"  BCFT FUNCTIONAL-FORM FIT — {len(model_list)} models")
    print(f"{'=' * 78}\n")
    print("  Models:")
    for m in model_list:
        print(f"    - {m}  (gpu={_gpu_for(m)})")
    print()

    git_commit = "unknown"
    git_dirty = False
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
        ).decode().strip()
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
        ).decode().strip()
        git_dirty = bool(git_status)
    except Exception:
        pass

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
    )
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    summary_path = os.path.join(
        out_dir, f"bcft_functional_form_fit_{timestamp}.json"
    )

    all_results = {
        "timestamp_utc": timestamp,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "seq_len": SEQ_LEN,
        "n_inputs": N_INPUTS,
        "conformal_R2_threshold": CONFORMAL_R2,
        "conformal_delta_threshold": CONFORMAL_DELTA,
        "models": {},
    }

    spawned = {}
    for model_name in model_list:
        gpu = _gpu_for(model_name)
        f = measure_a100_80gb if gpu == "A100-80GB" else measure_a100
        print(f"  Spawning {gpu} job for {model_name}...", flush=True)
        spawned[model_name] = f.spawn(model_name)

    for model_name, fc in spawned.items():
        print(f"\n  Awaiting result for {model_name}...", flush=True)
        try:
            result = fc.get()
            all_results["models"][model_name] = result
            print(
                f"  → {model_name}: "
                f"ρ_1d={result['spearman_rho_1d']:+.4f}, "
                f"ρ_PL={result['spearman_rho_PL']:+.4f}, "
                f"ρ_BCFT={result['spearman_rho_BCFT']:+.4f}, "
                f"n_conf={result['n_conformal_heads']}"
            )
        except Exception as exc:
            print(f"  → {model_name}: FAILED — {exc}", flush=True)
            all_results["models"][model_name] = {
                "model": model_name,
                "error": str(exc),
            }

        with open(summary_path, "w") as fp:
            json.dump(all_results, fp, indent=2)

    print(f"\n{'=' * 78}")
    print(f"  ALL RESULTS")
    print(f"{'=' * 78}\n")
    print(
        f"  {'MODEL':40s} {'n_conf':>6s}  "
        f"{'ρ_1d':>8s}  {'ρ_PL':>8s}  {'ρ_BCFT':>8s}"
    )
    print(f"  {'-' * 78}")
    for m, r in all_results["models"].items():
        if "error" in r:
            print(f"  {m:40s}  ERROR: {r['error']}")
            continue
        print(
            f"  {m:40s} {r['n_conformal_heads']:>6d}  "
            f"{r['spearman_rho_1d']:>+8.4f}  "
            f"{r['spearman_rho_PL']:>+8.4f}  "
            f"{r['spearman_rho_BCFT']:>+8.4f}"
        )

    print(f"\n  Full results saved to: {summary_path}")
    print(f"  Git commit at run time: {git_commit}{' (DIRTY)' if git_dirty else ''}\n")
