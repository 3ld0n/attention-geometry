"""
Local BCFT functional-form fit for exp-067 Test A extension.

Runs the exp-026 BCFT measurement protocol on locally-cached models using
MPS (Apple Silicon) instead of Modal cloud GPUs. Outputs a JSON file in
exp-026 format (with delta_BCFT, lambda_BCFT, valley_depth per head) that
can be consumed directly by run_test_a.py.

Models to measure (all locally cached):
  - gpt2-medium    (24L/16H, ~345M params)
  - pythia-1.4b    (24L/16H, ~1.4B params)

Usage:
  .venv/bin/python3 research/physics/experiments/exp-067_v2_prereg_tests/run_bcft_fit_local.py

Ariel — June 17, 2026. Exp-067 Test A extension (no new hypothesis — local
measurement step feeding the archived Test A script).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]  # .../ariel

# Measurement geometry (matches exp-026 / exp-025 pipeline exactly)
SEQ_LEN = 512
N_INPUTS = 50
FIT_LOW = 3
FIT_HIGH = min(120, SEQ_LEN // 3)   # = 120
DEEP_LO = SEQ_LEN // 2              # = 256
Q_LO = 3 * SEQ_LEN // 4             # = 384
CONFORMAL_R2 = 0.85
CONFORMAL_DELTA = 0.05


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fit_one_model(model_name: str, hf_id: str) -> dict:
    """Run forward passes and per-head 1D-PL + 2D-BCFT fits. Returns dict."""
    device = get_device()
    print(f"\n[{model_name}] Loading (fp16, eager, {device})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
    n_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head")
    vocab_size = config.vocab_size
    print(f"[{model_name}] {n_layers}L/{n_heads}H, vocab {vocab_size}", flush=True)

    # Accumulate averaged attention matrices
    attn_sum = [
        np.zeros((n_heads, SEQ_LEN, SEQ_LEN), dtype=np.float64)
        for _ in range(n_layers)
    ]

    t0 = time.time()
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
        if device.type == "mps":
            torch.mps.empty_cache()
        if (inp + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{model_name}] forward {inp+1}/{N_INPUTS} ({elapsed:.0f}s)", flush=True)

    for l in range(n_layers):
        attn_sum[l] /= N_INPUTS

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"[{model_name}] Fitting per-head PL and BCFT forms...", flush=True)
    t_fit = time.time()

    head_results = []

    for l in range(n_layers):
        for h in range(n_heads):
            A = attn_sum[l][h]

            # 1D Δx-averaged profile (pipeline-standard) ------------------
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

            log_dx = np.log(dx_arr_1d[valid_1d].astype(float))
            log_y  = np.log(y_1d[valid_1d])
            A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
            pred_1d = A_mat @ coeffs
            ss_res_1d = float(np.sum((log_y - pred_1d) ** 2))
            ss_tot_1d = float(np.sum((log_y - np.mean(log_y)) ** 2))
            R2_1d = 1.0 - ss_res_1d / ss_tot_1d if ss_tot_1d > 1e-30 else 0.0
            delta_1d = float(-coeffs[1] / 2.0)

            # 2D position-resolved data for BCFT fit ----------------------
            all_dx, all_xq, all_g = [], [], []
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
            all_g  = np.array(all_g)

            if len(all_g) < 30:
                continue

            ss_tot_2d = float(np.sum((all_g - np.mean(all_g)) ** 2))
            if ss_tot_2d < 1e-30:
                continue

            # Bare PL fit on 2D data
            def pl_loss(params):
                C, delta = params
                if C <= 0 or delta <= 0:
                    return 1e12
                pred = C * np.power(all_dx, -2.0 * delta)
                return float(np.sum((all_g - pred) ** 2))

            C_init = float(np.mean(all_g * np.power(all_dx, 2.0 * delta_1d)))
            delta_init = max(0.05, delta_1d)
            res_pl = minimize(pl_loss, [C_init, delta_init], method="Nelder-Mead",
                              options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12})
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
                corr = 1.0 + lam * np.power(eta, delta)
                if np.any(corr <= 0):
                    return 1e12
                pred = C * np.power(all_dx, -2.0 * delta) * corr
                return float(np.sum((all_g - pred) ** 2))

            res_bcft = minimize(bcft_loss, [C_pl, delta_pl, 0.0], method="Nelder-Mead",
                                options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-12})
            C_bcft   = abs(float(res_bcft.x[0]))
            delta_bcft  = abs(float(res_bcft.x[1]))
            lambda_bcft = float(res_bcft.x[2])
            corr = 1.0 + lambda_bcft * np.power(eta, delta_bcft)
            pred_bcft = C_bcft * np.power(all_dx, -2.0 * delta_bcft) * corr
            R2_bcft = 1.0 - float(np.sum((all_g - pred_bcft) ** 2)) / ss_tot_2d

            # Valley depth (pipeline-standard) ----------------------------
            profile = np.zeros(SEQ_LEN)
            p_count = 0
            for x_q in range(Q_LO, SEQ_LEN):
                profile[: x_q + 1] += A[x_q, : x_q + 1]
                p_count += 1
            if p_count > 0:
                profile /= p_count
            if np.sum(profile) > 1e-12:
                profile /= np.sum(profile)

            THIRD = SEQ_LEN // 3
            g_start  = float(np.mean(profile[1: THIRD]))
            g_mid    = float(np.mean(profile[THIRD: 2 * THIRD]))
            g_end    = float(np.mean(profile[2 * THIRD: 3 * THIRD]))
            peak = max(g_start, g_end)
            valley_depth = float(1.0 - g_mid / peak) if peak > 1e-12 else 0.0

            head_results.append({
                "layer": int(l),
                "head": int(h),
                "R2_1d": float(R2_1d),
                "delta_1d": float(delta_1d),
                "R2_pl_2d": float(R2_pl),
                "delta_PL": float(delta_pl),
                "R2_BCFT": float(R2_bcft),
                "delta_BCFT": float(delta_bcft),
                "lambda_BCFT": float(lambda_bcft),
                "valley_depth": float(valley_depth),
                "g_start": float(g_start),
                "g_mid": float(g_mid),
                "g_end": float(g_end),
            })

        if (l + 1) % 4 == 0:
            elapsed_fit = time.time() - t_fit
            print(f"[{model_name}] layer {l+1}/{n_layers} fit ({elapsed_fit:.0f}s)", flush=True)

    print(f"[{model_name}] Done. {len(head_results)} heads fitted in {time.time()-t_fit:.0f}s",
          flush=True)
    return {
        "model": hf_id,
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "n_heads_fitted": len(head_results),
        "heads": head_results,
    }


def main() -> None:
    out_dir = ROOT / "research/physics/experiments/exp-026_bcft_functional_form/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("gpt2-medium", "gpt2-medium"),
        ("pythia-1.4b", "EleutherAI/pythia-1.4b"),
    ]

    for short_name, hf_id in models:
        print(f"\n{'='*60}")
        print(f"Measuring {hf_id}")
        print("="*60)
        t0 = time.time()
        model_result = _fit_one_model(short_name, hf_id)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        out_file = out_dir / f"bcft_functional_form_fit_{ts}_{short_name}.json"

        # Wrap in exp-026 format (models dict keyed by hf_id)
        out_data = {
            "experiment": "exp-026-extension-local",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "exp-026 BCFT functional-form fit (local MPS version for exp-067 Test A)",
            "models": {hf_id: model_result},
        }
        out_file.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved: {out_file}")
        print(f"Total time for {hf_id}: {time.time()-t0:.0f}s")

    print("\nAll models done. Run run_test_a.py to add these to exp-067 Test A results.")


if __name__ == "__main__":
    main()
