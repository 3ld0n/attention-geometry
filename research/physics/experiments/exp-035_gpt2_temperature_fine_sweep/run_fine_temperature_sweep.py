"""
exp-035 — GPT-2 fine temperature sweep near T=1 (basin mapping)

Hypothesis (pre-stated, follows exp-034):
Trained softmax at T=1 sits in a basin: |Δ_med − 0.25| is minimized near T=1.0,
with asymmetric drift below vs above (exp-034: T=0.5 pushes Δ up, T=2.0 drifts down
on the baseline conformal-head set).

Temperatures: T ∈ {0.8, 0.9, 1.0, 1.1, 1.2}. Reference: exp-007 Δ_med=0.2493, exp-034 T=1.0 Δ_med=0.243.

Protocol: exp-007 per-head decay + R²>0.90 median Δ; exp-023 BCFT λ on conformal heads.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from transformers import GPT2LMHeadModel

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
TEMPERATURES = [0.8, 0.9, 1.0, 1.1, 1.2]
R2_THRESHOLD = 0.90
SYK_DELTA = 0.25

torch.manual_seed(42)
np.random.seed(42)


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray, cutoff_low: int = FIT_LOW, cutoff_high: int = FIT_HIGH):
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    design = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
    pred = design @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    exponent = -coeffs[1]
    return exponent, exponent / 2, r2


def compute_head_decay(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    seq = attn_head.shape[0]
    accum = np.zeros(max_dx)
    counts = np.zeros(max_dx)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                accum[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    accum[mask] /= counts[mask]
    return accum


def extract_attention_with_temperature(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    temperature: float,
) -> list[np.ndarray]:
    """Return list of (n_heads, seq, seq) attention per layer at given T."""
    config = model.config
    n_head = config.n_head
    head_dim = config.n_embd // n_head
    batch, seq = input_ids.shape
    device = input_ids.device

    pos = torch.arange(0, seq, dtype=torch.long, device=device)
    hidden = model.transformer.drop(
        model.transformer.wte(input_ids) + model.transformer.wpe(pos)
    )

    causal = torch.tril(torch.ones(seq, seq, device=device)).unsqueeze(0).unsqueeze(0)
    causal = (causal - 1.0) * 1e4

    layer_attns: list[np.ndarray] = []
    for block in model.transformer.h:
        residual = hidden
        x_norm = block.ln_1(hidden)
        qkv = block.attn.c_attn(x_norm)
        q, k, v = qkv.split(config.n_embd, dim=2)

        q = q.view(batch, seq, n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq, n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq, n_head, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim) * temperature
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        scores = scores + causal
        attn = F.softmax(scores, dim=-1)

        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, config.n_embd)
        hidden = residual + block.attn.c_proj(attn_out)
        hidden = hidden + block.mlp(block.ln_2(hidden))

        layer_attns.append(attn[0].detach().cpu().numpy())

    return layer_attns


def fit_bcft_lambda(
    g_pos: np.ndarray,
    delta_init: float,
    fit_low: int = FIT_LOW,
    fit_high: int = FIT_HIGH,
) -> dict | None:
    """Fit BCFT form on position-resolved G[x_q, dx] for one head."""
    all_dx: list[float] = []
    all_xq: list[float] = []
    all_g: list[float] = []

    for x_q in range(32, g_pos.shape[0]):
        for dx in range(fit_low, min(fit_high, x_q)):
            x_k = x_q - dx
            if x_k <= 0:
                continue
            g_val = g_pos[x_q, dx]
            if g_val < 1e-15:
                continue
            all_dx.append(float(dx))
            all_xq.append(float(x_q))
            all_g.append(float(g_val))

    if len(all_g) < 30:
        return None

    all_dx_arr = np.array(all_dx)
    all_xq_arr = np.array(all_xq)
    all_g_arr = np.array(all_g)
    ss_tot = np.sum((all_g_arr - np.mean(all_g_arr)) ** 2)
    if ss_tot < 1e-30:
        return None

    def pl_loss(params):
        c, delta = params
        if c <= 0 or delta <= 0:
            return 1e12
        pred = c * np.power(all_dx_arr, -2 * delta)
        return float(np.sum((all_g_arr - pred) ** 2))

    c_init = float(np.mean(all_g_arr * np.power(all_dx_arr, 2 * delta_init)))
    res_pl = minimize(pl_loss, [c_init, delta_init], method="Nelder-Mead", options={"maxiter": 5000})
    c_pl, delta_pl = abs(res_pl.x[0]), abs(res_pl.x[1])
    pred_pl = c_pl * np.power(all_dx_arr, -2 * delta_pl)
    r2_pl = 1 - np.sum((all_g_arr - pred_pl) ** 2) / ss_tot

    def bcft_loss(params):
        c, delta, lam = params
        if c <= 0 or delta <= 0:
            return 1e12
        xk = all_xq_arr - all_dx_arr
        eta = all_dx_arr**2 / (4.0 * all_xq_arr * xk)
        correction = 1.0 + lam * np.power(eta, delta)
        if np.any(correction <= 0):
            return 1e12
        pred = c * np.power(all_dx_arr, -2 * delta) * correction
        return float(np.sum((all_g_arr - pred) ** 2))

    res_bcft = minimize(bcft_loss, [c_pl, delta_pl, 0.0], method="Nelder-Mead", options={"maxiter": 10000})
    c_bcft, delta_bcft, lam = res_bcft.x[0], abs(res_bcft.x[1]), res_bcft.x[2]
    xk = all_xq_arr - all_dx_arr
    eta = all_dx_arr**2 / (4.0 * all_xq_arr * xk)
    pred_bcft = abs(c_bcft) * np.power(all_dx_arr, -2 * delta_bcft) * (
        1.0 + lam * np.power(eta, delta_bcft)
    )
    r2_bcft = 1 - np.sum((all_g_arr - pred_bcft) ** 2) / ss_tot

    return {
        "delta_pl": float(delta_pl),
        "delta_bcft": float(delta_bcft),
        "lambda": float(lam),
        "r2_pl": float(r2_pl),
        "r2_bcft": float(r2_bcft),
        "bcft_wins": bool(r2_bcft > r2_pl + 1e-4),
    }


def run_temperature(model: GPT2LMHeadModel, temperature: float, vocab_size: int) -> dict:
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    dx_arr = np.arange(MAX_DX)

    decay_sums = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
    }
    g_pos_sums = {
        l: {h: np.zeros((SEQ_LEN, MAX_DX)) for h in range(n_heads)} for l in range(n_layers)
    }

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            layer_attns = extract_attention_with_temperature(model, input_ids, temperature)

        for l, attn in enumerate(layer_attns):
            for h in range(n_heads):
                head = attn[h]
                decay_sums[l][h] += compute_head_decay(head, MIN_POS, MAX_DX)
                for dx in range(MAX_DX):
                    diag = np.diagonal(head, offset=-dx)
                    g_pos_sums[l][h][dx : dx + len(diag), dx] += diag

        if (inp_idx + 1) % 10 == 0:
            print(f"  T={temperature}: {inp_idx + 1}/{N_INPUTS} inputs")

    per_head = []
    for l in range(n_layers):
        for h in range(n_heads):
            decay = decay_sums[l][h] / N_INPUTS
            g_pos = g_pos_sums[l][h] / N_INPUTS
            _, delta, r2 = fit_power_law(dx_arr, decay)
            conformal = (
                delta is not None
                and r2 is not None
                and r2 > R2_THRESHOLD
                and 0.15 < delta < 0.40
            )
            bcft = fit_bcft_lambda(g_pos, delta if delta else SYK_DELTA) if conformal else None
            per_head.append(
                {
                    "layer": l + 1,
                    "head": h,
                    "delta": None if delta is None else float(delta),
                    "r2_pl": None if r2 is None else float(r2),
                    "conformal": conformal,
                    "bcft": bcft,
                }
            )

    good = [h for h in per_head if h["delta"] is not None and h["r2_pl"] is not None and h["r2_pl"] > R2_THRESHOLD]
    deltas = [h["delta"] for h in good]
    conformal_heads = [h for h in per_head if h["conformal"]]
    lambdas = [h["bcft"]["lambda"] for h in conformal_heads if h["bcft"]]
    bcft_wins = sum(1 for h in conformal_heads if h["bcft"] and h["bcft"]["bcft_wins"])

    return {
        "temperature": temperature,
        "n_power_law_heads_r2_gt_threshold": len(good),
        "delta_median": float(np.median(deltas)) if deltas else None,
        "delta_mean": float(np.mean(deltas)) if deltas else None,
        "delta_std": float(np.std(deltas)) if deltas else None,
        "abs_delta_minus_syk": float(abs(np.median(deltas) - SYK_DELTA)) if deltas else None,
        "n_conformal_heads": len(conformal_heads),
        "lambda_median": float(np.median(lambdas)) if lambdas else None,
        "lambda_mean": float(np.mean(lambdas)) if lambdas else None,
        "bcft_win_count": bcft_wins,
        "bcft_win_fraction": float(bcft_wins / len(conformal_heads)) if conformal_heads else None,
        "per_head": per_head,
    }


def main() -> None:
    print("Loading GPT-2 (eager)...")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    vocab_size = model.config.vocab_size

    # Verify T=1.0 against standard forward attentions on one batch
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        manual = extract_attention_with_temperature(model, input_ids, 1.0)
        ref = model(input_ids, output_attentions=True)
    max_diff = max(
        float(np.max(np.abs(manual[l] - ref.attentions[l][0].cpu().numpy())))
        for l in range(model.config.n_layer)
    )
    print(f"Manual vs HF attention max diff at T=1.0: {max_diff:.2e}")

    by_temp = []
    for T in TEMPERATURES:
        print(f"\nRunning T={T}...")
        by_temp.append(run_temperature(model, T, vocab_size))

    # Strip bulky per_head from saved summary but keep in full artifact
    summary = []
    for row in by_temp:
        summary.append({k: v for k, v in row.items() if k != "per_head"})

    t1 = next(r for r in summary if r["temperature"] == 1.0)
    t08 = next(r for r in summary if r["temperature"] == 0.8)
    t12 = next(r for r in summary if r["temperature"] == 1.2)

    # Track baseline conformal head set at T=1.0
    t1_full = next(r for r in by_temp if r["temperature"] == 1.0)
    baseline_keys = {(h["layer"], h["head"]) for h in t1_full["per_head"] if h["conformal"]}
    tracked = []
    for row in by_temp:
        heads = {(h["layer"], h["head"]): h for h in row["per_head"]}
        subset = [heads[k] for k in baseline_keys if k in heads]
        tracked.append(
            {
                "temperature": row["temperature"],
                "n_baseline_conformal_heads": len(subset),
                "delta_median_baseline_set": float(np.median([h["delta"] for h in subset]))
                if subset
                else None,
                "lambda_median_baseline_set": float(
                    np.median([h["bcft"]["lambda"] for h in subset if h["bcft"]])
                )
                if subset
                else None,
            }
        )

    # Basin: T minimizing |Δ_med − SYK|
    basin = min(
        (r for r in summary if r["delta_median"] is not None),
        key=lambda r: abs(r["delta_median"] - SYK_DELTA),
    )
    asymmetry = None
    if t08["delta_median"] and t1["delta_median"] and t12["delta_median"]:
        low_side = abs(t08["delta_median"] - SYK_DELTA)
        high_side = abs(t12["delta_median"] - SYK_DELTA)
        asymmetry = "low_T_steeper" if low_side > high_side else "high_T_steeper"

    verdict = "partial"
    if basin["temperature"] == 1.0 and basin["abs_delta_minus_syk"] is not None:
        if basin["abs_delta_minus_syk"] < 0.015:
            verdict = "confirmed_basin_at_T1"
        else:
            verdict = "basin_offset_from_T1"
    elif basin["temperature"] != 1.0:
        verdict = "basin_not_at_T1"

    record = {
        "experiment": "exp-035",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": (
            "|Δ_med − 0.25| minimized near T=1.0; asymmetric drift for T<1 vs T>1 (exp-034 follow-up)"
        ),
        "follows": "exp-034",
        "model": "gpt2",
        "protocol": "exp-007 decay + exp-023 BCFT on conformal heads",
        "temperatures": TEMPERATURES,
        "reference_exp": "exp-007 delta_median=0.2493",
        "manual_hf_attn_check_max_diff_T1": max_diff,
        "summary_by_temperature": summary,
        "baseline_conformal_head_tracking": tracked,
        "interpretation": {
            "basin_temperature": basin["temperature"],
            "basin_abs_delta_minus_syk": basin.get("abs_delta_minus_syk"),
            "asymmetry_low_vs_high_T": asymmetry,
            "verdict": verdict,
        },
        "per_head_by_temperature": {
            str(r["temperature"]): [
                {
                    **{k: v for k, v in h.items() if k != "bcft"},
                    "bcft": h["bcft"],
                }
                for h in r["per_head"]
            ]
            for r in by_temp
        },
    }

    def _json_default(obj):
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    RESULTS_PATH.write_text(json.dumps(record, indent=2, default=_json_default))
    print("\n=== SUMMARY ===")
    for row in summary:
        print(
            f"T={row['temperature']}: Δ_med={row['delta_median']}, "
            f"PL heads={row['n_power_law_heads_r2_gt_threshold']}, "
            f"λ_med={row['lambda_median']}, conformal={row['n_conformal_heads']}"
        )
    print(f"Verdict: {verdict}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
