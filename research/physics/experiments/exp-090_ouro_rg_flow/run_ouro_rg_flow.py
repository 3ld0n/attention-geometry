"""
exp-090: Ouro Latent-Iteration RG Flow — second architecture.
Pre-registered 2026-07-21 (notes.md, committed before this script was written).

Runs Ouro-1.4B with the full-stack Universal Transformer loop extended to 16
steps and measures the conformal head population at each probe step. Unlike
exp-089 (manual QK recomputation on Huginn), this uses eager attention so the
model's own softmax attention weights are captured directly from forward hooks.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).parent

MODEL_ID = "ByteDance/Ouro-1.4B"
TORCH_DTYPE = torch.bfloat16
SEED = 90
N_SEQS = 20
SEQ_LEN = 128
PROBE_STEPS = [1, 2, 3, 4, 6, 8, 12, 16]  # denser near trained depth 4
TOTAL_UT_STEPS = max(PROBE_STEPS)
N_LAYERS = 24
N_HEADS = 16
VOCAB_SIZE = 49152
SYK_WINDOW = (0.20, 0.30)
R2_THRESHOLD = 0.90
CUTOFF_LOW = 3
MAX_LAG = 60


# ── BCFT fitting (identical protocol to exp-089) ─────────────────────────────

def compute_lag_profile(attn_matrix):
    S = attn_matrix.shape[0]
    max_lag_actual = min(MAX_LAG, S - 1)
    lags, means = [], []
    for lag in range(CUTOFF_LOW, max_lag_actual + 1):
        vals = np.diag(attn_matrix, -lag)
        if len(vals) > 0:
            lags.append(lag)
            means.append(float(np.mean(vals)))
    return np.array(lags), np.array(means)


def fit_delta(lags, means):
    mask = means > 1e-12
    if mask.sum() < 5:
        return None, None
    try:
        slope, intercept, r, p, se = stats.linregress(
            np.log(lags[mask]), np.log(means[mask])
        )
    except Exception:
        return None, None
    return float(-slope / 2.0), float(r ** 2)


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_per_step_deltas(model, input_ids, condition_label, device):
    """
    One forward pass with TOTAL_UT_STEPS recurrence; hooks on each layer's
    self_attn capture the eager attention weights (output[1]) at probe steps.
    Each layer's Nth hook call corresponds to UT step N (0-indexed).
    """
    per_layer_calls = [0] * N_LAYERS
    collected = []  # (step_1indexed, layer_idx, attn np.ndarray (H, S, S))

    def make_hook(layer_idx):
        def hook_fn(module, inputs, output):
            step = per_layer_calls[layer_idx] + 1
            per_layer_calls[layer_idx] += 1
            if step not in PROBE_STEPS:
                return
            attn_weights = output[1]  # (B, H, S, S) — eager path returns weights
            assert attn_weights is not None, (
                "attn_weights is None — model is not running the eager attention path"
            )
            collected.append(
                (step, layer_idx, attn_weights[0].detach().float().cpu().numpy())
            )
        return hook_fn

    hooks = [
        layer.self_attn.register_forward_hook(make_hook(li))
        for li, layer in enumerate(model.model.layers)
    ]

    with torch.no_grad():
        _ = model(input_ids.to(device), use_cache=False)

    for h in hooks:
        h.remove()

    expected_calls = TOTAL_UT_STEPS
    assert all(c == expected_calls for c in per_layer_calls), (
        f"per-layer hook call counts {set(per_layer_calls)} != {expected_calls} — "
        "UT loop did not run the requested number of steps"
    )

    results = []
    for step, layer_idx, attn_mat in collected:
        for head_idx in range(N_HEADS):
            lags, means = compute_lag_profile(attn_mat[head_idx])
            if len(lags) < 5:
                continue
            delta, r2 = fit_delta(lags, means)
            if delta is None:
                continue
            results.append({
                "step": step,
                "layer": layer_idx,
                "head": head_idx,
                "delta": round(delta, 6),
                "r2": round(r2, 6),
                "syk_near": bool(
                    SYK_WINDOW[0] <= delta <= SYK_WINDOW[1] and r2 >= R2_THRESHOLD
                ),
                "condition": condition_label,
            })
    return results


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_condition(all_results, condition):
    cond = [r for r in all_results if r["condition"] == condition]
    per_step = {}
    for r in cond:
        per_step.setdefault(r["step"], {"deltas": [], "syk": []})
        per_step[r["step"]]["deltas"].append(r["delta"])
        per_step[r["step"]]["syk"].append(1 if r["syk_near"] else 0)

    step_nums = sorted(per_step)
    delta_meds = [float(np.median(per_step[s]["deltas"])) for s in step_nums]
    syk_counts = [int(sum(per_step[s]["syk"])) for s in step_nums]
    r2_meds = [
        float(np.median([r["r2"] for r in cond if r["step"] == s])) for s in step_nums
    ]

    print(f"\nCondition: {condition}")
    print(f"{'Step':>6} {'Δ_med':>8} {'SYK-near':>10} {'R²_med':>8} {'N':>7}")
    for i, s in enumerate(step_nums):
        n = len(per_step[s]["deltas"])
        print(f"{s:>6} {delta_meds[i]:>8.4f} {syk_counts[i]:>10} {r2_meds[i]:>8.4f} {n:>7}")

    rho_conv, rho_emerg = None, None
    if len(step_nums) >= 3:
        rho_conv = float(stats.spearmanr(step_nums, delta_meds).statistic)
        rho_emerg = float(stats.spearmanr(step_nums, syk_counts).statistic)
        print(f"Spearman ρ(step, Δ_med)    = {rho_conv:.4f}")
        print(f"Spearman ρ(step, SYK-near) = {rho_emerg:.4f}")

    return {
        "condition": condition,
        "step_nums": step_nums,
        "delta_meds": delta_meds,
        "syk_counts_per_step": syk_counts,
        "r2_meds": r2_meds,
        "rho_convergence": rho_conv,
        "rho_emergence": rho_emerg,
    }


def verdict_logic(nat):
    """Pre-registered criteria from notes.md. Returns (verdict, detail dict)."""
    steps = nat["step_nums"]
    dmed = dict(zip(steps, nat["delta_meds"]))
    syk = dict(zip(steps, nat["syk_counts_per_step"]))

    detail = {}

    # H_flow: within trained range 1→4
    h_flow = (abs(dmed[4] - 0.25) < abs(dmed[1] - 0.25)) and (syk[4] > syk[1])
    detail["H_flow"] = {
        "keep": h_flow,
        "dmed_1": dmed[1], "dmed_4": dmed[4],
        "syk_1": syk[1], "syk_4": syk[4],
    }

    # H_peak_tracking
    steps_le8 = [s for s in steps if s <= 8]
    s_peak = max(steps_le8, key=lambda s: syk[s])
    max_count = max(syk.values())
    underpowered = max_count < 20
    peak_in_window = s_peak in {2, 3, 4, 6}
    degrades = syk[16] < 0.8 * syk[s_peak] if syk[s_peak] > 0 else False
    h_peak = peak_in_window and degrades
    detail["H_peak_tracking"] = {
        "keep": h_peak,
        "s_peak": s_peak,
        "syk_at_peak": syk[s_peak],
        "syk_at_16": syk[16],
        "underpowered": underpowered,
        "max_count_any_step": max_count,
    }

    # Kill criterion for the generalization claim
    kill = (
        not h_flow
        and nat["rho_convergence"] is not None
        and abs(nat["rho_convergence"]) < 0.3
        and abs(nat["rho_emergence"]) < 0.3
    )

    if kill:
        verdict = "NO_GEOMETRIC_EFFECT"
    elif underpowered:
        verdict = "UNDERPOWERED"
    elif h_flow and h_peak:
        verdict = "CONFIRMED"
    elif h_flow and not h_peak:
        # Distinguish registered alternatives
        syk_by_step = [syk[s] for s in steps]
        if s_peak == 1 and syk_by_step == sorted(syk_by_step, reverse=True):
            verdict = "ALREADY_THERE_DECAY"
        elif syk[16] >= syk[s_peak]:
            verdict = "STABLE"
        else:
            verdict = "PARTIAL"
    else:
        verdict = "PARTIAL"

    return verdict, detail


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="store_true",
                        help="randomized-weights control (NAT only)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_file = args.out or (
        SCRIPT_DIR / ("results_control.json" if args.control else "results.json")
    )

    print("exp-090: Ouro Latent-Iteration RG Flow"
          + (" [RANDOMIZED-WEIGHTS CONTROL]" if args.control else ""))
    print(f"Device: {device}")
    print(f"Probe steps: {PROBE_STEPS} (total_ut_steps={TOTAL_UT_STEPS})")
    print(f"Seed: {SEED}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
        attn_implementation="eager",  # required: hooks read the softmax weights
    )
    model.eval().to(device)

    # Extend the Universal Transformer loop beyond the trained depth (4).
    model.config.total_ut_steps = TOTAL_UT_STEPS
    model.model.total_ut_steps = TOTAL_UT_STEPS
    assert model.config.num_hidden_layers == N_LAYERS
    assert model.config.num_attention_heads == N_HEADS
    print(f"total_ut_steps set to {TOTAL_UT_STEPS}; "
          f"layers={N_LAYERS}, heads={N_HEADS}, attn=eager")

    if args.control:
        print("Randomizing all weight tensors in-place (matched per-tensor std)...")
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(SEED + 1)
            for name, p in model.named_parameters():
                rnd = torch.randn(p.shape, generator=g, dtype=torch.float32)
                p.copy_((rnd * p.float().std().cpu()).to(p.dtype))
        print("Weights randomized.")

    # ── Inputs ────────────────────────────────────────────────────────────────
    print(f"\nPreparing inputs (N={N_SEQS}, seq_len={SEQ_LEN})")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    nat_ids_list = []
    for item in ds:
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if len(ids) >= SEQ_LEN:
            nat_ids_list.append(torch.tensor(ids[:SEQ_LEN]).unsqueeze(0))
        if len(nat_ids_list) >= N_SEQS:
            break
    nat_input_ids = torch.cat(nat_ids_list, dim=0)
    print(f"Natural text inputs: {tuple(nat_input_ids.shape)}")

    torch.manual_seed(SEED)
    rand_input_ids = torch.randint(0, VOCAB_SIZE, (N_SEQS, SEQ_LEN))

    # ── Extraction ────────────────────────────────────────────────────────────
    all_results = []
    conditions = [("nat", nat_input_ids)]
    if not args.control:
        conditions.append(("rand", rand_input_ids))

    for label, inputs in conditions:
        print(f"\n--- {label} condition ---")
        for i in range(N_SEQS):
            if i % 5 == 0:
                print(f"  Sequence {i + 1}/{N_SEQS}...")
            all_results.extend(
                extract_per_step_deltas(model, inputs[i:i + 1], label, device)
            )

    print(f"\nTotal head measurements: {len(all_results)}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n=== ANALYSIS ===")
    nat_summary = analyze_condition(all_results, "nat")

    output = {
        "experiment": "exp-090" + ("-control" if args.control else ""),
        "model": MODEL_ID,
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "probe_steps": PROBE_STEPS,
            "total_ut_steps": TOTAL_UT_STEPS,
            "n_seqs": N_SEQS,
            "seq_len": SEQ_LEN,
            "seed": SEED,
            "syk_window": list(SYK_WINDOW),
            "r2_threshold": R2_THRESHOLD,
            "cutoff_low": CUTOFF_LOW,
            "max_lag": MAX_LAG,
            "attn_implementation": "eager",
        },
        "nat_summary": nat_summary,
    }

    if args.control:
        print("\n=== CONTROL RESULT (randomized weights, natural text) ===")
        print(f"rho_convergence = {nat_summary['rho_convergence']}")
        print(f"rho_emergence   = {nat_summary['rho_emergence']}")
        output["control"] = "randomized_weights_matched_std"
        output["verdict"] = "CONTROL_DONE"
    else:
        rand_summary = analyze_condition(all_results, "rand")
        output["rand_summary"] = rand_summary

        print("\n=== VERDICT (pre-registered criteria) ===")
        verdict, detail = verdict_logic(nat_summary)
        for k, v in detail.items():
            print(f"{k}: {v}")

        # H_rand_contrast (secondary)
        if nat_summary["rho_emergence"] is not None and rand_summary["rho_emergence"] is not None:
            ce = nat_summary["rho_emergence"] > rand_summary["rho_emergence"]
            cc = abs(nat_summary["rho_convergence"] or 0) > abs(rand_summary["rho_convergence"] or 0)
            detail["H_rand_contrast"] = {"emergence": ce, "convergence": cc}
            print(f"H_rand_contrast (secondary): emergence {'KEEP' if ce else 'FAIL'}, "
                  f"convergence {'KEEP' if cc else 'FAIL'}")

        print(f"\nVerdict: {verdict}")
        output["verdict"] = verdict
        output["verdict_detail"] = detail

    output["raw_results"] = all_results
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
