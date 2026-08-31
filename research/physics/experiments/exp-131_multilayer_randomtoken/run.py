"""exp-131 — Theory-of-A Level 3: Multi-Layer Decomposition Under Random-Token Census.

Pre-registration committed to attention-geometry at 84cfefb before this script
was written or any forward passes run.

Tests whether the random-token census protocol reproduces exp-117's σ_delta = 0.249,
and decomposes the accumulated delta into per-layer contributions.

Ariel — 2026-08-31.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"

spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
sys.path.insert(0, str(EXP112.parent / "exp-107_natural_text_bilocal"))
exp112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp112)

pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW = exp112.WINDOW    # lags 8..256, 249 elements

SEQ_LEN = 512
N_SEQS = 50
SEED = 42
VOCAB_SIZE = 50257
PREREG_COMMIT = "84cfefb"


def r2_of_slope(profile: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, profile, rcond=None)
    y_pred = X @ c
    ss_tot = float(((profile - profile.mean()) ** 2).sum())
    ss_res = float(((profile - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def perseq_cosine_profile(arr_seq: np.ndarray) -> np.ndarray:
    """Per-row cosine profile for ONE sequence. arr_seq: (512, 768)."""
    norms = np.linalg.norm(arr_seq, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = arr_seq / norms
    C = M @ M.T
    return pooled_window_profile(C)


def sigma_meanfirst(mean_arr: np.ndarray) -> tuple[float, float]:
    """σ from mean-over-sequences array (exp-128 protocol)."""
    norms = np.linalg.norm(mean_arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = mean_arr / norms
    C = M @ M.T
    profile = pooled_window_profile(C)
    sigma = -ols_slope(profile, WINDOW)
    r2 = r2_of_slope(profile, WINDOW)
    return float(sigma), float(r2)


def main() -> None:
    print("exp-131: Multi-Layer Decomposition Under Random-Token Census", flush=True)
    print(f"Pre-registration commit: {PREREG_COMMIT}\n", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32,
        attn_implementation="eager").to(device).eval()

    # Random-token sequences — same protocol as exp-107
    rng = np.random.RandomState(SEED)
    seqs_np = rng.randint(0, VOCAB_SIZE, size=(N_SEQS, SEQ_LEN))
    seqs = torch.tensor(seqs_np, dtype=torch.long, device=device)
    print(f"Random-token sequences: {N_SEQS} × {SEQ_LEN}  seed={SEED}", flush=True)

    # Per-sequence profile accumulators
    profile_acc_perseq = {
        "delta_total":       np.zeros(len(WINDOW), dtype=np.float64),
        "delta_after_attn0": np.zeros(len(WINDOW), dtype=np.float64),
        "delta_after_block0": np.zeros(len(WINDOW), dtype=np.float64),
        "delta_after_attn1": np.zeros(len(WINDOW), dtype=np.float64),
        "attn0":             np.zeros(len(WINDOW), dtype=np.float64),
        "mlp0":              np.zeros(len(WINDOW), dtype=np.float64),
        "attn1":             np.zeros(len(WINDOW), dtype=np.float64),
        "mlp1":              np.zeros(len(WINDOW), dtype=np.float64),
    }
    # Mean-first accumulators
    acc_mean = {k: np.zeros((SEQ_LEN, 768), dtype=np.float64)
                for k in ["h0", "attn0", "mlp0", "attn1", "mlp1"]}

    handles = []
    _buf: dict[str, np.ndarray] = {}

    def hook_h0(module, args):
        _buf["h0"] = args[0].detach().float().cpu().numpy()[0]

    def make_attn_hook(key):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            _buf[key] = out.detach().float().cpu().numpy()[0]
        return hook

    def make_mlp_hook(key):
        def hook(module, args, output):
            out = output if not isinstance(output, tuple) else output[0]
            _buf[key] = out.detach().float().cpu().numpy()[0]
        return hook

    handles.append(model.transformer.h[0].register_forward_pre_hook(hook_h0))
    handles.append(model.transformer.h[0].attn.register_forward_hook(
        make_attn_hook("attn0")))
    handles.append(model.transformer.h[0].mlp.register_forward_hook(
        make_mlp_hook("mlp0")))
    handles.append(model.transformer.h[1].attn.register_forward_hook(
        make_attn_hook("attn1")))
    handles.append(model.transformer.h[1].mlp.register_forward_hook(
        make_mlp_hook("mlp1")))

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            if (i + 1) % 10 == 0:
                print(f"  Sequence {i+1}/{N_SEQS}...", flush=True)
            _buf.clear()
            model(seq.unsqueeze(0))

            h0_s    = _buf["h0"].astype(np.float64)
            attn0_s = _buf["attn0"].astype(np.float64)
            mlp0_s  = _buf["mlp0"].astype(np.float64)
            attn1_s = _buf["attn1"].astype(np.float64)
            mlp1_s  = _buf["mlp1"].astype(np.float64)

            # Accumulate mean-first
            for key, arr in [("h0", h0_s), ("attn0", attn0_s), ("mlp0", mlp0_s),
                              ("attn1", attn1_s), ("mlp1", mlp1_s)]:
                acc_mean[key] += arr / N_SEQS

            # Per-sequence profiles
            da_attn0  = attn0_s
            da_block0 = attn0_s + mlp0_s
            da_attn1  = attn0_s + mlp0_s + attn1_s
            da_total  = attn0_s + mlp0_s + attn1_s + mlp1_s

            for key, arr in [
                ("delta_total",        da_total),
                ("delta_after_attn0",  da_attn0),
                ("delta_after_block0", da_block0),
                ("delta_after_attn1",  da_attn1),
                ("attn0",              attn0_s),
                ("mlp0",               mlp0_s),
                ("attn1",              attn1_s),
                ("mlp1",               mlp1_s),
            ]:
                profile_acc_perseq[key] += perseq_cosine_profile(arr) / N_SEQS

    for h in handles:
        h.remove()

    print("\nComputing statistics...", flush=True)

    # Per-sequence results
    print("\n=== Per-sequence cosine protocol ===", flush=True)
    perseq_results = {}
    for key, profile in profile_acc_perseq.items():
        sigma = -ols_slope(profile, WINDOW)
        r2 = r2_of_slope(profile, WINDOW)
        v8, v32, v128, v256 = (float(profile[0]), float(profile[24]),
                                float(profile[120]), float(profile[-1]))
        perseq_results[key] = {
            "sigma": float(sigma), "r2": float(r2),
            "profile_dx8": v8, "profile_dx32": v32,
            "profile_dx128": v128, "profile_dx256": v256,
        }
        print(f"  {key:25s}  sigma={sigma:+.4f}  R²={r2:.4f}  "
              f"C[8]={v8:.3f}  C[256]={v256:.3f}", flush=True)

    # Mean-first results
    print("\n=== Mean-first cosine protocol ===", flush=True)
    meanfirst_results = {}
    delta_mf = (acc_mean["attn0"] + acc_mean["mlp0"]
                + acc_mean["attn1"] + acc_mean["mlp1"])
    for key, arr in [("attn0", acc_mean["attn0"]), ("mlp0", acc_mean["mlp0"]),
                     ("attn1", acc_mean["attn1"]), ("mlp1", acc_mean["mlp1"]),
                     ("delta_total", delta_mf)]:
        s, r = sigma_meanfirst(arr)
        meanfirst_results[key] = {"sigma": s, "r2": r}
        print(f"  {key:25s}  sigma={s:+.4f}  R²={r:.4f}", flush=True)

    # Registered verdicts
    sigma_dt = perseq_results["delta_total"]["sigma"]
    c8_dt = perseq_results["delta_total"]["profile_dx8"]
    sigma_b0 = perseq_results["delta_after_block0"]["sigma"]
    sigma_attn0 = perseq_results["attn0"]["sigma"]

    sigma_wikitext_perseq = 0.129    # exp-130
    sigma_attn0_wikitext = 0.183     # exp-130

    P1_ok = (0.22 <= sigma_dt <= 0.28) and (c8_dt >= 0.90)
    P2_ok = sigma_dt > sigma_wikitext_perseq
    P3_ok = sigma_attn0 > sigma_attn0_wikitext
    P4_ok = sigma_dt > sigma_b0   # cumulative grows: total > block0 alone? (reversed in WikiText)

    K1_fired = (sigma_dt < 0.20) and (c8_dt < 0.85)

    print(f"\n=== Registered verdicts ===", flush=True)
    print(f"  P1 (σ∈[0.22,0.28] & C[8]≥0.90):  "
          f"σ={sigma_dt:.4f}, C[8]={c8_dt:.3f}  → {'OK' if P1_ok else 'FAIL'}", flush=True)
    print(f"  P2 (σ > {sigma_wikitext_perseq}):              "
          f"σ={sigma_dt:.4f}  → {'OK' if P2_ok else 'FAIL'}", flush=True)
    print(f"  P3 (σ_attn0 > {sigma_attn0_wikitext}):         "
          f"σ={sigma_attn0:.4f}  → {'OK' if P3_ok else 'FAIL'}", flush=True)
    print(f"  P4 (σ_total > σ_block0):           "
          f"{sigma_dt:.4f} vs {sigma_b0:.4f}  → {'OK' if P4_ok else 'FAIL'}", flush=True)
    print(f"  K1 (σ<0.20 AND C[8]<0.85):         {'FIRED' if K1_fired else 'ok'}", flush=True)

    if K1_fired:
        overall = "inconclusive"
    elif P1_ok and P2_ok:
        overall = "confirmed"
    elif P2_ok:
        overall = "partial"
    else:
        overall = "falsified"

    print(f"\n  Overall verdict: {overall.upper()}", flush=True)

    print(f"\n=== Comparison with exp-117 ===", flush=True)
    e117 = {"sigma": 0.249, "dx8": 0.950, "dx32": 0.931, "dx128": 0.656, "dx256": 0.359}
    e130 = {"sigma": 0.129, "dx8": 0.683}
    e128 = {"sigma": 0.189}
    print(f"  exp-117 (random, unknown protocol):  σ=0.249  C[8]=0.950  C[256]=0.359", flush=True)
    print(f"  exp-128 (WikiText, mean-first):       σ=0.189", flush=True)
    print(f"  exp-130 (WikiText, per-seq):          σ=0.129  C[8]=0.683", flush=True)
    print(f"  exp-131 (random, per-seq):            "
          f"σ={sigma_dt:.3f}  "
          f"C[8]={c8_dt:.3f}  "
          f"C[32]={perseq_results['delta_total']['profile_dx32']:.3f}  "
          f"C[256]={perseq_results['delta_total']['profile_dx256']:.3f}", flush=True)

    results = {
        "exp": "exp-131",
        "date": "2026-08-31",
        "prereg_commit": PREREG_COMMIT,
        "model": "gpt2",
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "data": "random-token census (uniform [0, 50257), seed=42)",
        "perseq_cosine": perseq_results,
        "meanfirst_cosine": meanfirst_results,
        "exp117_comparison": {
            "sigma": 0.249, "protocol": "unknown (weight-only analysis, likely random-token)",
            "profile_dx8": 0.950, "profile_dx32": 0.931,
            "profile_dx128": 0.656, "profile_dx256": 0.359,
        },
        "wikitext_comparison": {
            "exp128_sigma_meanfirst": 0.189,
            "exp130_sigma_perseq": 0.129, "exp130_c8": 0.683,
        },
        "registered_verdicts": {
            "P1": {"ok": P1_ok, "sigma": float(sigma_dt), "c8": float(c8_dt),
                   "criterion": "sigma in [0.22,0.28] and C[8] >= 0.90"},
            "P2": {"ok": P2_ok, "sigma": float(sigma_dt),
                   "criterion": f"sigma > {sigma_wikitext_perseq} (exp-130)"},
            "P3": {"ok": P3_ok, "sigma_attn0": float(sigma_attn0),
                   "criterion": f"sigma_attn0 > {sigma_attn0_wikitext} (exp-130)"},
            "P4": {"ok": P4_ok, "sigma_total": float(sigma_dt),
                   "sigma_block0": float(sigma_b0),
                   "criterion": "sigma_total > sigma_block0"},
        },
        "kill_conditions": {
            "K1": {"fired": K1_fired, "criterion": "sigma < 0.20 AND C[8] < 0.85"},
        },
        "overall_verdict": overall,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
