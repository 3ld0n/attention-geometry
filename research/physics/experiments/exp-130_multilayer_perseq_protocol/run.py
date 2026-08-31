"""exp-130 — Theory-of-A Level 3: Per-Sequence σ_delta Protocol.

Pre-registration committed to attention-geometry at 7a672d9 before this script
was written or any forward passes were run.

Tests whether computing σ_delta per-sequence (then averaging the profile)
reproduces exp-117's σ_delta = 0.249, fixing the protocol difference identified
in exp-129.

Ariel — 2026-08-31.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
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
N_SEQS = 100
PREREG_COMMIT = "7a672d9"


def r2_of_slope(profile: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, profile, rcond=None)
    y_pred = X @ c
    ss_tot = float(((profile - profile.mean()) ** 2).sum())
    ss_res = float(((profile - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def perseq_cosine_profile(arr_seq: np.ndarray) -> np.ndarray:
    """Compute pooled cosine-similarity profile for one sequence.
    arr_seq: (512, 768) residual stream component for ONE sequence.
    Returns: (249,) profile of pooled cosine similarities.
    """
    norms = np.linalg.norm(arr_seq, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = arr_seq / norms                              # (512, 768)
    C = M @ M.T                                      # (512, 512)
    return pooled_window_profile(C)                  # (249,)


def load_wikitext_sequences(tokenizer, n_seqs: int, seq_len: int) -> torch.Tensor:
    print("Loading WikiText-103 validation...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n".join(line for line in ds["text"] if line.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor(ids, dtype=torch.long)
    seqs = []
    for i in range(n_seqs):
        start = i * seq_len
        end = start + seq_len
        if end > len(ids):
            break
        seqs.append(ids[start:end])
    out = torch.stack(seqs[:n_seqs])
    print(f"Loaded {len(seqs)} sequences.", flush=True)
    return out


def main() -> None:
    print("exp-130: Per-Sequence σ_delta Protocol", flush=True)
    print(f"Pre-registration commit: {PREREG_COMMIT}\n", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32,
        attn_implementation="eager").to(device).eval()

    seqs = load_wikitext_sequences(tokenizer, N_SEQS, SEQ_LEN).to(device)
    actual_n = len(seqs)

    # Accumulators for per-sequence profiles (249-element arrays)
    # Accumulated as mean over sequences
    profile_acc = {
        "delta_total":       np.zeros(len(WINDOW), dtype=np.float64),
        "delta_after_attn0": np.zeros(len(WINDOW), dtype=np.float64),
        "delta_after_block0": np.zeros(len(WINDOW), dtype=np.float64),
        "delta_after_attn1": np.zeros(len(WINDOW), dtype=np.float64),
        "attn0":             np.zeros(len(WINDOW), dtype=np.float64),
        "mlp0":              np.zeros(len(WINDOW), dtype=np.float64),
        "attn1":             np.zeros(len(WINDOW), dtype=np.float64),
        "mlp1":              np.zeros(len(WINDOW), dtype=np.float64),
    }
    # Also track first-dx and last-dx values for exp-117 comparison
    # dx=8 is WINDOW[0], dx=256 is WINDOW[-1]
    handles = []
    _buf: dict[str, np.ndarray] = {}

    def hook_h0(module, args):
        _buf["h0"] = args[0].detach().float().cpu().numpy()[0]  # (512, 768)

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
                print(f"  Sequence {i+1}/{actual_n}...", flush=True)
            _buf.clear()
            model(seq.unsqueeze(0))

            # Per-sequence components
            h0_s    = _buf["h0"].astype(np.float64)    # (512, 768)
            attn0_s = _buf["attn0"].astype(np.float64)
            mlp0_s  = _buf["mlp0"].astype(np.float64)
            attn1_s = _buf["attn1"].astype(np.float64)
            mlp1_s  = _buf["mlp1"].astype(np.float64)

            # Cumulative deltas per sequence
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
                profile_acc[key] += perseq_cosine_profile(arr) / actual_n

    for h in handles:
        h.remove()

    print("\nComputing statistics...", flush=True)

    results_sigma = {}
    for key, profile in profile_acc.items():
        sigma = -ols_slope(profile, WINDOW)
        r2 = r2_of_slope(profile, WINDOW)
        # Report values at key lags (dx=8→idx 0, dx=32→idx 24, dx=128→idx 120, dx=256→idx 248)
        v8   = float(profile[0])
        v32  = float(profile[24])
        v128 = float(profile[120])
        v256 = float(profile[-1])
        results_sigma[key] = {
            "sigma": float(sigma), "r2": float(r2),
            "profile_dx8": v8, "profile_dx32": v32,
            "profile_dx128": v128, "profile_dx256": v256,
        }
        print(f"  {key:25s}  sigma={sigma:+.4f}  R²={r2:.4f}  "
              f"C[8]={v8:.3f}  C[256]={v256:.3f}", flush=True)

    sigma_dt = results_sigma["delta_total"]["sigma"]
    r2_dt = results_sigma["delta_total"]["r2"]
    c8_dt = results_sigma["delta_total"]["profile_dx8"]
    sigma_b0 = results_sigma["delta_after_block0"]["sigma"]

    P1_ok = (0.22 <= sigma_dt <= 0.28) and (c8_dt >= 0.90)
    P2_ok = sigma_dt > 0.189    # > exp-128 mean-first result
    P3_ok = sigma_b0 > 0.226    # > exp-128 mean-first block-0 result

    K1_fired = (sigma_dt < 0.20) and (c8_dt < 0.85)

    print(f"\n=== Registered verdicts ===", flush=True)
    print(f"  P1 (σ∈[0.22,0.28] & C[8]≥0.90):  "
          f"σ={sigma_dt:.4f}, C[8]={c8_dt:.3f}  → {'OK' if P1_ok else 'FAIL'}", flush=True)
    print(f"  P2 (σ > 0.189):                   σ={sigma_dt:.4f}  → {'OK' if P2_ok else 'FAIL'}", flush=True)
    print(f"  P3 (σ_block0 > 0.226):            σ={sigma_b0:.4f}  → {'OK' if P3_ok else 'FAIL'}", flush=True)
    print(f"  K1 (σ<0.20 AND C[8]<0.85):        {'FIRED' if K1_fired else 'ok'}", flush=True)

    if K1_fired:
        overall = "inconclusive"
    elif P1_ok and P2_ok:
        overall = "confirmed"
    elif P2_ok:
        overall = "partial"
    else:
        overall = "falsified"

    print(f"\n  Overall verdict: {overall.upper()}", flush=True)

    # Exp-117 comparison
    print(f"\n=== Comparison with exp-117 ===", flush=True)
    print(f"  exp-117: σ_delta=0.249  C[8]=0.950  C[32]=0.931  C[128]=0.656  C[256]=0.359", flush=True)
    print(f"  exp-130: σ_delta={sigma_dt:.3f}  "
          f"C[8]={c8_dt:.3f}  "
          f"C[32]={results_sigma['delta_total']['profile_dx32']:.3f}  "
          f"C[128]={results_sigma['delta_total']['profile_dx128']:.3f}  "
          f"C[256]={results_sigma['delta_total']['profile_dx256']:.3f}", flush=True)

    results = {
        "exp": "exp-130",
        "date": "2026-08-31",
        "prereg_commit": PREREG_COMMIT,
        "model": "gpt2",
        "n_seqs": actual_n,
        "seq_len": SEQ_LEN,
        "data": "wikitext-103-raw-v1 validation, 100 sequences",
        "protocol": "per-sequence cosine similarity, then averaged profile",
        "sigma_r2": results_sigma,
        "exp117_comparison": {
            "sigma": 0.249,
            "profile_dx8": 0.950, "profile_dx32": 0.931,
            "profile_dx128": 0.656, "profile_dx256": 0.359,
        },
        "exp128_comparison": {
            "sigma_delta_total_mean_first": 0.1892,
        },
        "registered_verdicts": {
            "P1": {"ok": P1_ok, "sigma": float(sigma_dt), "c8": float(c8_dt),
                   "criterion": "sigma in [0.22,0.28] and C[8] >= 0.90"},
            "P2": {"ok": P2_ok, "sigma": float(sigma_dt),
                   "criterion": "sigma > 0.189 (exp-128 mean-first)"},
            "P3": {"ok": P3_ok, "sigma_block0": float(sigma_b0),
                   "criterion": "sigma_block0 > 0.226"},
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
