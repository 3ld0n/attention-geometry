"""exp-128 — Theory-of-A Level 3: Multi-Layer Residual Stream Decomposition.

Pre-registration committed to attention-geometry at ce1934c before this script
was written or any forward passes were run.

Tests whether the position-correlated delta at L2H1's input (σ_delta ≈ 0.249,
exp-117) arises from constructive accumulation of per-layer attention and MLP
writes across blocks 0 and 1.

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
WINDOW = exp112.WINDOW          # lags 8..256
NW = exp112.NW                  # 249

SEQ_LEN = 512
DEEP_LO = 256
FIT_LO, FIT_HI = 8, 256
N_SEQS = 100                    # sequences for positional mean estimate
SEED = 42

PREREG_COMMIT = "ce1934c"


def r2_of_slope(profile: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, profile, rcond=None)
    y_pred = X @ c
    ss_tot = float(((profile - profile.mean()) ** 2).sum())
    ss_res = float(((profile - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def sigma_of(mean_residual: np.ndarray) -> tuple[float, float]:
    """Compute position-correlation slope and R² for a (512, 768) positional mean.

    Protocol (pre-registered):
      1. Per-row normalize: X_n[i] = X[i] / ||X[i]||
      2. Cosine similarity matrix: C = X_n @ X_n^T   (512 x 512)
      3. pooled_window_profile(C) -> 249-element array
      4. sigma = -ols_slope(profile, WINDOW), R2 from log-log OLS fit
    """
    norms = np.linalg.norm(mean_residual, axis=-1, keepdims=True)  # (512, 1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_n = mean_residual / norms                                      # (512, 768)
    C = X_n @ X_n.T                                                  # (512, 512)
    profile = pooled_window_profile(C)                               # (249,)
    sigma = -ols_slope(profile, WINDOW)
    r2 = r2_of_slope(profile, WINDOW)
    return float(sigma), float(r2)


def load_wikitext_sequences(tokenizer, n_seqs: int, seq_len: int) -> torch.Tensor:
    """Load WikiText-103 validation sequences (same protocol as exp-118)."""
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

    out = torch.stack(seqs[:n_seqs])           # (n_seqs, seq_len)
    print(f"Loaded {len(seqs)} sequences of length {seq_len}.", flush=True)
    return out


def main() -> None:
    print("exp-128: Multi-Layer Residual Stream Decomposition", flush=True)
    print(f"Pre-registration commit: {PREREG_COMMIT} (pushed before this script)\n",
          flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32,
        attn_implementation="eager").to(device).eval()
    cfg = AutoConfig.from_pretrained("gpt2")
    print(f"Model: gpt2  n_layer={cfg.num_hidden_layers}  "
          f"n_head={cfg.num_attention_heads}  d_model={cfg.hidden_size}", flush=True)

    seqs = load_wikitext_sequences(tokenizer, N_SEQS, SEQ_LEN).to(device)
    actual_n = len(seqs)
    print(f"Running {actual_n} sequences.", flush=True)

    # Accumulators: mean over sequences, kept as numpy float64 for precision
    acc = {
        "h0":     np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64),
        "attn0":  np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64),
        "mlp0":   np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64),
        "attn1":  np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64),
        "mlp1":   np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64),
        "h2":     np.zeros((SEQ_LEN, cfg.hidden_size), dtype=np.float64),
    }
    handles = []
    _buf: dict[str, np.ndarray] = {}

    # Hook: capture block 0 input (= embedding output)
    def hook_h0(module, args):
        x = args[0]
        _buf["h0"] = x.detach().float().cpu().numpy()

    # Hook: capture block ℓ attention output (before residual add)
    def make_attn_hook(key: str):
        def hook(module, args, output):
            # GPT-2 attention returns (attn_output, ...) where attn_output includes c_proj
            out = output[0] if isinstance(output, tuple) else output
            _buf[key] = out.detach().float().cpu().numpy()
        return hook

    # Hook: capture block ℓ MLP output (before residual add)
    def make_mlp_hook(key: str):
        def hook(module, args, output):
            out = output if not isinstance(output, tuple) else output[0]
            _buf[key] = out.detach().float().cpu().numpy()
        return hook

    # Hook: capture block 2 input (= output of block 1)
    def hook_h2(module, args):
        x = args[0]
        _buf["h2"] = x.detach().float().cpu().numpy()

    handles.append(model.transformer.h[0].register_forward_pre_hook(hook_h0))
    handles.append(model.transformer.h[0].attn.register_forward_hook(
        make_attn_hook("attn0")))
    handles.append(model.transformer.h[0].mlp.register_forward_hook(
        make_mlp_hook("mlp0")))
    handles.append(model.transformer.h[1].attn.register_forward_hook(
        make_attn_hook("attn1")))
    handles.append(model.transformer.h[1].mlp.register_forward_hook(
        make_mlp_hook("mlp1")))
    handles.append(model.transformer.h[2].register_forward_pre_hook(hook_h2))

    # Forward passes — accumulate positional means
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            if (i + 1) % 10 == 0:
                print(f"  Sequence {i+1}/{actual_n}...", flush=True)
            _buf.clear()
            model(seq.unsqueeze(0))  # (1, 512)

            for key in acc:
                arr = _buf.get(key)
                if arr is None:
                    raise RuntimeError(f"Hook for '{key}' did not fire on seq {i}")
                # arr shape: (1, 512, 768) or (512, 768)
                if arr.ndim == 3:
                    arr = arr[0]    # (512, 768)
                acc[key] += arr.astype(np.float64) / actual_n

    for h in handles:
        h.remove()

    print("\nForward passes complete. Computing statistics...", flush=True)

    # Verify decomposition identity: h2 ≈ h0 + attn0 + mlp0 + attn1 + mlp1
    reconstructed = (acc["h0"] + acc["attn0"] + acc["mlp0"]
                     + acc["attn1"] + acc["mlp1"])
    residual_err = np.linalg.norm(reconstructed - acc["h2"]) / np.linalg.norm(acc["h2"])
    print(f"Decomposition verification (||reconstructed - h2|| / ||h2||): "
          f"{residual_err:.2e}", flush=True)

    # Compute position-correlation slopes for each component
    components = {
        "h0":            acc["h0"],
        "attn0":         acc["attn0"],
        "mlp0":          acc["mlp0"],
        "attn1":         acc["attn1"],
        "mlp1":          acc["mlp1"],
        "h2":            acc["h2"],
        # Cumulative deltas
        "delta_after_attn0":   acc["attn0"],                                      # just attn^(0)
        "delta_after_block0":  acc["attn0"] + acc["mlp0"],                        # Δ₀
        "delta_after_attn1":   acc["attn0"] + acc["mlp0"] + acc["attn1"],         # Δ₁
        "delta_total":         acc["attn0"] + acc["mlp0"] + acc["attn1"] + acc["mlp1"],  # Δ_total
    }

    results_sigma = {}
    for name, tensor in components.items():
        sigma, r2 = sigma_of(tensor)
        results_sigma[name] = {"sigma": sigma, "r2": r2}
        print(f"  {name:25s}  sigma={sigma:+.4f}  R²={r2:.4f}", flush=True)

    # Registered predictions evaluation
    sigma_delta_total = results_sigma["delta_total"]["sigma"]
    r2_delta_total = results_sigma["delta_total"]["r2"]
    sigma_attn0 = results_sigma["attn0"]["sigma"]
    sigma_attn1 = results_sigma["attn1"]["sigma"]
    sigma_delta0 = results_sigma["delta_after_block0"]["sigma"]

    P1_ok = (0.20 <= sigma_delta_total <= 0.30) and (r2_delta_total >= 0.70)
    P2_ok = sigma_attn0 > 0.05
    P3_ok = sigma_attn1 > 0.05
    P4_ok = sigma_delta0 > 0.05

    K1_fired = sigma_delta_total < 0.15
    K2_fired = sigma_delta_total > 0.35
    K3_fired = (sigma_attn0 <= 0.0) and (sigma_attn1 <= 0.0)

    print(f"\n=== Registered verdicts ===", flush=True)
    print(f"  P1 (σ_total ∈ [0.20, 0.30], R²≥0.70):  "
          f"σ={sigma_delta_total:.4f}, R²={r2_delta_total:.4f}  "
          f"→ {'OK' if P1_ok else 'FAIL'}", flush=True)
    print(f"  P2 (σ_attn0 > 0.05):                   "
          f"σ={sigma_attn0:.4f}  → {'OK' if P2_ok else 'FAIL'}", flush=True)
    print(f"  P3 (σ_attn1 > 0.05):                   "
          f"σ={sigma_attn1:.4f}  → {'OK' if P3_ok else 'FAIL'}", flush=True)
    print(f"  P4 (σ_Δ₀ > 0.05):                      "
          f"σ={sigma_delta0:.4f}  → {'OK' if P4_ok else 'FAIL'}", flush=True)
    print(f"  K1 (σ_total < 0.15): {'FIRED' if K1_fired else 'ok'}", flush=True)
    print(f"  K2 (σ_total > 0.35): {'FIRED' if K2_fired else 'ok'}", flush=True)
    print(f"  K3 (both attn slopes ≤ 0): {'FIRED' if K3_fired else 'ok'}", flush=True)

    if K1_fired or K2_fired:
        overall = "inconclusive"
    elif P1_ok and (P2_ok or P3_ok):
        overall = "confirmed"
    elif P1_ok:
        overall = "partial"
    else:
        overall = "falsified"

    print(f"\n  Overall verdict: {overall.upper()}", flush=True)

    # Save results
    results = {
        "exp": "exp-128",
        "date": "2026-08-31",
        "prereg_commit": PREREG_COMMIT,
        "model": "gpt2",
        "n_seqs": actual_n,
        "seq_len": SEQ_LEN,
        "data": "wikitext-103-raw-v1 validation, 100 non-overlapping 512-token windows",
        "decomposition_verification": {
            "relative_error": float(residual_err),
            "passes": bool(residual_err < 1e-4),
        },
        "sigma_r2": results_sigma,
        "registered_verdicts": {
            "P1": {"ok": P1_ok, "sigma": float(sigma_delta_total),
                   "r2": float(r2_delta_total),
                   "criterion": "sigma in [0.20, 0.30] and R2 >= 0.70"},
            "P2": {"ok": P2_ok, "sigma": float(sigma_attn0),
                   "criterion": "sigma(attn0) > 0.05"},
            "P3": {"ok": P3_ok, "sigma": float(sigma_attn1),
                   "criterion": "sigma(attn1) > 0.05"},
            "P4": {"ok": P4_ok, "sigma": float(sigma_delta0),
                   "criterion": "sigma(delta after block 0) > 0.05"},
        },
        "kill_conditions": {
            "K1": {"fired": K1_fired, "criterion": "sigma_total < 0.15"},
            "K2": {"fired": K2_fired, "criterion": "sigma_total > 0.35"},
            "K3": {"fired": K3_fired, "criterion": "sigma_attn0 <= 0 AND sigma_attn1 <= 0"},
        },
        "overall_verdict": overall,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)

    # Save positional mean arrays for analysis-only follow-up (exp-129)
    np.savez_compressed(
        HERE / "positional_means.npz",
        h0=acc["h0"].astype(np.float32),
        attn0=acc["attn0"].astype(np.float32),
        mlp0=acc["mlp0"].astype(np.float32),
        attn1=acc["attn1"].astype(np.float32),
        mlp1=acc["mlp1"].astype(np.float32),
        h2=acc["h2"].astype(np.float32),
    )
    print(f"Saved positional_means.npz (for exp-129 analysis)", flush=True)


if __name__ == "__main__":
    main()
