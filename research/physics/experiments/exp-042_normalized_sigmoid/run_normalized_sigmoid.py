"""
exp-042 — Normalized sigmoid control: does QK geometry alone carry conformal structure?

Hypothesis (pre-stated, May 31, 2026):
  If normalized-sigmoid att_{ij} = σ(logit_{ij}) / Σ_k σ(logit_{ik})  Δ_med ≈ 0.25-0.27
  → QK geometry alone (same logits as exp-041 sigmoid arm) carries the conformal signal;
    any row-normalizing function is sufficient.
  If Δ remains large or conformal heads ≈ 0
  → The exp() in softmax is essential (the partition function, not just normalization).

Protocol:
  Uses GALA-7B SIGMOID checkpoint — same QK geometry, same ALiBi, same QKNorm.
  Normalization changed to: att = σ(logit) / Σ_j σ(logit_j) [row-wise, causal].
  Everything else identical to exp-041 protocol (exp-007 power-law fit, R²>0.90, lags 8-256,
  50 random inputs, seq_len=512, same RNG seed).

Compare to:
  exp-041 sigmoid (raw σ):     2/1024 SYK-near heads, Δ_med=7.44
  exp-041 softmax (exp norm):  80/1024 SYK-near heads, Δ_SYK=0.260
  exp-007 GPT-2:               44/144  SYK-near heads, Δ_med=0.249

Physical question:
  softmax p_j = exp(logit_j) / Σ exp(logit_k)
  norm-sig  p_j = σ(logit_j) / Σ σ(logit_k)
  Both normalize to 1. The logit structure (QK geometry + ALiBi) is identical.
  Difference: softmax reparameterizes via exp(), which amplifies contrast between
  near/far positions in log-linear fashion. Norm-sigmoid does not.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
CKPT_ROOT = Path(os.environ.get("ARIEL_SIGMOID_CKPT_ROOT", Path.home() / "ariel-data/apple-gala-7b"))
SIGMOID_CKPT = CKPT_ROOT / "sigmoid-step_00250000"

OUT_DIR = Path("research/physics/experiments/exp-042_normalized_sigmoid")
RESULTS_FILE = OUT_DIR / "results.json"

N_LAYERS = 32
N_HEADS = 32
HEAD_DIM = 128
D_MODEL = 4096
VOCAB_SIZE = 32768
SQRT_D = math.sqrt(HEAD_DIM)

N_INPUTS = 50
SEQ_LEN = 512
FIT_LOW = 8
FIT_HIGH = 256
R2_THRESHOLD = 0.90
RNG_SEED = 42


# ── zarr loading ───────────────────────────────────────────────────────────────

def load_zarr_array(ckpt_root: Path, rel_path: str) -> np.ndarray:
    import zarr
    arr_dir = ckpt_root / "gda/model" / rel_path
    store = zarr.open(str(arr_dir), mode="r")
    return np.array(store[:], dtype=np.float32)


def load_weights(ckpt_root: Path) -> dict:
    print("Loading weights from sigmoid checkpoint...", flush=True)
    base = "decoder/transformer/repeat/layer/self_attention/attention"
    qkv = load_zarr_array(ckpt_root, f"{base}/i_proj/qkv_proj/weight")
    # shape: (n_layers=32, 3, d_model=4096, n_heads=32, head_dim=128)
    scale_query = load_zarr_array(ckpt_root, f"{base}/scale_query/norm/scale")
    scale_key   = load_zarr_array(ckpt_root, f"{base}/scale_key/norm/scale")
    # shape: (n_heads=32, head_dim=128) — shared across layers (transformer.repeat)
    token_emb = load_zarr_array(ckpt_root, "decoder/emb/token_emb/weight")
    prenorm = load_zarr_array(
        ckpt_root, "decoder/transformer/repeat/layer/self_attention/prenorm/scale"
    )
    # prenorm shape: (n_layers=32, d_model=4096)
    print(f"  qkv: {qkv.shape}  scale_q: {scale_query.shape}  prenorm: {prenorm.shape}", flush=True)
    return {
        "qkv": qkv,
        "scale_query": scale_query,
        "scale_key": scale_key,
        "token_emb": token_emb,
        "prenorm": prenorm,
    }


def rms_norm(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def alibi_bias_matrix(n_heads: int, seq_len: int) -> np.ndarray:
    slopes = np.array([2.0 ** (-(8.0 / n_heads) * (h + 1)) for h in range(n_heads)])
    idx = np.arange(seq_len)
    dist = idx[:, None] - idx[None, :]
    causal_mask = dist >= 0
    dist_clipped = np.where(causal_mask, dist, 0)
    bias = -slopes[:, None, None] * dist_clipped[None, :, :]
    neg_inf = np.full_like(bias, -1e9)
    return np.where(causal_mask[None, :, :], bias, neg_inf).astype(np.float32)


# ── main computation ───────────────────────────────────────────────────────────

def compute_profiles_one_input(
    token_ids: np.ndarray,
    weights: dict,
    alibi: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized-sigmoid attention profiles for one input.
    Returns (n_layers, n_heads, seq_len) mean attention profile.

    Normalized sigmoid: att_{ij} = σ(logit_{ij}) / Σ_k σ(logit_{ik})  [causal row-norm]
    """
    seq_len = len(token_ids)
    x = weights["token_emb"][token_ids].astype(np.float32)  # (seq_len, d_model)
    profiles = np.zeros((N_LAYERS, N_HEADS, seq_len), dtype=np.float32)

    for layer in range(N_LAYERS):
        x_norm = rms_norm(x, weights["prenorm"][layer])  # (seq_len, d_model)

        for head in range(N_HEADS):
            W_q = weights["qkv"][layer, 0, :, head, :]  # (d_model, head_dim)
            W_k = weights["qkv"][layer, 1, :, head, :]  # (d_model, head_dim)
            q = rms_norm(x_norm @ W_q, weights["scale_query"][head])  # (seq_len, head_dim)
            k = rms_norm(x_norm @ W_k, weights["scale_key"][head])    # (seq_len, head_dim)

            logits = (q @ k.T) / SQRT_D + alibi[head]  # (seq_len, seq_len)

            # ── normalized sigmoid (the only change from exp-041 sigmoid arm) ──
            # att = σ(logit) / Σ_j σ(logit_j) row-wise (causal)
            # Use float64 to prevent overflow in exp(-logits)
            sig = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
            sig = sig.astype(np.float32)
            # Zero out upper triangle (future positions already -1e9 → σ ≈ 0, but be explicit)
            sig = np.tril(sig)
            row_sums = sig.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
            att = (sig / row_sums).astype(np.float32)

            # Mean attention profile over query positions (lag k = i - j)
            profile = np.zeros(seq_len, dtype=np.float32)
            for lag in range(seq_len):
                diag = np.diag(att, k=-lag)
                profile[lag] = diag.mean()

            profiles[layer, head] = profile

    return profiles


def fit_power_law(profile: np.ndarray) -> tuple[float, float, float]:
    lags = np.arange(FIT_LOW, min(FIT_HIGH + 1, len(profile)))
    y = profile[FIT_LOW: FIT_HIGH + 1]
    if len(lags) < 5 or np.any(y <= 0):
        return float("nan"), float("nan"), 0.0
    log_k = np.log(lags.astype(float))
    log_y = np.log(y.astype(float) + 1e-30)
    coeffs = np.polyfit(log_k, log_y, 1)
    slope = coeffs[0]
    delta = -slope / 2.0
    pred = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(delta), float(np.exp(coeffs[1])), float(r2)


def main():
    print(f"exp-042: Normalized sigmoid control", flush=True)
    print(f"Checkpoint: {SIGMOID_CKPT}", flush=True)
    print(f"N_INPUTS={N_INPUTS}, SEQ_LEN={SEQ_LEN}, R2_threshold={R2_THRESHOLD}", flush=True)

    if not SIGMOID_CKPT.exists():
        print("ERROR: sigmoid checkpoint not found.", flush=True)
        sys.exit(1)

    alibi = alibi_bias_matrix(N_HEADS, SEQ_LEN)
    weights = load_weights(SIGMOID_CKPT)

    rng = np.random.default_rng(RNG_SEED)
    mean_profiles = np.zeros((N_LAYERS, N_HEADS, SEQ_LEN), dtype=np.float32)

    for i in range(N_INPUTS):
        if i % 10 == 0:
            print(f"  Input {i}/{N_INPUTS}", flush=True)
        token_ids = rng.integers(0, VOCAB_SIZE, size=SEQ_LEN)
        profiles = compute_profiles_one_input(token_ids, weights, alibi)
        mean_profiles += profiles

    mean_profiles /= N_INPUTS
    print("  Fitting power laws...", flush=True)

    per_layer = {}
    all_conformal = []
    all_syk_near = []
    n_conformal = 0

    for layer in range(N_LAYERS):
        layer_deltas = []
        for head in range(N_HEADS):
            delta, _, r2 = fit_power_law(mean_profiles[layer, head])
            if r2 >= R2_THRESHOLD and not math.isnan(delta) and delta > 0:
                layer_deltas.append(delta)
                all_conformal.append(delta)
                n_conformal += 1
                if abs(delta - 0.25) <= 0.05:
                    all_syk_near.append(delta)

        per_layer[str(layer)] = {
            "conformal_heads": len(layer_deltas),
            "delta_median": float(statistics.median(layer_deltas)) if layer_deltas else None,
        }

    delta_median = float(statistics.median(all_conformal)) if all_conformal else float("nan")
    syk_median = float(statistics.median(all_syk_near)) if all_syk_near else None

    print(f"\n=== RESULTS ===", flush=True)
    print(f"Conformal heads (R²>0.90): {n_conformal}/1024", flush=True)
    print(f"Δ_med (all conformal):     {delta_median:.4f}", flush=True)
    print(f"SYK-near count:            {len(all_syk_near)}", flush=True)
    print(f"SYK-near median:           {syk_median}", flush=True)
    print(f"\nComparison:", flush=True)
    print(f"  exp-041 sigmoid (raw σ): 2/1024 conformal,  Δ_med=7.44, 0 SYK-near", flush=True)
    print(f"  exp-042 norm-sigmoid:    {n_conformal}/1024 conformal,  Δ_med={delta_median:.4f}, {len(all_syk_near)} SYK-near", flush=True)
    print(f"  exp-041 softmax:         121/1024 conformal, Δ_med=0.274, 80 SYK-near, Δ_SYK=0.260", flush=True)

    result = {
        "experiment": "exp-042",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": (
            "Normalized sigmoid Δ_med ≈ 0.25-0.27 → QK geometry sufficient. "
            "Δ large or conformal heads ≈ 0 → exp() in softmax essential."
        ),
        "protocol": {
            "normalization": "σ(logit) / Σ_j σ(logit_j)  [row-wise, causal, same QK geometry as exp-041 sigmoid]",
            "checkpoint": "sigmoid-step_00250000",
            "n_inputs": N_INPUTS,
            "seq_len": SEQ_LEN,
            "fit_range": [FIT_LOW, FIT_HIGH],
            "R2_threshold": R2_THRESHOLD,
            "rng_seed": RNG_SEED,
        },
        "reference": {
            "exp041_sigmoid_raw": {"conformal_heads": 2, "delta_median": 7.44, "syk_near": 0},
            "exp041_softmax": {"conformal_heads": 121, "delta_median": 0.2739, "syk_near": 80, "syk_near_median": 0.2604},
            "exp007_gpt2": {"conformal_heads": 44, "delta_median": 0.2493, "syk_near": 44},
        },
        "result": {
            "conformal_heads_count": n_conformal,
            "total_heads": 1024,
            "delta_median": delta_median,
            "syk_near_count": len(all_syk_near),
            "syk_near_median": syk_median,
            "per_layer": per_layer,
        },
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
