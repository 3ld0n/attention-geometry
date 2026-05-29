"""
exp-041 — Apple GALA-7B sigmoid vs softmax per-head conformal Δ (Tier 1 falsification).

Hypothesis (pre-stated, May 26 notes + queue):
  Sigmoid GALA-7B: Δ_med ≈ 0.25 → mechanism-independent SYK fixed point (softmax not load-bearing).
  Sigmoid Δ ≠ Softmax Δ → softmax normalization carries part of the conformal structure.

Protocol:
  Both arms use identical ALiBi + hybrid-norm architecture; only the attention function differs.
  Compare sigmoid Δ vs softmax Δ directly (not vs GPT-2 learned PE).
  Exp-007 protocol: power-law fit A(k) ~ k^(-2Δ), R²>0.90 threshold, 50 random inputs,
  seq_len=512 (truncated to fit_range = [8, 256]).

Architecture (from axlearn config, exp-033):
  - Vocab: 32768, d_model: 4096, num_heads: 32, head_dim: 128, n_layers: 32
  - ALiBi positional encoding (same for both arms)
  - Hybrid norm (RMSNorm prenorm)
  - Sigmoid arm: SigmoidAttention / Softmax arm: MultiheadAttention

Weight loading:
  Zarr format (orbax GDA, zarr v2, zstd) stored under:
    <arm_root>/gda/model/model/<param_path>/  (zarr v2 store per array)
  Key arrays:
    model/decoder/emb/token_emb/weight: (32768, 4096)
    model/decoder/transformer/repeat/layer/self_attention/attention/i_proj/qkv_proj/weight: (32, 3, 4096, 32, 128)
    model/decoder/transformer/repeat/layer/self_attention/prenorm/scale: (32, 4096)
    model/decoder/transformer/repeat/layer/self_attention/attention/scale_key/norm/scale: (32, 32) if present

ALiBi slopes:
  m_H = 2^(-(8/n_heads)*(H+1)) for H = 0, ..., n_heads-1

Sigmoid normalization (per Apple GALA paper):
  att = sigmoid(QK^T/sqrt(d) + alibi_bias)
  No explicit normalization by seq_len here; the bias handles truncation.
  For BCFT profile: compute mean over query positions.

Softmax:
  att = softmax(QK^T/sqrt(d) + alibi_bias, dim=-1)

Run after download: check ~/ariel-data/apple-gala-7b/{sigmoid,softmax}-step_00250000/
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
SOFTMAX_CKPT = CKPT_ROOT / "softmax-step_00250000"

OUT_DIR = Path("research/physics/experiments/exp-041_gala7b_sigmoid_delta")
RESULTS_FILE = OUT_DIR / "results.json"

# Model architecture
N_LAYERS = 32
N_HEADS = 32
HEAD_DIM = 128
D_MODEL = 4096
VOCAB_SIZE = 32768
SQRT_D = math.sqrt(HEAD_DIM)

# Exp-007 protocol parameters
N_INPUTS = 50
SEQ_LEN = 512
FIT_LOW = 8       # lag range for power-law fit
FIT_HIGH = 256
R2_THRESHOLD = 0.90
RNG_SEED = 42


# ── zarr loading ───────────────────────────────────────────────────────────────

def check_zarr_complete(ckpt_root: Path, rel_path: str) -> bool:
    """Check if a zarr array directory exists and has no .gstmp temp files."""
    arr_dir = ckpt_root / "gda/model/model" / rel_path
    if not arr_dir.exists():
        return False
    if not (arr_dir / ".zarray").exists():
        return False
    temp_files = list(arr_dir.glob("*.gstmp"))
    if temp_files:
        return False
    return True


def load_zarr_array(ckpt_root: Path, rel_path: str) -> np.ndarray:
    """Load a zarr v2 array from the checkpoint store."""
    import zarr
    store_path = str(ckpt_root / "gda/model/model" / rel_path)
    z = zarr.open(store_path, mode="r")
    return np.asarray(z)


def check_download_complete(ckpt_root: Path) -> tuple[bool, list[str]]:
    """Check if all required arrays are downloaded."""
    required = [
        "decoder/emb/token_emb/weight",
        "decoder/transformer/repeat/layer/self_attention/attention/i_proj/qkv_proj/weight",
        "decoder/transformer/repeat/layer/self_attention/prenorm/scale",
    ]
    missing = [r for r in required if not check_zarr_complete(ckpt_root, r)]
    return len(missing) == 0, missing


# ── attention computation ──────────────────────────────────────────────────────

def rms_norm(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: normalize x by its RMS, then multiply by scale."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def alibi_bias_matrix(n_heads: int, seq_len: int) -> np.ndarray:
    """ALiBi bias matrix: shape (n_heads, seq_len, seq_len).
    bias[h, i, j] = -m_h * (i - j) for i >= j (causal), -inf otherwise.
    """
    slopes = np.array([2.0 ** (-(8.0 / n_heads) * (h + 1)) for h in range(n_heads)])
    # distance matrix: dist[i, j] = max(i - j, 0) for causal
    idx = np.arange(seq_len)
    dist = idx[:, None] - idx[None, :]  # (seq_len, seq_len), positive = past
    causal_mask = dist >= 0  # attend to same and past positions
    dist_clipped = np.where(causal_mask, dist, 0)  # zero out future (will be -inf)
    # bias[h, i, j] = -slope[h] * dist[i, j] for causal positions
    bias = -slopes[:, None, None] * dist_clipped[None, :, :]  # (n_heads, seq_len, seq_len)
    # Apply causal mask: future positions get -inf
    neg_inf = np.full_like(bias, -1e9)
    bias = np.where(causal_mask[None, :, :], bias, neg_inf)
    return bias.astype(np.float32)


def compute_attention_profile(
    token_emb: np.ndarray,       # (vocab, d_model)
    qkv_weights: np.ndarray,     # (n_layers, 3, d_model, n_heads, head_dim)
    prenorm_scale: np.ndarray,   # (n_layers, d_model)
    token_ids: np.ndarray,       # (seq_len,)
    alibi: np.ndarray,           # (n_heads, seq_len, seq_len)
    use_sigmoid: bool,
) -> np.ndarray:
    """
    For one input sequence, compute per-head per-layer attention profiles.
    Returns: (n_layers, n_heads, seq_len) — mean attention weight over query positions.
    """
    seq_len = len(token_ids)
    x = token_emb[token_ids].astype(np.float32)  # (seq_len, d_model)

    profiles = np.zeros((N_LAYERS, N_HEADS, seq_len), dtype=np.float32)

    for layer in range(N_LAYERS):
        # RMSNorm
        x_norm = rms_norm(x, prenorm_scale[layer])  # (seq_len, d_model)

        for head in range(N_HEADS):
            # Q and K projections
            W_q = qkv_weights[layer, 0, :, head, :]  # (d_model, head_dim)
            W_k = qkv_weights[layer, 1, :, head, :]  # (d_model, head_dim)
            q = x_norm @ W_q   # (seq_len, head_dim)
            k = x_norm @ W_k   # (seq_len, head_dim)
            # Attention logits
            logits = (q @ k.T) / SQRT_D  # (seq_len, seq_len)
            # Add ALiBi bias for this head
            logits = logits + alibi[head]  # (seq_len, seq_len)
            # Attention function
            if use_sigmoid:
                att = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            else:
                # Softmax with causal masking (alibi already masks future via -1e9)
                logits_max = logits.max(axis=-1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                att = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-10)
            # Mean attention profile over query positions: att_profile[k] = mean_i att[i, i-k]
            # We compute the average diagonal at each lag
            profile = np.zeros(seq_len, dtype=np.float32)
            for lag in range(seq_len):
                diag = np.diag(att, k=-lag)  # query pos i attends to key pos i-lag
                profile[lag] = diag.mean()
            profiles[layer, head] = profile

    return profiles


# ── power-law fit ──────────────────────────────────────────────────────────────

def fit_power_law(profile: np.ndarray, fit_low: int, fit_high: int) -> tuple[float, float, float]:
    """
    Fit A(k) ~ k^(-2Δ) in log-log space for lags in [fit_low, fit_high].
    Returns (delta, lambda_scale, R²).
    """
    lags = np.arange(fit_low, min(fit_high + 1, len(profile)))
    y = profile[fit_low: fit_high + 1]
    if len(lags) < 5 or np.any(y <= 0):
        return float("nan"), float("nan"), 0.0
    log_k = np.log(lags)
    log_y = np.log(y + 1e-30)
    # Linear fit: log_y = -2*delta * log_k + log_lambda
    coeffs = np.polyfit(log_k, log_y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    delta = -slope / 2.0
    lambda_scale = math.exp(intercept)
    # R²
    log_y_pred = slope * log_k + intercept
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return float(delta), float(lambda_scale), float(r2)


# ── main ───────────────────────────────────────────────────────────────────────

def measure_arm(ckpt_root: Path, arm_name: str, use_sigmoid: bool) -> dict:
    """Measure conformal Δ for one arm (sigmoid or softmax)."""
    complete, missing = check_download_complete(ckpt_root)
    if not complete:
        print(f"  [{arm_name}] Download incomplete. Missing: {missing}")
        return {"status": "download_incomplete", "missing": missing}

    print(f"  [{arm_name}] Loading weights...")
    token_emb = load_zarr_array(ckpt_root, "decoder/emb/token_emb/weight")
    qkv_weights = load_zarr_array(
        ckpt_root,
        "decoder/transformer/repeat/layer/self_attention/attention/i_proj/qkv_proj/weight",
    )
    prenorm_scale = load_zarr_array(
        ckpt_root,
        "decoder/transformer/repeat/layer/self_attention/prenorm/scale",
    )
    print(f"  [{arm_name}] token_emb: {token_emb.shape}, qkv: {qkv_weights.shape}, prenorm: {prenorm_scale.shape}")

    # Build ALiBi bias matrix once (same for all inputs and layers)
    alibi = alibi_bias_matrix(N_HEADS, SEQ_LEN)

    # Generate random inputs
    rng = np.random.default_rng(RNG_SEED)
    all_token_ids = rng.integers(0, VOCAB_SIZE, size=(N_INPUTS, SEQ_LEN))

    # Accumulate attention profiles across inputs
    accumulated = np.zeros((N_LAYERS, N_HEADS, SEQ_LEN), dtype=np.float32)
    for inp_idx, token_ids in enumerate(all_token_ids):
        if inp_idx % 10 == 0:
            print(f"  [{arm_name}] input {inp_idx}/{N_INPUTS}...")
        prof = compute_attention_profile(token_emb, qkv_weights, prenorm_scale,
                                          token_ids, alibi, use_sigmoid)
        accumulated += prof
    accumulated /= N_INPUTS  # mean profile over inputs

    # Fit power law per head
    head_results = []
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            profile = accumulated[layer, head]
            delta, lam, r2 = fit_power_law(profile, FIT_LOW, FIT_HIGH)
            if not math.isnan(delta):
                head_results.append({
                    "layer": layer,
                    "head": head,
                    "delta": delta,
                    "lambda": lam,
                    "R2": r2,
                })

    conformal = [h for h in head_results if h["R2"] >= R2_THRESHOLD and h["delta"] > 0]
    deltas = [h["delta"] for h in conformal]
    delta_med = statistics.median(deltas) if deltas else None
    delta_mean = statistics.mean(deltas) if deltas else None

    # Per-layer medians
    per_layer = {}
    for layer in range(N_LAYERS):
        layer_conf = [h for h in conformal if h["layer"] == layer]
        layer_deltas = [h["delta"] for h in layer_conf]
        per_layer[str(layer)] = {
            "conformal_heads": len(layer_conf),
            "delta_median": round(statistics.median(layer_deltas), 4) if layer_deltas else None,
        }

    syk_near = [h for h in conformal if abs(h["delta"] - 0.25) <= 0.05]

    print(f"  [{arm_name}] R²>{R2_THRESHOLD}: {len(conformal)}/{len(head_results)} heads; "
          f"Δ_med={delta_med:.4f}" if delta_med else f"  [{arm_name}] No conformal heads")

    return {
        "status": "complete",
        "arm": arm_name,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "total_heads": N_LAYERS * N_HEADS,
        "conformal_heads_count": len(conformal),
        "delta_median": round(delta_med, 4) if delta_med else None,
        "delta_mean": round(delta_mean, 4) if delta_mean else None,
        "syk_near_count": len(syk_near),
        "syk_near_median": round(statistics.median([h["delta"] for h in syk_near]), 4) if syk_near else None,
        "per_layer": per_layer,
        "head_results": head_results,
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    print(f"exp-041: GALA-7B sigmoid vs softmax Δ — {timestamp}")
    print(f"Checkpoint root: {CKPT_ROOT}")

    for arm_name, ckpt_root in [("sigmoid", SIGMOID_CKPT), ("softmax", SOFTMAX_CKPT)]:
        complete, missing = check_download_complete(ckpt_root)
        print(f"  [{arm_name}] download {'COMPLETE' if complete else f'INCOMPLETE (missing: {missing})'}")

    result = {
        "experiment": "exp-041",
        "timestamp": timestamp,
        "hypothesis": (
            "GALA-7B sigmoid Δ_med ≈ 0.25 (same as softmax) → SYK fixed point mechanism-independent. "
            "If Δ differs: softmax normalization load-bearing."
        ),
        "protocol": {
            "n_inputs": N_INPUTS,
            "seq_len": SEQ_LEN,
            "fit_range": [FIT_LOW, FIT_HIGH],
            "R2_threshold": R2_THRESHOLD,
            "input_type": "random_token_ids [0, 32767]",
            "rng_seed": RNG_SEED,
            "alibi": "head-specific slopes 2^(-(8/n_heads)*(H+1))",
            "normalization": "RMSNorm prenorm applied",
        },
        "reference_deltas": {
            "gpt2_learned_pe": 0.2493,
            "syk_q4_prediction": 0.25,
            "olmo_alibi": 0.265,
            "pythia_rope": 0.358,
        },
        "sigmoid": None,
        "softmax": None,
        "verdict": None,
    }

    sigmoid_result = measure_arm(SIGMOID_CKPT, "sigmoid", use_sigmoid=True)
    result["sigmoid"] = sigmoid_result

    softmax_result = measure_arm(SOFTMAX_CKPT, "softmax", use_sigmoid=False)
    result["softmax"] = softmax_result

    # Verdict
    sig_delta = sigmoid_result.get("delta_median")
    sft_delta = softmax_result.get("delta_median")
    if sig_delta is not None and sft_delta is not None:
        diff = abs(sig_delta - sft_delta)
        syk_dist_sig = abs(sig_delta - 0.25)
        if diff < 0.03 and syk_dist_sig < 0.05:
            verdict = "confirmed_universality"
        elif diff >= 0.05:
            verdict = "falsified_mechanism_dependent"
        else:
            verdict = "inconclusive"
        result["verdict"] = verdict
        result["sigmoid_delta"] = sig_delta
        result["softmax_delta"] = sft_delta
        result["delta_difference"] = round(diff, 4)
        print(f"\n=== RESULT ===")
        print(f"Sigmoid Δ_med:  {sig_delta:.4f}")
        print(f"Softmax Δ_med:  {sft_delta:.4f}")
        print(f"Difference:     {diff:.4f}")
        print(f"Verdict: {verdict}")
    elif sigmoid_result.get("status") == "download_incomplete":
        result["verdict"] = "blocked_download_incomplete"
        print("\nDownload not complete — no Δ measured. Run again when ~/ariel-data/apple-gala-7b/ is ready.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
