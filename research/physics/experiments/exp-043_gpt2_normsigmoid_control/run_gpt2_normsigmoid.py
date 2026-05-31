"""
exp-043 — GPT-2 normalized sigmoid control

Hypothesis (pre-stated, May 31, 2026):
  GPT-2 norm-sigmoid Δ_SYK < 0.249 (softmax, exp-007).
  If true: the normalization-function bracket (norm-sig < SYK-pred < softmax)
    is PE-type-universal — holds for learned positional embeddings (GPT-2)
    as well as ALiBi (GALA-7B).
  If false (Δ_norm-sig ≥ 0.249): the bracket is architecture-specific to
    GALA-7B, possibly ALiBi-dependent.

Protocol: identical to exp-007 (GPT-2 base, seq_len=256, N_INPUTS=50, RNG_SEED=42,
  FIT_LOW=3, FIT_HIGH=50, R²>0.90) except softmax replaced by σ/Σσ normalization.
  Using exp-007 parameters so the comparison to Δ_SYK=0.249 is direct.

Implementation:
  Monkey-patch GPT2Attention._attn in each layer to use norm-sigmoid instead
  of F.softmax. The QK logits, causal masking, and attention_mask are
  identical to the standard forward pass. Only the normalization changes.

Compare to:
  exp-007  GPT-2 softmax:         44/144  conformal, Δ_med=0.2493, Δ_SYK=0.2493
  exp-042  GALA norm-sigmoid:    210/1024 SYK-near,  Δ_SYK=0.2233
  exp-041  GALA softmax:          80/1024 SYK-near,  Δ_SYK=0.2604

Physical question:
  Does learned PE produce the same bracket pattern as ALiBi?
  GALA result: σ/Σσ → Δ ≈ 0.22, exp/Σexp → Δ ≈ 0.26, SYK=0.25 sits between.
  GPT-2 prediction: σ/Σσ → Δ < 0.249 (softmax GPT-2).
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# ── constants ──────────────────────────────────────────────────────────────────
SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
R2_THRESHOLD = 0.90
RNG_SEED = 42
SYK_NEAR_TOL = 0.05  # |Δ - 0.25| ≤ 0.05

OUT_DIR = Path("research/physics/experiments/exp-043_gpt2_normsigmoid_control")
RESULTS_FILE = OUT_DIR / "results.json"


# ── normalized sigmoid normalization ──────────────────────────────────────────

def norm_sigmoid_2d(logits: torch.Tensor) -> torch.Tensor:
    """
    σ/Σσ normalization: att_{ij} = σ(logit_{ij}) / Σ_k σ(logit_{ik}).

    Input logits should already have causal masking applied (future positions
    set to -inf / finfo.min so σ ≈ 0 there). Normalization is then over the
    causal context only.
    """
    sig = torch.sigmoid(logits.float())
    row_sums = sig.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return (sig / row_sums).to(logits.dtype)


def patch_layer_attn(layer_module) -> None:
    """
    Monkey-patch one GPT2Attention module so its _attn uses σ/Σσ instead
    of F.softmax. The patch is applied in-place on the instance.
    """
    def _patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Replicate GPT-2 eager _attn up to the normalization step.
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # Scale
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
        if hasattr(self, "scale_attn_by_inverse_layer_idx") and self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # Causal mask (applies to non-cross-attention only)
        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value_t = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value_t)

        # Additional attention mask (padding etc.)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # ── normalized sigmoid instead of softmax ──────────────────────────
        attn_weights = norm_sigmoid_2d(attn_weights)

        # Dropout (eval mode → no-op) and head mask
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    layer_module._attn = MethodType(_patched_attn, layer_module)


# ── lag profile + power-law fitting ───────────────────────────────────────────

def compute_head_attention_decay(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    """Mean attention weight as function of lag (exp-007 method)."""
    seq = attn_head.shape[0]
    A = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A.astype(np.float32)


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray, low: int, high: int) -> tuple[float, float]:
    """Return (Δ, R²). Δ = -slope/2 from log-log linear fit."""
    mask = (dx_arr >= low) & (dx_arr < high) & (y_arr > 1e-20)
    if mask.sum() < 5:
        return float("nan"), 0.0
    log_x = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask].astype(float))
    A = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    delta = float(-coeffs[1] / 2.0)
    return delta, r2


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("exp-043: GPT-2 normalized sigmoid control", flush=True)
    print(f"  Protocol: SEQ_LEN={SEQ_LEN}, N_INPUTS={N_INPUTS}, "
          f"FIT=[{FIT_LOW},{FIT_HIGH}], R²>{R2_THRESHOLD}", flush=True)
    print(f"  Normalization: σ(logit) / Σσ(logit)  [causal, row-wise]", flush=True)

    print("Loading GPT-2 (eager)...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    n_layers = model.config.n_layer   # 12
    n_heads  = model.config.n_head    # 12
    vocab_size = model.config.vocab_size
    total_heads = n_layers * n_heads  # 144

    print(f"  {n_layers} layers × {n_heads} heads = {total_heads} heads", flush=True)

    # Patch every attention layer
    for block in model.transformer.h:
        patch_layer_attn(block.attn)
    print("  Attention patched: σ/Σσ normalization active", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    dx_arr = np.arange(MAX_DX)

    # Accumulators: [layer][head] → sum of A(dx) over inputs
    A_heads = {l: {h: np.zeros(MAX_DX, dtype=np.float64) for h in range(n_heads)}
               for l in range(n_layers)}

    print(f"Running {N_INPUTS} forward passes...", flush=True)
    for inp_idx in range(N_INPUTS):
        token_ids = torch.tensor(
            rng.integers(0, vocab_size, size=(1, SEQ_LEN)), dtype=torch.long
        )
        with torch.no_grad():
            outputs = model(token_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].float().numpy()  # (n_heads, seq, seq)
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)

        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS} done", flush=True)

    # Average
    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    # Fit power laws
    print("Fitting power laws...", flush=True)
    per_layer: dict[str, dict] = {}
    all_conformal: list[float] = []
    all_syk_near:  list[float] = []

    for l in range(n_layers):
        layer_deltas = []
        for h in range(n_heads):
            delta, r2 = fit_power_law(dx_arr, A_heads[l][h], FIT_LOW, FIT_HIGH)
            if r2 >= R2_THRESHOLD and not math.isnan(delta) and delta > 0:
                layer_deltas.append(delta)
                all_conformal.append(delta)
                if abs(delta - 0.25) <= SYK_NEAR_TOL:
                    all_syk_near.append(delta)

        per_layer[str(l)] = {
            "conformal_heads": len(layer_deltas),
            "delta_median": float(statistics.median(layer_deltas)) if layer_deltas else None,
        }

    n_conformal  = len(all_conformal)
    delta_median = float(statistics.median(all_conformal)) if all_conformal else float("nan")
    syk_median   = float(statistics.median(all_syk_near))  if all_syk_near  else None
    n_syk_near   = len(all_syk_near)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"Conformal heads (R²>{R2_THRESHOLD}): {n_conformal}/{total_heads}", flush=True)
    print(f"Δ_med (all conformal):              {delta_median:.4f}", flush=True)
    print(f"SYK-near (|Δ-0.25|≤{SYK_NEAR_TOL}):          {n_syk_near}/{total_heads}", flush=True)
    print(f"Δ_SYK (SYK-near median):            {syk_median}", flush=True)
    print(flush=True)
    print(f"Comparison:", flush=True)
    print(f"  exp-007 GPT-2 softmax:           44/144  conformal, Δ_med=0.2493, Δ_SYK=0.2493", flush=True)
    print(f"  exp-043 GPT-2 norm-sigmoid: {n_conformal:5d}/144  conformal, Δ_med={delta_median:.4f}, "
          f"Δ_SYK={syk_median}", flush=True)
    print(f"  exp-042 GALA norm-sigmoid:        210/1024 SYK-near, Δ_SYK=0.2233", flush=True)
    print(f"  exp-041 GALA softmax:              80/1024 SYK-near, Δ_SYK=0.2604", flush=True)
    print(flush=True)

    # Check prediction
    if syk_median is not None:
        if syk_median < 0.249:
            print(f"  ✓ PREDICTION CONFIRMED: GPT-2 norm-sigmoid Δ_SYK ({syk_median:.4f}) "
                  f"< GPT-2 softmax Δ_SYK (0.249)", flush=True)
            print(f"    Normalization-function bracket is PE-type-universal.", flush=True)
        else:
            print(f"  ✗ PREDICTION NOT CONFIRMED: GPT-2 norm-sigmoid Δ_SYK ({syk_median:.4f}) "
                  f"≥ GPT-2 softmax Δ_SYK (0.249)", flush=True)
            print(f"    Bracket may be architecture/PE-specific.", flush=True)
    else:
        print("  No SYK-near heads found — check fit range.", flush=True)

    per_layer_summary = []
    for l in range(n_layers):
        info = per_layer[str(l)]
        per_layer_summary.append({
            "layer": l,
            "conformal_heads": info["conformal_heads"],
            "delta_median": info["delta_median"],
        })
        print(f"  L{l:2d}: {info['conformal_heads']:2d} conformal  "
              f"Δ_med={info['delta_median']:.4f}" if info["delta_median"] is not None
              else f"  L{l:2d}: 0 conformal", flush=True)

    result = {
        "experiment": "exp-043",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": (
            "GPT-2 norm-sigmoid Δ_SYK < 0.249 (softmax, exp-007) → "
            "normalization-function bracket is PE-type-universal."
        ),
        "protocol": {
            "model": "gpt2",
            "normalization": "σ(logit) / Σσ(logit)  [row-wise, causal]",
            "seq_len": SEQ_LEN,
            "n_inputs": N_INPUTS,
            "fit_range": [FIT_LOW, FIT_HIGH],
            "R2_threshold": R2_THRESHOLD,
            "rng_seed": RNG_SEED,
            "note": "Same parameters as exp-007 for direct comparison.",
        },
        "reference": {
            "exp007_gpt2_softmax": {
                "conformal_heads": 44, "total_heads": 144,
                "delta_median": 0.2493, "syk_near": 44, "syk_near_median": 0.2493,
            },
            "exp042_gala_normsigmoid": {
                "conformal_heads": 378, "total_heads": 1024,
                "delta_median": 0.2653, "syk_near": 210, "syk_near_median": 0.2233,
            },
            "exp041_gala_softmax": {
                "conformal_heads": 121, "total_heads": 1024,
                "syk_near": 80, "syk_near_median": 0.2604,
            },
        },
        "result": {
            "conformal_heads_count": n_conformal,
            "total_heads": total_heads,
            "delta_median": delta_median,
            "syk_near_count": n_syk_near,
            "syk_near_median": syk_median,
            "per_layer": per_layer_summary,
        },
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
