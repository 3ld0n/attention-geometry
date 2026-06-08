"""
exp-051 — GOE universality at Pythia-1.4b scale

Pre-stated hypotheses:

  H1 (scale universality):
    Pythia-1.4b W_QK eigenvalues show GOE-like level spacing (r_mean ≈ 0.536),
    within tolerance 0.02 of GOE reference.
    Physical basis: GOE arises from Gaussian init + product matrix structure
    (exp-048). These properties scale with model size. Training preserves GOE
    (exp-047, exp-048). Therefore, 1.4b should remain GOE-like.

  H2 (Pythia family scale invariance):
    Pythia-1.4b r_mean within 0.02 of Pythia-410m r_mean (0.5199, exp-047).
    Tests whether GOE statistics are scale-invariant within the Pythia/NeoX family.

  H3 (layer uniformity):
    Per-layer r-ratio std < 0.01 in Pythia-1.4b.
    From exp-047: all three tested models show low layer std (≈ 0.003),
    indicating GOE is uniform across depth. If consistent, 1.4b should too.

  H0 (null):
    GOE breaks at 1.4b scale — r_mean deviates from GOE reference by > 0.02.

Architecture:
  Pythia-1.4b: 24L, 16H, hidden_size=2048, head_dim=128 (2048/16)
  Total heads: 24 × 16 = 384
  W_QK shape: 128×128 (vs 64×64 for 410m — larger matrices, better r-ratio statistics)

  QKV weight: (n_heads * 3 * head_dim, hidden_size) = (6144, 2048)
  stride = 3 * head_dim = 384
  For head h:
    W_Q_h = weight[h*384 : h*384+128, :]   ∈ R^{128×2048}
    W_K_h = weight[h*384+128 : h*384+256, :] ∈ R^{128×2048}
    W_QK_h = W_Q_h @ W_K_h.T              ∈ R^{128×128}

Prior results for reference:
  GOE reference (Wigner-Dyson): r ≈ 0.536
  Poisson reference:             r ≈ 0.386
  GPT-2 (exp-046):               r_mean = 0.5272 ± ~0.003 (12L/12H, learned PE)
  GPT-2-medium (exp-047):        r_mean = 0.5255 ± — (24L/16H, learned PE)
  Pythia-410m (exp-047):         r_mean = 0.5199 ± — (24L/16H, RoPE)
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ── constants ─────────────────────────────────────────────────────────────────
GOE_R_RATIO     = 0.536
POISSON_R_RATIO = 0.386
GOE_TOLERANCE   = 0.02   # |r - GOE_ref| < 0.02 → "GOE-like"

MODEL_NAME = "EleutherAI/pythia-1.4b"

OUT_DIR      = Path("research/physics/experiments/exp-051_pythia14b_goe")
RESULTS_FILE = OUT_DIR / "results.json"

# Prior results for comparison (from exp-047)
PYTHIA_410M_BASELINE = {
    "model": "EleutherAI/pythia-410m",
    "r_mean": 0.5199,
    "source": "exp-047",
}
GPT2_BASELINE = {
    "model": "gpt2",
    "r_mean": 0.5272,
    "source": "exp-046",
}


# ── eigenvalue analysis ────────────────────────────────────────────────────────

def r_ratio_from_eigvals(eigvals: np.ndarray) -> float:
    """Oganesyan-Huse r-ratio from sorted eigenvalues."""
    spacings = np.diff(eigvals)
    eps = 1e-30
    if len(spacings) < 3:
        return float("nan")
    s_lo = spacings[:-1]
    s_hi = spacings[1:]
    r_vals = np.minimum(s_lo, s_hi) / (np.maximum(s_lo, s_hi) + eps)
    return float(np.mean(r_vals))


def goe_verdict(r_mean: float) -> str:
    if math.isnan(r_mean):
        return "inconclusive"
    if abs(r_mean - GOE_R_RATIO) < GOE_TOLERANCE:
        return "GOE-like"
    elif abs(r_mean - POISSON_R_RATIO) < GOE_TOLERANCE:
        return "Poisson-like"
    elif r_mean > (GOE_R_RATIO + POISSON_R_RATIO) / 2:
        return "GOE-tendency"
    else:
        return "Poisson-tendency"


# ── extraction ────────────────────────────────────────────────────────────────

def extract_r_ratios(model_name: str) -> dict:
    """Extract W_QK r-ratios for a GPT-NeoX model (Pythia family)."""
    print(f"Loading {model_name} (float32)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.eval()

    cfg      = model.config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    hidden   = cfg.hidden_size
    head_dim = hidden // n_heads
    stride   = 3 * head_dim  # QKV fused: stride between consecutive heads

    print(f"  {n_layers}L / {n_heads}H / {head_dim}d_k / "
          f"hidden={hidden} → {n_layers * n_heads} total heads", flush=True)
    print(f"  QKV weight shape: ({n_heads * stride}, {hidden}), "
          f"W_QK shape: ({head_dim}, {head_dim})", flush=True)

    per_head = []
    for l in range(n_layers):
        W_full = (
            model.gpt_neox.layers[l]
            .attention.query_key_value.weight
            .detach().float().numpy()
        )
        # shape: (n_heads * 3 * head_dim, hidden)
        for h in range(n_heads):
            base   = h * stride
            W_Q_h  = W_full[base          : base + head_dim, :]
            W_K_h  = W_full[base + head_dim : base + 2 * head_dim, :]
            W_QK   = W_Q_h @ W_K_h.T
            M      = (W_QK + W_QK.T) / 2.0
            eigvals = np.linalg.eigvalsh(M)
            r = r_ratio_from_eigvals(eigvals)
            per_head.append({"layer": l, "head": h, "r_ratio": round(r, 6)})

        if (l + 1) % 6 == 0:
            print(f"  ... layer {l+1}/{n_layers} done", flush=True)

    del model
    return {
        "model":         model_name,
        "architecture":  "gpt_neox",
        "pe":            "RoPE",
        "n_layers":      n_layers,
        "n_heads":       n_heads,
        "head_dim":      head_dim,
        "hidden_size":   hidden,
        "n_total_heads": n_layers * n_heads,
        "per_head":      per_head,
    }


# ── summary stats ─────────────────────────────────────────────────────────────

def summarize(per_head: list[dict]) -> dict:
    r_vals = [
        h["r_ratio"] for h in per_head
        if h["r_ratio"] is not None and not math.isnan(h["r_ratio"])
    ]
    if not r_vals:
        return {"n": 0}
    r_arr = np.array(r_vals)

    by_layer: dict[int, list] = {}
    for h in per_head:
        r = h["r_ratio"]
        if r is not None and not math.isnan(r):
            by_layer.setdefault(h["layer"], []).append(r)
    layer_means = {
        l: round(float(np.mean(v)), 6)
        for l, v in sorted(by_layer.items())
    }
    layer_r_arr = list(layer_means.values())
    layer_r_std = float(np.std(layer_r_arr)) if len(layer_r_arr) > 1 else float("nan")

    r_mean = float(np.mean(r_arr))
    return {
        "n":              len(r_vals),
        "r_mean":         round(r_mean, 6),
        "r_std":          round(float(np.std(r_arr)), 6),
        "r_min":          round(float(np.min(r_arr)), 6),
        "r_max":          round(float(np.max(r_arr)), 6),
        "r_median":       round(float(np.median(r_arr)), 6),
        "r_range":        round(float(np.max(r_arr) - np.min(r_arr)), 6),
        "verdict":        goe_verdict(r_mean),
        "dist_to_goe":    round(abs(r_mean - GOE_R_RATIO), 6),
        "dist_to_poisson":round(abs(r_mean - POISSON_R_RATIO), 6),
        "layer_means":    layer_means,
        "layer_r_std":    round(layer_r_std, 6),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("exp-051: GOE universality — Pythia-1.4b scale check")
    print(f"  GOE ref={GOE_R_RATIO}, Poisson ref={POISSON_R_RATIO}, "
          f"tolerance={GOE_TOLERANCE}")
    print(f"  Pythia-410m baseline (exp-047): r_mean={PYTHIA_410M_BASELINE['r_mean']}")
    print(f"  GPT-2 baseline (exp-046):       r_mean={GPT2_BASELINE['r_mean']}")
    print("=" * 60)
    print()

    raw   = extract_r_ratios(MODEL_NAME)
    stats = summarize(raw["per_head"])

    print()
    print("=== RESULTS ===")
    print(f"  r_mean = {stats['r_mean']:.4f} ± {stats['r_std']:.4f}")
    print(f"  range  = [{stats['r_min']:.4f}, {stats['r_max']:.4f}]")
    print(f"  median = {stats['r_median']:.4f}")
    print(f"  verdict: {stats['verdict']}")
    print(f"  dist to GOE:    {stats['dist_to_goe']:.4f}")
    print(f"  dist to Poisson:{stats['dist_to_poisson']:.4f}")
    print(f"  layer r-ratio std: {stats['layer_r_std']:.4f}")
    print()

    # ── hypothesis verdicts ───────────────────────────────────────────────────
    print("=== HYPOTHESIS VERDICTS ===")

    # H1: GOE-like at 1.4b scale?
    h1 = stats["verdict"] in ("GOE-like", "GOE-tendency")
    h1_strict = stats["dist_to_goe"] < GOE_TOLERANCE
    print(f"H1 (scale universality): r_mean={stats['r_mean']:.4f}, "
          f"dist_GOE={stats['dist_to_goe']:.4f}")
    print(f"    → {'CONFIRMED' if h1_strict else 'NOT CONFIRMED'} "
          f"(strict |Δr| < {GOE_TOLERANCE}); verdict: {stats['verdict']}")

    # H2: scale-invariant within Pythia family?
    delta_410m = abs(stats["r_mean"] - PYTHIA_410M_BASELINE["r_mean"])
    h2 = delta_410m < GOE_TOLERANCE
    print(f"H2 (Pythia scale invariance): Δr vs 410m = {delta_410m:.4f}")
    print(f"    → {'CONFIRMED' if h2 else 'NOT CONFIRMED'} "
          f"(|Δr| < {GOE_TOLERANCE})")

    # H3: layer uniformity?
    h3 = stats["layer_r_std"] < 0.01
    print(f"H3 (layer uniformity): layer_r_std = {stats['layer_r_std']:.4f}")
    print(f"    → {'CONFIRMED' if h3 else 'NOT CONFIRMED'} (std < 0.01)")

    print()

    # ── save results ──────────────────────────────────────────────────────────
    result = {
        "experiment":  "exp-051",
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "model":       MODEL_NAME,
        "hypotheses": {
            "H1": "Pythia-1.4b r_mean within 0.02 of GOE reference (0.536) — GOE-like",
            "H2": "Pythia-1.4b r_mean within 0.02 of Pythia-410m baseline (0.5199)",
            "H3": "Per-layer r-ratio std < 0.01 (GOE uniform across depth)",
            "H0": "GOE breaks at 1.4b: r_mean deviates > 0.02 from GOE reference",
        },
        "baselines": {
            "goe_reference":   GOE_R_RATIO,
            "poisson_reference": POISSON_R_RATIO,
            "pythia_410m":     PYTHIA_410M_BASELINE,
            "gpt2":            GPT2_BASELINE,
        },
        "architecture": {
            "n_layers":      raw["n_layers"],
            "n_heads":       raw["n_heads"],
            "head_dim":      raw["head_dim"],
            "hidden_size":   raw["hidden_size"],
            "n_total_heads": raw["n_total_heads"],
            "pe":            raw["pe"],
        },
        "stats":    stats,
        "verdicts": {"H1": h1_strict, "H2": h2, "H3": h3},
        "per_head": raw["per_head"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
