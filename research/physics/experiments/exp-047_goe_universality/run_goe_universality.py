"""
exp-047 — GOE universality cross-model check

Pre-stated hypotheses:

  H1 (GOE universality):
    Pythia-410m W_QK eigenvalues show GOE-like level spacing (r-ratio ≈ 0.536)
    across all heads, similar to GPT-2 (exp-046: r-ratio 0.525–0.529).
    If confirmed: GOE weight chaos is model-family-independent.

  H2 (Architecture independence):
    The GOE r-ratio for Pythia-410m (RoPE, GPT-NeoX) is statistically
    indistinguishable from GPT-2 (learned PE, GPT-2 arch).
    Tolerance: |Δr| < 0.02 (< 4% of GOE reference 0.536).

  H3 (Scale invariance):
    GPT-2-medium (24L, 16H, 64d_k — same arch, larger scale) shows
    the same GOE pattern as GPT-2 (12L, 12H, 64d_k).
    Tests whether GOE is scale-invariant within a family.

  H0 (null):
    GOE was GPT-2-specific: other models show Poisson or random statistics.

Physical background:
  exp-046 found that ALL 144 GPT-2 W_QK matrices have GOE-like level spacing
  (r-ratio 0.525–0.529, GOE ref = 0.536, Poisson ref = 0.386). Conformal and
  non-conformal heads were indistinguishable (0.525 vs 0.529). This suggests
  GOE is a universal property of trained transformer QK weight matrices, not
  a marker of conformal structure.

  If GOE is universal, it means: all trained transformers sit in SYK-chaotic
  weight space, and the position-space conformal structure (power-law decay,
  Δ ≈ 0.25) is an additional layer of organization *on top of* a universal
  weight-space GOE background. This is a coherent physical picture.

  If GOE fails for Pythia or GPT-2-medium, the exp-046 result may be
  GPT-2-specific (possibly an artifact of the learned PE or Conv1D structure).

W_QK extraction:

  GPT-2 (Conv1D):
    c_attn.weight shape (768, 2304) = (d_model, 3*d_model)
    output = input @ weight + bias  (Conv1D convention — opposite of nn.Linear)
    Q_h = input @ weight[:, h*64:(h+1)*64]
    K_h = input @ weight[:, 768+h*64:768+(h+1)*64]
    W_Q_h = weight[:, h*64:(h+1)*64]  ∈ R^{768×64}
    W_K_h = weight[:, 768+h*64:768+(h+1)*64]  ∈ R^{768×64}
    W_QK_h = W_Q_h.T @ W_K_h  ∈ R^{64×64}

  Pythia / GPT-NeoX (nn.Linear, fused QKV):
    query_key_value.weight shape (3072, 1024) = (n_heads*3*head_dim, hidden_size)
    output = input @ weight.T + bias
    View: (batch, seq, n_heads=16, 3*head_dim=192), chunk(3, dim=-1) → Q,K,V
    W_Q_h = weight[h*192 : h*192+64, :]    ∈ R^{64×1024}
    W_K_h = weight[h*192+64 : h*192+128, :]  ∈ R^{64×1024}
    W_QK_h = W_Q_h @ W_K_h.T  ∈ R^{64×64}

Compare to:
  exp-046: GPT-2 (12L, 12H, 64d_k, learned PE)
    r-ratio all heads: 0.525–0.529 (GOE ≈ 0.536, Poisson ≈ 0.386)
    conformal vs non-conformal: indistinguishable
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoModelForCausalLM

# ── constants ─────────────────────────────────────────────────────────────────
GOE_R_RATIO     = 0.536
POISSON_R_RATIO = 0.386
GOE_TOLERANCE   = 0.02   # |r - GOE_ref| < 0.02 to call "GOE-like"

OUT_DIR      = Path("research/physics/experiments/exp-047_goe_universality")
RESULTS_FILE = OUT_DIR / "results.json"

# GPT-2 baseline from exp-046
GPT2_BASELINE = {
    "model": "gpt2",
    "n_layers": 12,
    "n_heads": 12,
    "head_dim": 64,
    "pe": "learned",
    "r_ratio_mean": 0.5272,   # mean across all 144 heads
    "r_ratio_min": 0.525,
    "r_ratio_max": 0.529,
    "n_total_heads": 144,
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


def goe_universality_verdict(
    r_mean: float, r_std: float, n: int, ref_r: float = GOE_R_RATIO
) -> str:
    """Classify r-ratio result: GOE, Poisson, intermediate, or inconclusive."""
    if math.isnan(r_mean):
        return "inconclusive"
    dist_goe     = abs(r_mean - GOE_R_RATIO)
    dist_poisson = abs(r_mean - POISSON_R_RATIO)
    if dist_goe < GOE_TOLERANCE:
        return "GOE-like"
    elif dist_poisson < GOE_TOLERANCE:
        return "Poisson-like"
    elif r_mean > (GOE_R_RATIO + POISSON_R_RATIO) / 2:
        return "GOE-tendency"
    else:
        return "Poisson-tendency"


# ── model-specific extractors ──────────────────────────────────────────────────

def extract_gpt2_r_ratios(model_name: str) -> dict:
    """Extract W_QK r-ratios for GPT-2 family models."""
    print(f"  Loading {model_name} (eager)...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(
        model_name, attn_implementation="eager"
    )
    model.eval()

    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_model  = model.config.n_embd
    head_dim = d_model // n_heads

    print(f"  {model_name}: {n_layers}L / {n_heads}H / {head_dim}d_k "
          f"({n_layers*n_heads} total heads)", flush=True)

    per_head = []
    for l in range(n_layers):
        W_full = model.transformer.h[l].attn.c_attn.weight.detach().float().numpy()
        # Conv1D: output = input @ weight + bias → weight shape (d_model, 3*d_model)
        for h in range(n_heads):
            W_Q_h = W_full[:, h * head_dim          : (h + 1) * head_dim]
            W_K_h = W_full[:, d_model + h * head_dim : d_model + (h + 1) * head_dim]
            W_QK  = W_Q_h.T @ W_K_h
            M     = (W_QK + W_QK.T) / 2.0
            eigvals = np.linalg.eigvalsh(M)
            r = r_ratio_from_eigvals(eigvals)
            per_head.append({"layer": l, "head": h, "r_ratio": round(r, 6)})

    del model
    return {
        "model": model_name,
        "architecture": "gpt2",
        "pe": "learned",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_total_heads": n_layers * n_heads,
        "per_head": per_head,
    }


def extract_neox_r_ratios(model_name: str) -> dict:
    """Extract W_QK r-ratios for GPT-NeoX (Pythia) family models."""
    print(f"  Loading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.eval()

    cfg      = model.config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    hidden   = cfg.hidden_size
    head_dim = hidden // n_heads
    # QKV fused weight: (n_heads * 3 * head_dim, hidden) = (n_heads*3*head_dim, hidden)
    # For head h: Q = weight[h*3*head_dim : h*3*head_dim+head_dim, :]
    #             K = weight[h*3*head_dim+head_dim : h*3*head_dim+2*head_dim, :]
    stride = 3 * head_dim   # stride between consecutive heads in the weight

    print(f"  {model_name}: {n_layers}L / {n_heads}H / {head_dim}d_k "
          f"({n_layers*n_heads} total heads)", flush=True)

    pe = "RoPE"  # Pythia uses RoPE (rotary position embeddings)

    per_head = []
    for l in range(n_layers):
        W_full = model.gpt_neox.layers[l].attention.query_key_value.weight.detach().float().numpy()
        # shape: (n_heads * 3 * head_dim, hidden) = (3072, 1024) for Pythia-410m
        for h in range(n_heads):
            base = h * stride
            W_Q_h = W_full[base          : base + head_dim, :]        # (64, 1024)
            W_K_h = W_full[base + head_dim : base + 2 * head_dim, :]  # (64, 1024)
            W_QK  = W_Q_h @ W_K_h.T    # (64, 64)
            M     = (W_QK + W_QK.T) / 2.0
            eigvals = np.linalg.eigvalsh(M)
            r = r_ratio_from_eigvals(eigvals)
            per_head.append({"layer": l, "head": h, "r_ratio": round(r, 6)})

    del model
    return {
        "model": model_name,
        "architecture": "gpt_neox",
        "pe": pe,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_total_heads": n_layers * n_heads,
        "per_head": per_head,
    }


# ── summary statistics ─────────────────────────────────────────────────────────

def summarize(per_head: list[dict]) -> dict:
    r_vals = [h["r_ratio"] for h in per_head
              if h["r_ratio"] is not None and not math.isnan(h["r_ratio"])]
    if not r_vals:
        return {"n": 0}
    r_arr   = np.array(r_vals)
    r_mean  = float(np.mean(r_arr))
    r_std   = float(np.std(r_arr))
    r_min   = float(np.min(r_arr))
    r_max   = float(np.max(r_arr))
    r_med   = float(np.median(r_arr))

    verdict = goe_universality_verdict(r_mean, r_std, len(r_vals))

    # Layer-level: per-layer mean r-ratio
    by_layer: dict[int, list] = {}
    for h in per_head:
        r = h["r_ratio"]
        if r is not None and not math.isnan(r):
            by_layer.setdefault(h["layer"], []).append(r)
    layer_means = {l: round(float(np.mean(v)), 6) for l, v in sorted(by_layer.items())}
    layer_r_arr = list(layer_means.values())
    layer_r_std = float(np.std(layer_r_arr)) if layer_r_arr else float("nan")

    return {
        "n": len(r_vals),
        "r_mean": round(r_mean, 6),
        "r_std":  round(r_std, 6),
        "r_min":  round(r_min, 6),
        "r_max":  round(r_max, 6),
        "r_median": round(r_med, 6),
        "r_range": round(r_max - r_min, 6),
        "verdict": verdict,
        "dist_to_goe": round(abs(r_mean - GOE_R_RATIO), 6),
        "dist_to_poisson": round(abs(r_mean - POISSON_R_RATIO), 6),
        "layer_means": layer_means,
        "layer_r_std": round(layer_r_std, 6),
    }


def compare_to_gpt2(stats: dict, model_name: str) -> dict:
    """Assess whether r-ratio matches GPT-2 baseline from exp-046."""
    delta = abs(stats["r_mean"] - GPT2_BASELINE["r_ratio_mean"])
    within_tolerance = (delta < GOE_TOLERANCE)
    return {
        "delta_from_gpt2": round(delta, 6),
        "within_tolerance": within_tolerance,
        "tolerance_used": GOE_TOLERANCE,
        "verdict": (
            f"matches_gpt2 (Δr={delta:.4f} < {GOE_TOLERANCE})" if within_tolerance
            else f"differs_from_gpt2 (Δr={delta:.4f} ≥ {GOE_TOLERANCE})"
        ),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("exp-047: GOE universality cross-model check")
    print(f"  GOE ref={GOE_R_RATIO}, Poisson ref={POISSON_R_RATIO}, "
          f"tolerance={GOE_TOLERANCE}")
    print(f"  GPT-2 baseline (exp-046): r_mean≈{GPT2_BASELINE['r_ratio_mean']}")
    print()

    models_to_run = [
        ("pythia_410m", "EleutherAI/pythia-410m", "neox"),
        ("gpt2_medium", "gpt2-medium", "gpt2"),
    ]

    all_results = {}

    for key, name, arch in models_to_run:
        print(f"=== {name} ===")
        try:
            if arch == "gpt2":
                raw = extract_gpt2_r_ratios(name)
            else:
                raw = extract_neox_r_ratios(name)

            stats = summarize(raw["per_head"])
            vs_gpt2 = compare_to_gpt2(stats, name)

            print(f"  r-ratio: mean={stats['r_mean']:.4f} ± {stats['r_std']:.4f}  "
                  f"range=[{stats['r_min']:.4f},{stats['r_max']:.4f}]")
            print(f"  verdict: {stats['verdict']}")
            print(f"  vs GPT-2 baseline: {vs_gpt2['verdict']}")
            print(f"  layer std(r-ratio): {stats['layer_r_std']:.4f}  "
                  f"(low → uniform across layers)")
            print()

            all_results[key] = {
                "model":      raw["model"],
                "architecture": raw["architecture"],
                "pe":          raw["pe"],
                "n_layers":    raw["n_layers"],
                "n_heads":     raw["n_heads"],
                "head_dim":    raw["head_dim"],
                "n_total_heads": raw["n_total_heads"],
                "stats":       stats,
                "vs_gpt2":     vs_gpt2,
                "per_head":    raw["per_head"],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[key] = {"error": str(e)}

    # ── hypothesis verdicts ───────────────────────────────────────────────────
    print("=" * 60)
    print("=== HYPOTHESIS VERDICTS ===")

    # H1: Pythia-410m GOE-like?
    p410 = all_results.get("pythia_410m", {})
    if "stats" in p410:
        s = p410["stats"]
        h1_verdict = (s["verdict"] in ("GOE-like", "GOE-tendency"))
        print(f"H1 (Pythia-410m GOE): r_mean={s['r_mean']:.4f}  "
              f"→ {'CONFIRMED' if h1_verdict else 'NOT CONFIRMED'} ({s['verdict']})")
    else:
        h1_verdict = None
        print(f"H1: inconclusive (run error)")

    # H2: Architecture independence
    if "stats" in p410:
        h2_verdict = p410["vs_gpt2"]["within_tolerance"]
        print(f"H2 (architecture independence): {p410['vs_gpt2']['verdict']}")
        print(f"    → {'CONFIRMED' if h2_verdict else 'NOT CONFIRMED'}")
    else:
        h2_verdict = None
        print("H2: inconclusive (run error)")

    # H3: GPT-2-medium scale invariance
    gpt2m = all_results.get("gpt2_medium", {})
    if "stats" in gpt2m:
        s = gpt2m["stats"]
        h3_verdict = gpt2m["vs_gpt2"]["within_tolerance"]
        print(f"H3 (GPT-2-medium scale): r_mean={s['r_mean']:.4f}  "
              f"→ {'CONFIRMED' if h3_verdict else 'NOT CONFIRMED'} ({gpt2m['vs_gpt2']['verdict']})")
    else:
        h3_verdict = None
        print(f"H3: inconclusive (run error)")

    # ── save ─────────────────────────────────────────────────────────────────
    result = {
        "experiment": "exp-047",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H1": "Pythia-410m (RoPE/NeoX) W_QK r-ratio ≈ 0.536 (GOE-like) across all heads",
            "H2": "Pythia-410m r-ratio within 0.02 of GPT-2 baseline (architecture-independent)",
            "H3": "GPT-2-medium r-ratio within 0.02 of GPT-2 (scale-invariant within family)",
            "H0": "GOE was GPT-2-specific; other models show Poisson or weaker pattern",
        },
        "protocol": {
            "analysis": "weights-only (no forward passes)",
            "matrix": "W_QK_sym = (W_Q^T W_K + W_K^T W_Q) / 2  ∈ R^{d_k×d_k}",
            "r_ratio": "Oganesyan-Huse r-ratio on sorted eigenvalue spacings",
            "goe_reference": GOE_R_RATIO,
            "poisson_reference": POISSON_R_RATIO,
            "tolerance_H2_H3": GOE_TOLERANCE,
            "gpt2_baseline": GPT2_BASELINE,
        },
        "models": all_results,
        "verdicts": {
            "H1": h1_verdict,
            "H2": h2_verdict,
            "H3": h3_verdict,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
