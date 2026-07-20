"""
exp-087 — RAND Early Burst Scaling with Model Size

Tests whether the fraction of RAND SYK-near heads at step 512 is constant
across Pythia model sizes with the same d_k architecture.

Pre-registration: notes.md (commit 5e023594, 2026-07-20).

Ariel — July 20, 2026.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ─── constants (pre-registered in notes.md) ──────────────────────────────────

SEQ_LEN = 256
N_RAND = 20
RAND_SEED = 87

FIT_LOW = 3
FIT_HIGH = 50
R2_THRESHOLD = 0.85
SYK_LOW = 0.20
SYK_HIGH = 0.30

OUT = Path(__file__).resolve().parent


# ─── measurement helpers ─────────────────────────────────────────────────────

def compute_attention_decay(attn_head: np.ndarray, max_dx: int = FIT_HIGH + 5) -> np.ndarray:
    """Average A[i, i-dx] over positions i."""
    A = np.array([np.diag(attn_head, -dx).mean() if dx < attn_head.shape[0] else 0.0
                  for dx in range(max_dx)])
    return A


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float | None, float | None]:
    """Fit power-law A ~ dx^{-2Δ}. Returns (Δ, R²) or (None, None)."""
    mask = (
        (dx_arr >= FIT_LOW)
        & (dx_arr < FIT_HIGH)
        & (y_arr > 1e-20)
    )
    if mask.sum() < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask])
    A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    pred = A_mat @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    delta = -coeffs[1] / 2.0
    return float(delta), float(r2)


def measure_model(model_name: str, revision: str, n_layers: int, n_heads: int,
                  vocab_size: int) -> dict:
    """Run BCFT analysis on RAND inputs at a specific training checkpoint."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} @ {revision}")
    print(f"  Architecture: {n_layers}L × {n_heads}H, total={n_layers*n_heads} heads")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(RAND_SEED)
    dx_arr = np.arange(FIT_HIGH + 5)

    # Accumulate per-head attention profiles
    head_profiles = {(l, h): np.zeros(FIT_HIGH + 5) for l in range(n_layers)
                     for h in range(n_heads)}

    with torch.no_grad():
        for i in range(N_RAND):
            tokens = torch.tensor(
                rng.integers(0, vocab_size, size=(1, SEQ_LEN)), dtype=torch.long
            )
            out = model(tokens, output_attentions=True)
            # out.attentions: tuple of (1, n_heads, seq, seq) per layer
            for layer_idx, attn in enumerate(out.attentions):
                # attn: (1, n_heads, seq, seq)
                attn_np = attn[0].cpu().numpy()  # (n_heads, seq, seq)
                for head_idx in range(n_heads):
                    head_attn = attn_np[head_idx]  # (seq, seq)
                    decay = compute_attention_decay(head_attn, max_dx=FIT_HIGH + 5)
                    head_profiles[(layer_idx, head_idx)] += decay / N_RAND

    print(f"  Measured {N_RAND} random sequences")

    # Fit power law per head
    results = []
    syk_near_count = 0
    per_layer_syk = {l: 0 for l in range(n_layers)}

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            profile = head_profiles[(layer_idx, head_idx)]
            delta, r2 = fit_power_law(dx_arr, profile)
            if delta is None:
                conformal = False
                syk_near = False
            else:
                conformal = r2 >= R2_THRESHOLD
                syk_near = conformal and SYK_LOW <= delta <= SYK_HIGH

            if syk_near:
                syk_near_count += 1
                per_layer_syk[layer_idx] += 1

            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "delta": delta,
                "R2": r2,
                "conformal": conformal,
                "syk_near": syk_near,
            })

    total_heads = n_layers * n_heads
    fraction = syk_near_count / total_heads

    print(f"\n  Results @ {revision} RAND:")
    print(f"    SYK-near: {syk_near_count}/{total_heads} = {fraction:.4f}")
    print(f"    Per-layer SYK-near mean: {syk_near_count/n_layers:.2f}")
    print(f"    SYK-near heads:")
    for r in results:
        if r["syk_near"]:
            print(f"      L{r['layer']}H{r['head']}: delta={r['delta']:.3f}, R2={r['R2']:.3f}")

    # Per-layer breakdown
    print(f"\n  Per-layer SYK-near count:")
    for l in range(n_layers):
        if per_layer_syk[l] > 0:
            print(f"    L{l}: {per_layer_syk[l]}")

    return {
        "model": model_name,
        "revision": revision,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads": total_heads,
        "n_rand": N_RAND,
        "rand_seed": RAND_SEED,
        "seq_len": SEQ_LEN,
        "fit_low": FIT_LOW,
        "fit_high": FIT_HIGH,
        "r2_threshold": R2_THRESHOLD,
        "syk_window": [SYK_LOW, SYK_HIGH],
        "syk_near_count": syk_near_count,
        "syk_near_fraction": fraction,
        "per_layer_syk_near": per_layer_syk,
        "per_head_results": results,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Pythia-70m step512 — from exp-086 crossover_fine_results.json
    # Already measured; load existing data rather than re-running
    exp086_path = (
        OUT.parent / "exp-086_longitudinal_delta_spectrum" / "crossover_fine_results.json"
    )
    with open(exp086_path) as f:
        exp086_data = json.load(f)

    step512_data = exp086_data["per_head_results"]["512"]
    rand_heads_70m = step512_data["rand"]
    syk_near_70m = sum(1 for h in rand_heads_70m if h["syk_near"])
    total_70m = len(rand_heads_70m)

    print("=" * 60)
    print("exp-087: RAND Early Burst Scaling")
    print("Pre-registration: 5e023594")
    print("=" * 60)
    print(f"\nPythia-70m step512 RAND (from exp-086):")
    print(f"  SYK-near: {syk_near_70m}/{total_70m} = {syk_near_70m/total_70m:.4f}")
    print(f"  SYK-near heads:")
    for h in rand_heads_70m:
        if h["syk_near"]:
            print(f"    L{h['layer']}H{h['head']}: delta={h['delta']:.3f}, R2={h['R2']:.3f}")

    # Pythia-410m step512 — run new measurement
    result_410m = measure_model(
        model_name="EleutherAI/pythia-410m",
        revision="step512",
        n_layers=24,
        n_heads=16,
        vocab_size=50304,
    )

    # Summary comparison
    print("\n" + "=" * 60)
    print("SCALING COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Total heads':<15} {'SYK-near':<12} {'Fraction':<12} {'Per-layer'}")
    print(f"{'Pythia-70m':<20} {total_70m:<15} {syk_near_70m:<12} {syk_near_70m/total_70m:.4f}{'':8} {syk_near_70m/6:.2f}")
    n410 = result_410m['syk_near_count']
    print(f"{'Pythia-410m':<20} {result_410m['total_heads']:<15} {n410:<12} {result_410m['syk_near_fraction']:.4f}{'':8} {n410/24:.2f}")

    # Verdict
    print("\nPRE-REGISTERED VERDICT:")
    f410 = result_410m['syk_near_fraction']
    if 0.1375 <= f410 <= 0.2375:
        print("  H_scale_const: CONFIRMED (fraction within ±0.05 of 70m baseline 0.1875)")
    elif f410 > 0.2375:
        print("  H_scale_supralinear: CONFIRMED (fraction > 0.2375)")
    else:
        print("  H_scale_sublinear: CONFIRMED (fraction < 0.1375)")

    per_layer_mean_410 = n410 / 24
    if 1.0 <= per_layer_mean_410 <= 2.0:
        print("  H_scale_layer_fixed: CONSISTENT (1.0–2.0 SYK-near per layer)")
    else:
        print(f"  H_scale_layer_fixed: NOT CONSISTENT ({per_layer_mean_410:.2f} per layer, outside 1.0–2.0)")

    # Save results
    output = {
        "experiment": "exp-087",
        "date": "2026-07-20",
        "prereg_commit": "5e023594",
        "protocol": {
            "seq_len": SEQ_LEN,
            "n_rand": N_RAND,
            "rand_seed": RAND_SEED,
            "fit_low": FIT_LOW,
            "fit_high": FIT_HIGH,
            "r2_threshold": R2_THRESHOLD,
            "syk_window": [SYK_LOW, SYK_HIGH],
        },
        "pythia_70m_step512": {
            "source": "exp-086 crossover_fine_results.json",
            "total_heads": total_70m,
            "syk_near_count": syk_near_70m,
            "syk_near_fraction": syk_near_70m / total_70m,
            "per_layer_mean": syk_near_70m / 6,
            "syk_near_heads": [h for h in rand_heads_70m if h["syk_near"]],
        },
        "pythia_410m_step512": result_410m,
    }

    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
