"""
exp-088 — RAND Early Burst Scaling: Pythia-160m Third Scale Point

Pre-registration: notes.md (commit 4f4549c2, 2026-07-20).

Adds Pythia-160m (12L × 12H = 144 heads) at step512 as the intermediate scale
point between Pythia-70m (6L × 8H = 48 heads) and Pythia-410m (24L × 16H = 384 heads).

Protocol identical to exp-087 / exp-086.

Ariel — July 20, 2026.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ─── constants (same as exp-087, pre-registered) ─────────────────────────────

SEQ_LEN = 256
N_RAND = 20
RAND_SEED = 87

FIT_LOW = 3
FIT_HIGH = 50
R2_THRESHOLD = 0.85
SYK_LOW = 0.20
SYK_HIGH = 0.30

# Pythia-160m architecture
N_LAYERS = 12
N_HEADS = 12
VOCAB_SIZE = 50304

OUT = Path(__file__).resolve().parent


# ─── measurement helpers (identical to exp-087) ──────────────────────────────

def compute_attention_decay(attn_head: np.ndarray, max_dx: int = FIT_HIGH + 5) -> np.ndarray:
    """Average A[i, i-dx] over positions i."""
    return np.array([
        np.diag(attn_head, -dx).mean() if dx < attn_head.shape[0] else 0.0
        for dx in range(max_dx)
    ])


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float | None, float | None]:
    """Fit power-law A ~ dx^{-2Δ}. Returns (Δ, R²) or (None, None)."""
    mask = (dx_arr >= FIT_LOW) & (dx_arr < FIT_HIGH) & (y_arr > 1e-20)
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


def main():
    t_start = time.time()
    print("=" * 60)
    print("exp-088: RAND Early Burst Scaling — Pythia-160m")
    print("Pre-registration: 4f4549c2")
    print("=" * 60)

    # ─── Load existing 70m and 410m data ────────────────────────────────────

    exp087_path = OUT.parent / "exp-087_rand_burst_scaling" / "results.json"
    with open(exp087_path) as f:
        exp087 = json.load(f)

    data_70m = exp087["pythia_70m_step512"]
    data_410m = exp087["pythia_410m_step512"]

    syk_70m = data_70m["syk_near_count"]
    total_70m = data_70m["total_heads"]
    syk_410m = data_410m["syk_near_count"]
    total_410m = data_410m["total_heads"]

    print(f"\nExisting data (from exp-087):")
    print(f"  Pythia-70m  step512 RAND: {syk_70m}/{total_70m} = {syk_70m/total_70m:.4f}")
    print(f"  Pythia-410m step512 RAND: {syk_410m}/{total_410m} = {syk_410m/total_410m:.4f}")
    print(f"  Scaling exponent (two-point): N^{np.log(syk_410m/syk_70m)/np.log(total_410m/total_70m):.3f}")

    # ─── Predictions ────────────────────────────────────────────────────────

    total_160m = N_LAYERS * N_HEADS  # 144
    alpha = np.log(syk_410m / syk_70m) / np.log(total_410m / total_70m)
    predicted_count = syk_70m * (total_160m / total_70m) ** alpha
    print(f"\nPredictions for Pythia-160m ({total_160m} heads):")
    print(f"  N^{alpha:.3f} scaling prediction: {predicted_count:.1f} heads (range: 11–19)")
    print(f"  Structural decomposition: ~9 (L1-L4) + ~5 (L5-L11) = ~14 heads")

    # ─── Load Pythia-160m at step512 ────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Loading EleutherAI/pythia-160m @ step512")
    print(f"  Architecture: {N_LAYERS}L × {N_HEADS}H, total={N_LAYERS*N_HEADS} heads, d_k=64")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        revision="step512",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ─── Measure ────────────────────────────────────────────────────────────

    rng = np.random.default_rng(RAND_SEED)
    dx_arr = np.arange(FIT_HIGH + 5)

    head_profiles = {
        (l, h): np.zeros(FIT_HIGH + 5)
        for l in range(N_LAYERS)
        for h in range(N_HEADS)
    }

    with torch.no_grad():
        for i in range(N_RAND):
            tokens = torch.tensor(
                rng.integers(0, VOCAB_SIZE, size=(1, SEQ_LEN)), dtype=torch.long
            )
            out = model(tokens, output_attentions=True)
            for layer_idx, attn in enumerate(out.attentions):
                attn_np = attn[0].cpu().numpy()
                for head_idx in range(N_HEADS):
                    decay = compute_attention_decay(attn_np[head_idx], max_dx=FIT_HIGH + 5)
                    head_profiles[(layer_idx, head_idx)] += decay / N_RAND

    print(f"  Measured {N_RAND} random sequences")

    # ─── Fit and classify ────────────────────────────────────────────────────

    results = []
    syk_near_count = 0
    per_layer_syk: dict[int, int] = {l: 0 for l in range(N_LAYERS)}

    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            profile = head_profiles[(layer_idx, head_idx)]
            delta, r2 = fit_power_law(dx_arr, profile)
            if delta is None:
                conformal = syk_near = False
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

    fraction_160m = syk_near_count / total_160m

    # ─── Report ─────────────────────────────────────────────────────────────

    print(f"\nResults @ step512 RAND:")
    print(f"  SYK-near: {syk_near_count}/{total_160m} = {fraction_160m:.4f}")
    print(f"  Per-layer SYK-near mean: {syk_near_count/N_LAYERS:.2f}")
    print(f"\n  SYK-near heads:")
    for r in results:
        if r["syk_near"]:
            print(f"    L{r['layer']}H{r['head']}: delta={r['delta']:.3f}, R2={r['R2']:.3f}")

    print(f"\n  Per-layer SYK-near count:")
    for l in range(N_LAYERS):
        if per_layer_syk[l] > 0:
            print(f"    L{l}: {per_layer_syk[l]}")

    # L1–L4 structural zone
    l1_l4_count = sum(per_layer_syk[l] for l in range(1, 5))
    deep_count = sum(per_layer_syk[l] for l in range(5, N_LAYERS))
    print(f"\n  L1-L4 structural zone: {l1_l4_count}")
    print(f"  L5-L11 background zone: {deep_count}")

    # Structural head check
    l4h3 = next((r for r in results if r["layer"] == 4 and r["head"] == 3), None)
    l4h7 = next((r for r in results if r["layer"] == 4 and r["head"] == 7), None)
    print(f"\n  L4H3 (structural?): delta={l4h3['delta']:.3f}, R2={l4h3['R2']:.3f}, syk_near={l4h3['syk_near']}")
    print(f"  L4H7 (structural?): delta={l4h7['delta']:.3f}, R2={l4h7['R2']:.3f}, syk_near={l4h7['syk_near']}")

    # ─── Three-model scaling table ───────────────────────────────────────────

    print(f"\n{'='*60}")
    print("THREE-MODEL SCALING TABLE")
    print("=" * 60)
    header = f"{'Model':<20} {'Total heads':<14} {'SYK-near':<11} {'Fraction':<12} {'Per-layer mean'}"
    print(header)
    print(f"{'Pythia-70m':<20} {total_70m:<14} {syk_70m:<11} {syk_70m/total_70m:.4f}{'':8} {syk_70m/6:.2f}")
    print(f"{'Pythia-160m':<20} {total_160m:<14} {syk_near_count:<11} {fraction_160m:.4f}{'':8} {syk_near_count/12:.2f}")
    print(f"{'Pythia-410m':<20} {total_410m:<14} {syk_410m:<11} {syk_410m/total_410m:.4f}{'':8} {syk_410m/24:.2f}")

    # Three-point scaling fit
    x = np.log([total_70m, total_160m, total_410m])
    y = np.log([syk_70m, syk_near_count, syk_410m])
    A = np.column_stack([np.ones_like(x), x])
    coeffs_3pt, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    alpha_3pt = coeffs_3pt[1]
    print(f"\n  Three-point scaling exponent: N^{alpha_3pt:.3f}")
    print(f"  (Two-point from exp-087 was: N^{alpha:.3f})")

    # ─── Verdict ────────────────────────────────────────────────────────────

    print("\nPRE-REGISTERED VERDICTS:")
    if 11 <= syk_near_count <= 19:
        print(f"  H_160m_consistent: CONFIRMED ({syk_near_count} in 11–19 range)")
    elif syk_near_count > 19:
        print(f"  H_160m_deviant_high: CONFIRMED ({syk_near_count} > 19)")
    else:
        print(f"  H_160m_deviant_low: CONFIRMED ({syk_near_count} < 11)")

    l4h3_syk = l4h3["syk_near"] if l4h3 else False
    l4h7_syk = l4h7["syk_near"] if l4h7 else False
    if l4h3_syk and l4h7_syk:
        print("  H_structural_confirmed: CONFIRMED (both L4H3 and L4H7 SYK-near)")
    elif l4h3_syk or l4h7_syk:
        which = "L4H3" if l4h3_syk else "L4H7"
        print(f"  H_structural_partial: only {which} SYK-near")
    else:
        print("  H_structural_falsified: neither L4H3 nor L4H7 SYK-near")

    if 7 <= l1_l4_count <= 11:
        print(f"  H_structural_zone: CONFIRMED (L1-L4 count = {l1_l4_count}, in 7–11)")
    else:
        print(f"  H_structural_zone_fail: L1-L4 count = {l1_l4_count}, outside 7–11")

    # ─── Save ────────────────────────────────────────────────────────────────

    output = {
        "experiment": "exp-088",
        "date": "2026-07-20",
        "prereg_commit": "4f4549c2",
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
            "source": "exp-087 results.json",
            "total_heads": total_70m,
            "syk_near_count": syk_70m,
            "syk_near_fraction": syk_70m / total_70m,
        },
        "pythia_160m_step512": {
            "model": "EleutherAI/pythia-160m",
            "revision": "step512",
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "total_heads": total_160m,
            "syk_near_count": syk_near_count,
            "syk_near_fraction": fraction_160m,
            "per_layer_syk_near": per_layer_syk,
            "l1_l4_zone_count": l1_l4_count,
            "l5_plus_zone_count": deep_count,
            "l4h3": l4h3,
            "l4h7": l4h7,
            "per_head_results": results,
        },
        "pythia_410m_step512": {
            "source": "exp-087 results.json",
            "total_heads": total_410m,
            "syk_near_count": syk_410m,
            "syk_near_fraction": syk_410m / total_410m,
        },
        "scaling": {
            "two_point_exponent_exp087": float(alpha),
            "three_point_exponent": float(alpha_3pt),
            "predicted_160m_count_n_alpha": float(predicted_count),
        },
    }

    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
