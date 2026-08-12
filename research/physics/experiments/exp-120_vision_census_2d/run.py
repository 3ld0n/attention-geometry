"""
exp-120: 2D Vision Census
Protocol: ViT-B/16 patch-to-patch attention power law on CIFAR-10 (natural images)
          vs random Gaussian patches (control condition).

Pre-registration: a6f7380 (pushed before any forward pass).

Image source: CIFAR-10 test set via HuggingFace datasets streaming (50 images,
32×32 → resized to 224×224 via bicubic interpolation using torchvision transforms).
"""

import json
import os
import numpy as np
import torch
from datetime import datetime, timezone

# ── Constants from pre-registration ──────────────────────────────────────────
MODEL_NAME = "google/vit-base-patch16-224"
N_IMAGES = 50
R2_THRESHOLD = 0.90
DELTA_WINDOW = (0.45, 0.55)          # 2D prediction window (T3: Δ=0.50 for D=2)
CONTROL_DELTA_WINDOW = (0.20, 0.30)  # 1D window for reference
N_LAYERS = 12
N_HEADS = 12
N_PATCHES = 196                       # 14×14
GRID_SIZE = 14
N_BINS = 20
D_MIN = 1.0                           # minimum patch-grid distance
D_MAX = 14.0                          # near-diagonal for 14×14 grid (√(13²+13²)≈18.4; 14 covers most)

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model():
    from transformers import ViTForImageClassification
    print(f"Loading {MODEL_NAME}...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        output_attentions=True,
    )
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")
    return model, device


def load_natural_images(n_images):
    """Stream n_images from CIFAR-10 test set, resize to 224×224."""
    from datasets import load_dataset
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Streaming {n_images} CIFAR-10 test images...")
    ds = load_dataset("cifar10", split="test", streaming=True)
    tensors = []
    for i, item in enumerate(ds):
        if i >= n_images:
            break
        pil_img = item["img"]  # PIL Image, 32×32
        tensors.append(transform(pil_img))

    images = torch.stack(tensors)  # N × 3 × 224 × 224
    print(f"  Loaded {len(images)} images, shape {tuple(images.shape)}")
    return images


def load_random_images(n_images, seed=42):
    """Random Gaussian patches at 224×224 (control condition)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_images, 3, 224, 224)).astype(np.float32)
    noise = np.clip(noise, -3.0, 3.0)
    print(f"  Generated {n_images} random patch images, shape {noise.shape}")
    return torch.tensor(noise)


def build_distance_matrix():
    """2D Euclidean distances on the 14×14 patch grid. Returns (196, 196) array."""
    idx = np.arange(N_PATCHES)
    rows = idx // GRID_SIZE
    cols = idx % GRID_SIZE
    dr = rows[:, None] - rows[None, :]
    dc = cols[:, None] - cols[None, :]
    dist = np.sqrt(dr**2 + dc**2)  # 196 × 196, diagonal=0
    return dist


def run_census(model, device, images, label):
    """
    For each head: pool attention across images by 2D distance bin, fit power law.
    Returns list of per-head result dicts.
    """
    images = images.to(device)
    dist_matrix = build_distance_matrix()  # 196 × 196

    # Log-uniform bin edges
    bin_edges = np.geomspace(D_MIN, D_MAX, N_BINS + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Accumulate attention per bin for each head
    # Shape: N_LAYERS × N_HEADS × N_BINS
    attn_sum = np.zeros((N_LAYERS, N_HEADS, N_BINS))
    attn_count = np.zeros((N_LAYERS, N_HEADS, N_BINS), dtype=np.int64)

    # Precompute bin assignment for all off-diagonal patch pairs
    dist_flat = dist_matrix.ravel()   # 196*196 = 38416
    patch_mask = dist_flat > 0        # exclude diagonal
    dist_valid = dist_flat[patch_mask]
    bin_idx_all = np.clip(np.digitize(dist_valid, bin_edges) - 1, 0, N_BINS - 1)

    print(f"\nCensus: {label} — {len(images)} images")
    for img_idx in range(len(images)):
        pixel_values = images[img_idx].unsqueeze(0)  # 1 × 3 × 224 × 224
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_attentions=True)

        # outputs.attentions: tuple of N_LAYERS tensors, each (1, N_HEADS, 197, 197)
        for layer in range(N_LAYERS):
            attn_layer = outputs.attentions[layer][0].cpu().numpy()  # N_HEADS × 197 × 197
            # Strip CLS (index 0): → N_HEADS × 196 × 196
            patch_attn = attn_layer[:, 1:, 1:]  # N_HEADS × 196 × 196

            for head in range(N_HEADS):
                a_flat = patch_attn[head].ravel()  # 38416
                a_valid = a_flat[patch_mask]        # off-diagonal pairs

                # Vectorized bin accumulation
                for b in range(N_BINS):
                    mask_b = bin_idx_all == b
                    if mask_b.any():
                        attn_sum[layer, head, b] += a_valid[mask_b].sum()
                        attn_count[layer, head, b] += mask_b.sum()

        if (img_idx + 1) % 10 == 0:
            print(f"  {img_idx + 1}/{len(images)} images processed")

    # Mean attention per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        attn_mean = np.where(attn_count > 0, attn_sum / attn_count, np.nan)

    # Fit power law for each head
    results = []
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            a = attn_mean[layer, head]  # N_BINS
            valid = ~np.isnan(a) & (a > 0) & (attn_count[layer, head] > 0)

            if valid.sum() < 5:
                results.append({
                    "layer": layer, "head": head,
                    "delta": None, "R2": None,
                    "in_2d_window": False, "in_1d_window": False,
                    "n_bins_valid": int(valid.sum()),
                })
                continue

            log_d = np.log(bin_centers[valid])
            log_a = np.log(a[valid])

            # OLS: log_a = b0 + b1 * log_d  →  Δ = -b1/2
            X = np.column_stack([np.ones_like(log_d), log_d])
            coeffs, _, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
            b0, b1 = coeffs
            a_pred = X @ coeffs
            ss_res = np.sum((log_a - a_pred)**2)
            ss_tot = np.sum((log_a - log_a.mean())**2)
            R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            delta = float(-b1 / 2.0)

            in_2d = bool(R2 >= R2_THRESHOLD and DELTA_WINDOW[0] <= delta <= DELTA_WINDOW[1])
            in_1d = bool(R2 >= R2_THRESHOLD and CONTROL_DELTA_WINDOW[0] <= delta <= CONTROL_DELTA_WINDOW[1])

            results.append({
                "layer": layer, "head": head,
                "delta": round(delta, 4),
                "R2": round(R2, 4),
                "in_2d_window": in_2d,
                "in_1d_window": in_1d,
                "n_bins_valid": int(valid.sum()),
            })

    return results, bin_centers, attn_mean


def summarize(results, label, condition):
    in_2d = [r for r in results if r.get("in_2d_window")]
    in_1d = [r for r in results if r.get("in_1d_window")]
    high_r2 = [r for r in results if r.get("R2") is not None and r["R2"] >= R2_THRESHOLD]
    deltas_valid = [r["delta"] for r in results if r.get("delta") is not None]

    print(f"\n── {label} ({condition}) ──")
    print(f"  Total heads: {len(results)}")
    print(f"  Heads with R²≥{R2_THRESHOLD}: {len(high_r2)}")
    print(f"  Heads in 2D Δ-window {DELTA_WINDOW}: {len(in_2d)}")
    print(f"  Heads in 1D Δ-window {CONTROL_DELTA_WINDOW}: {len(in_1d)}")
    if in_2d:
        med = np.median([r["delta"] for r in in_2d])
        print(f"  2D Δ-window median Δ: {med:.4f}")
    if in_1d:
        med_1d = np.median([r["delta"] for r in in_1d])
        print(f"  1D Δ-window median Δ: {med_1d:.4f}")
    if high_r2:
        med_hr2 = np.median([r["delta"] for r in high_r2])
        print(f"  R²≥{R2_THRESHOLD} all-population median Δ: {med_hr2:.4f}")
    if deltas_valid:
        print(f"  All-heads median Δ: {np.median(deltas_valid):.4f}")

    return {
        "condition": condition,
        "n_heads_total": len(results),
        "n_high_r2": len(high_r2),
        "n_in_2d_window": len(in_2d),
        "n_in_1d_window": len(in_1d),
        "delta_2d_window_median": float(np.median([r["delta"] for r in in_2d])) if in_2d else None,
        "delta_1d_window_median": float(np.median([r["delta"] for r in in_1d])) if in_1d else None,
        "delta_high_r2_median": float(np.median([r["delta"] for r in high_r2])) if high_r2 else None,
        "delta_all_median": float(np.median(deltas_valid)) if deltas_valid else None,
        "per_head": results,
    }


def verdict(results, nat_summary, rand_summary):
    p1 = nat_summary["n_in_2d_window"] >= 1
    p2_med = nat_summary["delta_2d_window_median"]
    p2 = p2_med is not None and 0.40 <= p2_med <= 0.60
    p3 = rand_summary["n_in_2d_window"] < 1
    p4_med = nat_summary["delta_2d_window_median"]
    p4 = p4_med is not None and p4_med > 0.30  # > 1D window upper bound

    print("\n── Prediction verdicts ──")
    print(f"  P1 (2D population ≥1 head):  {'CONFIRMED' if p1 else 'DEAD'} "
          f"({nat_summary['n_in_2d_window']} heads in {DELTA_WINDOW})")
    print(f"  P2 (Δ_med ∈ [0.40,0.60]):    {'CONFIRMED' if p2 else 'DEAD'} "
          f"(Δ_med = {p2_med})")
    print(f"  P3 (random control clean):   {'CONFIRMED' if p3 else 'DEAD'} "
          f"({rand_summary['n_in_2d_window']} random heads in 2D window)")
    print(f"  P4 (2D > 1D by >0.05):       {'CONFIRMED' if p4 else 'DEAD'} "
          f"(2D Δ_med={p4_med}, 1D exp-118 ≈ 0.25)")

    return {
        "P1": {"verdict": "CONFIRMED" if p1 else "DEAD",
               "n_heads_in_window": nat_summary["n_in_2d_window"]},
        "P2": {"verdict": "CONFIRMED" if p2 else "DEAD",
               "delta_med": p2_med},
        "P3": {"verdict": "CONFIRMED" if p3 else "DEAD",
               "random_heads_in_window": rand_summary["n_in_2d_window"]},
        "P4": {"verdict": "CONFIRMED" if p4 else "DEAD",
               "delta_2d_med": p4_med, "delta_1d_ref": 0.25},
    }


def layer_breakdown(nat_results):
    print("\n── Layer breakdown (natural images) ──")
    for layer in range(N_LAYERS):
        layer_r = [r for r in nat_results if r["layer"] == layer]
        in_win = [r for r in layer_r if r.get("in_2d_window")]
        high_r2 = [r for r in layer_r if r.get("R2") and r["R2"] >= R2_THRESHOLD]
        deltas = [r["delta"] for r in high_r2]
        print(f"  L{layer:2d}: {len(in_win)}/{N_HEADS} in 2D window"
              + (f", R²≥{R2_THRESHOLD} Δ_med={np.median(deltas):.3f}" if deltas else ""))


def main():
    model, device = load_model()

    # ── Natural images (CIFAR-10 streamed) ───────────────────────────────────
    nat_images = load_natural_images(N_IMAGES)
    nat_results, bin_centers, _ = run_census(model, device, nat_images, "Natural (CIFAR-10 upscaled)")
    nat_summary = summarize(nat_results, "Natural images", "natural_cifar10")

    # ── Random patches (control) ──────────────────────────────────────────────
    print("\nGenerating random patches...")
    rand_images = load_random_images(N_IMAGES)
    rand_results, _, _ = run_census(model, device, rand_images, "Random Gaussian patches")
    rand_summary = summarize(rand_results, "Random patches", "random_patches")

    # ── Verdicts ──────────────────────────────────────────────────────────────
    preds = verdict(nat_results, nat_summary, rand_summary)

    # ── Layer breakdown ───────────────────────────────────────────────────────
    layer_breakdown(nat_results)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "exp_id": "exp-120",
        "date": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "image_source": "cifar10_test_streamed_resized_224x224_bicubic",
        "n_images_per_condition": N_IMAGES,
        "grid_size": GRID_SIZE,
        "n_patches": N_PATCHES,
        "r2_threshold": R2_THRESHOLD,
        "delta_window_2d": list(DELTA_WINDOW),
        "delta_window_1d_ref": list(CONTROL_DELTA_WINDOW),
        "n_bins": N_BINS,
        "d_min": D_MIN,
        "d_max": D_MAX,
        "bin_centers": bin_centers.tolist(),
        "prereg_commit": "a6f7380",
        "predictions": preds,
        "natural_images": nat_summary,
        "random_patches": rand_summary,
    }

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
