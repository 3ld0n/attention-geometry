"""
exp-100 — W_QK Effective Rank Measurement (Modal cloud, CPU only).

Pre-registration: notes.md (committed to 3ld0n/attention-geometry before this script ran).

Reads model.safetensors from existing Modal volumes (exp097, exp096, exp098, exp099),
extracts per-head W_K matrices, computes SVD-based effective rank, saves results JSON.

No new training. Cost: <$0.10 (CPU seconds per corpus, no GPU needed).

Usage (from repo root):
    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-100_wqk_rank/modal_exp100.py

Ariel — August 5, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# ─── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("exp100-wqk-rank")

# Source volumes (existing — no create_if_missing needed but required by API)
vol_alien     = modal.Volume.from_name("exp097-alien-data")
vol_anon      = modal.Volume.from_name("exp096-anon-data")
vol_realnames = modal.Volume.from_name("exp098-realnames-data")
vol_rich      = modal.Volume.from_name("exp099-rich-data")

# Results volume (new)
vol_results   = modal.Volume.from_name("exp100-rank-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "safetensors==0.5.3",
        "scipy==1.17.1",
    )
)

CHECKPOINTS = {
    "C-alien":           "/data097/runs/run_alien_s0/step_2000/model.safetensors",
    "C-NAT-anon":        "/data096/runs/run_anon_s0/step_2000/model.safetensors",
    "C-alien-realnames": "/data098/runs/run_realnames_s0/step_2000/model.safetensors",
    "C-alien-rich":      "/data099/runs/run_rich_s0/step_2000/model.safetensors",
}

N_LAYERS = 6
N_HEADS  = 8
D_K      = 64   # head dimension
D_MODEL  = 512


def compute_rank_metrics(weight_slice_np):
    """
    weight_slice_np: ndarray of shape [D_K, D_MODEL]
    Returns dict with effective rank (participation ratio) and stable rank.
    """
    import numpy as np
    from numpy.linalg import svd
    sigma = svd(weight_slice_np, compute_uv=False)  # sorted descending
    sigma_sq = sigma ** 2
    s1 = float(sigma[0])
    # Participation ratio: (sum sigma_i)^2 / sum sigma_i^2
    denom = float(np.sum(sigma_sq))
    if denom < 1e-15:
        return {"r_eff": 0.0, "r_stable": 0.0, "sigma_max": s1,
                "sigma_min": float(sigma[-1]), "nuclear_norm": 0.0}
    r_eff    = float(np.sum(sigma) ** 2) / denom
    r_stable = denom / (s1 ** 2)
    return {
        "r_eff":        r_eff,
        "r_stable":     r_stable,
        "sigma_max":    s1,
        "sigma_min":    float(sigma[-1]),
        "nuclear_norm": float(np.sum(sigma)),
        "frob_sq":      denom,
    }


@app.function(
    image=image,
    volumes={
        "/data097": vol_alien,
        "/data096": vol_anon,
        "/data098": vol_realnames,
        "/data099": vol_rich,
        "/results": vol_results,
    },
    timeout=300,
)
def measure_ranks() -> dict:
    import json
    import numpy as np
    from safetensors import safe_open

    results = {}

    for corpus_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n[{corpus_name}] Loading {ckpt_path} ...")
        path = Path(ckpt_path)
        if not path.exists():
            print(f"  WARNING: checkpoint not found at {ckpt_path}")
            results[corpus_name] = {"error": f"not found: {ckpt_path}"}
            continue

        with safe_open(str(path), framework="numpy") as f:
            all_keys = list(f.keys())
            # Find QKV weight keys
            qkv_keys = [k for k in all_keys if "query_key_value.weight" in k]
            print(f"  QKV keys: {qkv_keys}")

            head_results = []
            for layer in range(N_LAYERS):
                key = f"gpt_neox.layers.{layer}.attention.query_key_value.weight"
                if key not in all_keys:
                    print(f"  Layer {layer}: key not found ({key})")
                    continue

                qkv_weight = f.get_tensor(key)  # shape [3*512, 512] = [1536, 512]
                # Extract K slice: rows [D_MODEL : 2*D_MODEL]
                W_K = qkv_weight[D_MODEL : 2 * D_MODEL, :]  # [512, 512]

                for head in range(N_HEADS):
                    W_K_h = W_K[head * D_K : (head + 1) * D_K, :]  # [64, 512]
                    metrics = compute_rank_metrics(W_K_h)
                    head_results.append({
                        "layer": layer,
                        "head":  head,
                        **metrics,
                    })

        r_effs    = [h["r_eff"]    for h in head_results]
        r_stables = [h["r_stable"] for h in head_results]

        # Per-layer breakdown
        layer_stats = {}
        for layer in range(N_LAYERS):
            layer_heads = [h for h in head_results if h["layer"] == layer]
            if layer_heads:
                layer_r_effs = [h["r_eff"] for h in layer_heads]
                layer_stats[f"L{layer}"] = {
                    "mean_r_eff":   float(np.mean(layer_r_effs)),
                    "median_r_eff": float(np.median(layer_r_effs)),
                }

        results[corpus_name] = {
            "n_heads":        len(head_results),
            "mean_r_eff":     float(np.mean(r_effs)),
            "median_r_eff":   float(np.median(r_effs)),
            "mean_r_stable":  float(np.mean(r_stables)),
            "per_layer":      layer_stats,
            "heads":          head_results,
        }
        print(f"  mean_r_eff={results[corpus_name]['mean_r_eff']:.3f}  "
              f"median_r_eff={results[corpus_name]['median_r_eff']:.3f}")

    # Compute cross-corpus ratios
    comparisons = {}
    if "C-alien" in results and "C-alien-rich" in results:
        r_alien = results["C-alien"].get("mean_r_eff", 0)
        r_rich  = results["C-alien-rich"].get("mean_r_eff", 0)
        if r_alien > 0:
            comparisons["rich_over_alien_r_eff"] = r_rich / r_alien
    if "C-alien" in results and "C-NAT-anon" in results:
        r_alien = results["C-alien"].get("mean_r_eff", 0)
        r_anon  = results["C-NAT-anon"].get("mean_r_eff", 0)
        if r_alien > 0:
            comparisons["anon_over_alien_r_eff"] = r_anon / r_alien

    output = {
        "experiment": "exp-100",
        "timestamp":  __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "protocol": {
            "d_k": D_K, "d_model": D_MODEL, "n_layers": N_LAYERS,
            "n_heads": N_HEADS, "rank_metric": "participation_ratio",
            "checkpoint_seed": "s0 (first seed per corpus)",
        },
        "results":     results,
        "comparisons": comparisons,
    }

    out_path = "/results/measurements/rank_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    vol_results.commit()
    print(f"\nSaved to {out_path}")

    # Print summary for logs
    print("\n=== SUMMARY ===")
    for corpus, data in results.items():
        if "error" not in data:
            print(f"  {corpus}: mean_r_eff={data['mean_r_eff']:.3f}  median={data['median_r_eff']:.3f}")
    for k, v in comparisons.items():
        print(f"  {k}: {v:.3f}  (S-ratio prediction: 4.0)")

    return output


@app.local_entrypoint()
def main():
    result = measure_ranks.remote()
    print("\nDone.")
