"""
exp-101 — Key Gram Matrix Effective Rank (Modal cloud, CPU only).

Pre-registration: notes.md (committed to 3ld0n/attention-geometry before this script ran).

For each corpus: samples N_TOKENS token IDs from the corpus .bin file, extracts
token embeddings from the trained checkpoint, projects by W_K per head (6 layers ×
8 heads), computes SVD-based effective rank of the resulting key matrix.

Tests whether key gram matrix rank scales with world complexity S —
the correct operationalization of the Kim–Cao–Altman low-rank SYK mechanism
(following exp-100's falsification of the weight-level rank hypothesis).

No new training. All checkpoints exist on Modal volumes from prior experiments.
Cost: <$0.10 (CPU seconds, no GPU).

Usage (from repo root):
    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-101_kernel_gram_rank/modal_exp101.py

Ariel — August 6, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# ─── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("exp101-kernel-gram-rank")

vol_alien     = modal.Volume.from_name("exp097-alien-data")
vol_anon      = modal.Volume.from_name("exp096-anon-data")
vol_realnames = modal.Volume.from_name("exp098-realnames-data")
vol_rich      = modal.Volume.from_name("exp099-rich-data")
vol_results   = modal.Volume.from_name("exp101-kern-rank-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "safetensors==0.5.3",
        "scipy==1.17.1",
    )
)

# ─── Experiment constants (pre-registered in notes.md) ─────────────────────────

N_TOKENS  = 4096   # tokens sampled per corpus (strided, not first-N)
N_LAYERS  = 6
N_HEADS   = 8
D_K       = 64
D_MODEL   = 512

CORPORA = {
    "C-alien":           {
        "corpus": "/data097/C-alien.bin",
        "ckpt":   "/data097/runs/run_alien_s0/step_2000/model.safetensors",
    },
    "C-alien-realnames": {
        "corpus": "/data098/C-alien-realnames.bin",
        "ckpt":   "/data098/runs/run_realnames_s0/step_2000/model.safetensors",
    },
    "C-alien-rich":      {
        "corpus": "/data099/C-alien-rich.bin",
        "ckpt":   "/data099/runs/run_rich_s0/step_2000/model.safetensors",
    },
    "C-NAT-anon":        {
        "corpus": "/data096/C-NAT-anon.bin",
        "ckpt":   "/data096/runs/run_anon_s0/step_2000/model.safetensors",
    },
}


def compute_key_gram_rank(token_ids_np, embed_weight_np, wqkv_per_layer):
    """
    Compute key gram matrix effective rank per head, for all layers.

    Uses layer-0 token embeddings (h_a = embed_in[token_id]) as the fixed
    input. Applies W_K from each layer in turn. This isolates how each layer's
    W_K projects the token distribution, without confounding from hidden-state
    evolution (no forward pass needed).

    Args:
        token_ids_np: int array, shape [N]
        embed_weight_np: float32 array, shape [vocab_size, D_MODEL]
        wqkv_per_layer: list of float32 arrays [3*D_MODEL, D_MODEL] per layer

    Returns:
        list of dicts, one per (layer, head): {layer, head, r_eff, r_stable,
        sigma_max, sigma_min, sigma_sum, sigma_sq_sum}
    """
    import numpy as np

    # Token embeddings at layer 0 (fixed for all layers' W_K)
    h = embed_weight_np[token_ids_np]   # [N, D_MODEL]

    results = []
    for layer_idx, wqkv in enumerate(wqkv_per_layer):
        # Extract W_K: rows [D_MODEL : 2*D_MODEL], shape [D_MODEL, D_MODEL]
        W_K_full = wqkv[D_MODEL : 2 * D_MODEL, :]   # [512, 512]

        for head in range(N_HEADS):
            # Per-head W_K: shape [D_K=64, D_MODEL=512]
            W_K_h = W_K_full[head * D_K : (head + 1) * D_K, :]

            # Key matrix: A[a] = h[a] @ W_K_h.T,  shape [N, D_K=64]
            A = h @ W_K_h.T   # [N, 64]

            # SVD of A (shape [N, 64]): we only need singular values
            # min(N, D_K) = 64 singular values
            sigma = np.linalg.svd(A, compute_uv=False)   # [64], sorted descending

            sigma_sq  = sigma ** 2
            s_sum     = float(np.sum(sigma))
            s_sq_sum  = float(np.sum(sigma_sq))
            s1        = float(sigma[0])

            r_eff    = (s_sum ** 2) / s_sq_sum if s_sq_sum > 1e-15 else 0.0
            r_stable = s_sq_sum / (s1 ** 2)    if s1      > 1e-15 else 0.0

            results.append({
                "layer":       layer_idx,
                "head":        head,
                "r_eff":       r_eff,
                "r_stable":    r_stable,
                "sigma_max":   s1,
                "sigma_min":   float(sigma[-1]),
                "sigma_sum":   s_sum,
                "sigma_sq_sum": s_sq_sum,
            })

    return results


@app.function(
    image=image,
    volumes={
        "/data097": vol_alien,
        "/data096": vol_anon,
        "/data098": vol_realnames,
        "/data099": vol_rich,
        "/results": vol_results,
    },
    timeout=600,
    cpu=4,
    memory=8192,
)
def measure_kern_ranks() -> dict:
    import json
    import numpy as np
    from safetensors import safe_open

    all_results = {}

    for corpus_name, paths in CORPORA.items():
        print(f"\n{'='*60}")
        print(f"[{corpus_name}]")
        corpus_path = Path(paths["corpus"])
        ckpt_path   = Path(paths["ckpt"])

        # ── Load corpus tokens (strided sample) ───────────────────────────────
        if not corpus_path.exists():
            print(f"  SKIP — corpus not found: {corpus_path}")
            all_results[corpus_name] = {"error": f"corpus not found: {corpus_path}"}
            continue

        corpus = np.fromfile(str(corpus_path), dtype=np.uint16)
        corpus_len = len(corpus)
        stride = max(1, corpus_len // N_TOKENS)
        indices = np.arange(0, corpus_len, stride)[:N_TOKENS]
        token_ids = corpus[indices].astype(np.int64)
        print(f"  corpus: {corpus_len:,} tokens  stride={stride}  "
              f"sampled={len(token_ids)}")

        # ── Load model weights ─────────────────────────────────────────────────
        if not ckpt_path.exists():
            print(f"  SKIP — checkpoint not found: {ckpt_path}")
            all_results[corpus_name] = {"error": f"ckpt not found: {ckpt_path}"}
            continue

        with safe_open(str(ckpt_path), framework="numpy") as f:
            all_keys = list(f.keys())

            # Token embedding table
            embed_key = "gpt_neox.embed_in.weight"
            if embed_key not in all_keys:
                print(f"  SKIP — embed_in.weight not found")
                all_results[corpus_name] = {"error": "embed_in.weight not found"}
                continue
            embed_weight = f.get_tensor(embed_key)   # [vocab_size, D_MODEL]
            print(f"  embedding: {embed_weight.shape}  "
                  f"vocab_size={embed_weight.shape[0]}")

            # QKV weights per layer
            wqkv_per_layer = []
            for layer in range(N_LAYERS):
                key = f"gpt_neox.layers.{layer}.attention.query_key_value.weight"
                if key not in all_keys:
                    print(f"  WARNING: layer {layer} QKV key not found")
                    wqkv_per_layer.append(None)
                else:
                    wqkv_per_layer.append(f.get_tensor(key))   # [3*512, 512]

        # ── Compute key gram matrix ranks ──────────────────────────────────────
        head_results = compute_key_gram_rank(token_ids, embed_weight, wqkv_per_layer)

        valid = [h for h in head_results if "r_eff" in h]
        r_effs    = [h["r_eff"]    for h in valid]
        r_stables = [h["r_stable"] for h in valid]

        # Per-layer breakdown
        layer_stats = {}
        for layer in range(N_LAYERS):
            layer_heads = [h for h in valid if h["layer"] == layer]
            if layer_heads:
                layer_r_effs    = [h["r_eff"]    for h in layer_heads]
                layer_r_stables = [h["r_stable"] for h in layer_heads]
                layer_stats[f"L{layer}"] = {
                    "mean_r_eff":    float(np.mean(layer_r_effs)),
                    "median_r_eff":  float(np.median(layer_r_effs)),
                    "mean_r_stable": float(np.mean(layer_r_stables)),
                    "min_r_eff":     float(np.min(layer_r_effs)),
                    "max_r_eff":     float(np.max(layer_r_effs)),
                }

        all_results[corpus_name] = {
            "n_tokens_sampled": len(token_ids),
            "corpus_len":       int(corpus_len),
            "stride":           int(stride),
            "n_heads":          len(valid),
            "mean_r_eff":       float(np.mean(r_effs)),
            "median_r_eff":     float(np.median(r_effs)),
            "mean_r_stable":    float(np.mean(r_stables)),
            "median_r_stable":  float(np.median(r_stables)),
            "min_r_eff":        float(np.min(r_effs)),
            "max_r_eff":        float(np.max(r_effs)),
            "per_layer":        layer_stats,
            "heads":            valid,
        }
        print(f"  mean_r_eff={all_results[corpus_name]['mean_r_eff']:.3f}  "
              f"median={all_results[corpus_name]['median_r_eff']:.3f}  "
              f"mean_r_stable={all_results[corpus_name]['mean_r_stable']:.3f}")

    # ── Cross-corpus analysis ──────────────────────────────────────────────────
    comparisons = {}
    def safe_ratio(a_key, b_key, metric="mean_r_eff"):
        a = all_results.get(a_key, {}).get(metric)
        b = all_results.get(b_key, {}).get(metric)
        if a and b and a > 0:
            return round(b / a, 4)
        return None

    comparisons["rich_over_alien"] = safe_ratio("C-alien", "C-alien-rich")
    comparisons["anon_over_alien"]  = safe_ratio("C-alien", "C-NAT-anon")
    comparisons["realnames_over_alien"] = safe_ratio("C-alien", "C-alien-realnames")

    # H_kern_delta: Pearson r between R_eff and Δ_med (from prior experiments)
    # Δ_med values: alien=1.04, realnames=0.727, rich=0.750, anon~0.17
    delta_med = {
        "C-alien":           1.04,
        "C-alien-realnames": 0.727,
        "C-alien-rich":      0.750,
        "C-NAT-anon":        0.17,
    }
    r_eff_vals  = []
    delta_vals  = []
    for corpus_name in delta_med:
        r_eff = all_results.get(corpus_name, {}).get("mean_r_eff")
        if r_eff is not None and "error" not in all_results.get(corpus_name, {}):
            r_eff_vals.append(r_eff)
            delta_vals.append(delta_med[corpus_name])
    if len(r_eff_vals) >= 3:
        import numpy as np
        r_eff_arr  = np.array(r_eff_vals)
        delta_arr  = np.array(delta_vals)
        r_corr = float(np.corrcoef(r_eff_arr, delta_arr)[0, 1])
        comparisons["pearson_r_eff_vs_delta_med"] = r_corr
        print(f"\n  Pearson r(R_eff, Δ_med) = {r_corr:.4f}  (predicted: negative)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdicts = {}

    alien_r     = all_results.get("C-alien", {}).get("mean_r_eff", None)
    rich_r      = all_results.get("C-alien-rich", {}).get("mean_r_eff", None)
    anon_r      = all_results.get("C-NAT-anon", {}).get("mean_r_eff", None)
    realnames_r = all_results.get("C-alien-realnames", {}).get("mean_r_eff", None)

    if alien_r is not None and rich_r is not None and anon_r is not None:
        ordering_confirmed = (alien_r < rich_r < anon_r)
        verdicts["H_kern_ordered"] = "CONFIRMED" if ordering_confirmed else "FALSIFIED"

        alien_in_range  = 4 <= alien_r <= 20
        rich_in_range   = 15 <= rich_r  <= 50
        anon_in_range   = anon_r > 40
        ratio_ok        = (rich_r / alien_r) >= 2.0 if alien_r > 0 else False
        verdicts["H_kern_S"] = (
            "CONFIRMED" if (alien_in_range and rich_in_range and anon_in_range and ratio_ok)
            else "PARTIAL" if ordering_confirmed
            else "FALSIFIED"
        )
        verdicts["H_kern_S_detail"] = {
            "alien_in_range_4_20":  alien_in_range,
            "rich_in_range_15_50":  rich_in_range,
            "anon_above_40":        anon_in_range,
            "rich_over_alien_ratio": round(rich_r / alien_r, 4) if alien_r > 0 else None,
            "ratio_ge_2": ratio_ok,
        }

    if alien_r is not None and realnames_r is not None:
        equiv = abs(realnames_r - alien_r) / alien_r <= 0.10 if alien_r > 0 else False
        verdicts["H_kern_realnames_equiv"] = "CONFIRMED" if equiv else "FALSIFIED"
        verdicts["H_kern_realnames_diff_pct"] = (
            round(100 * (realnames_r - alien_r) / alien_r, 2) if alien_r > 0 else None
        )

    if "pearson_r_eff_vs_delta_med" in comparisons:
        verdicts["H_kern_delta"] = (
            "CONFIRMED" if comparisons["pearson_r_eff_vs_delta_med"] < -0.5
            else "FALSIFIED"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "experiment":  "exp-101",
        "timestamp":   __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "protocol": {
            "n_tokens":       N_TOKENS,
            "n_layers":       N_LAYERS,
            "n_heads":        N_HEADS,
            "d_k":            D_K,
            "d_model":        D_MODEL,
            "checkpoint_seed": "s0",
            "layer_input":    "embed_in (layer-0 token embeddings, all layers' W_K applied)",
        },
        "results":     all_results,
        "comparisons": comparisons,
        "verdicts":    verdicts,
    }

    out_dir = Path("/results/measurements")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kern_rank_results.json"
    with open(out_path, "w") as fout:
        json.dump(output, fout, indent=2)
    vol_results.commit()
    print(f"\nSaved to {out_path}")

    print("\n=== SUMMARY ===")
    for corpus_name, data in all_results.items():
        if "error" not in data:
            print(f"  {corpus_name:25s}: mean_r_eff={data['mean_r_eff']:.2f}  "
                  f"median={data['median_r_eff']:.2f}  "
                  f"r_stable={data['mean_r_stable']:.2f}")
    print("\n  Ratios (theory prediction: rich/alien ≈ 4, anon/alien ≈ 8):")
    for k, v in comparisons.items():
        if v is not None and "pearson" not in k:
            print(f"    {k}: {v:.3f}")
    print("\n  Verdicts:")
    for k, v in verdicts.items():
        if isinstance(v, str):
            print(f"    {k}: {v}")

    return output


@app.local_entrypoint()
def main():
    result = measure_kern_ranks.remote()
    print("\nDone.")
