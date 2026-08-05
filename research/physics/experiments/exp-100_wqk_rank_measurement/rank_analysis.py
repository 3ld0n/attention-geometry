"""
exp-100 — W_QK Effective Rank Measurement (local, no new training).

Pre-registration: notes.md (committed before this script ran).

Theory: melonic threshold derivation (notes/2026-08-03_melonic_threshold_derivation.md §2–3).
Primary prediction (H_rank_gap): mean stable_rank(W_K, C-alien) << mean stable_rank(W_K, C-NAT).

Downloads final checkpoints (step_2000) from Modal volumes,
computes SVD of W_K (and W_Q, W_V) for each layer, reports stable rank and
participation ratio.

Usage (from repo root):
    .venv/bin/python3 research/physics/experiments/exp-100_wqk_rank_measurement/rank_analysis.py

Ariel — 2026-08-04. Pre-registered before first run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── configuration ───────────────────────────────────────────────────────────────

# (volume_name, run_subdirectory, corpus_label, seed_label)
CORPORA: list[tuple[str, str, str, str]] = [
    # C-alien: exp-097
    ("exp097-alien-data",    "runs/run_alien_s0/step_2000", "C-alien",         "s0"),
    ("exp097-alien-data",    "runs/run_alien_s1/step_2000", "C-alien",         "s1"),
    ("exp097-alien-data",    "runs/run_alien_s2/step_2000", "C-alien",         "s2"),
    # C-NAT: exp-062
    ("exp062-data",          "runs/run_CNAT_s0/step_2000",  "C-NAT",           "s0"),
    ("exp062-data",          "runs/run_CNAT_s1/step_2000",  "C-NAT",           "s1"),
    ("exp062-data",          "runs/run_CNAT_s2/step_2000",  "C-NAT",           "s2"),
    # C-NAT-anon: exp-096
    ("exp096-anon-data",     "runs/run_anon_s0/step_2000",  "C-NAT-anon",      "s0"),
    ("exp096-anon-data",     "runs/run_anon_s1/step_2000",  "C-NAT-anon",      "s1"),
    ("exp096-anon-data",     "runs/run_anon_s2/step_2000",  "C-NAT-anon",      "s2"),
    # C-alien-realnames: exp-098
    ("exp098-realnames-data","runs/run_realnames_s0/step_2000","C-alien-realnames","s0"),
    ("exp098-realnames-data","runs/run_realnames_s1/step_2000","C-alien-realnames","s1"),
    ("exp098-realnames-data","runs/run_realnames_s2/step_2000","C-alien-realnames","s2"),
]

# Only these files needed for rank analysis (skip opt.pt / generation_config.json)
CHECKPOINT_FILES = ["model.safetensors", "config.json"]

# GPT-NeoX QKV split: W shape [3*H, H] = [1536, 512] for H=512
HIDDEN_SIZE = 512


# ── download helpers ─────────────────────────────────────────────────────────────

def modal_volume_get(volume: str, remote_path: str, local_path: Path) -> None:
    """Download a single file from a Modal volume."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["python3", "-m", "modal", "volume", "get",
         volume, remote_path, str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"modal volume get failed for {volume}/{remote_path}:\n"
            f"{result.stderr}"
        )


def download_checkpoint(volume: str, remote_dir: str, local_dir: Path) -> None:
    """Download checkpoint files needed for rank analysis."""
    for fname in CHECKPOINT_FILES:
        remote = f"{remote_dir}/{fname}"
        local  = local_dir / fname
        if not local.exists():
            print(f"  downloading {volume}/{remote} …", flush=True)
            modal_volume_get(volume, remote, local)
        else:
            print(f"  cached {local}", flush=True)


# ── weight extraction ────────────────────────────────────────────────────────────

def load_qkv_weights(ckpt_dir: Path) -> list[dict[str, np.ndarray]]:
    """
    Load W_Q, W_K, W_V for each layer from a GPT-NeoX checkpoint.

    Returns a list of dicts, one per layer, with keys 'W_Q', 'W_K', 'W_V'.
    Each matrix is float64, shape [hidden_size, hidden_size].
    """
    # safetensors is the saved format
    try:
        from safetensors import safe_open
    except ImportError:
        # fallback: load via transformers
        return _load_via_transformers(ckpt_dir)

    layers: dict[int, dict[str, np.ndarray]] = {}
    with safe_open(str(ckpt_dir / "model.safetensors"),
                   framework="numpy", device="cpu") as f:
        for key in f.keys():
            # key looks like:
            # gpt_neox.layers.{L}.attention.query_key_value.weight
            if "attention.query_key_value.weight" not in key:
                continue
            parts = key.split(".")
            l_idx = int(parts[2])  # layers.{L}
            w = f.get_tensor(key).astype(np.float64)  # [3*H, H]
            H = HIDDEN_SIZE
            layers[l_idx] = {
                "W_Q": w[0:H,   :],
                "W_K": w[H:2*H, :],
                "W_V": w[2*H:,  :],
            }
    return [layers[i] for i in sorted(layers)]


def _load_via_transformers(ckpt_dir: Path) -> list[dict[str, np.ndarray]]:
    """Fallback: load weights via transformers (slower)."""
    import torch
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
        str(ckpt_dir), torch_dtype=torch.float32,
    )
    H = HIDDEN_SIZE
    result = []
    for layer in model.gpt_neox.layers:
        w = layer.attention.query_key_value.weight.detach().float().numpy().astype(np.float64)
        result.append({
            "W_Q": w[0:H,   :],
            "W_K": w[H:2*H, :],
            "W_V": w[2*H:,  :],
        })
    return result


# ── rank metrics ─────────────────────────────────────────────────────────────────

def effective_rank_metrics(W: np.ndarray) -> dict[str, float]:
    """
    Compute effective rank metrics for matrix W.

    - stable_rank:  sum(S^2) / S_max^2  — range [1, min(m,n)]
    - pr:           (sum(S))^2 / (n * sum(S^2))  — participation ratio in [0,1]
    - entropy_rank: exp(-sum(p_i * log(p_i))) where p_i = S_i^2 / sum(S^2)
    """
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    S = S.astype(np.float64)
    s2 = S ** 2
    stable_rank   = float(s2.sum() / s2.max())
    n = len(S)
    pr            = float(S.sum() ** 2 / (n * s2.sum()))
    # entropy-based effective rank
    p = s2 / s2.sum()
    ent = -float(np.sum(p * np.log(p + 1e-300)))
    entropy_rank  = float(np.exp(ent))
    return {
        "stable_rank":  stable_rank,
        "pr":           pr,
        "entropy_rank": entropy_rank,
        "S_max":        float(S.max()),
        "S_min":        float(S.min()),
        "frobenius":    float(np.sqrt(s2.sum())),
    }


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results: dict = {"corpora": {}, "summary": {}}

    with tempfile.TemporaryDirectory(prefix="exp100_ckpts_") as tmpdir:
        tmp = Path(tmpdir)

        for volume, remote_dir, corpus, seed in CORPORA:
            run_key = f"{corpus}/{seed}"
            print(f"\n── {run_key} ({volume}) ──", flush=True)
            local_ckpt = tmp / corpus / seed
            download_checkpoint(volume, remote_dir, local_ckpt)

            try:
                layer_weights = load_qkv_weights(local_ckpt)
            except Exception as e:
                print(f"  ERROR loading weights: {e}", file=sys.stderr)
                results["corpora"].setdefault(corpus, {})[seed] = {"error": str(e)}
                continue

            n_layers = len(layer_weights)
            layer_results = []
            for l_idx, wdict in enumerate(layer_weights):
                rec: dict = {"layer": l_idx}
                for mat_name in ("W_Q", "W_K", "W_V"):
                    rec[mat_name] = effective_rank_metrics(wdict[mat_name])
                layer_results.append(rec)
                sr_k = rec["W_K"]["stable_rank"]
                sr_q = rec["W_Q"]["stable_rank"]
                print(f"  L{l_idx}: W_K stable_rank={sr_k:.2f}  W_Q stable_rank={sr_q:.2f}",
                      flush=True)

            results["corpora"].setdefault(corpus, {})[seed] = {
                "volume": volume,
                "remote_dir": remote_dir,
                "n_layers": n_layers,
                "layers": layer_results,
                # per-seed summary statistics
                "mean_stable_rank_WK": float(
                    np.mean([lr["W_K"]["stable_rank"] for lr in layer_results])),
                "median_stable_rank_WK": float(
                    np.median([lr["W_K"]["stable_rank"] for lr in layer_results])),
                "per_layer_stable_rank_WK": [
                    lr["W_K"]["stable_rank"] for lr in layer_results],
            }

    # ── summary across corpora ────────────────────────────────────────────────────
    print("\n\n══ SUMMARY ══\n", flush=True)
    corpus_means: dict[str, list[float]] = {}
    for corpus, seeds in results["corpora"].items():
        vals = [v["mean_stable_rank_WK"] for v in seeds.values()
                if "mean_stable_rank_WK" in v]
        if vals:
            corpus_means[corpus] = vals
            print(f"{corpus}: mean stable_rank(W_K) per seed = {[f'{v:.2f}' for v in vals]}")
            print(f"         aggregate mean = {np.mean(vals):.2f}  "
                  f"std = {np.std(vals):.2f}\n")

    # ── verdict ───────────────────────────────────────────────────────────────────
    if "C-alien" in corpus_means and "C-NAT" in corpus_means:
        alien_vals = corpus_means["C-alien"]
        nat_vals   = corpus_means["C-NAT"]
        alien_mean = float(np.mean(alien_vals))
        nat_mean   = float(np.mean(nat_vals))
        gap_pct    = 100.0 * (nat_mean - alien_mean) / nat_mean if nat_mean > 0 else 0.0

        # Pre-registered kill criteria from notes.md:
        # CONFIRMED: C-alien median < C-NAT median for all 3 seeds
        n_alien_seeds_below = sum(1 for a, n in zip(alien_vals, nat_vals) if a < n)
        confirmed  = (n_alien_seeds_below == len(alien_vals)) and (alien_mean < nat_mean)
        falsified  = all(a >= n for a, n in zip(alien_vals, nat_vals))

        verdict_str = (
            "H_rank_gap CONFIRMED" if confirmed
            else "H_rank_gap FALSIFIED" if falsified
            else "H_rank_gap INCONCLUSIVE"
        )
        print(f"Primary verdict: {verdict_str}")
        print(f"  C-alien mean={alien_mean:.2f}, C-NAT mean={nat_mean:.2f}, "
              f"gap={gap_pct:.1f}%")
        results["summary"]["verdict"]  = verdict_str
        results["summary"]["alien_mean_stable_rank_WK"] = alien_mean
        results["summary"]["nat_mean_stable_rank_WK"]   = nat_mean
        results["summary"]["gap_pct"]  = gap_pct
        results["summary"]["n_alien_seeds_below_nat_mean"] = n_alien_seeds_below

    # ── secondary: realnames vs alien ─────────────────────────────────────────────
    if "C-alien-realnames" in corpus_means:
        rn_mean = float(np.mean(corpus_means["C-alien-realnames"]))
        print(f"\nH_rank_realnames: C-alien-realnames mean={rn_mean:.2f} "
              f"vs C-alien mean={alien_mean:.2f}")
        results["summary"]["realnames_mean_stable_rank_WK"] = rn_mean

    # ── secondary: anon vs nat ────────────────────────────────────────────────────
    if "C-NAT-anon" in corpus_means:
        anon_mean = float(np.mean(corpus_means["C-NAT-anon"]))
        print(f"H_rank_anon: C-NAT-anon mean={anon_mean:.2f} "
              f"vs C-NAT mean={nat_mean:.2f}")
        results["summary"]["anon_mean_stable_rank_WK"] = anon_mean

    # ── save results ──────────────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
