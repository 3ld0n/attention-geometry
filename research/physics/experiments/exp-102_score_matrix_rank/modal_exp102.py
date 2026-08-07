"""
exp-102 — Sequence-Level Attention Score Matrix Effective Rank (Modal cloud, GPU).

Pre-registration: notes.md (committed to 3ld0n/attention-geometry before this script ran).
Theoretical frame: notes/2026-08-07_tau_chaos_product_formula.md

For each of 4 trained corpora, loads the seed-0 checkpoint, samples N_SEQS
random sequences of length SEQ_LEN, runs forward passes with output_attentions=True,
and computes the SVD-based effective rank of the attention weight matrix A_{ij}
(softmax output) per head per layer. Averages over all contexts.

The attention weight matrix A_{ij} = softmax(q_i·k_j/√d_k) is the correct
sequence-level operationalization of the KCA coupling rank γ_eff = R/N.

Tests:
  H_score_ordered : R_eff^score ordered alien < rich < anon
  H_score_S       : R_eff^score(C-alien) ~ S ≈ 8 (not ~34 like the token proxy)
  H_tau_gain      : τ_chaos = m₂ × R_eff/d_k gives ≥30× discrimination
  H_score_delta   : Pearson r(R_eff^score, Δ_med) more negative than exp-101's −0.91
  H_realnames_equiv: R_eff^score(alien) ≈ R_eff^score(realnames) (vocabulary inert)

No new training. All checkpoints exist on Modal volumes from prior experiments.
Estimated cost: ~$1.50 (GPU, bf16 forward passes).

Usage (from repo root):
    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-102_score_matrix_rank/modal_exp102.py

Ariel — August 7, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# ─── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("exp102-score-matrix-rank")

vol_alien     = modal.Volume.from_name("exp097-alien-data")
vol_anon      = modal.Volume.from_name("exp096-anon-data")
vol_realnames = modal.Volume.from_name("exp098-realnames-data")
vol_rich      = modal.Volume.from_name("exp099-rich-data")
vol_results   = modal.Volume.from_name("exp102-score-rank-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "torch==2.12.0",
        "transformers==5.8.1",
        "scipy==1.17.1",
    )
)

# ─── Experiment constants (pre-registered in notes.md) ─────────────────────────

N_SEQS   = 512    # sequences per corpus
SEQ_LEN  = 64    # tokens per sequence (shorter than census: [64×64] score matrix)
RNG_SEED = 42    # pre-registered

N_LAYERS = 6
N_HEADS  = 8
D_K      = 64    # head dimension
D_MODEL  = 512

# Δ_med from prior experiments (for H_score_delta; pre-registered values)
DELTA_MED_PRIOR = {
    "C-alien":           1.04,
    "C-alien-realnames": 0.727,
    "C-alien-rich":      0.750,
    "C-NAT-anon":        0.17,
}

# m₂ from corpus_functional.py (IDF-weighted; for τ_chaos product formula)
M2_PRIOR = {
    "C-alien":           0.74,    # proxy units from melonic note §6.4
    "C-alien-realnames": 0.74,    # assumed same as C-alien (same generator)
    "C-alien-rich":      0.75,
    "C-NAT-anon":        13.2,
}

CORPORA = {
    "C-alien": {
        "corpus":   "/data097/C-alien.bin",
        "ckpt":     "/data097/runs/run_alien_s0/step_2000",
        "alphabet": "/data097/alphabet.json",
    },
    "C-alien-realnames": {
        "corpus":   "/data098/C-alien-realnames.bin",
        "ckpt":     "/data098/runs/run_realnames_s0/step_2000",
        "alphabet": "/data098/alphabet.json",
    },
    "C-alien-rich": {
        "corpus":   "/data099/C-alien-rich.bin",
        "ckpt":     "/data099/runs/run_rich_s0/step_2000",
        "alphabet": "/data099/alphabet.json",
    },
    "C-NAT-anon": {
        "corpus":   "/data096/C-NAT-anon.bin",
        "ckpt":     "/data096/runs/run_anon_s0/step_2000",
        "alphabet": None,  # full vocab
    },
}


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/data097": vol_alien,
        "/data096": vol_anon,
        "/data098": vol_realnames,
        "/data099": vol_rich,
        "/results": vol_results,
    },
    timeout=1800,   # 30 min: forward passes only, no training
    memory=16384,
)
def measure_score_rank() -> dict:
    import json as _json
    import numpy as np
    import torch
    from pathlib import Path as _Path
    from transformers import GPTNeoXForCausalLM

    device = torch.device("cuda")
    rng = np.random.default_rng(RNG_SEED)

    all_results = {}

    for corpus_name, paths in CORPORA.items():
        print(f"\n{'='*60}")
        print(f"[{corpus_name}]")

        corpus_path  = _Path(paths["corpus"])
        ckpt_dir     = _Path(paths["ckpt"])
        alphabet_path = paths["alphabet"]

        # ── Validate paths ─────────────────────────────────────────────────────
        if not corpus_path.exists():
            print(f"  SKIP — corpus not found: {corpus_path}")
            all_results[corpus_name] = {"error": f"corpus not found: {corpus_path}"}
            continue
        if not ckpt_dir.exists():
            print(f"  SKIP — checkpoint dir not found: {ckpt_dir}")
            all_results[corpus_name] = {"error": f"ckpt not found: {ckpt_dir}"}
            continue

        # ── Token draw pool ────────────────────────────────────────────────────
        if alphabet_path is not None and _Path(alphabet_path).exists():
            draw_pool = np.array(_json.loads(_Path(alphabet_path).read_text())["ids"])
            print(f"  alphabet: {len(draw_pool)} ids")
        else:
            # Use a small pool for NAT-anon since corpus tokens cover most of vocab
            corpus_raw = np.fromfile(str(corpus_path), dtype=np.uint16)
            # Use unique tokens in the corpus as the draw pool
            draw_pool = np.unique(corpus_raw[:100_000]).astype(np.int64)
            print(f"  full-vocab proxy: {len(draw_pool)} unique tokens in first 100k")
            del corpus_raw

        # ── Load model ─────────────────────────────────────────────────────────
        print(f"  loading model from {ckpt_dir}")
        model = GPTNeoXForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        print(f"  model loaded: {model.config.num_hidden_layers}L "
              f"{model.config.num_attention_heads}H")

        # ── Generate sequences and measure ────────────────────────────────────
        # R_eff accumulator: [n_layers, n_heads] per-context, averaged
        r_eff_sum    = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
        r_stable_sum = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
        n_valid      = 0

        for seq_idx in range(N_SEQS):
            ids = rng.choice(draw_pool, size=SEQ_LEN, replace=True)
            x   = torch.tensor(ids[None, :], dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(x, output_attentions=True)

            # out.attentions: tuple of n_layers tensors, each [1, n_heads, SEQ_LEN, SEQ_LEN]
            for layer_idx in range(N_LAYERS):
                a_t = out.attentions[layer_idx]
                # a_t shape: [1, n_heads, seq_len, seq_len]
                a_np = a_t[0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]

                for head_idx in range(N_HEADS):
                    A = a_np[head_idx]   # [seq_len, seq_len], stochastic matrix

                    # SVD of the attention weight matrix
                    # We want singular values of the full matrix (not just one row)
                    sigma = np.linalg.svd(A, compute_uv=False)   # [seq_len], descending

                    s_sum    = float(np.sum(sigma))
                    s_sq_sum = float(np.sum(sigma ** 2))
                    s1       = float(sigma[0])

                    r_eff    = (s_sum ** 2) / s_sq_sum if s_sq_sum > 1e-15 else 0.0
                    r_stable = s_sq_sum / (s1 ** 2)    if s1      > 1e-15 else 0.0

                    r_eff_sum[layer_idx, head_idx]    += r_eff
                    r_stable_sum[layer_idx, head_idx] += r_stable

            n_valid += 1
            del out

            if (seq_idx + 1) % 100 == 0:
                print(f"  progress: {seq_idx + 1}/{N_SEQS} sequences", flush=True)

        print(f"  finished {n_valid} sequences")
        del model

        # ── Aggregate results ─────────────────────────────────────────────────
        r_eff_mean    = r_eff_sum    / n_valid   # [n_layers, n_heads]
        r_stable_mean = r_stable_sum / n_valid

        # Per-layer summary
        per_layer = {}
        for layer_idx in range(N_LAYERS):
            layer_r = r_eff_mean[layer_idx]
            layer_s = r_stable_mean[layer_idx]
            per_layer[f"L{layer_idx}"] = {
                "mean_r_eff":    float(np.mean(layer_r)),
                "median_r_eff":  float(np.median(layer_r)),
                "mean_r_stable": float(np.mean(layer_s)),
                "min_r_eff":     float(np.min(layer_r)),
                "max_r_eff":     float(np.max(layer_r)),
            }

        # Global (all heads × all layers)
        all_r_eff    = r_eff_mean.flatten()
        all_r_stable = r_stable_mean.flatten()

        corpus_mean_r_eff    = float(np.mean(all_r_eff))
        corpus_median_r_eff  = float(np.median(all_r_eff))
        corpus_mean_r_stable = float(np.mean(all_r_stable))

        print(f"  corpus mean_r_eff = {corpus_mean_r_eff:.4f}  "
              f"median = {corpus_median_r_eff:.4f}  "
              f"mean_r_stable = {corpus_mean_r_stable:.4f}")

        # Per-head raw data for Pearson correlation
        heads = []
        for layer_idx in range(N_LAYERS):
            for head_idx in range(N_HEADS):
                heads.append({
                    "layer":     layer_idx,
                    "head":      head_idx,
                    "r_eff":     float(r_eff_mean[layer_idx, head_idx]),
                    "r_stable":  float(r_stable_mean[layer_idx, head_idx]),
                })

        all_results[corpus_name] = {
            "n_seqs":           n_valid,
            "seq_len":          SEQ_LEN,
            "mean_r_eff":       corpus_mean_r_eff,
            "median_r_eff":     corpus_median_r_eff,
            "mean_r_stable":    corpus_mean_r_stable,
            "min_r_eff":        float(np.min(all_r_eff)),
            "max_r_eff":        float(np.max(all_r_eff)),
            "per_layer":        per_layer,
            "heads":            heads,
        }

    # ── Cross-corpus analysis ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CROSS-CORPUS ANALYSIS")

    comparisons = {}

    def safe_ratio(a_key: str, b_key: str, metric: str = "mean_r_eff") -> float | None:
        a = all_results.get(a_key, {}).get(metric)
        b = all_results.get(b_key, {}).get(metric)
        if a and b and a > 0:
            return round(b / a, 4)
        return None

    comparisons["rich_over_alien"]     = safe_ratio("C-alien", "C-alien-rich")
    comparisons["anon_over_alien"]     = safe_ratio("C-alien", "C-NAT-anon")
    comparisons["realnames_over_alien"] = safe_ratio("C-alien", "C-alien-realnames")

    # H_score_delta: Pearson r(R_eff^score, Δ_med)
    r_eff_vals, delta_vals = [], []
    for cname, delta in DELTA_MED_PRIOR.items():
        r_eff = all_results.get(cname, {}).get("mean_r_eff")
        if r_eff is not None and "error" not in all_results.get(cname, {}):
            r_eff_vals.append(r_eff)
            delta_vals.append(delta)
    if len(r_eff_vals) >= 3:
        import numpy as _np
        r_eff_arr = _np.array(r_eff_vals)
        delta_arr = _np.array(delta_vals)
        r_corr = float(_np.corrcoef(r_eff_arr, delta_arr)[0, 1])
        comparisons["pearson_r_eff_vs_delta_med"] = r_corr
        print(f"  Pearson r(R_eff^score, Δ_med) = {r_corr:.4f}")
        print(f"  exp-101 token proxy:              -0.9100  (reference)")

    # τ_chaos product formula: τ = m₂ × R_eff / d_k
    tau_chaos = {}
    for cname in M2_PRIOR:
        r_eff = all_results.get(cname, {}).get("mean_r_eff")
        m2    = M2_PRIOR[cname]
        if r_eff is not None and "error" not in all_results.get(cname, {}):
            tau = m2 * r_eff / D_K
            tau_chaos[cname] = round(tau, 6)
    if tau_chaos:
        tau_list = list(tau_chaos.values())
        print(f"\n  τ_chaos (m₂ × R_eff / d_k):")
        for k, v in tau_chaos.items():
            print(f"    {k}: {v:.6f}")
        if "C-alien" in tau_chaos and "C-NAT-anon" in tau_chaos and tau_chaos["C-alien"] > 0:
            tau_ratio = round(tau_chaos["C-NAT-anon"] / tau_chaos["C-alien"], 2)
            comparisons["tau_chaos_ratio_anon_over_alien"] = tau_ratio
            comparisons["m2_only_ratio"]                   = round(M2_PRIOR["C-NAT-anon"] / M2_PRIOR["C-alien"], 2)
            print(f"\n  τ_chaos discrimination: {tau_ratio}×")
            print(f"  m₂ alone discrimination: {comparisons['m2_only_ratio']}×")
            print(f"  H_tau_gain threshold: ≥30×")
            comparisons["H_tau_gain"] = "CONFIRMED" if tau_ratio >= 30 else "FALSIFIED"

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdicts = {}

    alien_r     = all_results.get("C-alien", {}).get("mean_r_eff")
    rich_r      = all_results.get("C-alien-rich", {}).get("mean_r_eff")
    anon_r      = all_results.get("C-NAT-anon", {}).get("mean_r_eff")
    realnames_r = all_results.get("C-alien-realnames", {}).get("mean_r_eff")

    if alien_r is not None and rich_r is not None and anon_r is not None:
        ordering_ok = (alien_r < rich_r < anon_r)
        verdicts["H_score_ordered"] = "CONFIRMED" if ordering_ok else "FALSIFIED"
        print(f"\n  H_score_ordered: {verdicts['H_score_ordered']}")
        print(f"    alien={alien_r:.3f}  rich={rich_r:.3f}  anon={anon_r:.3f}")

        # H_score_S: C-alien R_eff^score ~ 8 (not ~34 like token proxy)
        if alien_r is not None:
            score_s_confirmed = alien_r < 20   # kill condition: R_eff > 20
            verdicts["H_score_S"] = "CONFIRMED" if score_s_confirmed else "FALSIFIED"
            print(f"  H_score_S: {verdicts['H_score_S']}  (alien R_eff={alien_r:.2f}, kill if > 20)")

    if alien_r is not None and realnames_r is not None and alien_r > 0:
        ratio = abs(realnames_r - alien_r) / alien_r
        verdicts["H_realnames_equiv"] = "CONFIRMED" if ratio <= 0.10 else "FALSIFIED"
        print(f"  H_realnames_equiv: {verdicts['H_realnames_equiv']}  "
              f"(ratio={ratio:.4f}, threshold 0.10)")

    if "pearson_r_eff_vs_delta_med" in comparisons:
        r_score = comparisons["pearson_r_eff_vs_delta_med"]
        r_token = -0.91  # exp-101 value
        verdicts["H_score_delta"] = (
            "CONFIRMED" if abs(r_score) > abs(r_token) else "FALSIFIED"
        )
        print(f"  H_score_delta: {verdicts['H_score_delta']}  "
              f"(r_score={r_score:.4f} vs r_token=-0.91)")

    if "H_tau_gain" in comparisons:
        verdicts["H_tau_gain"] = comparisons["H_tau_gain"]
        print(f"  H_tau_gain: {verdicts['H_tau_gain']}")

    # ── Final results JSON ─────────────────────────────────────────────────────
    final = {
        "experiment":   "exp-102",
        "date":         "2026-08-07",
        "protocol": {
            "n_seqs":   N_SEQS,
            "seq_len":  SEQ_LEN,
            "rng_seed": RNG_SEED,
            "d_k":      D_K,
        },
        "corpora":      {k: {ek: ev for ek, ev in v.items() if ek != "heads"}
                         for k, v in all_results.items()},
        "comparisons":  comparisons,
        "tau_chaos":    tau_chaos,
        "verdicts":     verdicts,
    }

    out_path = _Path("/results/exp102_results.json")
    out_path.write_text(_json.dumps(final, indent=2))
    print(f"\nResults written to {out_path}")
    vol_results.commit()

    # Also write per-head data
    heads_path = _Path("/results/exp102_heads.json")
    heads_path.write_text(_json.dumps(
        {k: v.get("heads", []) for k, v in all_results.items()
         if "error" not in v},
        indent=1
    ))
    vol_results.commit()
    print(f"Per-head data written to {heads_path}")

    return final


@app.local_entrypoint()
def main():
    result = measure_score_rank.remote()
    print("\n" + "="*70)
    print("FINAL VERDICTS:")
    for h, v in result.get("verdicts", {}).items():
        print(f"  {h}: {v}")
    print("\nτ_chaos values:")
    for k, v in result.get("tau_chaos", {}).items():
        print(f"  {k}: {v:.6f}")
