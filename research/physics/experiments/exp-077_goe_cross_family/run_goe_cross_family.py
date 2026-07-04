"""
exp-077 — GOE universality across model families.

See notes.md for pre-stated hypotheses (H1 family universality, H2 GQA-not-special,
H3 layer uniformity incl. Pythia-2.8b layers 22-27, H0 null).

Protocol identical to exp-046/047/051: symmetrized W_QK per head, eigvalsh,
Oganesyan-Huse r-ratio. Weights-only, CPU, no forward passes. Models freed
sequentially (largest is longchat-13b: ~52 GB fp32 on a 128 GB machine).

Usage:
  .venv/bin/python3 research/physics/experiments/exp-077_goe_cross_family/run_goe_cross_family.py

Ariel — July 4, 2026, night session.
"""

from __future__ import annotations

import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

GOE_R_RATIO = 0.536
POISSON_R_RATIO = 0.386
TOLERANCE = 0.02

OUT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = OUT_DIR / "results.json"

# Baselines from prior experiments (all-heads r_mean)
BASELINES = {
    "gpt2 (exp-046)": 0.5272,
    "gpt2-medium (exp-047)": 0.5255,
    "pythia-410m (exp-047)": 0.5199,
    "pythia-1.4b (exp-051)": 0.5235,
    "untrained gpt2 (exp-048)": 0.5288,
}


def r_ratio_from_eigvals(eigvals: np.ndarray) -> float:
    spacings = np.diff(eigvals)
    if len(spacings) < 3:
        return float("nan")
    s_lo, s_hi = spacings[:-1], spacings[1:]
    r_vals = np.minimum(s_lo, s_hi) / (np.maximum(s_lo, s_hi) + 1e-30)
    return float(np.mean(r_vals))


def head_r(W_Q_h: np.ndarray, W_K_h: np.ndarray) -> float:
    """Both inputs (d_k, hidden): W_QK = W_Q_h @ W_K_h.T, symmetrized."""
    W_QK = W_Q_h @ W_K_h.T
    M = (W_QK + W_QK.T) / 2.0
    return r_ratio_from_eigvals(np.linalg.eigvalsh(M))


# ── extractors ─────────────────────────────────────────────────────────────────

def extract_gpt2(model_name: str) -> dict:
    model = GPT2LMHeadModel.from_pretrained(
        model_name, attn_implementation="eager", low_cpu_mem_usage=True,
        torch_dtype=torch.float32)
    model.eval()
    n_layers, n_heads = model.config.n_layer, model.config.n_head
    d_model = model.config.n_embd
    head_dim = d_model // n_heads
    per_head = []
    for l in range(n_layers):
        W_full = model.transformer.h[l].attn.c_attn.weight.detach().float().numpy()
        for h in range(n_heads):
            # Conv1D convention: columns are output dims → slice cols, transpose to (d_k, hidden)
            W_Q_h = W_full[:, h * head_dim:(h + 1) * head_dim].T
            W_K_h = W_full[:, d_model + h * head_dim: d_model + (h + 1) * head_dim].T
            per_head.append({"layer": l, "head": h,
                             "r_ratio": round(head_r(W_Q_h, W_K_h), 6)})
    meta = {"architecture": "gpt2", "pe": "learned", "attention": "MHA",
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}
    del model
    gc.collect()
    return {"meta": meta, "per_head": per_head}


def extract_neox(model_name: str) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    stride = 3 * head_dim
    per_head = []
    for l in range(n_layers):
        W_full = model.gpt_neox.layers[l].attention.query_key_value.weight.detach().float().numpy()
        for h in range(n_heads):
            base = h * stride
            W_Q_h = W_full[base: base + head_dim, :]
            W_K_h = W_full[base + head_dim: base + 2 * head_dim, :]
            per_head.append({"layer": l, "head": h,
                             "r_ratio": round(head_r(W_Q_h, W_K_h), 6)})
    meta = {"architecture": "gpt_neox", "pe": "RoPE", "attention": "MHA",
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}
    del model
    gc.collect()
    return {"meta": meta, "per_head": per_head}


def extract_gptneo(model_name: str) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_layers, cfg.num_heads
    head_dim = cfg.hidden_size // n_heads
    attn_types = cfg.attention_layers  # ["global","local",...]
    per_head = []
    for l in range(n_layers):
        attn = model.transformer.h[l].attn.attention
        Wq = attn.q_proj.weight.detach().float().numpy()  # (hidden, hidden), rows = out
        Wk = attn.k_proj.weight.detach().float().numpy()
        for h in range(n_heads):
            W_Q_h = Wq[h * head_dim:(h + 1) * head_dim, :]
            W_K_h = Wk[h * head_dim:(h + 1) * head_dim, :]
            per_head.append({"layer": l, "head": h, "layer_type": attn_types[l],
                             "r_ratio": round(head_r(W_Q_h, W_K_h), 6)})
    meta = {"architecture": "gpt_neo", "pe": "learned",
            "attention": "alternating local/global MHA",
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}
    del model
    gc.collect()
    return {"meta": meta, "per_head": per_head}


def extract_llama_like(model_name: str) -> dict:
    """Llama/Mistral: separate q_proj/k_proj; handles GQA via kv-head mapping."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    head_dim = cfg.hidden_size // n_heads
    group = n_heads // n_kv
    per_head = []
    for l in range(n_layers):
        attn = model.model.layers[l].self_attn
        Wq = attn.q_proj.weight.detach().float().numpy()  # (n_heads*d_k, hidden)
        Wk = attn.k_proj.weight.detach().float().numpy()  # (n_kv*d_k, hidden)
        for h in range(n_heads):
            kv = h // group
            W_Q_h = Wq[h * head_dim:(h + 1) * head_dim, :]
            W_K_h = Wk[kv * head_dim:(kv + 1) * head_dim, :]
            per_head.append({"layer": l, "head": h, "kv_head": kv,
                             "r_ratio": round(head_r(W_Q_h, W_K_h), 6)})
    meta = {"architecture": cfg.model_type, "pe": "RoPE",
            "attention": f"GQA {n_heads}Q/{n_kv}KV" if n_kv < n_heads else "MHA",
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
            "n_kv_heads": n_kv}
    del model
    gc.collect()
    return {"meta": meta, "per_head": per_head}


# ── summary ─────────────────────────────────────────────────────────────────────

def summarize(per_head: list[dict]) -> dict:
    r_vals = [h["r_ratio"] for h in per_head if not math.isnan(h["r_ratio"])]
    r = np.array(r_vals)
    by_layer: dict[int, list] = {}
    for h in per_head:
        if not math.isnan(h["r_ratio"]):
            by_layer.setdefault(h["layer"], []).append(h["r_ratio"])
    layer_means = {l: round(float(np.mean(v)), 6) for l, v in sorted(by_layer.items())}
    lm = np.array(list(layer_means.values()))
    r_mean = float(r.mean())
    dist_goe = abs(r_mean - GOE_R_RATIO)
    dist_poi = abs(r_mean - POISSON_R_RATIO)
    if dist_goe < TOLERANCE:
        verdict = "GOE-like"
    elif dist_poi < TOLERANCE:
        verdict = "Poisson-like"
    else:
        verdict = "GOE-tendency" if r_mean > (GOE_R_RATIO + POISSON_R_RATIO) / 2 else "Poisson-tendency"
    return {
        "n": len(r_vals),
        "r_mean": round(r_mean, 6), "r_std": round(float(r.std()), 6),
        "r_min": round(float(r.min()), 6), "r_max": round(float(r.max()), 6),
        "r_median": round(float(np.median(r)), 6),
        "verdict": verdict,
        "dist_to_goe": round(dist_goe, 6), "dist_to_poisson": round(dist_poi, 6),
        "layer_means": layer_means,
        "layer_r_std": round(float(lm.std()), 6),
    }


def main() -> None:
    print("exp-077: GOE universality across model families")
    print(f"  GOE ref={GOE_R_RATIO}  Poisson ref={POISSON_R_RATIO}  tol={TOLERANCE}")
    print(f"  baselines: {BASELINES}\n")

    models = [
        ("gpt_neo_2.7b", "EleutherAI/gpt-neo-2.7B", extract_gptneo),
        ("pythia_2.8b", "EleutherAI/pythia-2.8b", extract_neox),
        ("gpt2_large", "openai-community/gpt2-large", extract_gpt2),
        ("mistral_7b_v03", "mistralai/Mistral-7B-v0.3", extract_llama_like),
        ("longchat_13b_16k", "lmsys/longchat-13b-16k", extract_llama_like),
    ]

    all_results = {}
    for key, name, fn in models:
        print(f"=== {name} ===", flush=True)
        try:
            raw = fn(name)
            stats = summarize(raw["per_head"])
            entry = {"model": name, **raw["meta"], "stats": stats,
                     "per_head": raw["per_head"]}
            print(f"  {raw['meta']['n_layers']}L x {raw['meta']['n_heads']}H "
                  f"d_k={raw['meta']['head_dim']} ({raw['meta']['attention']})")
            print(f"  r_mean={stats['r_mean']:.4f} ± {stats['r_std']:.4f}  "
                  f"verdict={stats['verdict']}  layer_std={stats['layer_r_std']:.4f}")

            # GPT-Neo: split by layer type
            if key == "gpt_neo_2.7b":
                for lt in ("global", "local"):
                    sub = [h for h in raw["per_head"] if h.get("layer_type") == lt]
                    s = summarize(sub)
                    entry[f"stats_{lt}"] = {k: s[k] for k in
                                            ("n", "r_mean", "r_std", "verdict")}
                    print(f"    {lt}: r_mean={s['r_mean']:.4f} ({s['verdict']})")

            # Pythia-2.8b: layers 22-27 vs rest (H3 specific check)
            if key == "pythia_2.8b":
                lm = stats["layer_means"]
                anom = [lm[l] for l in range(22, 28)]
                rest = [v for l, v in lm.items() if not 22 <= l <= 27]
                d = abs(float(np.mean(anom)) - float(np.mean(rest)))
                entry["layers_22_27_check"] = {
                    "mean_22_27": round(float(np.mean(anom)), 6),
                    "mean_rest": round(float(np.mean(rest)), 6),
                    "abs_diff": round(d, 6),
                    "threshold_2x_layer_std": round(2 * stats["layer_r_std"], 6),
                    "deviates": bool(d > 2 * stats["layer_r_std"]),
                }
                print(f"    layers 22-27: {np.mean(anom):.4f} vs rest {np.mean(rest):.4f} "
                      f"(diff {d:.4f}, 2x layer_std {2*stats['layer_r_std']:.4f})")
            all_results[key] = entry
            print()
        except Exception as e:
            print(f"  ERROR: {e}\n")
            all_results[key] = {"model": name, "error": str(e)}
        gc.collect()

    # ── hypothesis verdicts ────────────────────────────────────────────────────
    print("=" * 60)
    print("=== HYPOTHESIS VERDICTS ===")
    ok = {k: v for k, v in all_results.items() if "stats" in v}

    h1 = all(v["stats"]["dist_to_goe"] < TOLERANCE for v in ok.values()) if ok else None
    for k, v in ok.items():
        print(f"  {k:18s} r_mean={v['stats']['r_mean']:.4f} "
              f"dist_GOE={v['stats']['dist_to_goe']:.4f} {v['stats']['verdict']}")
    print(f"H1 (family universality, all |r-0.536|<0.02): "
          f"{'CONFIRMED' if h1 else 'NOT CONFIRMED'}")

    h2 = None
    if "mistral_7b_v03" in ok:
        mha = [v["stats"]["r_mean"] for k, v in ok.items() if k != "mistral_7b_v03"]
        d = abs(ok["mistral_7b_v03"]["stats"]["r_mean"] - float(np.mean(mha)))
        h2 = d < TOLERANCE
        print(f"H2 (GQA not special): |r(Mistral) - mean(MHA)| = {d:.4f} "
              f"→ {'CONFIRMED' if h2 else 'NOT CONFIRMED'}")

    h3 = None
    if ok:
        h3_layers = all(v["stats"]["layer_r_std"] < 0.015 for v in ok.values())
        h3_2p8 = None
        if "pythia_2.8b" in ok and "layers_22_27_check" in ok["pythia_2.8b"]:
            h3_2p8 = not ok["pythia_2.8b"]["layers_22_27_check"]["deviates"]
        h3 = h3_layers and (h3_2p8 is not False)
        print(f"H3 (layer uniformity <0.015 all models: {h3_layers}; "
              f"pythia-2.8b L22-27 no deviation: {h3_2p8}) → "
              f"{'CONFIRMED' if h3 else 'NOT CONFIRMED'}")

    result = {
        "experiment": "exp-077",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H1": "All five families GOE-like (|r_mean - 0.536| < 0.02)",
            "H2": "Mistral GQA r_mean within 0.02 of MHA models' mean",
            "H3": "Layer uniformity: layer_r_std < 0.015 per model; pythia-2.8b layers 22-27 not deviating > 2x layer_std",
            "H0": "Some family Poisson-tendency or strong layer structure",
        },
        "protocol": {
            "analysis": "weights-only, CPU fp32, no forward passes",
            "matrix": "W_QK_sym = (W_Q_h W_K_h^T + transpose)/2 per head; GQA pairs query head h with kv head h//group",
            "r_ratio": "Oganesyan-Huse on sorted eigenvalue spacings",
            "references": {"GOE": GOE_R_RATIO, "Poisson": POISSON_R_RATIO,
                           "tolerance": TOLERANCE},
            "baselines": BASELINES,
        },
        "verdicts": {"H1": h1, "H2": h2, "H3": h3},
        "models": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
