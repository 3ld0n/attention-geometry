"""
exp-075 — Tradeoff: perplexity audit of the κ-operator on vicuna-13b-v1.5.

Pre-registration committed before this script was run (2026-07-09). See notes.md.

Measures: LM perplexity (mean NLL nats/tok) on wikitext-103-v1 test under the
locked κ-operator from exp-072 (8 target heads in layers 22–27, vicuna-13b).

The LITM task data (middle-retrieval + recency) is read from exp-072/results.json
and NOT re-run — it is already pre-registered and confirmed.

Ariel — July 9, 2026.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EXP072_DIR = HERE.parent / "exp-072_cloud_powered_slope_editing"
PREREG_PATH = EXP072_DIR / "prereg_locked.json"

# ── constants (match exp-072 projector protocol exactly) ──────────────────────
PROJ_SEQ = 512
PROJ_N = 50
PROJ_SEED = 68
N_PC = 8

# ── wikitext selection ────────────────────────────────────────────────────────
WIKI_N_PASSAGES = 50     # number of passages to use
WIKI_SEQ_LEN = 512       # tokens per passage

# ── verdict threshold ─────────────────────────────────────────────────────────
FLAT_THRESHOLD = 0.05    # |ΔNLL| ≤ this → H_flat confirmed

# ── κ grid ────────────────────────────────────────────────────────────────────
KAPPAS = [0.5, 1.0, 1.5, 2.0]


# ════════════════════════════════════════════════════════════════════════════
# Architecture helpers (Llama, identical to exp-072 cloud_slope_editing.py)
# ════════════════════════════════════════════════════════════════════════════

def _model_dims(model):
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    q_rows = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    head_dim = q_rows // n_heads
    return n_layers, n_heads, hidden, head_dim


def _input_ln(model, layer):
    return model.model.layers[layer].input_layernorm


def _q_proj_weight(model, layer):
    return model.model.layers[layer].self_attn.q_proj.weight


def _get_wq(model, layer, head, head_dim):
    W = _q_proj_weight(model, layer)
    return W[head * head_dim:(head + 1) * head_dim, :].detach().clone()


def _apply_edit(model, layer, head, head_dim, kappa, P_U, W_orig):
    device = W_orig.device
    dtype = W_orig.dtype
    hidden = W_orig.shape[1]
    P = torch.tensor(P_U, dtype=dtype, device=device)
    M = torch.eye(hidden, device=device, dtype=dtype) + (kappa - 1.0) * P
    W = _q_proj_weight(model, layer)
    with torch.no_grad():
        W[head * head_dim:(head + 1) * head_dim, :] = (W_orig @ M)


def _restore(model, layer, head, head_dim, W_orig):
    W = _q_proj_weight(model, layer)
    with torch.no_grad():
        W[head * head_dim:(head + 1) * head_dim, :] = W_orig


# ════════════════════════════════════════════════════════════════════════════
# Projector computation (exact port from exp-072 _compute_projectors)
# ════════════════════════════════════════════════════════════════════════════

def _compute_projectors(model, needed_layers, device):
    _, _, hidden, _ = _model_dims(model)
    rng = np.random.default_rng(PROJ_SEED)
    vocab = model.config.vocab_size
    acc = {layer: np.zeros((PROJ_SEQ, hidden), dtype=np.float64) for layer in needed_layers}
    for i in range(PROJ_N):
        ids = torch.tensor(rng.integers(0, vocab, size=(1, PROJ_SEQ)),
                           dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            for layer in needed_layers:
                h = out.hidden_states[layer]
                ln_out = _input_ln(model, layer)(h)
                acc[layer] += ln_out[0].cpu().float().numpy()
        del out
        if (i + 1) % 10 == 0:
            print(f"    projector pass {i + 1}/{PROJ_N}", flush=True)
    projectors = {}
    for layer in needed_layers:
        xbar = acc[layer] / PROJ_N
        xbar -= xbar.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(xbar, full_matrices=False)
        U = vt[:N_PC].T
        projectors[layer] = (U @ U.T).astype(np.float32)
    return projectors


# ════════════════════════════════════════════════════════════════════════════
# Wikitext passage selection
# ════════════════════════════════════════════════════════════════════════════

def _load_wikitext_passages(tokenizer):
    """Load first WIKI_N_PASSAGES non-overlapping 512-token chunks from wikitext-103-v1 test.

    wikitext-103-v1 is stored per-line in HuggingFace. Standard practice is to
    concatenate all test text, tokenize the full concatenation, then chunk into
    fixed-length blocks. This matches the GPT-2 wikitext-103 evaluation protocol.
    """
    from datasets import load_dataset
    print("Loading wikitext-103-v1 test split...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")

    # Concatenate all non-empty lines with newlines
    full_text = "\n".join(ex["text"] for ex in ds if ex["text"].strip())
    print(f"  concatenated test text: {len(full_text):,} chars", flush=True)

    # Tokenize the full concatenation (no truncation)
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    all_ids = enc.input_ids[0]  # shape (total_tokens,)
    print(f"  total tokens: {all_ids.shape[0]:,}", flush=True)

    # Chunk into non-overlapping WIKI_SEQ_LEN-token blocks
    n_chunks = all_ids.shape[0] // WIKI_SEQ_LEN
    if n_chunks < WIKI_N_PASSAGES:
        raise RuntimeError(f"Only {n_chunks} full {WIKI_SEQ_LEN}-token chunks available "
                           f"in wikitext-103 test. Need {WIKI_N_PASSAGES}.")

    passages = []
    for i in range(WIKI_N_PASSAGES):
        chunk = all_ids[i * WIKI_SEQ_LEN:(i + 1) * WIKI_SEQ_LEN].unsqueeze(0)
        passages.append(chunk)

    print(f"  {len(passages)} passages selected (each {WIKI_SEQ_LEN} tokens)", flush=True)
    return passages


# ════════════════════════════════════════════════════════════════════════════
# NLL measurement
# ════════════════════════════════════════════════════════════════════════════

def _measure_nll(model, passages, device):
    """Mean NLL (nats/token) over all passages under the current model weights."""
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for ids in passages:
            ids = ids.to(device)
            out = model(ids, labels=ids)
            # out.loss is mean cross-entropy over all (L-1) next-token predictions
            n_toks = ids.shape[1] - 1   # tokens predicted (first token is context only)
            total_nll += out.loss.item() * n_toks
            total_tokens += n_toks
    return total_nll / total_tokens


def _run_condition(label, model, passages, device, edit_heads, kappa, head_dim, projectors):
    """Apply κ-edit to edit_heads, measure NLL, restore."""
    orig = {}
    for h in edit_heads:
        l, hd_i = h["layer"], h["head"]
        W_orig = _get_wq(model, l, hd_i, head_dim).to(device)
        P_U = projectors[l]
        _apply_edit(model, l, hd_i, head_dim, kappa, P_U, W_orig)
        orig[(l, hd_i)] = W_orig
    t0 = time.time()
    nll = _measure_nll(model, passages, device)
    for (l, hd_i), W_orig in orig.items():
        _restore(model, l, hd_i, head_dim, W_orig)
    elapsed = time.time() - t0
    print(f"  [{label} κ={kappa}] NLL={nll:.5f} nats/tok ({elapsed:.0f}s)", flush=True)
    return nll


# ════════════════════════════════════════════════════════════════════════════
# Load exp-072 LITM results for the tradeoff table
# ════════════════════════════════════════════════════════════════════════════

def _load_litm_summary():
    with open(EXP072_DIR / "results.json") as f:
        res = json.load(f)
    base_acc = res["base_accuracy_by_pos"]
    out = {
        "base_middle": round(base_acc[2], 4),
        "base_recency": round(base_acc[4], 4),
        "base_primacy": round(base_acc[0], 4),
        "base_V_task": round(res["conditions"][0]["V_task"], 5),
        "by_kappa": {}
    }
    for cond in res["conditions"]:
        k = cond["kappa"]
        label = cond["label"]
        acc = cond["accuracy_by_pos"]
        out["by_kappa"][f"{label}_k{k}"] = {
            "label": label, "kappa": k,
            "middle": round(acc[2], 4),
            "recency": round(acc[4], 4),
            "primacy": round(acc[0], 4),
            "V_task": round(cond["V_task"], 5),
        }
    return out


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    from datetime import datetime, timezone
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not PREREG_PATH.exists():
        sys.exit(f"ERROR: pre-reg not found at {PREREG_PATH}")

    with open(PREREG_PATH) as f:
        prereg = json.load(f)

    model_name = prereg["model"]
    target_heads = prereg["target_heads"]
    sham_heads = prereg["sham_heads"]
    head_dim = prereg["dims"]["head_dim"]
    needed_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Target heads: {len(target_heads)} (layers {sorted(set(h['layer'] for h in target_heads))})", flush=True)
    print(f"Sham heads:   {len(sham_heads)} (layers {sorted(set(h['layer'] for h in sham_heads))})", flush=True)

    # Load model in bf16 (per program brief: bf16 OK for tradeoff experiment)
    print("\nLoading model in bf16...", flush=True)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    print(f"  loaded in {time.time() - t_load:.0f}s", flush=True)

    # Load wikitext passages
    passages = _load_wikitext_passages(tok)

    # Compute projectors (same LOCKED protocol as exp-072)
    print("\nComputing projectors...", flush=True)
    t_proj = time.time()
    projectors = _compute_projectors(model, needed_layers, device)
    print(f"  done in {time.time() - t_proj:.0f}s", flush=True)

    # Baseline (unedited)
    print("\nMeasuring baseline NLL (unedited)...", flush=True)
    t0 = time.time()
    nll_baseline = _measure_nll(model, passages, device)
    print(f"  baseline NLL={nll_baseline:.5f} nats/tok ({time.time()-t0:.0f}s)", flush=True)

    # κ conditions on target heads
    nll_by_kappa = {}
    print("\nκ-sweep on target heads...", flush=True)
    for kappa in KAPPAS:
        nll = _run_condition("target", model, passages, device,
                             target_heads, kappa, head_dim, projectors)
        nll_by_kappa[kappa] = nll

    # κ=1.5 sham condition
    print("\nSham condition (κ=1.5)...", flush=True)
    nll_sham_15 = _run_condition("sham", model, passages, device,
                                 sham_heads, 1.5, head_dim, projectors)

    # Load LITM summary from exp-072
    litm = _load_litm_summary()

    # Compute deltas
    nll_10 = nll_by_kappa[1.0]
    delta_nll = {k: round(nll_by_kappa[k] - nll_10, 6) for k in KAPPAS}
    delta_nll_sham = round(nll_sham_15 - nll_10, 6)

    # Apply verdict logic
    delta_05 = abs(delta_nll[0.5])
    delta_15 = abs(delta_nll[1.5])
    delta_20 = abs(delta_nll[2.0])
    h_flat_confirmed = (delta_05 <= FLAT_THRESHOLD and
                        delta_15 <= FLAT_THRESHOLD and
                        delta_20 <= FLAT_THRESHOLD)
    h_redistribute_confirmed = delta_05 > FLAT_THRESHOLD

    # From exp-072: middle↑ (+1 item) and recency↔/↑ at κ=0.5 — already confirmed
    litm_k05 = litm["by_kappa"].get("target_k0.5", {})
    middle_improved = (litm_k05.get("middle", 0) > litm["base_middle"])
    recency_maintained = (litm_k05.get("recency", 0) >= litm["base_recency"] - 1/40)

    if h_flat_confirmed and middle_improved and recency_maintained:
        verdict = "CLEAN_WIN"
    elif h_redistribute_confirmed:
        verdict = "REDISTRIBUTION"
    else:
        verdict = "PARTIAL_WIN"  # middle↑, recency↔, perplexity ambiguous

    sham_specificity = abs(delta_nll_sham) < abs(delta_nll[1.5])

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    results = {
        "experiment": "exp-075",
        "timestamp_utc": ts,
        "model": model_name,
        "device": str(device),
        "dtype": "bfloat16",
        "prereg_commit": prereg.get("preregistration_commit", "e6fbaecf"),
        "n_passages": len(passages),
        "seq_len": WIKI_SEQ_LEN,
        "dataset": "wikitext-103-v1 test (first 50 passages ≥512 tokens)",
        "flat_threshold": FLAT_THRESHOLD,
        "perplexity": {
            "nll_baseline_unedited": round(nll_baseline, 6),
            "nll_by_kappa": {str(k): round(v, 6) for k, v in nll_by_kappa.items()},
            "delta_nll_vs_k10": {str(k): v for k, v in delta_nll.items()},
            "nll_sham_k15": round(nll_sham_15, 6),
            "delta_nll_sham_vs_k10": delta_nll_sham,
        },
        "litm_summary_from_exp072": litm,
        "hypotheses": {
            "H_flat": {
                "description": "All |delta_NLL| <= 0.05 nats/tok",
                "confirmed": h_flat_confirmed,
                "delta_05": delta_05,
                "delta_15": delta_15,
                "delta_20": delta_20,
            },
            "H_redistribute": {
                "description": "|delta_NLL(0.5)| > 0.05 nats/tok",
                "confirmed": h_redistribute_confirmed,
            },
        },
        "sham_specificity": {
            "confirmed": sham_specificity,
            "note": "|delta_sham| < |delta_target_k1.5| means the effect is head-specific",
            "delta_target_k15": abs(delta_nll[1.5]),
            "delta_sham_k15": abs(delta_nll_sham),
        },
        "verdict": verdict,
        "verdict_components": {
            "perplexity_holds": h_flat_confirmed,
            "middle_improved": middle_improved,
            "recency_maintained": recency_maintained,
        },
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {out_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60)
    print(f"EXP-075 RESULT: {verdict}")
    print("=" * 60)
    print(f"  Baseline NLL (unedited): {nll_baseline:.5f} nats/tok")
    print(f"  κ=1.0 (identity edit):   {nll_by_kappa[1.0]:.5f} "
          f"(Δ={delta_nll[1.0]:+.5f})")
    print(f"  κ=0.5 (flatten):         {nll_by_kappa[0.5]:.5f} "
          f"(Δ={delta_nll[0.5]:+.5f})")
    print(f"  κ=1.5 (sharpen):         {nll_by_kappa[1.5]:.5f} "
          f"(Δ={delta_nll[1.5]:+.5f})")
    print(f"  κ=2.0 (over-sharpen):    {nll_by_kappa[2.0]:.5f} "
          f"(Δ={delta_nll[2.0]:+.5f})")
    print(f"  κ=1.5 sham:              {nll_sham_15:.5f} "
          f"(Δ={delta_nll_sham:+.5f})")
    print()
    print("  H_flat confirmed:", h_flat_confirmed)
    print("  H_redistribute confirmed:", h_redistribute_confirmed)
    print("  Sham specificity:", sham_specificity)
    print()
    print("  LITM at κ=0.5 (from exp-072):")
    print(f"    Middle:  {litm['base_middle']:.3f} → {litm_k05.get('middle', '?'):.3f} "
          f"({'↑' if middle_improved else '↔/↓'})")
    print(f"    Recency: {litm['base_recency']:.3f} → {litm_k05.get('recency', '?'):.3f} "
          f"({'↔/↑' if recency_maintained else '↓'})")
    print()
    print(f"  → VERDICT: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
