"""
exp-145 — Preparatory characterization: κ̃_K for GPT-2 medium

Computes κ̃_K (positional-field read gain, relative to isotropic) for every
attention head in GPT-2 medium. This is the same metric from exp-137 on GPT-2
small. Output identifies:
  - Δ-window heads (low κ̃_K; cross-check against exp-118's 59 heads)
  - Steep/local heads (high κ̃_K; candidates for suppression in the battery)

This script runs BEFORE the pre-registration and is purely instrumental —
no hypothesis is tested here. Results feed the selection of target heads in
prereg.md.

Ariel — September 17, 2026, ~4:45 AM MDT, solo.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN  = 512
N_INPUTS = 50
SEED     = 42

D_MODEL  = 1024   # GPT-2 medium
D_HEAD   = 64     # same as GPT-2 small
N_LAYERS = 24
N_HEADS  = 16

MODEL_ID = "gpt2-medium"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Δ-window heads from exp-118 (WikiText text-native census, GPT-2 medium)
# These 59 heads had wiki_delta ∈ [0.20, 0.30] with R² ≥ 0.90.
def _load_wiki_heads() -> set:
    exp118_path = (Path(__file__).resolve().parent.parent /
                   "exp-118_wikitext_census_cross_family" / "results.json")
    data = json.load(open(exp118_path))
    for entry in data:
        if entry.get("model") == "gpt2-medium":
            return set(tuple(h) for h in entry["wiki_heads"])
    raise ValueError("gpt2-medium not found in exp-118 results")

WIKI_HEADS_EXP118 = _load_wiki_heads()


def compute_all_positional_fields(model: GPT2LMHeadModel,
                                   rng: np.random.Generator) -> dict[int, np.ndarray]:
    """
    Compute positional fields for ALL layers in a single set of 50 forward passes.
    Each forward pass contributes to all layers simultaneously via hooks.
    Returns {layer: delta (SEQ_LEN, D_MODEL), centered}.
    """
    model.eval()
    tok_ids = rng.integers(0, model.config.vocab_size, size=(N_INPUTS, SEQ_LEN))
    tokens  = torch.tensor(tok_ids, dtype=torch.long, device=DEVICE)

    # Accumulate activations across all layers
    layer_acts: dict[int, list] = {ell: [] for ell in range(N_LAYERS)}

    handles = []
    for ell in range(N_LAYERS):
        def make_hook(layer_idx):
            def hook_fn(mod, inp, out):
                layer_acts[layer_idx].append(out.detach().cpu().float().numpy())
            return hook_fn
        handles.append(
            model.transformer.h[ell].ln_1.register_forward_hook(make_hook(ell))
        )

    with torch.no_grad():
        model(tokens)

    for h in handles:
        h.remove()

    fields = {}
    for ell in range(N_LAYERS):
        acts  = np.concatenate(layer_acts[ell], axis=0)   # (N_INPUTS, SEQ_LEN, D_MODEL)
        xbar  = acts.mean(axis=0)                          # (SEQ_LEN, D_MODEL)
        delta = xbar - xbar.mean(axis=0, keepdims=True)
        fields[ell] = delta.astype(np.float64)

    return fields


def get_wk(model: GPT2LMHeadModel, layer: int, head: int) -> np.ndarray:
    """W_K for (layer, head): shape (D_MODEL, D_HEAD)."""
    W = model.transformer.h[layer].attn.c_attn.weight.detach().cpu().double().numpy()
    offset = D_MODEL + head * D_HEAD
    return W[:, offset: offset + D_HEAD]


def compute_kappa(W_K: np.ndarray, delta: np.ndarray) -> float:
    """κ̃(W_K) — isotropic-normalised positional capture (exp-137 formula)."""
    reads      = delta @ W_K
    cap        = float((np.linalg.norm(reads, axis=1) ** 2).sum() /
                       (np.linalg.norm(delta, axis=1) ** 2).sum())
    norm_factor = float((W_K ** 2).sum()) / D_MODEL
    return cap / norm_factor if norm_factor > 1e-14 else 0.0


def main():
    print(f"Loading {MODEL_ID}...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    rng = np.random.default_rng(SEED)

    print(f"Computing positional fields for all {N_LAYERS} layers in one pass batch...",
          flush=True)
    fields = compute_all_positional_fields(model, rng)
    print("Positional fields computed.", flush=True)

    results = []
    for layer in range(N_LAYERS):
        delta = fields[layer]
        for head in range(N_HEADS):
            W_K   = get_wk(model, layer, head)
            kappa = compute_kappa(W_K, delta)
            in_wiki = (layer, head) in WIKI_HEADS_EXP118
            results.append({
                "layer": layer,
                "head":  head,
                "kappa_K": round(kappa, 4),
                "wiki_head": in_wiki,
            })

    # Save full results
    out_path = Path(__file__).resolve().parent / "kappa_characterization.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL_ID, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                   "n_inputs": N_INPUTS, "seq_len": SEQ_LEN, "seed": SEED,
                   "results": results}, f, indent=2)
    print(f"Saved to {out_path}", flush=True)

    # Summary: top steep/local candidates and Δ-window κ̃_K range
    sorted_by_kappa = sorted(results, key=lambda r: r["kappa_K"], reverse=True)

    print("\n── Top 20 heads by κ̃_K (steep/local candidates) ──")
    for r in sorted_by_kappa[:20]:
        wiki = " [WIKI]" if r["wiki_head"] else ""
        print(f"  L{r['layer']:2d}H{r['head']:2d}: κ̃_K = {r['kappa_K']:.3f}{wiki}")

    wiki_results = [r for r in results if r["wiki_head"]]
    print(f"\n── Δ-window heads (59 from exp-118) κ̃_K range ──")
    print(f"  min: {min(r['kappa_K'] for r in wiki_results):.4f}")
    print(f"  max: {max(r['kappa_K'] for r in wiki_results):.4f}")
    print(f"  median: {np.median([r['kappa_K'] for r in wiki_results]):.4f}")

    print("\n── 20 Δ-window heads with LOWEST κ̃_K (amplification candidates) ──")
    wiki_sorted = sorted(wiki_results, key=lambda r: r["kappa_K"])
    for r in wiki_sorted[:20]:
        print(f"  L{r['layer']:2d}H{r['head']:2d}: κ̃_K = {r['kappa_K']:.4f}")

    print("\nCharacterization complete.")


if __name__ == "__main__":
    main()
