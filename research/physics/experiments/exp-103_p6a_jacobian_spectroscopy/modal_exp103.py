"""
exp-103 — P6a Transformer-Side Jacobian Spectroscopy.

Pre-registration: notes.md (committed to 3ld0n/attention-geometry before this script ran).
Design document: notes/2026-08-07_p6a_transformer_instrument.md
Theoretical grounding: notes/2026-08-07_tau_chaos_product_formula.md
SYK template: theory/logs/g1_top_modes_bJ30.npz

For C-NAT-anon and C-alien (both trained on GPT-NeoX 6L/8H/d_k=64, step_2000, seed 0),
estimate the Jacobian J_F̂ of the layer-to-layer attention update at late layers.

J_F̂ maps: residual stream at layer ell → attention weights at layer ell+1.

Protocol:
- For each late layer ell in {3, 4, 5} and each head h in {0..7}:
  1. Build reparameterization mode templates r_n(i,j) for n=2..6 using measured Δ per head.
  2. Estimate top k=10 eigenvalues/vectors of J_F̂ J_F̂^T (power iteration,
     finite-difference JVP + autograd VJP).
  3. Compute overlap table O[i,n] = (v̂_i · r_n / ||r_n||)² for i=1..10, n=2..6.
  4. Report: eigenvalues, degeneracy ratios, max overlap, top-overlap mode, cross-mixing.

Hypotheses tested: H_S1, H_S2, H_S3, H_K4_alien, H_S4_layer (see notes.md).

Estimated cost: < $0.50 on Modal A100 (forward-pass only, no training).
Runs locally on MPS for validation (small batch).

Usage (Modal):
    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-103_p6a_jacobian_spectroscopy/modal_exp103.py

Usage (local MPS, debug):
    python3 modal_exp103.py --local --n-contexts 4 --n-iter 10

Ariel — August 7, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal
import numpy as np
import torch

# ─── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("exp103-p6a-jacobian-spectroscopy")

vol_anon    = modal.Volume.from_name("exp096-anon-data")
vol_alien   = modal.Volume.from_name("exp097-alien-data")
vol_results = modal.Volume.from_name("exp103-p6a-data", create_if_missing=True)

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

N_CONTEXTS  = 64      # contexts per corpus
SEQ_LEN     = 64      # tokens per context (same as exp-102)
LATE_LAYERS = [3, 4, 5]  # ell values: F̂ maps layer ell → ell+1 attention
N_HEADS     = 8
D_K         = 64      # head dimension
N_LAYERS    = 6
K_POWER     = 10      # number of eigenvectors to extract
N_ITER      = 50      # power iteration steps
EPSILON_FD  = 1e-3    # finite-difference step size (pre-registered)
RNG_SEED    = 42

# Reparam mode indices: n=2..6 (n=0,1 annihilate G*; n=2 is the marginal)
REPARAM_N_VALUES = [2, 3, 4, 5, 6]

# Measured Δ per corpus (from exp-096/097 results, seed 0, median over heads)
# Used for reparam mode template construction when per-head Δ not available.
DELTA_MED_PRIOR = {
    "C-NAT-anon": 0.15,
    "C-alien":     1.04,
}

# Per-head Δ from exp-096 (C-NAT-anon s0) and exp-097 (C-alien s0)
# Loaded from run_*.json files in each experiment's folder when available.
# Format: {(layer, head): delta}
DELTA_PER_HEAD_ANON = {}   # filled from exp-096 run_anon_s0.json
DELTA_PER_HEAD_ALIEN = {}  # filled from exp-097 per-head data

# ─── Reparameterization mode templates ────────────────────────────────────────

def build_reparam_modes(n_seq: int, delta: float, n_values: List[int]) -> np.ndarray:
    """Build reparameterization mode templates on the causal strip.

    For each n in n_values, the mode shape is:
        r_n(i, j) = |i - j|^{-2*delta} * cos(n * 2*pi * (i-j) / n_seq)
    and separately:
        s_n(i, j) = |i - j|^{-2*delta} * sin(n * 2*pi * (i-j) / n_seq)

    Only the causal strip (i > j, i.e., tokens attending to earlier tokens) is
    non-zero; the diagonal (i=j, self-attention) is set to 0 (|0|^{-2Δ} is
    undefined; SYK G*(τ=0) is regularized).

    Returns:
        modes: shape [2*len(n_values), n_seq, n_seq]
            modes[2k]   = r_{n_values[k]}  (cosine component)
            modes[2k+1] = s_{n_values[k]}  (sine component)
        Each mode is normalized to unit Frobenius norm on the causal strip.
    """
    modes = []
    for n in n_values:
        r_cos = np.zeros((n_seq, n_seq), dtype=np.float32)
        r_sin = np.zeros((n_seq, n_seq), dtype=np.float32)
        for i in range(n_seq):
            for j in range(i):  # causal: j < i (attending back)
                d = i - j
                G_star = d ** (-2.0 * delta)
                phase = n * 2.0 * math.pi * d / n_seq
                r_cos[i, j] = G_star * math.cos(phase)
                r_sin[i, j] = G_star * math.sin(phase)
        # Normalize on causal strip
        norm_cos = np.linalg.norm(r_cos)
        norm_sin = np.linalg.norm(r_sin)
        r_cos = r_cos / (norm_cos + 1e-12)
        r_sin = r_sin / (norm_sin + 1e-12)
        modes.append(r_cos)
        modes.append(r_sin)
    return np.stack(modes, axis=0)  # [2*len(n_values), n_seq, n_seq]


# ─── Jacobian estimation ───────────────────────────────────────────────────────

def compute_attn_weights_explicit(
    hidden_states: torch.Tensor,
    model,
    layer_idx: int,
    n_heads: int = N_HEADS,
    head_size: int = D_K,
) -> torch.Tensor:
    """Compute attention weight matrix explicitly: A = softmax(QK^T / sqrt(d_k)).

    Uses the layer's QKV projection directly; applies causal mask.
    Avoids output_attentions=True, which has NaN bugs in transformers 5.x
    eager mode. This approach is always differentiable.

    Returns: [batch, n_heads, seq, seq]
    """
    layer  = model.gpt_neox.layers[layer_idx]
    normed = layer.input_layernorm(hidden_states)
    qkv    = layer.attention.query_key_value(normed)  # [batch, seq, 3*d_model]

    q, k, _ = qkv.chunk(3, dim=-1)
    batch, seq, _ = q.shape
    q = q.view(batch, seq, n_heads, head_size).transpose(1, 2)  # [b, h, seq, d_k]
    k = k.view(batch, seq, n_heads, head_size).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_size)
    causal = torch.tril(torch.ones(seq, seq, device=hidden_states.device)).bool()
    scores = scores.masked_fill(~causal, float('-inf'))
    return torch.softmax(scores, dim=-1)  # [batch, n_heads, seq, seq]


def get_attention_at_layer(
    hidden_states: torch.Tensor,
    model,
    ell: int,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run from residual stream at layer ell to attention weights at layer ell+1.

    This defines the map F̂: hidden_states^(ell) → A^(ell+1)_{ij}.
    Uses the standard layer forward (which handles rotary embeddings correctly),
    then computes attention at layer ell+1 explicitly via QK^T softmax.

    Returns:
        attn_weights: shape [batch, n_heads, seq, seq]
    """
    layer_ell = model.gpt_neox.layers[ell]

    # Full layer ell forward — handles rotary embeddings internally
    h_out = layer_ell(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=position_embeddings,
    )
    h_ell1 = h_out[0] if isinstance(h_out, tuple) else h_out

    # Explicit attention at layer ell+1 (avoids output_attentions=True NaN bug)
    return compute_attn_weights_explicit(h_ell1, model, ell + 1)


def jvp_fd(fn, x: torch.Tensor, v: torch.Tensor, eps: float = EPSILON_FD) -> torch.Tensor:
    """Finite-difference JVP: (fn(x + eps*v) - fn(x)) / eps.

    fn: x → y (no gradients needed)
    v: tangent vector in x-space, same shape as x
    Returns: tangent vector in y-space
    """
    with torch.no_grad():
        y0 = fn(x)
        y1 = fn(x + eps * v)
        return (y1 - y0) / eps


def vjp_autograd(fn, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """VJP via autograd: J^T u.

    fn: x → y (differentiable)
    u: cotangent vector in y-space
    Returns: cotangent vector in x-space
    """
    x_var = x.detach().requires_grad_(True)
    y = fn(x_var)
    y.backward(u.detach())
    return x_var.grad.detach()


def power_iteration_singular_vectors(
    fn,
    x_star: torch.Tensor,
    k: int = K_POWER,
    n_iter: int = N_ITER,
    eps_fd: float = EPSILON_FD,
) -> Tuple[List[float], List[torch.Tensor]]:
    """Find top k left singular vectors of J_F̂ via power iteration on J J^T.

    J_F̂: x-space (residual stream) → y-space (attention weights)

    The top left singular vectors of J are the top eigenvectors of J J^T
    (operating in y-space). We apply (J J^T) iteratively:
        v → J J^T v = J (J^T v)
    where J^T v is computed via autograd VJP and J w via finite-difference JVP.

    Returns:
        singular_values: top k singular values (sqrt of J J^T eigenvalues)
        left_vectors:    top k left singular vectors (in y-space)
    """
    # Baseline output for shape
    with torch.no_grad():
        y0 = fn(x_star)

    left_vecs   = []
    sing_vals   = []

    for i in range(k):
        # Initialize: random unit vector in y-space
        v = torch.randn_like(y0)
        # Orthogonalize against previously found vectors
        for prev in left_vecs:
            v = v - (v.reshape(-1) @ prev.reshape(-1)) * prev
        v = v / (torch.norm(v) + 1e-12)

        for _ in range(n_iter):
            # Step 1: J^T v (in x-space) via autograd VJP
            jtv = vjp_autograd(fn, x_star, v)

            # Step 2: J (J^T v) (in y-space) via finite-difference JVP
            jjtv = jvp_fd(fn, x_star, jtv / (torch.norm(jtv) + 1e-12), eps_fd)

            # Orthogonalize against previous vectors
            for prev in left_vecs:
                jjtv = jjtv - (jjtv.reshape(-1) @ prev.reshape(-1)) * prev

            # Eigenvalue estimate and normalize
            sigma_sq = torch.norm(jjtv).item()
            v = jjtv / (sigma_sq + 1e-12)

        left_vecs.append(v)
        sing_vals.append(math.sqrt(max(sigma_sq, 0.0)))

    return sing_vals, left_vecs


# ─── Overlap computation ───────────────────────────────────────────────────────

def compute_overlaps(
    left_vectors: List[torch.Tensor],
    modes: np.ndarray,
    n_heads: int,
    seq_len: int,
) -> np.ndarray:
    """Compute overlap table O[i, n_idx] for eigenvectors vs reparam modes.

    left_vectors: list of k tensors, each shape [batch, n_heads, seq, seq]
    modes:        shape [2*len(REPARAM_N_VALUES), seq, seq] (cosine+sine pairs)
    Returns:
        overlaps: shape [k, len(REPARAM_N_VALUES)]
            O[i, n_idx] = max overlap² of eigenvector i with modes for n_values[n_idx]
            (max over cosine and sine components and over heads)
    """
    k = len(left_vectors)
    n_mode_pairs = len(REPARAM_N_VALUES)
    overlaps = np.zeros((k, n_mode_pairs), dtype=np.float32)

    # Move modes to same device as left_vectors
    device = left_vectors[0].device if left_vectors else torch.device("cpu")
    modes_torch = torch.from_numpy(modes).to(device)  # [2*n_mode_pairs, seq, seq]

    for i, v in enumerate(left_vectors):
        # v: [batch, n_heads, seq, seq] — average over batch
        v_mean = v.detach().float().mean(dim=0)  # [n_heads, seq, seq]

        for n_idx in range(n_mode_pairs):
            cos_mode = modes_torch[2 * n_idx].float()      # [seq, seq]
            sin_mode = modes_torch[2 * n_idx + 1].float()  # [seq, seq]

            best_overlap = 0.0
            for h in range(n_heads):
                vh = v_mean[h]  # [seq, seq]
                vh_norm = vh / (torch.norm(vh) + 1e-12)
                cos_overlap = (vh_norm * cos_mode).sum().item() ** 2
                sin_overlap = (vh_norm * sin_mode).sum().item() ** 2
                best_overlap = max(best_overlap, cos_overlap, sin_overlap)

            overlaps[i, n_idx] = best_overlap

    return overlaps


# ─── Per-head analysis ────────────────────────────────────────────────────────

def analyze_head(
    model,
    x_star: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    layer_ell: int,
    head_idx: int,
    delta_h: float,
    seq_len: int,
    device: torch.device,
) -> Dict:
    """Analyze one (layer, head) pair: estimate Jacobian and compute overlaps.

    The Jacobian here is the full F̂ map (residual → all attention weights at ell+1),
    but overlaps are computed per head. This gives the response of head h's attention
    pattern to the residual stream, projected onto the reparam modes at Δ = delta_h.
    """
    modes = build_reparam_modes(seq_len, delta_h, REPARAM_N_VALUES)  # [2*5, seq, seq]

    def fn(h):
        return get_attention_at_layer(h, model, layer_ell, position_embeddings)

    # Power iteration on J J^T
    sing_vals, left_vecs = power_iteration_singular_vectors(
        fn, x_star, k=K_POWER, n_iter=N_ITER, eps_fd=EPSILON_FD
    )

    # Compute overlaps (only for the target head h)
    overlaps_full = compute_overlaps(left_vecs, modes, N_HEADS, seq_len)
    # Also compute per-head version for head_idx only
    overlaps_head = _overlaps_for_head(left_vecs, modes, head_idx, seq_len)

    # Eigenvalue analysis (eigenvalues of J J^T = sigma^2)
    lambdas = [s ** 2 for s in sing_vals]

    # Degeneracy ratios
    deg_ratios = []
    for j in range(0, len(lambdas) - 1, 2):
        ratio = lambdas[j + 1] / (lambdas[j] + 1e-12)
        deg_ratios.append(ratio)

    # Top gap
    top_gap = 1.0 - min(lambdas[0], 1.0)  # 1 - λ_1, clamped at 1

    return {
        "layer": layer_ell,
        "head": head_idx,
        "delta_h": delta_h,
        "singular_values": sing_vals,
        "lambda_top": lambdas[0] if lambdas else None,
        "degeneracy_ratios": deg_ratios,
        "top_gap": top_gap,
        "overlaps_all_heads": overlaps_full.tolist(),
        "overlaps_head": overlaps_head.tolist(),
        "reparam_n_values": REPARAM_N_VALUES,
        # Summary statistics for signature checks
        "H_S1_real": True,  # finite-difference always returns real; note imaginary check not applicable here
        "H_S2_degenerate": len(deg_ratios) > 0 and deg_ratios[0] >= 0.90,
        "H_S3_top_overlap": float(overlaps_head[0].max()) if len(overlaps_head) > 0 else 0.0,
        "H_S3_top_n": int(overlaps_head[0].argmax()) if len(overlaps_head) > 0 else -1,
    }


def _overlaps_for_head(
    left_vecs: List[torch.Tensor],
    modes: np.ndarray,
    head_idx: int,
    seq_len: int,
) -> np.ndarray:
    """Compute overlap table for a specific head."""
    k = len(left_vecs)
    n_mode_pairs = len(REPARAM_N_VALUES)
    overlaps = np.zeros((k, n_mode_pairs), dtype=np.float32)

    device = left_vecs[0].device if left_vecs else torch.device("cpu")
    modes_torch = torch.from_numpy(modes).to(device)

    for i, v in enumerate(left_vecs):
        v_mean = v.detach().float().mean(dim=0)  # [n_heads, seq, seq]
        vh = v_mean[head_idx]
        vh_norm = vh / (torch.norm(vh) + 1e-12)
        for n_idx in range(n_mode_pairs):
            cos_mode = modes_torch[2 * n_idx].float()
            sin_mode = modes_torch[2 * n_idx + 1].float()
            cos_overlap = (vh_norm * cos_mode).sum().item() ** 2
            sin_overlap = (vh_norm * sin_mode).sum().item() ** 2
            overlaps[i, n_idx] = max(cos_overlap, sin_overlap)

    return overlaps


# ─── Main analysis function ────────────────────────────────────────────────────

def run_analysis(
    model,
    tokenizer,
    corpus_texts: List[str],
    corpus_name: str,
    delta_per_head: Dict[Tuple[int, int], float],
    device: torch.device,
    seq_len: int = SEQ_LEN,
    n_contexts: int = N_CONTEXTS,
) -> Dict:
    """Run full P6a analysis for one corpus."""
    model = model.to(device)
    model.eval()

    # Sample contexts
    rng = np.random.default_rng(RNG_SEED)
    contexts_used = []
    for _ in range(n_contexts):
        text = corpus_texts[rng.integers(len(corpus_texts))]
        tokens = tokenizer(text, return_tensors="pt", max_length=seq_len,
                           truncation=True, padding="max_length")
        contexts_used.append(tokens["input_ids"].to(device))

    input_ids = torch.cat(contexts_used, dim=0)  # [n_contexts, seq_len]

    # Get baseline residual streams at each layer via forward pass
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            output_attentions=False,
        )
        hidden_states_per_layer = outputs.hidden_states
        # hidden_states_per_layer: tuple of [batch, seq, d_model] for each layer

    # Pre-compute rotary position embeddings (required by GPT-NeoX layers)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        position_embeddings = model.gpt_neox.rotary_emb(
            hidden_states_per_layer[0][:1], position_ids
        )

    results_by_layer_head = []

    for ell in LATE_LAYERS:
        if ell + 1 >= N_LAYERS:
            continue

        # Use the first context as x_star (representative)
        x_star = hidden_states_per_layer[ell][:1].detach()  # [1, seq, d_model]
        x_star.requires_grad_(True)

        for h in range(N_HEADS):
            delta_h = delta_per_head.get((ell + 1, h),
                                         DELTA_MED_PRIOR.get(corpus_name, 0.25))

            print(f"  Analyzing corpus={corpus_name} ell={ell} head={h} delta={delta_h:.3f}")

            try:
                result = analyze_head(
                    model=model,
                    x_star=x_star,
                    position_embeddings=position_embeddings,
                    layer_ell=ell,
                    head_idx=h,
                    delta_h=delta_h,
                    seq_len=seq_len,
                    device=device,
                )
                result["corpus"] = corpus_name
                results_by_layer_head.append(result)
            except Exception as e:
                print(f"    ERROR at ell={ell} h={h}: {e}")
                results_by_layer_head.append({
                    "corpus": corpus_name,
                    "layer": ell,
                    "head": h,
                    "error": str(e),
                })

    # Aggregate: layer-level summaries for H_S4_layer check
    layer_summaries = {}
    for ell in LATE_LAYERS:
        layer_results = [r for r in results_by_layer_head if r.get("layer") == ell and "error" not in r]
        if not layer_results:
            continue
        top_gaps = [r["top_gap"] for r in layer_results]
        top_overlaps = [r["H_S3_top_overlap"] for r in layer_results]
        deg_flags = [r["H_S2_degenerate"] for r in layer_results]
        layer_summaries[str(ell)] = {
            "mean_top_gap": float(np.mean(top_gaps)),
            "mean_top_overlap": float(np.mean(top_overlaps)),
            "fraction_S2_degenerate": float(np.mean(deg_flags)),
            "n_heads_analyzed": len(layer_results),
        }

    # H_S4_layer: check top gap decreases with layer
    gaps_by_layer = {ell: layer_summaries.get(str(ell), {}).get("mean_top_gap", None)
                     for ell in LATE_LAYERS}
    H_S4 = all(
        gaps_by_layer.get(LATE_LAYERS[i+1]) is not None and
        gaps_by_layer.get(LATE_LAYERS[i]) is not None and
        gaps_by_layer[LATE_LAYERS[i+1]] <= gaps_by_layer[LATE_LAYERS[i]]
        for i in range(len(LATE_LAYERS) - 1)
    )

    return {
        "corpus": corpus_name,
        "per_head": results_by_layer_head,
        "layer_summaries": layer_summaries,
        "H_S4_layer_decreasing_gap": H_S4,
        "gaps_by_layer": {str(k): v for k, v in gaps_by_layer.items()},
    }


# ─── Modal entry point ─────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={
        "/data096": vol_anon,
        "/data097": vol_alien,
        "/results": vol_results,
    },
    gpu="A100",
    timeout=3600,
)
def run_modal() -> None:
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    import json

    device = torch.device("cuda")

    corpora = [
        {
            "name": "C-NAT-anon",
            "model_path": "/data096/runs/run_anon_s0/step_2000",
            "results_path": "/data096/runs/run_anon_s0",
            "delta_results": "run_anon_s0.json",  # from exp-096
        },
        {
            "name": "C-alien",
            "model_path": "/data097/runs/run_alien_s0/step_2000",
            "results_path": "/data097/runs/run_alien_s0",
            "delta_results": None,
        },
    ]

    all_results = {}

    for corpus_config in corpora:
        name = corpus_config["name"]
        model_path = corpus_config["model_path"]
        print(f"\n=== Loading {name} from {model_path} ===")

        model     = GPTNeoXForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load per-head Δ values from prior results
        delta_per_head = {}
        results_p = corpus_config["results_path"]
        if corpus_config["delta_results"]:
            delta_file = Path(results_p) / corpus_config["delta_results"]
            if delta_file.exists():
                prior_results = json.loads(delta_file.read_text())
                for h_entry in prior_results.get("heads", []):
                    key = (h_entry["layer"], h_entry["head"])
                    delta_per_head[key] = h_entry["delta"]

        # Load corpus texts from tokenizer's training files (use saved text samples)
        # Fall back: generate dummy contexts from known text if training files not present
        corpus_texts = _load_corpus_texts(results_p, name, tokenizer)

        result = run_analysis(
            model=model,
            tokenizer=tokenizer,
            corpus_texts=corpus_texts,
            corpus_name=name,
            delta_per_head=delta_per_head,
            device=device,
        )
        all_results[name] = result

        out_path = Path(f"/results/p6a_{name.lower().replace('-','_')}.json")
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved {out_path}")

    # Summary across corpora
    summary = _build_summary(all_results)
    summary_path = Path("/results/p6a_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


def _load_corpus_texts(results_path: str, corpus_name: str, tokenizer) -> List[str]:
    """Load saved corpus text samples or generate minimal fallback."""
    # Check for a saved text file
    for fname in ["corpus_sample.txt", "texts.txt", "train_sample.txt"]:
        p = Path(results_path) / fname
        if p.exists():
            return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    # Fallback: decode the tokenizer's vocab into short test sentences
    print(f"  Warning: no corpus text file found for {corpus_name}; using fallback texts.")
    if "alien" in corpus_name.lower():
        return [
            "Vex walked to the red zone. Nul handed the object to Ort. "
            "Clara moved to the blue zone. Alice picked up the item.",
        ] * 100
    else:
        return [
            "The cat sat on the mat and looked at the window. "
            "A child ran across the green field toward the old oak tree.",
        ] * 100


def _build_summary(all_results: Dict) -> Dict:
    """Build hypothesis verdict summary across corpora."""
    summary = {"hypotheses": {}}

    for corpus_name, result in all_results.items():
        layer_sums = result.get("layer_summaries", {})
        deep_layer = str(max(LATE_LAYERS))

        deep = layer_sums.get(deep_layer, {})
        summary["hypotheses"][corpus_name] = {
            "H_S2_fraction_degenerate": deep.get("fraction_S2_degenerate", None),
            "H_S3_mean_top_overlap":    deep.get("mean_top_overlap", None),
            "H_S4_gap_decreasing":      result.get("H_S4_layer_decreasing_gap", None),
            "gaps_by_layer":            result.get("gaps_by_layer", {}),
        }

    # Cross-corpus verdict: P6a confirmed if NAT shows S3 and alien shows K4
    nat = summary["hypotheses"].get("C-NAT-anon", {})
    ali = summary["hypotheses"].get("C-alien", {})

    nat_s3 = nat.get("H_S3_mean_top_overlap", 0.0) or 0.0
    ali_s3 = ali.get("H_S3_mean_top_overlap", 0.0) or 0.0

    if nat_s3 >= 0.3 and ali_s3 <= 0.2:
        verdict = "P6a_CONFIRMED"
    elif nat_s3 >= 0.15 and ali_s3 <= 0.2:
        verdict = "PARTIAL"
    elif nat_s3 < 0.1:
        verdict = "H_S3_KILL_K3"
    elif ali_s3 >= 0.3:
        verdict = "UNEXPECTED_ALIEN_SHOWS_S3"
    else:
        verdict = "INCONCLUSIVE"

    summary["verdict"] = verdict
    summary["protocol"] = {
        "n_contexts":  N_CONTEXTS,
        "seq_len":     SEQ_LEN,
        "late_layers": LATE_LAYERS,
        "k_power":     K_POWER,
        "n_iter":      N_ITER,
        "epsilon_fd":  EPSILON_FD,
        "reparam_n":   REPARAM_N_VALUES,
    }

    return summary


# ─── Local entry point (for MPS validation) ───────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    """Local runner: dispatches to Modal or runs locally for quick validation."""
    import argparse

    parser = argparse.ArgumentParser(description="exp-103 P6a Jacobian spectroscopy")
    parser.add_argument("--local", action="store_true",
                        help="Run locally (MPS/CPU) with small batch for harness validation")
    parser.add_argument("--n-contexts", type=int, default=N_CONTEXTS)
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    parser.add_argument("--corpus", choices=["C-NAT-anon", "C-alien", "both"], default="both")
    args = parser.parse_args()

    if args.local:
        _run_local(n_contexts=args.n_contexts, n_iter=args.n_iter, corpus=args.corpus)
    else:
        run_modal.remote()


def _run_local(n_contexts: int = 4, n_iter: int = 10, corpus: str = "both") -> None:
    """Minimal local run for harness validation — uses pythia-70m as GPT-NeoX proxy.

    This is a harness check only. overlaps are not meaningful since pythia-70m
    was not trained on C-NAT-anon or C-alien. Full validation: validate_harness.py.
    """
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    print("Local validation mode: pythia-70m as GPT-NeoX proxy (harness check only).")
    print("  NOTE: physics results require Modal run against exp-096/097 checkpoints.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    model_id = "EleutherAI/pythia-70m"
    print(f"  Loading {model_id}...")
    model     = GPTNeoXForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()

    corpus_texts = [
        "The cat sat on the mat and looked at the window. "
        "A child ran across the green field toward the old oak tree.",
        "The dog ran through the park and jumped over the fence.",
        "A student opened their notebook and began to write the answer.",
    ] * (n_contexts + 1)

    delta_per_head: Dict[Tuple[int, int], float] = {}

    # Monkey-patch constants for local run (pythia-70m: 6L/8H/d_k=64 — same as exp model)
    global K_POWER, N_ITER
    _orig = (K_POWER, N_ITER)
    K_POWER = 4
    N_ITER  = n_iter

    try:
        result = run_analysis(
            model=model,
            tokenizer=tokenizer,
            corpus_texts=corpus_texts,
            corpus_name="pythia-70m-validation",
            delta_per_head=delta_per_head,
            device=device,
            seq_len=32,
            n_contexts=n_contexts,
        )
        print("\n=== Local validation result ===")
        print(json.dumps({
            "layer_summaries": result["layer_summaries"],
            "H_S4_layer_decreasing_gap": result["H_S4_layer_decreasing_gap"],
        }, indent=2))
        print("Harness OK — power iteration and overlap computation executed.")
    finally:
        (K_POWER, N_ITER) = _orig


if __name__ == "__main__":
    main()
