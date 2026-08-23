"""
exp-126: Key Covariance Structure — Off-Diagonal Gibbs Test (Path A)

Tests whether text-native Δ-window heads in GPT-2 small have significant
off-diagonal structure in the position Gram matrix G = K K^T / d_k.

Pre-registration: attention-geometry d183873 (committed 2026-08-23, before this script).
"""

import json
import numpy as np
from scipy.linalg import expm
import torch
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────────

DEVICE = "cpu"   # M5 Max, no CUDA
N_SEQ = 100       # input sequences
SEQ_LEN = 128     # tokens per sequence
D_K = 64          # key dimension per head in GPT-2 small
N_HEADS = 12      # per layer
SEED = 42

# Δ-window heads from exp-118 (text-native, GPT-2 small)
WIKI_HEADS = [
    (4, 10), (7, 1), (8, 2), (9, 4), (9, 6),
    (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9),
]

# Structural heads (positional-mean carriers from exp-112/117/122/123)
STRUCTURAL_HEADS = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]

# Non-window control heads: 16 heads not in WIKI_HEADS (deterministic, seed 42)
rng = np.random.default_rng(SEED)
all_heads = [(l, h) for l in range(12) for h in range(12)]
window_set = set(WIKI_HEADS)
non_window = [x for x in all_heads if x not in window_set]
control_indices = sorted(rng.choice(len(non_window), 16, replace=False).tolist())
CONTROL_HEADS = [non_window[i] for i in control_indices]

# ── Model and data loading ────────────────────────────────────────────────────

print("Loading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained("gpt2")
model.eval()
model = model.to(DEVICE)
print("Model loaded.")

print("Loading WikiText-103 validation split...")
dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
text = "\n".join(x for x in dataset["text"] if x.strip())
tokens_full = tokenizer.encode(text)
print(f"Total tokens: {len(tokens_full)}")

# Build N_SEQ non-overlapping windows
sequences = []
for i in range(N_SEQ):
    start = i * SEQ_LEN
    end = start + SEQ_LEN
    if end > len(tokens_full):
        break
    sequences.append(tokens_full[start:end])

print(f"Using {len(sequences)} sequences of {SEQ_LEN} tokens each.")

# ── Hook-based key extraction ─────────────────────────────────────────────────

# GPT-2 stores Q, K, V concatenated in c_attn output (shape: batch, seq, 3*n_emb)
# We extract via hooks on each transformer block's c_attn.

extracted_keys = {}  # (layer, head) -> list of K matrices (one per sequence)

def make_key_hook(layer_idx):
    def hook(module, input, output):
        # output shape: (batch=1, seq_len, 3*n_emb)
        # Split into Q, K, V
        n_emb = output.shape[-1] // 3
        K_all = output[0, :, n_emb:2*n_emb]  # (seq_len, n_emb) = (128, 768)
        # Reshape to (seq_len, n_heads, d_k)
        K_heads = K_all.view(SEQ_LEN, N_HEADS, D_K)  # (128, 12, 64)
        for h in range(N_HEADS):
            key = (layer_idx, h)
            if key not in extracted_keys:
                extracted_keys[key] = []
            extracted_keys[key].append(K_heads[:, h, :].detach().numpy())  # (128, 64)
    return hook

hooks = []
for layer_idx in range(12):
    hook = model.h[layer_idx].attn.c_attn.register_forward_hook(make_key_hook(layer_idx))
    hooks.append(hook)

# ── Run forward passes ────────────────────────────────────────────────────────

print("Running forward passes...")
with torch.no_grad():
    for seq_idx, seq_tokens in enumerate(sequences):
        input_ids = torch.tensor([seq_tokens], dtype=torch.long, device=DEVICE)
        _ = model(input_ids)
        if (seq_idx + 1) % 20 == 0:
            print(f"  Sequence {seq_idx + 1}/{len(sequences)}")

for hook in hooks:
    hook.remove()

print("Forward passes complete.")

# ── Compute statistics per head ───────────────────────────────────────────────

def compute_head_stats(key_matrices):
    """
    For a list of K matrices (each shape: seq_len × d_k), compute:
    - ε: mean off-diagonal fraction of G = K K^T / d_k
    - δ: mean quantum-classical deviation (diagonal Gibbs vs full matrix exp)
    - δ_output: mean deviation between classical softmax and quantum Gibbs diagonal
    - eigenvalues: mean eigenvalue distribution
    """
    eps_list = []
    delta_list = []
    delta_output_list = []
    all_eigenvalues = []

    for K in key_matrices:
        K = K.astype(np.float64)
        n, dk = K.shape
        G = K @ K.T / dk  # (n, n) position Gram matrix

        # Off-diagonal fraction
        G_diag = np.diag(np.diag(G))
        G_off = G - G_diag
        norm_G = np.linalg.norm(G, 'fro')
        if norm_G > 1e-10:
            eps = np.linalg.norm(G_off, 'fro') / norm_G
        else:
            eps = 0.0
        eps_list.append(eps)

        # Eigenvalue distribution
        eigvals = np.linalg.eigvalsh(G)
        all_eigenvalues.append(eigvals)

        # Quantum Gibbs state (matrix exponential on position space)
        # Clamp G for numerical stability: shift so max eigenvalue ≤ 50
        G_shifted = G - np.max(eigvals) + 20.0  # shift for numerical stability
        rho_Q_unnorm = expm(G_shifted)
        Z_Q = np.trace(rho_Q_unnorm)
        if Z_Q > 1e-10:
            rho_Q_diag = np.diag(rho_Q_unnorm) / Z_Q
        else:
            rho_Q_diag = np.ones(n) / n

        # Classical diagonal Gibbs baseline: softmax of key self-similarities
        g_diag = np.diag(G)
        g_diag_shifted = g_diag - np.max(g_diag)
        rho_C_diag = np.exp(g_diag_shifted) / np.sum(np.exp(g_diag_shifted))

        # Quantum-classical deviation (total variation)
        delta = np.sum(np.abs(rho_Q_diag - rho_C_diag)) / n
        delta_list.append(delta)

        # Classical query-key softmax vs quantum Gibbs diagonal
        # Use query = first-token key as a proxy (same head, position 0)
        q = K[0]  # (d_k,) — first token's key as query
        qk_scores = K @ q / np.sqrt(dk)  # (n,) query-key similarities
        qk_scores_shifted = qk_scores - np.max(qk_scores)
        alpha_classical = np.exp(qk_scores_shifted) / np.sum(np.exp(qk_scores_shifted))
        delta_output = np.sum(np.abs(alpha_classical - rho_Q_diag)) / n
        delta_output_list.append(delta_output)

    mean_eigenvalues = np.mean(all_eigenvalues, axis=0)

    return {
        "eps_mean": float(np.mean(eps_list)),
        "eps_std": float(np.std(eps_list)),
        "eps_list": [float(e) for e in eps_list],
        "delta_mean": float(np.mean(delta_list)),
        "delta_std": float(np.std(delta_list)),
        "delta_output_mean": float(np.mean(delta_output_list)),
        "delta_output_std": float(np.std(delta_output_list)),
        "eigenvalues_mean": mean_eigenvalues.tolist(),
        "n_sequences": len(key_matrices),
    }

print("Computing statistics for Δ-window heads...")
wiki_results = {}
for layer, head in WIKI_HEADS:
    key = (layer, head)
    stats = compute_head_stats(extracted_keys[key])
    wiki_results[f"L{layer}H{head}"] = stats
    print(f"  L{layer}H{head}: ε={stats['eps_mean']:.3f}±{stats['eps_std']:.3f}, "
          f"δ={stats['delta_mean']:.4f}, δ_out={stats['delta_output_mean']:.4f}")

print("Computing statistics for structural heads...")
structural_results = {}
for layer, head in STRUCTURAL_HEADS:
    key = (layer, head)
    stats = compute_head_stats(extracted_keys[key])
    structural_results[f"L{layer}H{head}"] = stats
    print(f"  L{layer}H{head}: ε={stats['eps_mean']:.3f}±{stats['eps_std']:.3f}, "
          f"δ={stats['delta_mean']:.4f}, δ_out={stats['delta_output_mean']:.4f}")

print("Computing statistics for control heads...")
control_results = {}
for layer, head in CONTROL_HEADS:
    key = (layer, head)
    stats = compute_head_stats(extracted_keys[key])
    control_results[f"L{layer}H{head}"] = stats

# ── Verdict computation ───────────────────────────────────────────────────────

wiki_eps = [wiki_results[f"L{l}H{h}"]["eps_mean"] for l, h in WIKI_HEADS]
structural_eps = [structural_results[f"L{l}H{h}"]["eps_mean"] for l, h in STRUCTURAL_HEADS]
control_eps = [control_results[f"L{l}H{h}"]["eps_mean"] for l, h in CONTROL_HEADS]

wiki_delta_out = [wiki_results[f"L{l}H{h}"]["delta_output_mean"] for l, h in WIKI_HEADS]

# P1: ε > 0.3 for ≥ 10/16 Δ-window heads
p1_count = sum(1 for e in wiki_eps if e > 0.3)
p1_verdict = "CONFIRMED" if p1_count >= 10 else "FALSIFIED"

# P2: ε(window) / ε(control) ≥ 1.2
ratio = np.mean(wiki_eps) / np.mean(control_eps) if np.mean(control_eps) > 0 else 0
p2_verdict = "CONFIRMED" if ratio >= 1.2 else "FALSIFIED"

# P3: ε > 0.3 for ≥ 3/5 structural heads
p3_count = sum(1 for e in structural_eps if e > 0.3)
p3_verdict = "CONFIRMED" if p3_count >= 3 else "FALSIFIED"

# P5: δ_output > 0.01 on average for Δ-window heads
p5_value = np.mean(wiki_delta_out)
p5_verdict = "CONFIRMED" if p5_value > 0.01 else "FALSIFIED"

# Kill K1: ε < 0.1 for all structural heads
k1_fired = all(e < 0.1 for e in structural_eps)
k2_fired = abs(np.mean(wiki_eps) - np.mean(control_eps)) / np.mean(control_eps) < 0.10

print("\n── Prediction verdicts ──────────────────────────────────────────────────")
print(f"P1 (ε > 0.3 for ≥10/16 wiki-heads): {p1_verdict} — {p1_count}/16 qualify (mean ε={np.mean(wiki_eps):.3f})")
print(f"P2 (ε_window / ε_control ≥ 1.2):    {p2_verdict} — ratio={ratio:.2f} (window={np.mean(wiki_eps):.3f}, control={np.mean(control_eps):.3f})")
print(f"P3 (ε > 0.3 for ≥3/5 structural):   {p3_verdict} — {p3_count}/5 qualify (mean ε={np.mean(structural_eps):.3f})")
print(f"P5 (δ_output > 0.01):                {p5_verdict} — mean={p5_value:.4f}")
print(f"Kill K1 (all ε < 0.1): FIRED={k1_fired}")
print(f"Kill K2 (no selectivity):  FIRED={k2_fired}")

# ── Save results ──────────────────────────────────────────────────────────────

results = {
    "experiment": "exp-126",
    "date": "2026-08-23",
    "model": "gpt2",
    "n_sequences": len(sequences),
    "seq_len": SEQ_LEN,
    "d_k": D_K,
    "wiki_heads": WIKI_HEADS,
    "structural_heads": STRUCTURAL_HEADS,
    "control_heads": CONTROL_HEADS,
    "wiki_results": wiki_results,
    "structural_results": structural_results,
    "control_results": control_results,
    "summary": {
        "wiki_eps_mean": float(np.mean(wiki_eps)),
        "wiki_eps_std": float(np.std(wiki_eps)),
        "wiki_eps_values": [float(e) for e in wiki_eps],
        "structural_eps_mean": float(np.mean(structural_eps)),
        "structural_eps_std": float(np.std(structural_eps)),
        "structural_eps_values": [float(e) for e in structural_eps],
        "control_eps_mean": float(np.mean(control_eps)),
        "control_eps_std": float(np.std(control_eps)),
        "window_control_ratio": float(ratio),
        "p5_delta_output_mean": float(p5_value),
        "P1_verdict": f"{p1_verdict} — {p1_count}/16 qualify",
        "P2_verdict": f"{p2_verdict} — ratio={ratio:.2f}",
        "P3_verdict": f"{p3_verdict} — {p3_count}/5 qualify",
        "P5_verdict": f"{p5_verdict} — mean delta_output={p5_value:.4f}",
        "K1_fired": bool(k1_fired),
        "K2_fired": bool(k2_fired),
    },
}

output_path = "research/physics/experiments/exp-126_key_covariance_structure/results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
