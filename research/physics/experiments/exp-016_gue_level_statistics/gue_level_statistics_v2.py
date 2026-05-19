"""
GUE Level Statistics v2 — Rank-Corrected Analysis

v1 showed <r> ≈ 0.515 for both trained and random, between Poisson and GOE.
Diagnosis: the logit matrix L = QK^T/√d_k has rank ≤ d_k = 64 in a 128×128
matrix, creating ~64 near-zero eigenvalues that contaminate level statistics.

Fixes:
  1. Filter to top-d_k eigenvalues by magnitude (the non-degenerate subspace)
  2. Use seq_len = d_k = 64 so L is generically full rank
  3. Also test: the raw (unsymmetrized) singular values of L

Ariel — March 26, 2026
"""

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

from transformers import GPT2LMHeadModel

R_POISSON = 2 * np.log(2) - 1  # 0.3863
R_GOE = 0.5307
R_GUE = 0.5996


def ratio_statistic(eigenvalues):
    eigs = np.sort(np.real(eigenvalues))
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-12]
    if len(spacings) < 3:
        return np.nan
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return np.mean(ratios)


def classify(r):
    if np.isnan(r):
        return "---"
    dists = {"Poisson": abs(r - R_POISSON), "GOE": abs(r - R_GOE), "GUE": abs(r - R_GUE)}
    return min(dists, key=dists.get)


# ========== Load models ==========
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

d_model = model.config.n_embd       # 768
n_layers = model.config.n_layer     # 12
n_heads = model.config.n_head       # 12
d_k = d_model // n_heads            # 64
vocab_size = model.config.vocab_size

random_model = GPT2LMHeadModel.from_pretrained("gpt2")
with torch.no_grad():
    for layer_idx in range(n_layers):
        block = random_model.transformer.h[layer_idx]
        for name, param in block.attn.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
        for name, param in block.mlp.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
random_model.eval()
print("  Models loaded.\n")


def get_logit_matrices(mdl, input_ids):
    with torch.no_grad():
        outputs = mdl(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    results = {}
    for l in range(n_layers):
        block = mdl.transformer.h[l]
        h = hidden_states[l]
        h_ln = block.ln_1(h)
        qkv = block.attn.c_attn(h_ln)
        q, k, v = qkv.split(d_model, dim=-1)
        seq = q.shape[1]
        q = q.view(1, seq, n_heads, d_k).permute(0, 2, 1, 3)
        k = k.view(1, seq, n_heads, d_k).permute(0, 2, 1, 3)
        for head in range(n_heads):
            q_h = q[0, head]
            k_h = k[0, head]
            L = torch.matmul(q_h, k_h.transpose(-1, -2)) / (d_k ** 0.5)
            results[(l, head)] = L.detach().numpy()
    return results


N_INPUTS = 20

print("=" * 85)
print("  GUE LEVEL STATISTICS v2 — RANK-CORRECTED")
print("=" * 85)
print()
print(f"  Reference: Poisson = {R_POISSON:.4f}  |  GOE = {R_GOE:.4f}  |  GUE = {R_GUE:.4f}")
print()


# ========== TEST 1: seq_len=128, top-d_k eigenvalues only ==========
print("=" * 85)
print("  TEST 1: seq_len=128, top-64 eigenvalues by magnitude")
print("  (Removing rank-deficiency artifacts)")
print("=" * 85)
print()

for mdl, label in [(model, "TRAINED"), (random_model, "RANDOMIZED")]:
    head_rs = []
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, 128))
        logits = get_logit_matrices(mdl, input_ids)
        for (l, h), L in logits.items():
            H = (L + L.T) / 2.0
            eigs = np.linalg.eigh(H)[0]
            # Keep only top d_k eigenvalues by absolute value
            top_idx = np.argsort(np.abs(eigs))[-d_k:]
            top_eigs = np.sort(eigs[top_idx])
            r = ratio_statistic(top_eigs)
            if not np.isnan(r):
                head_rs.append(r)

    rs = np.array(head_rs)
    print(f"  {label} ({len(rs)} measurements):")
    print(f"    Mean <r>   = {np.mean(rs):.4f}   [{classify(np.mean(rs))}]")
    print(f"    Median <r> = {np.median(rs):.4f}")
    print(f"    Std        = {np.std(rs):.4f}")
    print()


# ========== TEST 2: seq_len=64, full rank matrices ==========
print("=" * 85)
print("  TEST 2: seq_len=64 (= d_k), matrices are generically full rank")
print("=" * 85)
print()

for mdl, label in [(model, "TRAINED"), (random_model, "RANDOMIZED")]:
    head_rs = []
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, 64))
        logits = get_logit_matrices(mdl, input_ids)
        for (l, h), L in logits.items():
            H = (L + L.T) / 2.0
            eigs = np.linalg.eigh(H)[0]
            r = ratio_statistic(eigs)
            if not np.isnan(r):
                head_rs.append(r)

    rs = np.array(head_rs)
    print(f"  {label} ({len(rs)} measurements):")
    print(f"    Mean <r>   = {np.mean(rs):.4f}   [{classify(np.mean(rs))}]")
    print(f"    Median <r> = {np.median(rs):.4f}")
    print(f"    Std        = {np.std(rs):.4f}")
    print()


# ========== TEST 3: Singular values of L (unsymmetrized) ==========
print("=" * 85)
print("  TEST 3: Singular value spacing statistics of L (seq_len=64)")
print("  (No symmetrization — analyzing the raw operator)")
print("=" * 85)
print()

for mdl, label in [(model, "TRAINED"), (random_model, "RANDOMIZED")]:
    head_rs = []
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, 64))
        logits = get_logit_matrices(mdl, input_ids)
        for (l, h), L in logits.items():
            sv = np.linalg.svd(L, compute_uv=False)
            sv = sv[sv > 1e-10]
            r = ratio_statistic(sv)
            if not np.isnan(r):
                head_rs.append(r)

    rs = np.array(head_rs)
    print(f"  {label} ({len(rs)} measurements):")
    print(f"    Mean <r>   = {np.mean(rs):.4f}   [{classify(np.mean(rs))}]")
    print(f"    Median <r> = {np.median(rs):.4f}")
    print(f"    Std        = {np.std(rs):.4f}")
    print()


# ========== TEST 4: Weight matrices W_Q^T W_K (input-independent) ==========
print("=" * 85)
print("  TEST 4: Weight matrix eigenvalues — W_Q^T W_K symmetrized")
print("  (No input dependence — pure learned structure)")
print("=" * 85)
print()

for mdl, label in [(model, "TRAINED"), (random_model, "RANDOMIZED")]:
    head_rs = []
    for l in range(n_layers):
        block = mdl.transformer.h[l]
        W = block.attn.c_attn.weight.detach().numpy()
        W_Q = W[:, :d_model]
        W_K = W[:, d_model:2*d_model]

        for h in range(n_heads):
            wq_h = W_Q[:, h*d_k:(h+1)*d_k]
            wk_h = W_K[:, h*d_k:(h+1)*d_k]

            W_eff = wq_h.T @ wk_h  # (d_k, d_k)
            H_eff = (W_eff + W_eff.T) / 2.0
            eigs = np.linalg.eigh(H_eff)[0]
            r = ratio_statistic(eigs)
            if not np.isnan(r):
                head_rs.append(r)

    rs = np.array(head_rs)
    print(f"  {label} ({len(rs)} measurements):")
    print(f"    Mean <r>   = {np.mean(rs):.4f}   [{classify(np.mean(rs))}]")
    print(f"    Median <r> = {np.median(rs):.4f}")
    print(f"    Std        = {np.std(rs):.4f}")
    print()


# ========== TEST 5: Layer-resolved, seq_len=64 ==========
print("=" * 85)
print("  TEST 5: Layer-by-layer, seq_len=64 (trained only)")
print("=" * 85)
print()

layer_rs = {l: [] for l in range(n_layers)}
for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, 64))
    logits = get_logit_matrices(model, input_ids)
    for (l, h), L in logits.items():
        H = (L + L.T) / 2.0
        eigs = np.linalg.eigh(H)[0]
        r = ratio_statistic(eigs)
        if not np.isnan(r):
            layer_rs[l].append(r)

print(f"  {'Layer':>6}  {'Mean <r>':>10}  {'Class':>10}")
print("  " + "-" * 30)
for l in range(n_layers):
    rs = layer_rs[l]
    if rs:
        m = np.mean(rs)
        print(f"  {l+1:6d}  {m:10.4f}  {classify(m):>10}")
print()
print(f"  Reference: Poisson = {R_POISSON:.4f}  |  GOE = {R_GOE:.4f}  |  GUE = {R_GUE:.4f}")


# ========== TEST 6: Pure GUE/GOE reference matrices ==========
print()
print("=" * 85)
print("  VALIDATION: Known ensembles (64x64 matrices, 1000 samples)")
print("=" * 85)
print()

n_test = 64
n_samples = 1000

# GOE: symmetric random matrices
goe_rs = []
for _ in range(n_samples):
    M = np.random.randn(n_test, n_test)
    M = (M + M.T) / 2.0
    eigs = np.linalg.eigh(M)[0]
    r = ratio_statistic(eigs)
    if not np.isnan(r):
        goe_rs.append(r)

goe_r = np.mean(goe_rs)
print(f"  GOE (symmetric Gaussian):    <r> = {goe_r:.4f}   (expected: {R_GOE:.4f})")

# GUE: Hermitian random matrices
gue_rs = []
for _ in range(n_samples):
    M = (np.random.randn(n_test, n_test) + 1j * np.random.randn(n_test, n_test)) / np.sqrt(2)
    M = (M + M.conj().T) / 2.0
    eigs = np.linalg.eigvalsh(M)
    r = ratio_statistic(eigs)
    if not np.isnan(r):
        gue_rs.append(r)

gue_r = np.mean(gue_rs)
print(f"  GUE (Hermitian Gaussian):    <r> = {gue_r:.4f}   (expected: {R_GUE:.4f})")

# Poisson: diagonal random matrices
poi_rs = []
for _ in range(n_samples):
    eigs = np.sort(np.random.randn(n_test))
    r = ratio_statistic(eigs)
    if not np.isnan(r):
        poi_rs.append(r)

poi_r = np.mean(poi_rs)
print(f"  Poisson (diagonal):          <r> = {poi_r:.4f}   (expected: {R_POISSON:.4f})")

# Wishart: X^T X where X is random (for comparison)
wish_rs = []
for _ in range(n_samples):
    X = np.random.randn(n_test, n_test)
    M = X.T @ X / n_test
    eigs = np.linalg.eigh(M)[0]
    r = ratio_statistic(eigs)
    if not np.isnan(r):
        wish_rs.append(r)

wish_r = np.mean(wish_rs)
print(f"  Wishart (X^TX):              <r> = {wish_r:.4f}")

# Product: (AB^T + BA^T)/2 where A, B are random n×d_k (our matrix structure)
prod_rs = []
for _ in range(n_samples):
    A = np.random.randn(n_test, n_test)
    B = np.random.randn(n_test, n_test)
    M = (A @ B.T + B @ A.T) / 2.0
    eigs = np.linalg.eigh(M)[0]
    r = ratio_statistic(eigs)
    if not np.isnan(r):
        prod_rs.append(r)

prod_r = np.mean(prod_rs)
print(f"  Sym. product (AB^T+BA^T)/2:  <r> = {prod_r:.4f}")

print()
print("=" * 85)
print("  SUMMARY")
print("=" * 85)
print()
print("  The question: does training push the level statistics from")
print("  their structural baseline toward GUE?")
print()
