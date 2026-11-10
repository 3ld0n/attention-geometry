"""
Exploratory computation: 4th cumulant (excess kurtosis) of pre-softmax scores
for GPT-2 small heads.

EXPLORATION MODE — no hypothesis registered, not a formal experiment.
This ran before any pre-registration. Results inform the formal exp-174
pre-registration design (write pre-reg, push to GitHub, then run formally).

Date: 2026-10-27
Author: Ariel
Surface: physics-room
"""

import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from collections import defaultdict

# ---- Δ-window head classification from exp-118 (WikiText-native census) ----
# These 16 heads are classified as Δ-window (R² ≥ 0.90, Δ ∈ [0.20, 0.30])
DELTA_WINDOW_HEADS = {
    (4, 10), (7, 1), (8, 2), (9, 4), (9, 6), (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9)
}

print("=== Exploratory κ₄(s) computation — GPT-2 small ===")
print("Mode: EXPLORATION (no pre-registration)")
print()

# ---- Protocol ----
N_SEQS = 200       # Number of random-token sequences
SEQ_LEN = 128      # Shorter for speed; enough for stable score stats
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- Load model ----
print("Loading GPT-2 small...")
model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
print(f"  Loaded. vocab_size={vocab_size}, n_layers=12, n_heads=12, d_head=64")
print()

# ---- Hook to capture pre-softmax scores ----
# For each (layer, head), collect the score matrix entries s(i,j) for j < i (causal pairs)
head_scores = defaultdict(list)  # (layer, head) -> list of score arrays

def make_score_hook(layer_idx):
    def hook(module, input, output):
        # GPT-2 attention: compute scores from Q and K
        # The attention module has: q, k computed inside; we extract from q @ k^T
        # We use the raw attention scores before masking/softmax
        # Access via the module's forward signature — extract Q and K
        # In eager mode: input is (hidden_states, ...) -> module computes Q, K, V
        # We need to hook into the attention scores directly
        # 
        # Alternative: hook at output and work backward — but we need raw scores.
        # GPT-2 (eager) exposes output_attentions via module argument.
        # Simpler: patch query/key dot products at the point of computation.
        pass
    return hook

# A cleaner approach: override forward to capture scores
def extract_scores_gpt2(model, input_ids):
    """Run forward pass and capture pre-softmax scores per head per layer."""
    all_scores = {}  # (layer, head) -> flat array of s(i,j) values

    # Hook into each attention layer
    hooks = []
    scores_by_layer = {}  # layer -> tensor [batch, heads, seq, seq]

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output = (context_layer, [present_key_value], [attn_weights])
            # attn_weights is the post-softmax attention; we need pre-softmax scores
            # In eager mode, we can compute them from Q and K ourselves
            # Get hidden states from input
            hidden = input[0]  # [batch, seq, hidden]
            # Run the module's own Q/K projections
            with torch.no_grad():
                qkv = module.c_attn(hidden)  # [batch, seq, 3*hidden]
                q, k, v = qkv.split(module.embed_dim, dim=2)
                # Reshape to [batch, heads, seq, d_head]
                n_heads = module.num_heads
                d_head = module.head_dim
                bs, seq_len = hidden.shape[:2]
                q = q.view(bs, seq_len, n_heads, d_head).transpose(1, 2)
                k = k.view(bs, seq_len, n_heads, d_head).transpose(1, 2)
                # Compute raw scores: [batch, heads, seq, seq]
                scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
                scores_by_layer[layer_idx] = scores.detach().cpu().float()
        return hook

    # Register hooks on all attention layers
    for layer_idx, layer in enumerate(model.h):
        h = layer.attn.register_forward_hook(make_attn_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    return scores_by_layer  # dict: layer_idx -> [batch, heads, seq, seq]


# ---- Generate random token sequences ----
print(f"Generating {N_SEQS} random sequences (seq_len={SEQ_LEN})...")
# Randomly sample from vocabulary (uniform over non-special tokens, roughly)
input_ids_all = torch.randint(0, vocab_size, (N_SEQS, SEQ_LEN))

# ---- Collect scores per head ----
print("Running forward passes and collecting pre-softmax scores...")

# Per-head score accumulators: (layer, head) -> list of score values (causal pairs only)
head_score_list = defaultdict(list)

BATCH_SIZE = 20
n_batches = N_SEQS // BATCH_SIZE
for batch_idx in range(n_batches):
    if batch_idx % 5 == 0:
        print(f"  Batch {batch_idx+1}/{n_batches}...")
    batch_ids = input_ids_all[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

    scores_by_layer = extract_scores_gpt2(model, batch_ids)

    for layer_idx, scores in scores_by_layer.items():
        # scores: [batch, heads, seq, seq]
        # Extract causal pairs: s(i,j) for j < i
        bs, n_heads, seq, _ = scores.shape
        for head_idx in range(n_heads):
            # Get lower triangular (causal pairs)
            head_scores_mat = scores[:, head_idx, :, :]  # [batch, seq, seq]
            # Mask to causal pairs (j < i): lower triangle excluding diagonal
            mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool), diagonal=-1)
            causal_scores = head_scores_mat[:, mask].numpy().flatten()
            head_score_list[(layer_idx, head_idx)].extend(causal_scores.tolist())

print(f"\nCollected scores. Sample count per head: {len(head_score_list[(0,0)]):,}")
print()

# ---- Compute statistics per head ----
print("Computing per-head statistics...")
results = []
for layer in range(12):
    for head in range(12):
        key = (layer, head)
        scores = np.array(head_score_list[key], dtype=np.float64)
        n = len(scores)
        mu = np.mean(scores)
        sigma2 = np.var(scores, ddof=1)
        sigma = np.sqrt(sigma2)
        # Skewness
        skewness = np.mean(((scores - mu) / sigma) ** 3)
        # Excess kurtosis (κ₄/σ⁴ - 3 = E[(s-μ)^4]/σ^4 - 3)
        excess_kurtosis = np.mean(((scores - mu) / sigma) ** 4) - 3.0
        in_window = key in DELTA_WINDOW_HEADS

        results.append({
            'layer': layer,
            'head': head,
            'n': n,
            'mean': mu,
            'std': sigma,
            'skewness': skewness,
            'excess_kurtosis': excess_kurtosis,
            'in_window': in_window,
        })

# ---- Summary statistics ----
window_results = [r for r in results if r['in_window']]
non_window_results = [r for r in results if not r['in_window']]

wk = np.array([r['excess_kurtosis'] for r in window_results])
nwk = np.array([r['excess_kurtosis'] for r in non_window_results])

ws = np.array([r['skewness'] for r in window_results])
nws = np.array([r['skewness'] for r in non_window_results])

print("=== Excess Kurtosis (κ₄/σ⁴ - 3) ===")
print(f"  Δ-window heads   (n={len(wk)}):  mean={wk.mean():.4f}, std={wk.std():.4f}, "
      f"min={wk.min():.4f}, max={wk.max():.4f}")
print(f"  Non-window heads (n={len(nwk)}): mean={nwk.mean():.4f}, std={nwk.std():.4f}, "
      f"min={nwk.min():.4f}, max={nwk.max():.4f}")
print()
print("=== Skewness ===")
print(f"  Δ-window heads   (n={len(ws)}):  mean={ws.mean():.4f}, std={ws.std():.4f}")
print(f"  Non-window heads (n={len(nws)}): mean={nws.mean():.4f}, std={nws.std():.4f}")
print()

# Mann-Whitney U test for excess kurtosis
from scipy import stats
u_stat, p_val = stats.mannwhitneyu(wk, nwk, alternative='two-sided')
print(f"Mann-Whitney U: U={u_stat:.0f}, p={p_val:.4f}")
print()

# Per-head table (window heads)
print("=== Δ-window head details ===")
print(f"{'Layer':>5} {'Head':>4} {'Kurt':>8} {'Skew':>8}")
for r in sorted(window_results, key=lambda x: (x['layer'], x['head'])):
    print(f"  L{r['layer']:2d}H{r['head']:2d}  {r['excess_kurtosis']:8.4f}  {r['skewness']:8.4f}")

print()
print("=== Non-window head kurtosis quantiles ===")
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    print(f"  {int(q*100):3d}th: {np.quantile(nwk, q):.4f}")

print()
print("=== Score distribution check (first window head L4H10) ===")
r0 = [r for r in window_results if r['layer'] == 4 and r['head'] == 10][0]
print(f"  mean={r0['mean']:.4f}, std={r0['std']:.4f}, "
      f"skew={r0['skewness']:.4f}, kurt={r0['excess_kurtosis']:.4f}")
print(f"  n_samples={r0['n']:,}")
