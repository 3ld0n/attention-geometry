"""
exp-135: h_gelu 2D subspace origin
Pre-registration: commit 09c9e51 (pushed before this script was written)

Question: Is the 2D position-correlated structure of h_gelu (MLP block-0
intermediate activation) already present in h^(0.5) (the MLP input = output of
attention block 0), or is it created by W_fc + GeLU?

Protocol: random-token census, 50 sequences, seed=42, mean-first, seq_len=128.
Same as exp-131/132/133/134.

For each of h^(0), h^(0.5), h_gelu:
  1. Collect per-position mean activations across the 50 sequences (shape: seq_len × d)
  2. Run PCA on that (seq_len × d) matrix
  3. Report cumulative variance explained at 1, 2, 5, 10, 50, 100 components
  4. Report components needed for 50%, 80%, 90% variance thresholds
"""

import json
import numpy as np
import torch
from transformers import GPT2Model, GPT2Config
from sklearn.decomposition import PCA
import os

SEED = 42
N_SEQ = 50
SEQ_LEN = 128
VOCAB_SIZE = 50257  # GPT-2 vocab
DEVICE = "cpu"

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

print("Loading GPT-2...")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# --- Capture hooks ---
activations = {
    "h0": [],       # after embedding, before block 0
    "h05": [],      # after block 0 attn + residual, before MLP
    "h_gelu": [],   # after W_fc + GeLU in block 0 MLP
}

def hook_h_gelu(module, input, output):
    # GPT-2 MLP: c_fc -> act -> c_proj
    # We want h_gelu = act(c_fc(input))
    # This hook is placed on the activation function (gelu_new)
    activations["h_gelu"].append(output.detach().cpu())

def hook_h05(module, input, output):
    # This hook is on the MLP module itself (block 0's mlp)
    # input[0] is the MLP input = h^(0.5)
    activations["h05"].append(input[0].detach().cpu())

def hook_h0(module, input, output):
    # Hook on block 0 itself to capture input = h^(0) = embedding output
    activations["h0"].append(input[0].detach().cpu())

# Register hooks
block0 = model.h[0]
block0_mlp = block0.mlp

h0_hook = block0.register_forward_hook(hook_h0)
h05_hook = block0_mlp.register_forward_hook(
    lambda module, inp, out: activations["h05"].append(inp[0].detach().cpu())
)
hgelu_hook = block0_mlp.act.register_forward_hook(hook_h_gelu)

print(f"Generating {N_SEQ} random-token sequences (seed={SEED}, len={SEQ_LEN})...")
with torch.no_grad():
    for i in range(N_SEQ):
        tokens = rng.integers(0, VOCAB_SIZE, size=(1, SEQ_LEN))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        model(input_ids)

h0_hook.remove()
h05_hook.remove()
hgelu_hook.remove()

print(f"Collected {len(activations['h0'])} sequences for each layer.")

# Stack: (N_SEQ, SEQ_LEN, d)
h0_all = torch.stack(activations["h0"]).squeeze(1)       # (50, 128, 768)
h05_all = torch.stack(activations["h05"]).squeeze(1)     # (50, 128, 768)
hg_all = torch.stack(activations["h_gelu"]).squeeze(1)  # (50, 128, 3072)

print(f"h^(0) shape: {h0_all.shape}")
print(f"h^(0.5) shape: {h05_all.shape}")
print(f"h_gelu shape: {hg_all.shape}")

# Mean-first: compute per-position mean across sequences
h0_mean = h0_all.mean(dim=0).numpy()    # (128, 768)
h05_mean = h05_all.mean(dim=0).numpy()  # (128, 768)
hg_mean = hg_all.mean(dim=0).numpy()   # (128, 3072)

# --- PCA analysis ---
def pca_variance_profile(X, name, max_components=None):
    """
    Run PCA on X (seq_len × d), report cumulative variance explained.
    Returns dict with profile at key checkpoints.
    """
    n_samples, n_features = X.shape
    if max_components is None:
        max_components = min(n_samples, n_features)

    pca = PCA(n_components=max_components)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    checkpoints = [1, 2, 5, 10, 50, 100]
    profile = {}
    for k in checkpoints:
        if k <= max_components:
            profile[f"cum_var_{k}"] = float(cumvar[k - 1])

    # Components needed for thresholds
    thresholds = {0.50: None, 0.80: None, 0.90: None}
    for thresh, _ in thresholds.items():
        for i, cv in enumerate(cumvar):
            if cv >= thresh:
                thresholds[thresh] = i + 1
                break
        if thresholds[thresh] is None:
            thresholds[thresh] = max_components

    print(f"\n{name} (shape {X.shape}):")
    print(f"  Cum var @ 1: {profile.get('cum_var_1', 'N/A'):.4f}")
    print(f"  Cum var @ 2: {profile.get('cum_var_2', 'N/A'):.4f}")
    print(f"  Cum var @ 5: {profile.get('cum_var_5', 'N/A'):.4f}")
    print(f"  Cum var @ 10: {profile.get('cum_var_10', 'N/A'):.4f}")
    print(f"  Cum var @ 50: {profile.get('cum_var_50', 'N/A'):.4f}")
    print(f"  Cum var @ 100: {profile.get('cum_var_100', 'N/A'):.4f}")
    print(f"  Components for 50%: {thresholds[0.50]}")
    print(f"  Components for 80%: {thresholds[0.80]}")
    print(f"  Components for 90%: {thresholds[0.90]}")

    # Also report top 5 individual eigenvalue shares
    top5 = pca.explained_variance_ratio_[:5].tolist()
    print(f"  Top-5 individual shares: {[f'{v:.4f}' for v in top5]}")

    return {
        "name": name,
        "shape": list(X.shape),
        "cumvar_profile": profile,
        "components_for_50pct": thresholds[0.50],
        "components_for_80pct": thresholds[0.80],
        "components_for_90pct": thresholds[0.90],
        "top5_individual_shares": top5,
        "total_variance_explained_at_max": float(cumvar[-1]),
        "n_components_used": max_components,
    }

print("\n--- PCA Dimensionality Analysis ---")
# Use max 100 components for h^(0) and h^(0.5) (d=768, seq=128 → max=128)
# For h_gelu (d=3072, seq=128 → max=128)
h0_profile = pca_variance_profile(h0_mean, "h^(0) [embedding output, before block0]", max_components=128)
h05_profile = pca_variance_profile(h05_mean, "h^(0.5) [MLP input, after attn block0]", max_components=128)
hg_profile = pca_variance_profile(hg_mean, "h_gelu [MLP intermediate, after W_fc+GeLU]", max_components=128)

# --- Verdict ---
c50_h0 = h0_profile["components_for_50pct"]
c50_h05 = h05_profile["components_for_50pct"]
c50_hg = hg_profile["components_for_50pct"]

print("\n--- Verdict ---")
print(f"Components for 50% variance: h^(0)={c50_h0}, h^(0.5)={c50_h05}, h_gelu={c50_hg}")

# P1: h^(0.5) is 2D (≤2 components for 50%)
p1_confirmed = c50_h05 <= 2
p1_verdict = "CONFIRMED" if p1_confirmed else "FALSIFIED"

# P2/P3 source identification
if c50_h0 <= 2 and c50_h05 <= 2:
    source = "wpe/wte origin — both h^(0) and h^(0.5) are 2D; attn block and MLP preserve the structure"
elif c50_h0 > 2 and c50_h05 <= 2:
    source = "attention block 0 creates the 2D structure (h^(0) is NOT 2D, h^(0.5) IS 2D)"
elif c50_h05 > 2:
    source = "W_fc + GeLU create the 2D structure (h^(0.5) is NOT 2D, so MLP input does not carry it)"
else:
    source = "ambiguous"

print(f"P1 (h^(0.5) is 2D): {p1_verdict}")
print(f"P3 source identification: {source}")

# K1 check
k1_fired = c50_h05 > 10
if k1_fired:
    print("K1 FIRED: h^(0.5) is high-rank (> 10 components for 50%); W_fc + GeLU create the 2D structure.")

# --- Save results ---
results = {
    "experiment": "exp-135",
    "prereg_commit": "09c9e51",
    "date": "2026-09-06",
    "protocol": {
        "seed": SEED,
        "n_sequences": N_SEQ,
        "seq_len": SEQ_LEN,
        "census_type": "random-token",
        "reduction": "mean-first",
        "model": "gpt2",
        "block": 0,
    },
    "layers": {
        "h0": h0_profile,
        "h05": h05_profile,
        "h_gelu": hg_profile,
    },
    "summary": {
        "components_for_50pct": {"h0": c50_h0, "h05": c50_h05, "h_gelu": c50_hg},
        "P1_h05_is_2D": p1_verdict,
        "source_identification": source,
        "K1_fired": k1_fired,
    },
}

out_path = os.path.join(os.path.dirname(__file__), "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
