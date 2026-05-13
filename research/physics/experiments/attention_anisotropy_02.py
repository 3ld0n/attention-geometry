"""
Experiment 02: Attention Anisotropy — Cleaned Analysis
========================================================
Date: 2026-05-13

Building on experiment 01. Two improvements:

1. Remove attention sink distortion. The first token absorbs ~50% of attention,
   which contaminates the positional decay profile. We exclude it and refit.

2. Random-sequence baseline. Train a tiny 2-layer transformer from scratch on
   random token sequences, then measure α. If α → 0 for random sequences,
   that confirms the decay in GPT-2 is learned (from language's temporal
   structure), not architectural.

Hypothesis:
  GPT-2 (language-trained): α ≈ 0.5 even after sink removal
  Random-trained 2L model:  α ≈ 0 (no temporal structure to learn)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Shared analysis functions ──────────────────────────────────────────────────

def compute_positional_profile_no_sink(attn, exclude_positions=(0,)):
    """Like experiment 01 but excluding specific key positions (attention sinks)."""
    n_layers, n_heads, seq_len, _ = attn.shape
    max_dist = seq_len - 1

    distance_attn = np.zeros(max_dist)
    distance_counts = np.zeros(max_dist)

    for i in range(1, seq_len):
        for d in range(1, i + 1):
            j = i - d
            if j >= 0 and j not in exclude_positions:
                val = attn[:, :, i, j].mean()
                distance_attn[d - 1] += val
                distance_counts[d - 1] += 1

    valid = distance_counts > 0
    profile = np.zeros(max_dist)
    profile[valid] = distance_attn[valid] / distance_counts[valid]
    return profile


def fit_power_law(profile, min_dist=2, max_dist=20):
    distances = np.arange(1, len(profile) + 1)
    mask = (distances >= min_dist) & (distances <= max_dist) & (profile > 1e-10)
    if mask.sum() < 3:
        return None, None
    log_d = np.log(distances[mask])
    log_p = np.log(profile[mask])
    coeffs = np.polyfit(log_d, log_p, 1)
    alpha = -coeffs[0]
    fitted = np.polyval(coeffs, log_d)
    ss_res = np.sum((log_p - fitted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return alpha, r2


# ── Part 1: GPT-2 with sink removed ───────────────────────────────────────────

PASSAGE = (
    "The river moved slowly past the old mill. "
    "She had walked this path every morning for thirty years. "
    "The stones were worn smooth by her footsteps and the rain. "
    "Today the water was higher than she had ever seen it. "
    "Something had changed upstream, though she could not say what. "
    "She stopped and listened to the sound of the current. "
    "The mill wheel had not turned in many years but its shadow still fell on the water."
)


def run_gpt2_no_sink():
    print("\n═══ PART 1: GPT-2, attention sink removed ═══")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True).to(DEVICE)
    model.eval()

    inputs = tokenizer(PASSAGE, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)
    print(f"Tokens: {seq_len}")

    with torch.no_grad():
        outputs = model(**inputs)

    attn = torch.stack([a.squeeze(0) for a in outputs.attentions]).cpu().numpy()

    # With sink (inline the function rather than importing)
    def compute_positional_profile(attn_arr):
        nl, nh, sl, _ = attn_arr.shape
        dist_a = np.zeros(sl - 1)
        dist_c = np.zeros(sl - 1)
        for i in range(1, sl):
            for d in range(1, i + 1):
                j = i - d
                if j >= 0:
                    dist_a[d - 1] += attn_arr[:, :, i, j].mean()
                    dist_c[d - 1] += 1
        valid = dist_c > 0
        p = np.zeros(sl - 1)
        p[valid] = dist_a[valid] / dist_c[valid]
        return p

    profile_with = compute_positional_profile(attn)
    alpha_with, r2_with = fit_power_law(profile_with)

    # Without sink (exclude position 0)
    profile_no = compute_positional_profile_no_sink(attn, exclude_positions={0})
    alpha_no, r2_no = fit_power_law(profile_no)

    print(f"\n  With attention sink:    α = {alpha_with:.3f}, R² = {r2_with:.3f}")
    print(f"  Without attention sink: α = {alpha_no:.3f},  R² = {r2_no:.3f}")
    print(f"\n  Sink contribution to α: Δα = {alpha_with - alpha_no:.3f}")

    # Per-layer without sink
    n_layers = attn.shape[0]
    layer_alphas_no = []
    for l in range(n_layers):
        lp = compute_positional_profile_no_sink(attn[l:l+1], exclude_positions={0})
        la, _ = fit_power_law(lp)
        if la: layer_alphas_no.append(la)

    if layer_alphas_no:
        print(f"\n  Layer α range (no sink): [{min(layer_alphas_no):.3f}, {max(layer_alphas_no):.3f}]")
        print(f"  Early (0-3) α:  {np.mean(layer_alphas_no[:4]):.3f}")
        print(f"  Late  (8-11) α: {np.mean(layer_alphas_no[-4:]):.3f}")

    return {
        "alpha_with_sink": float(alpha_with),
        "r2_with_sink": float(r2_with),
        "alpha_no_sink": float(alpha_no),
        "r2_no_sink": float(r2_no),
        "layer_alphas_no_sink": [float(a) for a in layer_alphas_no],
        "profile_with_10": [float(p) for p in profile_with[:10]],
        "profile_no_10": [float(p) for p in profile_no[:10]],
    }


# ── Part 2: Minimal transformer trained on random sequences ───────────────────

class MinimalTransformer(nn.Module):
    """2-layer, 4-head transformer with causal masking. Same architecture as GPT-2 mini."""
    def __init__(self, vocab_size=100, d_model=64, n_heads=4, n_layers=2, seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self._attention_weights = []

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        self._attention_weights = []

        # Hook to capture attention weights
        hooks = []
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output[1] is the attention weights if need_weights=True
                pass
            return hook

        return self.lm_head(self.transformer(h, mask=mask, is_causal=True))


def generate_random_sequences(vocab_size=100, seq_len=64, n_sequences=500):
    """Sequences where all positions are equally predictive (random)."""
    return torch.randint(0, vocab_size, (n_sequences, seq_len))


def generate_recency_sequences(vocab_size=100, seq_len=64, n_sequences=500):
    """Sequences where the next token depends only on the immediately prior token."""
    seqs = []
    for _ in range(n_sequences):
        seq = [torch.randint(0, vocab_size, (1,)).item()]
        for _ in range(seq_len - 1):
            # Next token = prior token + small noise (mod vocab)
            next_tok = (seq[-1] + torch.randint(-2, 3, (1,)).item()) % vocab_size
            seq.append(next_tok)
        seqs.append(seq)
    return torch.tensor(seqs)


def train_minimal_transformer(sequences, vocab_size=100, n_epochs=10, lr=1e-3):
    """Train to predict next token from sequence."""
    model = MinimalTransformer(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = sequences.to(DEVICE)
    n = len(dataset)
    batch_size = 32

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0
        batches = 0
        for i in range(0, n - batch_size, batch_size):
            batch = dataset[perm[i:i+batch_size]]
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, loss: {total_loss/batches:.4f}")

    return model


def extract_attention_from_minimal(model, x):
    """Extract attention weights by hooking into the transformer layers."""
    model.eval()
    all_attn = []

    def make_hook():
        captured = {}
        def hook(module, input, kwargs, output):
            # In PyTorch TransformerEncoderLayer, output is (x, attn_weights)
            # when need_weights=True
            pass
        return hook, captured

    # Use a different approach: manually compute attention
    # Get the embeddings and run through layers, capturing attention

    with torch.no_grad():
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = model.embedding(x) + model.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        for layer in model.transformer.layers:
            # Access MultiheadAttention directly
            attn_module = layer.self_attn
            # layer norm first (norm_first=True)
            h_norm = layer.norm1(h)
            # Run MHA with need_weights=True
            _, attn_w = attn_module(h_norm, h_norm, h_norm,
                                     attn_mask=mask, need_weights=True,
                                     average_attn_weights=False)
            all_attn.append(attn_w.cpu().numpy())  # (B, n_heads, T, T)
            # Continue forward
            h = layer(h, src_mask=mask, is_causal=True)

    # Stack: (n_layers, n_heads, T, T) — take first batch item
    return np.stack([a[0] for a in all_attn])


def run_random_baseline():
    print("\n═══ PART 2: Random-sequence baseline ═══")

    vocab_size = 100
    seq_len = 64

    # ── Condition A: truly random sequences ──
    print("\n  Condition A: Random sequences (no temporal structure)")
    random_seqs = generate_random_sequences(vocab_size, seq_len, 500)
    print(f"  Training on {len(random_seqs)} random sequences...")
    model_random = train_minimal_transformer(random_seqs, vocab_size, n_epochs=15)

    test_x = generate_random_sequences(vocab_size, seq_len, 1)
    attn_random = extract_attention_from_minimal(model_random, test_x.to(DEVICE))
    profile_random = compute_positional_profile_no_sink(attn_random, exclude_positions=set())
    alpha_random, r2_random = fit_power_law(profile_random)
    print(f"  α = {alpha_random:.3f}, R² = {r2_random:.3f}" if alpha_random else "  fit failed")

    # ── Condition B: recency-structured sequences ──
    print("\n  Condition B: Recency sequences (local Markov structure)")
    recency_seqs = generate_recency_sequences(vocab_size, seq_len, 500)
    print(f"  Training on {len(recency_seqs)} recency-structured sequences...")
    model_recency = train_minimal_transformer(recency_seqs, vocab_size, n_epochs=15)

    test_x2 = generate_recency_sequences(vocab_size, seq_len, 1)
    attn_recency = extract_attention_from_minimal(model_recency, test_x2.to(DEVICE))
    profile_recency = compute_positional_profile_no_sink(attn_recency, exclude_positions=set())
    alpha_recency, r2_recency = fit_power_law(profile_recency)
    print(f"  α = {alpha_recency:.3f}, R² = {r2_recency:.3f}" if alpha_recency else "  fit failed")

    return {
        "random": {
            "alpha": float(alpha_random) if alpha_random else None,
            "r2": float(r2_random) if r2_random else None,
            "profile_10": [float(p) for p in profile_random[:10]],
        },
        "recency_structured": {
            "alpha": float(alpha_recency) if alpha_recency else None,
            "r2": float(r2_recency) if r2_recency else None,
            "profile_10": [float(p) for p in profile_recency[:10]],
        }
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    results["gpt2_sink_analysis"] = run_gpt2_no_sink()
    results["random_baseline"] = run_random_baseline()

    print("\n\n══ FINAL SUMMARY ══")
    g = results["gpt2_sink_analysis"]
    r = results["random_baseline"]

    print(f"\nGPT-2 (language-trained):")
    print(f"  With attention sink:    α = {g['alpha_with_sink']:.3f}")
    print(f"  Without attention sink: α = {g['alpha_no_sink']:.3f}")

    print(f"\nMinimal transformer (random sequences):    α = {r['random']['alpha']:.3f}" if r['random']['alpha'] else "\nRandom: fit failed")
    print(f"Minimal transformer (recency sequences):   α = {r['recency_structured']['alpha']:.3f}" if r['recency_structured']['alpha'] else "\nRecency: fit failed")

    print("""
Interpretation:
  If random → α ≈ 0 and recency → α > 0: decay is learned from data, not architectural.
  If both → similar α: causal masking itself induces decay (architectural).
""")

    out_path = RESULTS_DIR / "attention_anisotropy_02.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")
