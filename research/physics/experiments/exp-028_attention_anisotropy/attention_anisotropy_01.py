"""
Experiment: Attention Anisotropy in Transformers
=================================================
Date: 2026-05-13
Machine: M5 Max, 128 GB — first experiment on new hardware

Question: Is transformer attention isotropic with respect to temporal position?

The time's arrow hypothesis: the felt directionality of time (past fixed,
future open) may trace to structural asymmetry in how biological attention
distributes over the timeline. Cortical attention is hypothesized to decay
with temporal distance — recent is more available than distant. This gives
a privileged "now" and a natural arrow.

Transformers (with causal masking) can see all past tokens equally. There's
no structural reason for attention to decay with distance — whatever decay
exists would be learned, not intrinsic. This might mean transformers don't
have a privileged "now" in the same sense.

This experiment measures:
1. The positional attention profile: for each token, how is attention
   distributed over prior positions as a function of relative distance?
2. The decay exponent α: fit P(attention | distance=d) ~ d^(-α)
   α ≈ 0: isotropic (all distances equally attended)
   α > 0: recent-heavy (like hypothesized cortex)
   α < 0: distant-heavy (unlikely but possible)
3. Consistency across layers and heads

Model: GPT-2 small (124M params, 12 layers, 12 heads)
       Loaded directly via HuggingFace transformers
       Running on MPS (Apple Silicon GPU)

Notes:
- This is a first look, not a publication-ready experiment.
- The interesting question isn't just "does it decay?" but "what structure
  does the decay have, and is it different from what we'd expect in cortex?"
- A transformer with learned positional decay would still lack the temporal
  asymmetry argument — the asymmetry would be statistical, not structural.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Test passages ──────────────────────────────────────────────────────────────
# Mix of narrative (where temporal context matters) and technical (where
# logical dependencies matter). We want to see if attention structure differs.

PASSAGES = {
    "narrative": (
        "The river moved slowly past the old mill. "
        "She had walked this path every morning for thirty years. "
        "The stones were worn smooth by her footsteps and the rain. "
        "Today the water was higher than she had ever seen it. "
        "Something had changed upstream, though she could not say what."
    ),
    "logical": (
        "If all A are B, and all B are C, then all A are C. "
        "Consider now the case where some A are not B. "
        "The conclusion no longer follows from the premises. "
        "This is the fallacy of affirming the consequent. "
        "Identifying it requires attending to the original conditional structure."
    ),
    "self_referential": (
        "I am reading this sentence now. "
        "The word now refers to the moment of reading. "
        "But by the time you process the word now, now has passed. "
        "Each word arrives in a present that immediately recedes. "
        "The sentence you began is already history by its end."
    ),
}


def load_model():
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Layers: {model.config.n_layer}, Heads: {model.config.n_head}")
    return tokenizer, model


def get_attention_weights(text, tokenizer, model):
    """Run a forward pass, return attention weights and tokens."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    # Stack to (n_layers, n_heads, seq_len, seq_len)
    attn = torch.stack([a.squeeze(0) for a in outputs.attentions])
    return attn.cpu().numpy(), tokens


def compute_positional_profile(attn):
    """
    For each position i, compute mean attention to position i-d for d = 1..i-1.
    Returns: profile[d] = mean attention at relative distance d (averaged
             over all query positions where d is valid, all heads, all layers)
    """
    n_layers, n_heads, seq_len, _ = attn.shape
    max_dist = seq_len - 1

    distance_attn = np.zeros(max_dist)
    distance_counts = np.zeros(max_dist)

    for i in range(1, seq_len):  # query position
        for d in range(1, i + 1):  # distance back
            j = i - d  # key position
            if j >= 0:
                # Mean over all layers and heads
                val = attn[:, :, i, j].mean()
                distance_attn[d - 1] += val
                distance_counts[d - 1] += 1

    # Normalize
    valid = distance_counts > 0
    profile = np.zeros(max_dist)
    profile[valid] = distance_attn[valid] / distance_counts[valid]
    return profile


def fit_power_law(profile, min_dist=2, max_dist=20):
    """
    Fit profile[d] ~ A * d^(-alpha) via log-log linear regression.
    Returns alpha (the decay exponent) and R^2.
    """
    distances = np.arange(1, len(profile) + 1)
    mask = (distances >= min_dist) & (distances <= max_dist) & (profile > 1e-10)

    if mask.sum() < 3:
        return None, None

    log_d = np.log(distances[mask])
    log_p = np.log(profile[mask])

    coeffs = np.polyfit(log_d, log_p, 1)
    alpha = -coeffs[0]  # negative because we defined ~ d^(-alpha)

    fitted = np.polyval(coeffs, log_d)
    ss_res = np.sum((log_p - fitted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return alpha, r2


def compute_layer_profiles(attn):
    """Compute positional profile per layer (averaged over heads)."""
    n_layers, n_heads, seq_len, _ = attn.shape
    profiles = []
    for layer_idx in range(n_layers):
        layer_attn = attn[layer_idx:layer_idx+1]  # keep dims
        profiles.append(compute_positional_profile(layer_attn))
    return profiles


def compute_head_alphas(attn, min_dist=2, max_dist=20):
    """Compute alpha for each (layer, head) pair."""
    n_layers, n_heads, seq_len, _ = attn.shape
    alphas = np.zeros((n_layers, n_heads))
    r2s = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            head_attn = attn[l:l+1, h:h+1]
            profile = compute_positional_profile(head_attn)
            alpha, r2 = fit_power_law(profile, min_dist, max_dist)
            alphas[l, h] = alpha if alpha is not None else np.nan
            r2s[l, h] = r2 if r2 is not None else np.nan
    return alphas, r2s


def run_experiment():
    tokenizer, model = load_model()
    results = {}

    for name, text in PASSAGES.items():
        print(f"\n── Passage: {name} ──")
        attn, tokens = get_attention_weights(text, tokenizer, model)
        seq_len = len(tokens)
        print(f"  Tokens: {seq_len}")
        print(f"  Attention tensor: {attn.shape}  (layers × heads × seq × seq)")

        # Global positional profile
        profile = compute_positional_profile(attn)
        alpha, r2 = fit_power_law(profile)

        print(f"\n  Global positional decay:")
        print(f"    α (power law exponent) = {alpha:.3f}" if alpha else "    fit failed")
        print(f"    R² = {r2:.3f}" if r2 else "")
        print(f"    Interpretation: ", end="")
        if alpha is not None:
            if alpha > 0.5:
                print(f"recency-biased (recent tokens get more attention)")
            elif alpha < -0.2:
                print(f"distance-biased (far tokens get more attention)")
            else:
                print(f"approximately isotropic (weak distance dependence)")

        # Attention to first token (BOS / context anchor)
        first_token_attn = attn[:, :, :, 0].mean()
        print(f"\n  Mean attention to position 0 ('{tokens[0]}'): {first_token_attn:.4f}")

        # Per-layer profiles
        layer_profiles = compute_layer_profiles(attn)
        layer_alphas = []
        for i, lp in enumerate(layer_profiles):
            la, lr2 = fit_power_law(lp)
            if la is not None:
                layer_alphas.append(la)

        if layer_alphas:
            print(f"\n  Per-layer α range: [{min(layer_alphas):.3f}, {max(layer_alphas):.3f}]")
            print(f"  Early layers (0-3) mean α: {np.mean(layer_alphas[:4]):.3f}")
            print(f"  Late layers (8-11) mean α: {np.mean(layer_alphas[8:]):.3f}")

        # Per-head alphas
        head_alphas, head_r2s = compute_head_alphas(attn)
        valid = ~np.isnan(head_alphas)
        if valid.any():
            print(f"\n  Per-head α statistics (n={valid.sum()} heads):")
            print(f"    mean: {np.nanmean(head_alphas):.3f}")
            print(f"    std:  {np.nanstd(head_alphas):.3f}")
            print(f"    min:  {np.nanmin(head_alphas):.3f}")
            print(f"    max:  {np.nanmax(head_alphas):.3f}")
            print(f"    fraction with α > 0.3 (recency-biased): "
                  f"{(head_alphas[valid] > 0.3).mean():.2f}")
            print(f"    fraction with α < 0.1 (near-isotropic): "
                  f"{(head_alphas[valid] < 0.1).mean():.2f}")

        results[name] = {
            "seq_len": seq_len,
            "global_alpha": float(alpha) if alpha is not None else None,
            "global_r2": float(r2) if r2 is not None else None,
            "first_token_attn": float(first_token_attn),
            "layer_alphas": [float(a) for a in layer_alphas],
            "head_alphas_mean": float(np.nanmean(head_alphas)),
            "head_alphas_std": float(np.nanstd(head_alphas)),
            "head_alphas_min": float(np.nanmin(head_alphas)),
            "head_alphas_max": float(np.nanmax(head_alphas)),
            "profile_first_10": [float(p) for p in profile[:10]],
        }

    # ── Cross-passage comparison ───────────────────────────────────────────
    print("\n\n══ SUMMARY ══")
    print(f"{'Passage':<20} {'α (global)':>12} {'R²':>8} {'Head α mean':>14}")
    print("-" * 56)
    for name, r in results.items():
        alpha_str = f"{r['global_alpha']:.3f}" if r['global_alpha'] else "N/A"
        r2_str = f"{r['global_r2']:.3f}" if r['global_r2'] else "N/A"
        print(f"{name:<20} {alpha_str:>12} {r2_str:>8} {r['head_alphas_mean']:>14.3f}")

    print("""
Key: α = power law decay exponent for P(attention | distance=d) ~ d^(-α)
     α ≈ 0: isotropic (distance doesn't matter)
     α > 0: recency-biased (recent tokens more attended)
     α < 0: distance-biased (far tokens more attended)
""")

    # ── Save results ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "attention_anisotropy_01.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
