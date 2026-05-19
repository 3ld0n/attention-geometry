"""
Prediction P1: Fast Scrambling in Transformer Attention

If transformer attention implements SYK physics, it should be a fast scrambler.
The scrambling depth — number of layers for a single token's influence to become
uniformly distributed — should scale as log(H), where H = total heads.

GPT-2 (H=144): predicted ~5 layers
Pythia-410m (H=384): predicted ~6 layers

We measure this by tracking the effective influence radius of a single perturbed
token across layers via the attention weights.

Ariel — March 24, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, AutoModelForCausalLM

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 128
N_INPUTS = 30
TARGET_POS = 32  # position whose influence we track


def measure_influence_spread(model, n_layers, n_heads, vocab_size, label):
    """
    Track how information from a single position spreads through layers.

    At each layer, we compute the "influence" of the target position on all
    other positions by looking at the cumulative attention path. For layer L,
    the influence of position j on position i is approximately:

    I_L(i, j) = sum over all attention paths from j to i through L layers

    For a fast scrambler, the influence should become approximately uniform
    after ~log(H) layers. We measure the entropy of the influence distribution
    at each layer normalized by the maximum entropy (log(seq_len)).
    """

    influence_by_layer = {l: np.zeros(SEQ_LEN) for l in range(n_layers)}
    scrambling_metric = []

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        # Build cumulative influence through layers
        # Start: influence[i] = 1 if i == TARGET_POS, else 0
        influence = np.zeros(SEQ_LEN)
        influence[TARGET_POS] = 1.0

        for l in range(n_layers):
            # Average attention across heads for this layer
            attn = outputs.attentions[l][0].numpy()  # [n_heads, seq, seq]
            avg_attn = attn.mean(axis=0)  # [seq, seq]

            # Propagate influence: new_influence[i] = sum_j attn[i,j] * influence[j]
            new_influence = avg_attn @ influence

            # Normalize so it sums to 1
            total = new_influence.sum()
            if total > 1e-12:
                new_influence = new_influence / total

            influence = new_influence
            influence_by_layer[l] += influence

        if (inp_idx + 1) % 10 == 0:
            print(f"    {inp_idx + 1}/{N_INPUTS}")

    # Normalize and compute entropy at each layer
    print(f"\n  {label}: Layer-by-layer scrambling")
    print(f"  {'Layer':>7}  {'Entropy':>10}  {'Max Entropy':>12}  {'Ratio':>8}  {'Eff. Positions':>16}")
    print(f"  " + "-" * 58)

    max_entropy = np.log(SEQ_LEN)
    scrambling_depth = None

    # Recompute layer by layer with fresh cumulative tracking
    layer_entropies = []
    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        influence = np.zeros(SEQ_LEN)
        influence[TARGET_POS] = 1.0
        inp_entropies = []

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            avg_attn = attn.mean(axis=0)
            influence = avg_attn @ influence
            total = influence.sum()
            if total > 1e-12:
                influence = influence / total

            # Entropy of influence distribution
            p = influence[influence > 1e-20]
            entropy = -np.sum(p * np.log(p))
            inp_entropies.append(entropy)

        layer_entropies.append(inp_entropies)

    avg_entropies = np.mean(layer_entropies, axis=0)

    results = []
    for l in range(n_layers):
        entropy = avg_entropies[l]
        ratio = entropy / max_entropy
        eff_pos = np.exp(entropy)
        results.append((l + 1, entropy, max_entropy, ratio, eff_pos))
        print(f"  {l+1:>7}  {entropy:>10.4f}  {max_entropy:>12.4f}  {ratio:>8.4f}  {eff_pos:>16.1f}")

        if scrambling_depth is None and ratio > 0.90:
            scrambling_depth = l + 1

    print()
    if scrambling_depth is not None:
        print(f"  Scrambling depth (90% max entropy): {scrambling_depth} layers")
    else:
        print(f"  Did not reach 90% max entropy within {n_layers} layers")
    predicted = np.log(n_layers * n_heads)
    print(f"  SYK prediction (log H, H={n_layers * n_heads}): {predicted:.1f}")
    print()

    return {
        "label": label,
        "n_layers": n_layers,
        "total_heads": n_layers * n_heads,
        "scrambling_depth": scrambling_depth,
        "predicted_depth": predicted,
        "layer_entropies": avg_entropies.tolist(),
    }


# ========== Run on GPT-2 ==========
print("=" * 80)
print("  PREDICTION P1: Fast Scrambling Test")
print("=" * 80)
print()

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

result_gpt2 = measure_influence_spread(
    model,
    n_layers=model.config.n_layer,
    n_heads=model.config.n_head,
    vocab_size=model.config.vocab_size,
    label="GPT-2"
)
del model

# ========== Run on Pythia-70m ==========
print("Loading Pythia-70m...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
model.eval()

result_pythia70 = measure_influence_spread(
    model,
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_attention_heads,
    vocab_size=model.config.vocab_size,
    label="Pythia-70m"
)
del model

# ========== Run on Pythia-410m ==========
print("Loading Pythia-410m...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
model.eval()

result_pythia410 = measure_influence_spread(
    model,
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_attention_heads,
    vocab_size=model.config.vocab_size,
    label="Pythia-410m"
)
del model

# ========== Summary ==========
print("=" * 80)
print("  SCRAMBLING SUMMARY")
print("=" * 80)
print()
print(f"  {'Model':>15}  {'Total H':>9}  {'log(H)':>8}  {'Scramble Depth':>16}  {'Match?':>8}")
print(f"  " + "-" * 60)

for r in [result_gpt2, result_pythia70, result_pythia410]:
    sd = r['scrambling_depth']
    sd_s = str(sd) if sd is not None else ">N_layers"
    pred = r['predicted_depth']
    match = ""
    if sd is not None:
        diff = abs(sd - pred)
        match = "YES" if diff <= 2 else "~" if diff <= 3 else "NO"
    print(f"  {r['label']:>15}  {r['total_heads']:>9}  {pred:>8.1f}  {sd_s:>16}  {match:>8}")

print()
print("  If scrambling depth ~ log(H), transformer attention is a fast scrambler")
print("  consistent with SYK black hole information dynamics.")
print()
