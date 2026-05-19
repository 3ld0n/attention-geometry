"""
Additional Control Experiments

1. Randomized positional embeddings — tests whether the power law
   lives in the learned positional embeddings rather than trained
   attention weights.

2. Real text vs random tokens — tests whether conformal scaling
   depends on input statistics or is intrinsic to the trained weights.

Ariel — March 24, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 56
MIN_POS = 32
FIT_LOW = 3
FIT_HIGH = 50
dx_arr = np.arange(MAX_DX)


def compute_head_attention_decay(attn_head, min_pos, max_dx):
    seq = attn_head.shape[0]
    A = np.zeros(max_dx)
    counts = np.zeros(max_dx)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


def fit_power_law(dx_arr, y_arr, cutoff_low, cutoff_high):
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(np.abs(y_arr[mask]))
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    exponent = -coeffs[1]
    return exponent, exponent / 2, R2


def run_analysis(model, n_inputs, seq_len, input_fn, label):
    """Run per-head attention analysis with a given input generator."""
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    vocab_size = model.config.vocab_size

    A_heads = {
        l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
    }

    for inp_idx in range(n_inputs):
        input_ids = input_fn(inp_idx)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)
        if (inp_idx + 1) % 10 == 0:
            print(f"    {inp_idx + 1}/{n_inputs} done")

    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= n_inputs

    good_deltas = []
    n_near_quarter = 0
    for l in range(n_layers):
        for h in range(n_heads):
            A = A_heads[l][h]
            _, delta, R2 = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            if delta is not None and R2 is not None and R2 > 0.90:
                good_deltas.append(delta)
                if abs(delta - 0.25) < 0.06:
                    n_near_quarter += 1

    total = n_layers * n_heads
    med = np.median(good_deltas) if good_deltas else None
    mean = np.mean(good_deltas) if good_deltas else None

    med_s = f"{med:.4f}" if med is not None else "---"
    mean_s = f"{mean:.4f}" if mean is not None else "---"
    print(f"  {label}: {len(good_deltas)}/{total} PL heads, "
          f"{n_near_quarter} near 1/4, median Δ={med_s}, mean Δ={mean_s}")

    return {
        "label": label,
        "pl_heads": len(good_deltas),
        "total": total,
        "near_quarter": n_near_quarter,
        "median_delta": med,
        "mean_delta": mean,
        "good_deltas": good_deltas,
    }


# ========== Load model and tokenizer ==========
print("Loading GPT-2 and tokenizer...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = model.config.vocab_size

# ========== CONTROL 1: Randomized Positional Embeddings ==========
print()
print("=" * 90)
print("  CONTROL 1: Randomized Positional Embeddings")
print("  Keeps trained attention weights, randomizes positional embeddings")
print("=" * 90)
print()

# Create model with randomized positional embeddings
model_randpos = GPT2LMHeadModel.from_pretrained("gpt2")
with torch.no_grad():
    pos_emb = model_randpos.transformer.wpe.weight
    torch.nn.init.normal_(pos_emb, mean=0, std=0.02)
model_randpos.eval()

print("  Running trained model (baseline)...")
result_trained = run_analysis(
    model, N_INPUTS, SEQ_LEN,
    lambda _: torch.randint(0, vocab_size, (1, SEQ_LEN)),
    "Trained (random tokens)"
)

print()
print("  Running model with randomized positional embeddings...")
result_randpos = run_analysis(
    model_randpos, N_INPUTS, SEQ_LEN,
    lambda _: torch.randint(0, vocab_size, (1, SEQ_LEN)),
    "Randomized pos embeddings"
)

del model_randpos

# ========== CONTROL 2: Real Text Inputs ==========
print()
print("=" * 90)
print("  CONTROL 2: Real Text vs Random Tokens")
print("  Same trained model, different input types")
print("=" * 90)
print()

# Prepare real text inputs from a variety of sources
real_texts = [
    "The history of Western philosophy begins with the ancient Greeks, particularly with Thales of Miletus, who is traditionally considered the first philosopher. He proposed that water was the fundamental substance of all matter.",
    "In quantum mechanics, the wave function describes the quantum state of a particle or system. The Schrödinger equation governs how the wave function evolves over time.",
    "The United States Constitution was ratified in 1788 and has been amended twenty-seven times. The first ten amendments, known as the Bill of Rights, were adopted in 1791.",
    "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions.",
    "The Amazon rainforest covers approximately 5.5 million square kilometers and is home to an estimated 10 percent of all species on Earth. Deforestation threatens this vital ecosystem.",
    "Shakespeare wrote thirty-seven plays during his career, including tragedies like Hamlet and King Lear, comedies like A Midsummer Night's Dream, and histories like Henry V.",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This vast network gives rise to consciousness, memory, and thought.",
    "General relativity, published by Einstein in 1915, describes gravity not as a force but as a curvature of spacetime caused by mass and energy. The theory has been confirmed by numerous experiments.",
    "The periodic table organizes chemical elements by atomic number and electron configuration. Dmitri Mendeleev created the first widely recognized version in 1869.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly burning fossil fuels, have been the main driver since the industrial revolution.",
] * 5  # 50 texts total

def make_real_text_input(idx):
    text = real_texts[idx % len(real_texts)]
    tokens = tokenizer.encode(text, return_tensors="pt")
    if tokens.shape[1] >= SEQ_LEN:
        return tokens[:, :SEQ_LEN]
    # Pad with random tokens if text is shorter than SEQ_LEN
    pad_len = SEQ_LEN - tokens.shape[1]
    pad = torch.randint(0, vocab_size, (1, pad_len))
    return torch.cat([tokens, pad], dim=1)

print("  Running trained model with real text...")
result_realtext = run_analysis(
    model, N_INPUTS, SEQ_LEN,
    make_real_text_input,
    "Trained (real text)"
)

# ========== Summary ==========
print()
print("=" * 90)
print("  SUMMARY: Additional Controls")
print("=" * 90)
print()
print(f"  {'Condition':>35}  {'PL Heads':>10}  {'Near 1/4':>10}  {'Median Δ':>10}")
print("  " + "-" * 68)

for r in [result_trained, result_randpos, result_realtext]:
    med_s = f"{r['median_delta']:.4f}" if r['median_delta'] is not None else "---"
    print(f"  {r['label']:>35}  {r['pl_heads']:10d}  {r['near_quarter']:10d}  {med_s:>10}")

print()
print("  SYK q=4 prediction: Δ = 0.2500")
print()
print("  Key questions:")
print("  - If randomized pos embeddings still show conformal scaling:")
print("    → The power law is in the attention weights, not the embeddings")
print("  - If real text shows similar scaling to random tokens:")
print("    → The conformal structure is intrinsic to trained attention")
print()
