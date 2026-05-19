"""
Pre-trained Model Conformal Scaling Test (GPT-2)

Does trained attention develop the conformal structure that random attention cannot?

Three prior experiments (March 24) confirmed: random attention produces NO conformal
structure at any coupling strength. It either freezes or homogenizes. The Second Law
dominates without directed work.

This experiment: load GPT-2 (trained on WebText), extract hidden states and attention
patterns, compute the two-point function G(Δx), and fit for power-law scaling.

Prediction (SYK, D=1): G(Δx) ~ |Δx|^{-2Δ} with Δ = 1/4.

If trained attention shows this and random attention does not, training creates
the conformal structure — directed work against entropy produces self-consistent order.

Ariel — March 24, 2026
"""

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ========== Load model ==========
print("Loading GPT-2...")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

d_model = model.config.n_embd       # 768
n_layers = model.config.n_layer     # 12
n_heads = model.config.n_head       # 12
d_k = d_model // n_heads            # 64
vocab_size = model.config.vocab_size # 50257

print(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_k={d_k}")
print(f"  vocab_size={vocab_size}")
print()

# ========== Experiment parameters ==========
SEQ_LEN = 256
N_INPUTS = 50
MAX_DX = 64
MIN_POS = 32     # skip early positions (insufficient causal context)
FIT_LOW = 3      # lower cutoff for power-law fit
FIT_HIGH = 56    # upper cutoff for power-law fit

LAYER_INDICES = [0, 1, 2, 3, 4, 6, 8, 10, 12]


# ========== Measurement functions ==========

def compute_two_point_normed(X, max_dx, min_pos):
    """G(Δx) = mean cosine similarity at separation Δx."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-10)
    n = X.shape[0]
    G = np.zeros(max_dx)
    for dx in range(max_dx):
        lo = max(min_pos, 0)
        hi = n - dx
        if hi <= lo:
            continue
        G[dx] = np.mean(np.sum(Xn[lo:hi] * Xn[lo + dx:hi + dx], axis=1))
    return G


def compute_two_point_raw(X, max_dx, min_pos):
    """G(Δx) = mean raw kernel K(x, x+Δx) / d."""
    d = X.shape[1]
    n = X.shape[0]
    G = np.zeros(max_dx)
    for dx in range(max_dx):
        lo = max(min_pos, 0)
        hi = n - dx
        if hi <= lo:
            continue
        G[dx] = np.mean(np.sum(X[lo:hi] * X[lo + dx:hi + dx], axis=1)) / d
    return G


def compute_attention_decay(attn_weights, min_pos):
    """
    Average attention weight as function of distance.
    attn_weights: (n_heads, seq_len, seq_len) — causal mask applied.
    Returns A(Δx) for Δx = 0, 1, ..., max_dx-1.
    """
    n_h, seq, _ = attn_weights.shape
    A = np.zeros(MAX_DX)
    counts = np.zeros(MAX_DX)
    for dx in range(MAX_DX):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += np.mean(attn_weights[:, i, j])
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


def fit_power_law(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    """Fit y ~ A * |dx|^{-exponent}. Return (exponent, Δ=exponent/2, R²)."""
    if cutoff_high is None:
        cutoff_high = len(y_arr)
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


def fit_exponential(dx_arr, y_arr, cutoff_low=3, cutoff_high=None):
    """Fit y ~ A * exp(-dx / ξ). Return (ξ, R²)."""
    if cutoff_high is None:
        cutoff_high = len(y_arr)
    mask = (dx_arr >= cutoff_low) & (dx_arr < cutoff_high) & (np.abs(y_arr) > 1e-20)
    if np.sum(mask) < 5:
        return None, None
    log_y = np.log(np.abs(y_arr[mask]))
    dx_vals = dx_arr[mask].astype(float)
    A = np.column_stack([np.ones_like(dx_vals), -dx_vals])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    xi = 1.0 / coeffs[1] if coeffs[1] > 1e-10 else float("inf")
    return xi, R2


# ========== Run experiment ==========
print("=" * 85)
print("  PRE-TRAINED MODEL CONFORMAL SCALING TEST (GPT-2 small)")
print(f"  {N_INPUTS} random token inputs, seq_len={SEQ_LEN}")
print(f"  Measuring G(Δx) for Δx = 0..{MAX_DX-1}, positions {MIN_POS}+")
print(f"  Power-law fit range: Δx = {FIT_LOW}..{FIT_HIGH}")
print("=" * 85)
print()

dx_arr = np.arange(MAX_DX)

# Accumulators
G_norm_sum = {l: np.zeros(MAX_DX) for l in LAYER_INDICES}
G_raw_sum = {l: np.zeros(MAX_DX) for l in LAYER_INDICES}
G_norm_all = {l: [] for l in LAYER_INDICES}

# Attention decay accumulators (layers 1-12)
A_sum = {l: np.zeros(MAX_DX) for l in range(1, n_layers + 1)}

for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

    hidden_states = outputs.hidden_states   # len n_layers+1, each (1, seq, d_model)
    attentions = outputs.attentions          # len n_layers, each (1, n_heads, seq, seq)

    for l in LAYER_INDICES:
        X = hidden_states[l][0].numpy()
        gn = compute_two_point_normed(X, MAX_DX, MIN_POS)
        gr = compute_two_point_raw(X, MAX_DX, MIN_POS)
        G_norm_sum[l] += gn
        G_raw_sum[l] += gr
        G_norm_all[l].append(gn.copy())

    for l in range(n_layers):
        attn = attentions[l][0].numpy()  # (n_heads, seq, seq)
        A_sum[l + 1] += compute_attention_decay(attn, MIN_POS)

    if (inp_idx + 1) % 10 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} inputs processed")

# Averages
G_norm_avg = {l: G_norm_sum[l] / N_INPUTS for l in LAYER_INDICES}
G_raw_avg = {l: G_raw_sum[l] / N_INPUTS for l in LAYER_INDICES}
G_norm_var = {
    l: np.var(np.array(G_norm_all[l]), axis=0)
    for l in LAYER_INDICES
}
A_avg = {l: A_sum[l] / N_INPUTS for l in range(1, n_layers + 1)}


# ========== Part 1: Hidden state two-point function (normalized) ==========
print()
print("=" * 85)
print("  PART 1: Hidden State Two-Point Function — Cosine Similarity G(Δx)")
print("=" * 85)
print()

for l in LAYER_INDICES:
    G = G_norm_avg[l]
    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, G, FIT_LOW, FIT_HIGH)
    xi_exp, R2_exp = fit_exponential(dx_arr, G, FIT_LOW, FIT_HIGH)

    name = "embedding" if l == 0 else f"layer {l:2d}"
    print(f"  {name}: G(1)={G[1]:.4f}  G(4)={G[4]:.4f}  G(8)={G[8]:.4f}  "
          f"G(16)={G[16]:.4f}  G(32)={G[32]:.4f}  G(48)={G[48]:.4f}")

    if exp_pw is not None:
        print(f"           Power law:   exponent={exp_pw:.4f}  Δ={delta_pw:.4f}  R²={R2_pw:.4f}")
    if xi_exp is not None:
        print(f"           Exponential: ξ={xi_exp:.2f}  R²={R2_exp:.4f}")
    if exp_pw is not None and xi_exp is not None:
        better = "POWER LAW" if R2_pw > R2_exp else "EXPONENTIAL"
        print(f"           Better fit: {better}  (ΔR² = {abs(R2_pw - R2_exp):.4f})")
    if delta_pw is not None and abs(delta_pw - 0.25) < 0.15:
        print(f"           *** Δ = {delta_pw:.4f} near SYK prediction 0.25 ***")
    print()


# ========== Part 2: Hidden state two-point function (raw kernel) ==========
print("=" * 85)
print("  PART 2: Hidden State Two-Point Function — Raw Kernel G(Δx)")
print("=" * 85)
print()

for l in LAYER_INDICES:
    G = G_raw_avg[l]
    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, G, FIT_LOW, FIT_HIGH)
    xi_exp, R2_exp = fit_exponential(dx_arr, G, FIT_LOW, FIT_HIGH)

    name = "embedding" if l == 0 else f"layer {l:2d}"
    print(f"  {name}: G(1)={G[1]:.4f}  G(4)={G[4]:.4f}  G(8)={G[8]:.4f}  "
          f"G(16)={G[16]:.4f}  G(32)={G[32]:.4f}  G(48)={G[48]:.4f}")

    if exp_pw is not None:
        print(f"           Power law:   exponent={exp_pw:.4f}  Δ={delta_pw:.4f}  R²={R2_pw:.4f}")
    if xi_exp is not None:
        print(f"           Exponential: ξ={xi_exp:.2f}  R²={R2_exp:.4f}")
    if exp_pw is not None and xi_exp is not None:
        better = "POWER LAW" if R2_pw > R2_exp else "EXPONENTIAL"
        print(f"           Better fit: {better}  (ΔR² = {abs(R2_pw - R2_exp):.4f})")
    if delta_pw is not None and abs(delta_pw - 0.25) < 0.15:
        print(f"           *** Δ = {delta_pw:.4f} near SYK prediction 0.25 ***")
    print()


# ========== Part 3: Attention weight decay ==========
print("=" * 85)
print("  PART 3: Attention Weight Decay — Average α(i, i-Δx) vs Δx")
print("=" * 85)
print()

for l in [1, 2, 4, 6, 8, 10, 12]:
    A = A_avg[l]
    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
    xi_exp, R2_exp = fit_exponential(dx_arr, A, FIT_LOW, FIT_HIGH)

    print(f"  layer {l:2d}: A(1)={A[1]:.4f}  A(4)={A[4]:.4f}  A(8)={A[8]:.4f}  "
          f"A(16)={A[16]:.4f}  A(32)={A[32]:.4f}")

    if exp_pw is not None:
        print(f"           Power law:   exponent={exp_pw:.4f}  Δ={delta_pw:.4f}  R²={R2_pw:.4f}")
    if xi_exp is not None:
        print(f"           Exponential: ξ={xi_exp:.2f}  R²={R2_exp:.4f}")
    if exp_pw is not None and xi_exp is not None:
        better = "POWER LAW" if R2_pw > R2_exp else "EXPONENTIAL"
        print(f"           Better fit: {better}")
    if delta_pw is not None and abs(delta_pw - 0.25) < 0.15:
        print(f"           *** Δ = {delta_pw:.4f} near SYK prediction 0.25 ***")
    print()


# ========== Part 4: Connected correlator (variance across inputs) ==========
print("=" * 85)
print("  PART 4: Connected Correlator — Var[G(Δx)] across inputs")
print("=" * 85)
print()

for l in LAYER_INDICES:
    V = G_norm_var[l]
    exp_pw, delta_pw, R2_pw = fit_power_law(dx_arr, V, FIT_LOW, FIT_HIGH)

    name = "embedding" if l == 0 else f"layer {l:2d}"
    print(f"  {name}: V(1)={V[1]:.2e}  V(4)={V[4]:.2e}  V(16)={V[16]:.2e}  V(32)={V[32]:.2e}")
    if exp_pw is not None:
        print(f"           Power law: exponent={exp_pw:.4f}  Δ_eff={delta_pw:.4f}  R²={R2_pw:.4f}")
    print()


# ========== Summary table ==========
print("=" * 85)
print("  SUMMARY: Conformal Dimension Δ at Each Layer")
print("=" * 85)
print()
print(f"  {'Layer':>12}  {'Δ (cosine)':>12}  {'R² (cos)':>10}  {'Δ (raw)':>10}  "
      f"{'R² (raw)':>10}  {'Δ (attn)':>10}  {'R² (attn)':>10}  {'Best fit':>10}")
print("  " + "-" * 90)

for l in LAYER_INDICES:
    _, dn, rn = fit_power_law(dx_arr, G_norm_avg[l], FIT_LOW, FIT_HIGH)
    _, dr, rr = fit_power_law(dx_arr, G_raw_avg[l], FIT_LOW, FIT_HIGH)

    if l > 0:
        _, da, ra = fit_power_law(dx_arr, A_avg[l], FIT_LOW, FIT_HIGH)
    else:
        da, ra = None, None

    name = "embedding" if l == 0 else f"layer {l}"
    dn_s = f"{dn:.4f}" if dn is not None else "---"
    rn_s = f"{rn:.4f}" if rn is not None else "---"
    dr_s = f"{dr:.4f}" if dr is not None else "---"
    rr_s = f"{rr:.4f}" if rr is not None else "---"
    da_s = f"{da:.4f}" if da is not None else "---"
    ra_s = f"{ra:.4f}" if ra is not None else "---"

    # Determine best fit
    _, _, r2_exp_cos = fit_power_law(dx_arr, G_norm_avg[l], FIT_LOW, FIT_HIGH)
    xi_e, r2_e = fit_exponential(dx_arr, G_norm_avg[l], FIT_LOW, FIT_HIGH)
    if r2_exp_cos is not None and r2_e is not None:
        best = "power law" if r2_exp_cos > r2_e else "exponent."
    else:
        best = "---"

    print(f"  {name:>12}  {dn_s:>12}  {rn_s:>10}  {dr_s:>10}  "
          f"{rr_s:>10}  {da_s:>10}  {ra_s:>10}  {best:>10}")

print()
print("  SYK prediction for D=1: Δ = 0.2500 (exponent = 0.5000)")
print()
print("  Random attention baseline (3 experiments, March 24):")
print("    NO conformal structure at any σ. Freezes or homogenizes.")
print("    If trained attention shows Δ ≈ 0.25 with high R²,")
print("    training creates conformal structure that entropy alone cannot.")
print()


# ========== Detailed tail behavior ==========
print("=" * 85)
print("  TAIL BEHAVIOR: Ratios at Large Separation")
print("=" * 85)
print()
print(f"  SYK Δ=1/4 prediction: G(16)/G(4) = {(16/4)**(-0.5):.4f},  "
      f"G(32)/G(4) = {(32/4)**(-0.5):.4f},  G(48)/G(4) = {(48/4)**(-0.5):.4f}")
print()
print(f"  {'Layer':>12}  {'G(16)/G(4)':>12}  {'G(32)/G(4)':>12}  {'G(48)/G(4)':>12}")
print("  " + "-" * 52)

for l in LAYER_INDICES:
    G = G_norm_avg[l]
    if abs(G[4]) > 1e-10:
        r16 = G[16] / G[4]
        r32 = G[32] / G[4]
        r48 = G[48] / G[4]
    else:
        r16 = r32 = r48 = float("nan")
    name = "embedding" if l == 0 else f"layer {l}"
    print(f"  {name:>12}  {r16:12.4f}  {r32:12.4f}  {r48:12.4f}")

print()
print("  (Ratios closer to SYK prediction = more conformal)")
print()
