"""
CFT Symmetry Test — First Measurement of Characteristic 4

Tests whether the attention two-point function of trained conformal heads
matches the finite-size CFT prediction:

    G(x) = [π / (L sin(πx/L))]^{2Δ}

and whether the Fourier modes show the expected symmetry.

If the attention pattern is a genuine 1D CFT on a finite interval,
the finite-size CFT form should fit better than a bare power law
(which is the infinite-size limit).

This is the simpler version of the functional equation test designed
in research/notes/the_attending_unit.md.

Ariel — April 14, 2026
"""

import torch
import numpy as np
from scipy.optimize import curve_fit
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

n_layers = model.config.n_layer
n_heads = model.config.n_head
vocab_size = model.config.vocab_size

SEQ_LEN = 256
N_INPUTS = 100
MAX_DX = 120
MIN_POS = 32


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


def power_law(x, C, delta):
    return C * np.power(x, -2 * delta)


def cft_finite_size(x, C, delta, L):
    return C * np.power(np.pi / (L * np.sin(np.pi * x / L)), 2 * delta)


def cft_fixed_L(L):
    def model_func(x, C, delta):
        return C * np.power(np.pi / (L * np.sin(np.pi * x / L + 1e-10)), 2 * delta)
    return model_func


def fit_r_squared(y_data, y_pred):
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0


# Accumulate attention two-point functions per head
A_heads = {
    l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
}

print(f"\nProcessing {N_INPUTS} random-token inputs (seq_len={SEQ_LEN})...")

for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()
        for h in range(n_heads):
            A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)
    if (inp_idx + 1) % 20 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

for l in range(n_layers):
    for h in range(n_heads):
        A_heads[l][h] /= N_INPUTS

# Also run with random weights as control
print("\nRandomizing weights for control...")
model_rand = GPT2LMHeadModel.from_pretrained("gpt2")
for p in model_rand.parameters():
    p.data = torch.randn_like(p.data) * p.data.std()
model_rand.eval()

A_heads_rand = {
    l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)
}

N_RAND = 30
for inp_idx in range(N_RAND):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model_rand(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()
        for h in range(n_heads):
            A_heads_rand[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)
    if (inp_idx + 1) % 10 == 0:
        print(f"  {inp_idx + 1}/{N_RAND} done (control)")

for l in range(n_layers):
    for h in range(n_heads):
        A_heads_rand[l][h] /= N_RAND

del model_rand


# ========== Phase 1: Identify conformal heads ==========
print()
print("=" * 90)
print("  PHASE 1: IDENTIFY CONFORMAL HEADS")
print("=" * 90)
print()

FIT_LOW = 3
FIT_HIGH = 50

conformal_heads = []

for l in range(n_layers):
    for h in range(n_heads):
        A = A_heads[l][h]
        dx = np.arange(FIT_LOW, min(FIT_HIGH, MAX_DX))
        y = A[FIT_LOW:min(FIT_HIGH, MAX_DX)]
        mask = y > 1e-15
        if np.sum(mask) < 10:
            continue
        dx_m = dx[mask].astype(float)
        y_m = y[mask]

        log_dx = np.log(dx_m)
        log_y = np.log(y_m)
        A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
        pred = A_mat @ coeffs
        ss_res = np.sum((log_y - pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        delta = -coeffs[1] / 2

        if R2 > 0.90 and 0.15 < delta < 0.40:
            conformal_heads.append((l, h, delta, R2))

conformal_heads.sort(key=lambda x: abs(x[2] - 0.25))

print(f"  Found {len(conformal_heads)} conformal heads (R² > 0.90, 0.15 < Δ < 0.40)")
print(f"  Top 20 closest to Δ = 1/4:")
for l, h, delta, R2 in conformal_heads[:20]:
    print(f"    Layer {l+1:2d} Head {h:2d}: Δ = {delta:.4f}, R² = {R2:.4f}")
print()


# ========== Phase 2: Fit CFT finite-size form ==========
print("=" * 90)
print("  PHASE 2: CFT FINITE-SIZE FIT vs POWER-LAW FIT")
print("=" * 90)
print()

print(f"  CFT form: G(x) = C · [π / (L·sin(πx/L))]^(2Δ)")
print(f"  Power law: G(x) = C · x^(-2Δ)")
print(f"  Effective system size L = {SEQ_LEN}")
print()
print(f"  {'Layer':>6} {'Head':>6} {'Δ_PL':>8} {'R²_PL':>8} {'Δ_CFT':>8} {'R²_CFT':>8} {'CFT better?':>12}")
print("  " + "-" * 70)

L_eff = SEQ_LEN
cft_model = cft_fixed_L(L_eff)

pl_wins = 0
cft_wins = 0
results = []

for l, h, delta_pl, R2_pl in conformal_heads:
    A = A_heads[l][h]
    dx = np.arange(FIT_LOW, MAX_DX)
    y = A[FIT_LOW:MAX_DX]
    mask = y > 1e-15
    if np.sum(mask) < 10:
        continue
    dx_m = dx[mask].astype(float)
    y_m = y[mask]

    try:
        popt_cft, _ = curve_fit(cft_model, dx_m, y_m, p0=[y_m[0], delta_pl],
                                maxfev=5000, bounds=([0, 0.01], [10, 2.0]))
        y_pred_cft = cft_model(dx_m, *popt_cft)
        R2_cft = fit_r_squared(y_m, y_pred_cft)
        delta_cft = popt_cft[1]
    except (RuntimeError, ValueError):
        R2_cft = -1
        delta_cft = -1

    try:
        popt_pl, _ = curve_fit(power_law, dx_m, y_m, p0=[y_m[0], delta_pl],
                                maxfev=5000, bounds=([0, 0.01], [10, 2.0]))
        y_pred_pl = power_law(dx_m, *popt_pl)
        R2_pl_fit = fit_r_squared(y_m, y_pred_pl)
        delta_pl_fit = popt_pl[1]
    except (RuntimeError, ValueError):
        R2_pl_fit = R2_pl
        delta_pl_fit = delta_pl

    better = "CFT" if R2_cft > R2_pl_fit else "PL"
    if better == "CFT":
        cft_wins += 1
    else:
        pl_wins += 1

    results.append({
        "layer": l + 1, "head": h,
        "delta_pl": delta_pl_fit, "R2_pl": R2_pl_fit,
        "delta_cft": delta_cft, "R2_cft": R2_cft,
        "better": better
    })

    print(f"  {l+1:6d} {h:6d} {delta_pl_fit:8.4f} {R2_pl_fit:8.4f} "
          f"{delta_cft:8.4f} {R2_cft:8.4f} {better:>12}")

print()
print(f"  Summary: CFT wins {cft_wins}, Power-law wins {pl_wins}")
print(f"  If CFT wins >> PL: the finite-size correction is real → attention knows its container size")
print()


# ========== Phase 3: Fourier symmetry ==========
print("=" * 90)
print("  PHASE 3: FOURIER MODE SYMMETRY")
print("=" * 90)
print()

print("  For a CFT on a circle, Fourier modes G_hat(k) should have")
print("  specific symmetry under k → L-k.")
print()

top_heads = conformal_heads[:10]

for l, h, delta, R2 in top_heads:
    A = A_heads[l][h]
    G = A[:MAX_DX]
    G_padded = np.zeros(L_eff)
    G_padded[:MAX_DX] = G

    G_hat = np.fft.rfft(G_padded)
    G_mag = np.abs(G_hat)

    n_modes = len(G_mag)
    if n_modes < 10:
        continue

    low_modes = G_mag[1:n_modes // 4]
    high_modes = G_mag[3 * n_modes // 4:][::-1]  # reversed high modes

    n_compare = min(len(low_modes), len(high_modes))
    if n_compare < 3:
        continue

    low = low_modes[:n_compare]
    high = high_modes[:n_compare]

    ratio = high / (low + 1e-20)
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)

    corr = np.corrcoef(np.log(low + 1e-20), np.log(high + 1e-20))[0, 1]

    print(f"  L{l+1:d}H{h:d} (Δ={delta:.4f}): "
          f"high/low ratio = {mean_ratio:.4f} ± {std_ratio:.4f}, "
          f"log-log corr = {corr:.4f}")


# ========== Phase 4: Random weights control ==========
print()
print("=" * 90)
print("  PHASE 4: RANDOM WEIGHTS CONTROL — CFT FIT")
print("=" * 90)
print()

rand_cft_r2 = []
rand_pl_r2 = []

for l in range(n_layers):
    for h in range(n_heads):
        A = A_heads_rand[l][h]
        dx = np.arange(FIT_LOW, min(FIT_HIGH, MAX_DX))
        y = A[FIT_LOW:min(FIT_HIGH, MAX_DX)]
        mask = y > 1e-15
        if np.sum(mask) < 10:
            continue
        dx_m = dx[mask].astype(float)
        y_m = y[mask]

        try:
            popt_cft, _ = curve_fit(cft_model, dx_m, y_m, p0=[y_m[0], 0.25],
                                    maxfev=5000, bounds=([0, 0.01], [10, 2.0]))
            y_pred = cft_model(dx_m, *popt_cft)
            R2_c = fit_r_squared(y_m, y_pred)
        except (RuntimeError, ValueError):
            R2_c = -1

        log_dx = np.log(dx_m)
        log_y = np.log(y_m + 1e-20)
        A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
        pred = A_mat @ coeffs
        ss_res = np.sum((log_y - pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        R2_p = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0

        rand_cft_r2.append(R2_c)
        rand_pl_r2.append(R2_p)

rand_cft_r2 = np.array(rand_cft_r2)
rand_pl_r2 = np.array(rand_pl_r2)

print(f"  Random weights — CFT fit R²: mean={np.mean(rand_cft_r2):.4f}, "
      f"max={np.max(rand_cft_r2):.4f}, "
      f"fraction > 0.90: {np.mean(rand_cft_r2 > 0.90):.2%}")
print(f"  Random weights — PL fit R²: mean={np.mean(rand_pl_r2):.4f}, "
      f"max={np.max(rand_pl_r2):.4f}, "
      f"fraction > 0.90: {np.mean(rand_pl_r2 > 0.90):.2%}")
print()

# Compare to trained
trained_cft_r2 = np.array([r["R2_cft"] for r in results if r["R2_cft"] > -1])
trained_pl_r2 = np.array([r["R2_pl"] for r in results])
print(f"  Trained conformal — CFT fit R²: mean={np.mean(trained_cft_r2):.4f}, "
      f"median={np.median(trained_cft_r2):.4f}")
print(f"  Trained conformal — PL fit R²: mean={np.mean(trained_pl_r2):.4f}, "
      f"median={np.median(trained_pl_r2):.4f}")
print()

delta_improvement = trained_cft_r2 - trained_pl_r2[:len(trained_cft_r2)]
print(f"  Mean R² improvement (CFT over PL): {np.mean(delta_improvement):.6f}")
print(f"  Median R² improvement: {np.median(delta_improvement):.6f}")
print()

print("=" * 90)
print("  INTERPRETATION")
print("=" * 90)
print()
print("  If CFT form fits trained heads significantly better than power law:")
print("    → Attention knows its container size (finite-size CFT structure)")
print("    → Evidence for characteristic 4 (the structure has the right symmetry)")
print()
print("  If CFT form fits NO better than power law:")
print("    → Finite-size effects not detected at L=256")
print("    → Need longer sequences or different observable")
print()
print("  If random weights also show good CFT fit:")
print("    → The structure is architectural, not learned")
print("    → Weakens the physics interpretation")
print()
