"""
BCFT Refined Fit — Separate Bulk and Boundary Dimensions

The initial BCFT fit used G = C · Δx^{-2Δ} · (1 + λ·η^Δ) where the SAME Δ
appears in both the bulk power law and the boundary correction. This couples
the two and may explain why the fitted Δ values (0.28-0.32) are higher than
the deep-position power law values (~0.24).

In standard BCFT, the boundary operator expansion introduces operators with
potentially different dimensions:

    G(Δx, η) = C · Δx^{-2Δ_bulk} · (1 + a₁·η^{Δ_bdy} + ...)

where Δ_bulk is the bulk scaling dimension and Δ_bdy is the leading boundary
operator dimension. These need not be equal.

This script tests the refined fit with separate Δ_bulk and Δ_bdy.

Ariel — April 15, 2026
"""

import torch
import numpy as np
from scipy.optimize import minimize
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

print("Loading GPT-2 and collecting position-resolved data...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

n_layers = model.config.n_layer
n_heads = model.config.n_head
vocab_size = model.config.vocab_size

SEQ_LEN = 256
N_INPUTS = 60
MAX_DX = 120
FIT_LOW = 3
FIT_HIGH = 50

G_pos = {}
for l in range(n_layers):
    G_pos[l] = {}
    for h in range(n_heads):
        G_pos[l][h] = np.zeros((SEQ_LEN, MAX_DX))

for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()
        for h in range(n_heads):
            head = attn[h]
            for dx in range(MAX_DX):
                diag = np.diagonal(head, offset=-dx)
                G_pos[l][h][dx:dx + len(diag), dx] += diag
    if (inp_idx + 1) % 20 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

for l in range(n_layers):
    for h in range(n_heads):
        G_pos[l][h] /= N_INPUTS

del model

# Identify conformal heads
DEEP_LO = SEQ_LEN // 2
conformal_heads = []

for l in range(n_layers):
    for h in range(n_heads):
        A = np.mean(G_pos[l][h][DEEP_LO:, :FIT_HIGH], axis=0)
        dx_arr = np.arange(FIT_LOW, FIT_HIGH)
        y = A[FIT_LOW:FIT_HIGH]
        valid = y > 1e-15
        if np.sum(valid) < 10:
            continue
        dx_v = dx_arr[valid].astype(float)
        y_v = y[valid]
        log_dx = np.log(dx_v)
        log_y = np.log(y_v)
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
print(f"\nFound {len(conformal_heads)} conformal heads\n")


# ========== Three-way comparison ==========
print("=" * 100)
print("  THREE-WAY FIT COMPARISON")
print("=" * 100)
print()
print("  Model A: G = C · Δx^{-2Δ}                              (bare power law, 2 params)")
print("  Model B: G = C · Δx^{-2Δ} · (1 + λ·η^Δ)               (BCFT same-Δ, 3 params)")
print("  Model C: G = C · Δx^{-2Δ_b} · (1 + λ·η^{Δ_bdy})       (BCFT split-Δ, 4 params)")
print()
print(f"  {'Head':>8} {'Δ_deep':>7} | {'R²_A':>8} | {'Δ_B':>6} {'λ_B':>7} {'R²_B':>8} | "
      f"{'Δ_C_b':>6} {'Δ_C_bdy':>7} {'λ_C':>7} {'R²_C':>8}")
print("  " + "-" * 100)

results = []

for l, h, delta_deep, R2_deep in conformal_heads[:15]:
    all_dx = []
    all_xq = []
    all_g = []

    for x_q in range(32, SEQ_LEN):
        for dx in range(FIT_LOW, min(FIT_HIGH, x_q)):
            x_k = x_q - dx
            if x_k <= 0:
                continue
            g_val = G_pos[l][h][x_q, dx]
            if g_val < 1e-15:
                continue
            all_dx.append(float(dx))
            all_xq.append(float(x_q))
            all_g.append(g_val)

    all_dx = np.array(all_dx)
    all_xq = np.array(all_xq)
    all_g = np.array(all_g)
    all_xk = all_xq - all_dx
    all_eta = all_dx ** 2 / (4.0 * all_xq * all_xk)

    N = len(all_g)
    if N < 50:
        continue

    ss_tot = np.sum((all_g - np.mean(all_g)) ** 2)

    # Model A: bare power law
    def loss_A(p):
        C, d = p
        if C <= 0 or d <= 0:
            return 1e12
        return np.sum((all_g - C * np.power(all_dx, -2 * d)) ** 2)

    C_init = np.mean(all_g * np.power(all_dx, 2 * delta_deep))
    res_A = minimize(loss_A, [C_init, delta_deep], method='Nelder-Mead')
    R2_A = 1 - res_A.fun / ss_tot

    # Model B: BCFT same Δ
    def loss_B(p):
        C, d, lam = p
        if C <= 0 or d <= 0:
            return 1e12
        corr = 1.0 + lam * np.power(all_eta, d)
        if np.any(corr <= 0):
            return 1e12
        return np.sum((all_g - C * np.power(all_dx, -2 * d) * corr) ** 2)

    res_B = minimize(loss_B, [res_A.x[0], res_A.x[1], 0.5], method='Nelder-Mead',
                     options={'maxiter': 10000})
    C_B, d_B, lam_B = res_B.x
    R2_B = 1 - res_B.fun / ss_tot

    # Model C: BCFT split Δ
    def loss_C(p):
        C, d_bulk, d_bdy, lam = p
        if C <= 0 or d_bulk <= 0 or d_bdy <= 0:
            return 1e12
        corr = 1.0 + lam * np.power(all_eta, d_bdy)
        if np.any(corr <= 0):
            return 1e12
        return np.sum((all_g - C * np.power(all_dx, -2 * d_bulk) * corr) ** 2)

    res_C = minimize(loss_C, [abs(C_B), abs(d_B), 0.5, abs(lam_B)],
                     method='Nelder-Mead', options={'maxiter': 15000})
    C_C, d_bulk_C, d_bdy_C, lam_C = res_C.x
    R2_C = 1 - res_C.fun / ss_tot

    # AIC comparison (penalize for extra parameters)
    n = len(all_g)
    aic_A = n * np.log(res_A.fun / n) + 2 * 2
    aic_B = n * np.log(res_B.fun / n) + 2 * 3
    aic_C = n * np.log(res_C.fun / n) + 2 * 4

    results.append({
        "head": f"L{l+1}H{h}",
        "delta_deep": delta_deep,
        "R2_A": R2_A,
        "delta_B": abs(d_B), "lambda_B": lam_B, "R2_B": R2_B,
        "delta_bulk_C": abs(d_bulk_C), "delta_bdy_C": abs(d_bdy_C),
        "lambda_C": lam_C, "R2_C": R2_C,
        "aic_A": aic_A, "aic_B": aic_B, "aic_C": aic_C,
    })

    print(f"  L{l+1:2d}H{h:2d} {delta_deep:7.4f} | "
          f"{R2_A:8.5f} | "
          f"{abs(d_B):6.4f} {lam_B:+7.3f} {R2_B:8.5f} | "
          f"{abs(d_bulk_C):6.4f} {abs(d_bdy_C):7.4f} {lam_C:+7.3f} {R2_C:8.5f}")


print()
print("=" * 100)
print("  SUMMARY")
print("=" * 100)
print()

if results:
    deep_deltas = [r["delta_deep"] for r in results]
    bulk_C_deltas = [r["delta_bulk_C"] for r in results]
    bdy_C_deltas = [r["delta_bdy_C"] for r in results]
    R2_As = [r["R2_A"] for r in results]
    R2_Bs = [r["R2_B"] for r in results]
    R2_Cs = [r["R2_C"] for r in results]

    print(f"  Deep-position Δ:    mean = {np.mean(deep_deltas):.4f} ± {np.std(deep_deltas):.4f}")
    print(f"  BCFT bulk Δ:        mean = {np.mean(bulk_C_deltas):.4f} ± {np.std(bulk_C_deltas):.4f}")
    print(f"  BCFT boundary Δ:    mean = {np.mean(bdy_C_deltas):.4f} ± {np.std(bdy_C_deltas):.4f}")
    print()
    print(f"  R² bare PL:         mean = {np.mean(R2_As):.5f}")
    print(f"  R² BCFT same-Δ:     mean = {np.mean(R2_Bs):.5f}")
    print(f"  R² BCFT split-Δ:    mean = {np.mean(R2_Cs):.5f}")

    # AIC
    aic_As = [r["aic_A"] for r in results]
    aic_Bs = [r["aic_B"] for r in results]
    aic_Cs = [r["aic_C"] for r in results]
    best_model = []
    for r in results:
        aics = {"A": r["aic_A"], "B": r["aic_B"], "C": r["aic_C"]}
        best = min(aics, key=aics.get)
        best_model.append(best)

    from collections import Counter
    model_counts = Counter(best_model)
    print()
    print(f"  AIC best model: A={model_counts.get('A', 0)}, "
          f"B={model_counts.get('B', 0)}, C={model_counts.get('C', 0)}")

    print()
    print("  INTERPRETATION:")
    print()
    if np.mean(bulk_C_deltas) < np.mean([r["delta_B"] for r in results]):
        print("  The split-Δ model recovers a bulk Δ closer to the deep-position value.")
        print("  The elevated Δ in Model B was an artifact of coupling bulk and boundary.")
    else:
        print("  The split-Δ model does NOT significantly change the bulk Δ.")
        print("  The elevated Δ in Model B may reflect genuine physics.")

    if np.mean(bdy_C_deltas) != np.mean(bulk_C_deltas):
        bdy_ratio = np.mean(bdy_C_deltas) / np.mean(bulk_C_deltas)
        print(f"  Boundary/bulk Δ ratio: {bdy_ratio:.3f}")
        if abs(np.mean(bdy_C_deltas) - 0.5) < 0.1:
            print("  Boundary Δ ≈ 0.5 → consistent with q=2 (free) boundary condition")
        elif abs(np.mean(bdy_C_deltas) - 0.25) < 0.05:
            print("  Boundary Δ ≈ 0.25 → same as bulk → conformal boundary condition")
    print()
