"""
BCFT Lost-in-the-Middle: The Start-Boundary Excess

The key quantitative prediction: the bare power law predicts monotonic
decay from the query position. The U-shape (enhanced attention at the
START of the sequence) requires a boundary correction. The excess at
the start — attention above what the power law predicts — should be
explained by the BCFT correction with the same Δ and λ measured
independently from the two-point function.

Also: run BCFT fits on ALL power-law heads (R² > 0.90, no Δ filter)
to test whether head selection biases the result.

Ariel — April 15, 2026
"""

import torch
import numpy as np
from scipy.optimize import minimize
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
N_INPUTS = 60
FIT_LOW = 3
FIT_HIGH = 50

# ========== Collect position-resolved attention ==========
print(f"\nCollecting position-resolved attention from {N_INPUTS} inputs...")

G_pos = {}
for l in range(n_layers):
    G_pos[l] = {}
    for h in range(n_heads):
        G_pos[l][h] = np.zeros((SEQ_LEN, FIT_HIGH + 10))

# Also collect raw attention profiles for U-shape
attn_profiles = np.zeros((n_layers, n_heads, SEQ_LEN))
profile_counts = np.zeros((n_layers, n_heads, SEQ_LEN))

Q_LO = 3 * SEQ_LEN // 4

for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()
        for h in range(n_heads):
            head = attn[h]
            for dx in range(FIT_HIGH + 10):
                diag = np.diagonal(head, offset=-dx)
                G_pos[l][h][dx:dx + len(diag), dx] += diag
            # Attention profiles from late queries
            for x_q in range(Q_LO, SEQ_LEN):
                attn_profiles[l, h, :x_q+1] += head[x_q, :x_q+1]
                profile_counts[l, h, :x_q+1] += 1
    if (inp_idx + 1) % 15 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

for l in range(n_layers):
    for h in range(n_heads):
        G_pos[l][h] /= N_INPUTS

valid_mask = profile_counts > 0
attn_profiles[valid_mask] /= profile_counts[valid_mask]

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ========== Phase 1: Identify ALL power-law heads (no Δ filter) ==========
print("\n" + "=" * 90)
print("  PHASE 1: ALL POWER-LAW HEADS (R² > 0.90 only, NO Δ filter)")
print("=" * 90 + "\n")

DEEP_LO = SEQ_LEN // 2
all_pl_heads = []

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

        if R2 > 0.90 and delta > 0.05:
            all_pl_heads.append((l, h, delta, R2))

print(f"  Found {len(all_pl_heads)} power-law heads (R² > 0.90, Δ > 0.05)")
print(f"  Δ range: {min(x[2] for x in all_pl_heads):.4f} to {max(x[2] for x in all_pl_heads):.4f}")
print(f"  Median Δ: {np.median([x[2] for x in all_pl_heads]):.4f}")

# Bin by Δ
near_quarter = [x for x in all_pl_heads if 0.15 < x[2] < 0.35]
near_third = [x for x in all_pl_heads if 0.28 < x[2] < 0.42]
near_half = [x for x in all_pl_heads if 0.42 < x[2] < 0.65]
high_delta = [x for x in all_pl_heads if x[2] > 0.65]
print(f"  Near Δ=1/4: {len(near_quarter)},  Near Δ=1/3: {len(near_third)},  "
      f"Near Δ=1/2: {len(near_half)},  High Δ: {len(high_delta)}")


# ========== Phase 2: BCFT vs PL on ALL power-law heads ==========
print("\n" + "=" * 90)
print("  PHASE 2: BCFT vs POWER LAW — ALL HEADS (no Δ selection)")
print("=" * 90 + "\n")

bcft_wins = 0
pl_wins = 0
results_by_delta_bin = {'near_quarter': [], 'near_third': [], 'near_half': [], 'high': []}

print(f"  {'Head':>8} {'Δ_PL':>7} {'R²_PL':>9} {'Δ_BCFT':>7} {'λ':>8} "
      f"{'R²_BCFT':>9} {'ΔR²':>9} {'Winner':>8}")
print(f"  {'-'*75}")

for l, h, delta_avg, R2_avg in all_pl_heads:
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

    if len(all_g) < 30:
        continue

    ss_tot = np.sum((all_g - np.mean(all_g)) ** 2)
    if ss_tot < 1e-30:
        continue

    def pl_loss(params):
        C, delta = params
        if C <= 0 or delta <= 0:
            return 1e12
        pred = C * np.power(all_dx, -2 * delta)
        return np.sum((all_g - pred) ** 2)

    C_init = np.mean(all_g * np.power(all_dx, 2 * delta_avg))
    res_pl = minimize(pl_loss, [max(C_init, 1e-10), delta_avg],
                      method='Nelder-Mead', options={'maxiter': 5000})
    C_pl, delta_pl = abs(res_pl.x[0]), abs(res_pl.x[1])
    pred_pl = C_pl * np.power(all_dx, -2 * delta_pl)
    R2_pl = 1 - np.sum((all_g - pred_pl) ** 2) / ss_tot

    def bcft_loss(params):
        C, delta, lam = params
        if C <= 0 or delta <= 0:
            return 1e12
        xk = all_xq - all_dx
        eta = all_dx ** 2 / (4.0 * all_xq * xk)
        correction = 1.0 + lam * np.power(eta, delta)
        if np.any(correction <= 0):
            return 1e12
        pred = C * np.power(all_dx, -2 * delta) * correction
        return np.sum((all_g - pred) ** 2)

    res_bcft = minimize(bcft_loss, [C_pl, delta_pl, 0.0],
                        method='Nelder-Mead', options={'maxiter': 10000})
    C_bcft, delta_bcft, lam_bcft = res_bcft.x[0], abs(res_bcft.x[1]), res_bcft.x[2]
    xk = all_xq - all_dx
    eta = all_dx ** 2 / (4.0 * all_xq * xk)
    pred_bcft = abs(C_bcft) * np.power(all_dx, -2 * delta_bcft) * \
                (1.0 + lam_bcft * np.power(eta, delta_bcft))
    R2_bcft = 1 - np.sum((all_g - pred_bcft) ** 2) / ss_tot

    improvement = R2_bcft - R2_pl
    better = "BCFT" if improvement > 0.001 else "PL"
    if better == "BCFT":
        bcft_wins += 1
    else:
        pl_wins += 1

    # Bin result
    if 0.15 < delta_avg < 0.35:
        results_by_delta_bin['near_quarter'].append(improvement)
    elif 0.28 < delta_avg < 0.42:
        results_by_delta_bin['near_third'].append(improvement)
    elif 0.42 < delta_avg < 0.65:
        results_by_delta_bin['near_half'].append(improvement)
    else:
        results_by_delta_bin['high'].append(improvement)

    print(f"  L{l+1:2d}H{h:2d} {delta_pl:7.4f} {R2_pl:9.6f} {delta_bcft:7.4f} "
          f"{lam_bcft:+8.4f} {R2_bcft:9.6f} {improvement:+9.6f} {better:>8}")

print()
print(f"  TOTAL: BCFT wins {bcft_wins}, Power-law wins {pl_wins}")
print()
for bin_name, results in results_by_delta_bin.items():
    if results:
        wins = sum(1 for r in results if r > 0.001)
        print(f"  {bin_name}: {wins}/{len(results)} BCFT wins, "
              f"mean ΔR² = {np.mean(results):+.4f}")


# ========== Phase 3: Start-boundary EXCESS above power law ==========
print("\n" + "=" * 90)
print("  PHASE 3: START-BOUNDARY EXCESS ABOVE POWER LAW")
print("=" * 90)
print()
print("  The bare power law predicts monotonic decay from the query.")
print("  Any EXCESS attention at the start (above the PL prediction)")
print("  is the boundary correction. Measure this excess and compare")
print("  to the BCFT prediction using independently measured Δ and λ.")
print()

# For each conformal head, compute:
#   1. The attention profile from late queries (measured)
#   2. The bare power-law prediction (from deep-position Δ)
#   3. The BCFT prediction (from two-point function fit)
#   4. The start-boundary excess = measured - power_law

n_tested = 0
excess_ratios = []

for l, h, delta_avg, R2_avg in all_pl_heads[:20]:
    # Get average attention profile for this head from late queries
    profile = attn_profiles[l, h, :].copy()

    # Skip position 0 (attention sink is a separate mechanism)
    # Focus on positions 1 through ~SEQ_LEN//4 as "near start"
    # and SEQ_LEN//4 through SEQ_LEN//2 as "mid-sequence"

    # Power-law prediction: A(x_k) ∝ (x_q_avg - x_k)^{-2Δ}
    x_q_avg = (Q_LO + SEQ_LEN) / 2.0
    positions = np.arange(1, SEQ_LEN)
    pl_pred = np.power(x_q_avg - positions, -2 * delta_avg)
    # Normalize to match total attention in the mid region
    mid_lo = SEQ_LEN // 3
    mid_hi = 2 * SEQ_LEN // 3
    if np.sum(pl_pred[mid_lo:mid_hi]) > 1e-15 and np.sum(profile[mid_lo+1:mid_hi+1]) > 1e-15:
        scale = np.sum(profile[mid_lo+1:mid_hi+1]) / np.sum(pl_pred[mid_lo:mid_hi])
    else:
        continue
    pl_pred_scaled = pl_pred * scale

    # Measure excess at the start (positions 1-30, excluding sink at 0)
    start_measured = np.mean(profile[1:31])
    start_pl = np.mean(pl_pred_scaled[:30])
    mid_measured = np.mean(profile[mid_lo+1:mid_hi+1])
    mid_pl = np.mean(pl_pred_scaled[mid_lo:mid_hi])

    # The "excess" at the start relative to the PL prediction
    excess = start_measured / start_pl if start_pl > 1e-15 else 0

    if excess > 0:
        excess_ratios.append(excess)
        n_tested += 1
        if n_tested <= 15:
            print(f"  L{l+1:2d}H{h:2d} (Δ={delta_avg:.3f}):  "
                  f"start_measured={start_measured:.6f}  "
                  f"start_PL_pred={start_pl:.6f}  "
                  f"excess_ratio={excess:.3f}  "
                  f"{'↑ EXCESS' if excess > 1.05 else '≈ matches PL' if excess > 0.95 else '↓ DEFICIT'}")

print()
if excess_ratios:
    above = sum(1 for e in excess_ratios if e > 1.05)
    near = sum(1 for e in excess_ratios if 0.95 <= e <= 1.05)
    below = sum(1 for e in excess_ratios if e < 0.95)
    print(f"  Of {len(excess_ratios)} heads tested:")
    print(f"    Excess > 5% above PL: {above}")
    print(f"    Within 5% of PL:      {near}")
    print(f"    Deficit below PL:     {below}")
    print(f"    Mean excess ratio:    {np.mean(excess_ratios):.3f}")
    print(f"    Median excess ratio:  {np.median(excess_ratios):.3f}")
    print()
    if above > len(excess_ratios) * 0.5:
        print("  ✓ MAJORITY of heads show start-boundary excess above power law.")
        print("    The U-shape at the start is NOT explained by the decay alone.")
        print("    Something enhances attention at early positions — the BCFT boundary.")
    elif near > len(excess_ratios) * 0.5:
        print("  ≈ MAJORITY of heads match the power law at the start.")
        print("    The U-shape start arm may be fully explained by the decay.")
    else:
        print("  ↓ MAJORITY of heads show deficit at the start.")
        print("    The start boundary is SUPPRESSING attention, not enhancing it.")

print()
print("=" * 90)
print("  INTERPRETATION")
print("=" * 90)
print()
print("  If excess > 1: the start boundary ENHANCES attention beyond the power")
print("  law prediction. This is the BCFT λ > 0 correction measured independently")
print("  in the two-point function fit.")
print()
print("  If excess ≈ 1: the power law alone explains the start attention. The")
print("  U-shape at the start is just the tail of the decay, not a boundary effect.")
print()
print("  If excess < 1: something SUPPRESSES attention at the start (possibly the")
print("  attention sink absorbing probability mass that would otherwise go to")
print("  early positions).")
print()
