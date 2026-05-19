"""
BCFT Boundary Test — Position-Dependent Corrections in Causal Attention

The periodic CFT test (cft_symmetry_test.py) showed the causal mask is a boundary.
The correct framework is BCFT. In BCFT on a half-line with boundary at x=0:

    G(x_q, x_k) = |x_q - x_k|^{-2Δ} · f(η)
    η = (x_q - x_k)² / (4·x_q·x_k)

Predictions:
  1. The correlator depends on query position x_q (not just separation Δx)
  2. Near the boundary: deviations from pure power law
  3. Data collapses when plotted against cross-ratio η
  4. BCFT form fits better than bare power law

Ariel — April 15, 2026
"""

import torch
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
np.random.seed(42)

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

n_layers = model.config.n_layer   # 12
n_heads = model.config.n_head     # 12
vocab_size = model.config.vocab_size

SEQ_LEN = 256
N_INPUTS = 60
MAX_DX = 120
FIT_LOW = 3
FIT_HIGH = 50

# G_diag[l][h][dx] stores the sum of attention at separation dx,
# resolved by query position: G_pos[l][h] is (SEQ_LEN, MAX_DX)
print(f"\nCollecting position-resolved attention from {N_INPUTS} inputs (seq_len={SEQ_LEN})...")

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
        attn = outputs.attentions[l][0].numpy()  # (n_heads, SEQ_LEN, SEQ_LEN)
        for h in range(n_heads):
            head = attn[h]
            for dx in range(MAX_DX):
                diag = np.diagonal(head, offset=-dx)
                G_pos[l][h][dx:dx + len(diag), dx] += diag
    if (inp_idx + 1) % 15 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

# Average over inputs
for l in range(n_layers):
    for h in range(n_heads):
        G_pos[l][h] /= N_INPUTS

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ========== Phase 1: Identify conformal heads ==========
print("\n" + "=" * 90)
print("  PHASE 1: IDENTIFY CONFORMAL HEADS (using deep-sequence positions)")
print("=" * 90 + "\n")

conformal_heads = []
DEEP_LO = SEQ_LEN // 2

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
print(f"  Found {len(conformal_heads)} conformal heads")
for l, h, delta, R2 in conformal_heads[:15]:
    print(f"    L{l+1:2d}H{h:2d}: Δ = {delta:.4f}, R² = {R2:.4f}")


# ========== Phase 2: Position-dependent power law ==========
print("\n" + "=" * 90)
print("  PHASE 2: POSITION-DEPENDENT POWER LAW")
print("=" * 90)
print()
print("  Fit Δx^{-2Δ} separately for near/mid/deep query positions.")
print("  BCFT predicts: same Δ but different amplitude C(x_q).")
print()

BINS = [
    ("near", 16, 48),
    ("mid", 80, 128),
    ("deep", 160, 240),
]

for l, h, delta_avg, R2_avg in conformal_heads[:10]:
    print(f"  --- L{l+1}H{h} (Δ_avg = {delta_avg:.4f}) ---")

    for bin_name, xq_lo, xq_hi in BINS:
        A = np.mean(G_pos[l][h][xq_lo:xq_hi, :FIT_HIGH], axis=0)

        dx_arr = np.arange(FIT_LOW, FIT_HIGH)
        y = A[FIT_LOW:FIT_HIGH]
        valid = y > 1e-15
        if np.sum(valid) < 8:
            print(f"    {bin_name:5s} xq=[{xq_lo},{xq_hi}]: insufficient data")
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
        C = np.exp(coeffs[0])

        print(f"    {bin_name:5s} xq=[{xq_lo},{xq_hi}]: "
              f"Δ = {delta:.4f}, C = {C:.6f}, R² = {R2:.4f}")
    print()


# ========== Phase 3: BCFT cross-ratio collapse ==========
print("=" * 90)
print("  PHASE 3: BCFT CROSS-RATIO ANALYSIS")
print("=" * 90)
print()
print("  η = Δx²/(4·x_q·x_k).  G_scaled = G · Δx^{2Δ}.")
print("  If BCFT: G_scaled depends on η.  If bare PL: G_scaled constant.")
print()


for l, h, delta_avg, R2_avg in conformal_heads[:10]:
    # Collect (η, G_scaled) pairs from position-resolved data
    etas = []
    g_scaled = []

    for x_q in range(32, SEQ_LEN):
        for dx in range(FIT_LOW, min(FIT_HIGH, x_q)):
            x_k = x_q - dx
            if x_k <= 0:
                continue
            g_val = G_pos[l][h][x_q, dx]
            if g_val < 1e-15:
                continue
            eta = dx ** 2 / (4.0 * x_q * x_k)
            scaled = g_val * (dx ** (2 * delta_avg))
            etas.append(eta)
            g_scaled.append(scaled)

    etas = np.array(etas)
    g_scaled = np.array(g_scaled)

    if len(etas) < 50:
        print(f"  L{l+1}H{h}: insufficient data ({len(etas)} points)")
        continue

    rho_eta, p_eta = spearmanr(etas, g_scaled)
    print(f"  L{l+1:2d}H{h:2d}: Spearman(G_scaled, η) = {rho_eta:+.4f}  "
          f"(p = {p_eta:.2e})", end="")

    # Bin by η for visual summary
    eta_sorted_idx = np.argsort(etas)
    n_bins = 6
    bin_size = len(eta_sorted_idx) // n_bins
    lo_val = np.mean(g_scaled[eta_sorted_idx[:bin_size]])
    hi_val = np.mean(g_scaled[eta_sorted_idx[-(bin_size):]])
    ratio = hi_val / lo_val if lo_val > 1e-15 else float('inf')
    print(f"  | G_scaled ratio (high η / low η) = {ratio:.4f}")


# ========== Phase 4: BCFT vs PL fit on full 2D data ==========
print("\n" + "=" * 90)
print("  PHASE 4: BCFT TWO-POINT FUNCTION vs BARE POWER LAW")
print("=" * 90)
print()
print("  G_BCFT = C · Δx^{-2Δ} · (1 + λ · η^Δ)")
print("  G_PL   = C · Δx^{-2Δ}")
print(f"  {'Head':>8} {'Δ_PL':>7} {'R²_PL':>9} {'Δ_BCFT':>7} {'λ':>8} "
      f"{'R²_BCFT':>9} {'ΔR²':>9} {'Winner':>8}")
print("  " + "-" * 75)

pl_wins = 0
bcft_wins = 0
all_improvements = []

for l, h, delta_avg, R2_avg in conformal_heads[:15]:
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

    # Fit bare power law
    def pl_loss(params):
        C, delta = params
        if C <= 0 or delta <= 0:
            return 1e12
        pred = C * np.power(all_dx, -2 * delta)
        return np.sum((all_g - pred) ** 2)

    C_init = np.mean(all_g * np.power(all_dx, 2 * delta_avg))
    res_pl = minimize(pl_loss, [C_init, delta_avg],
                      method='Nelder-Mead', options={'maxiter': 5000})
    C_pl, delta_pl = abs(res_pl.x[0]), abs(res_pl.x[1])
    pred_pl = C_pl * np.power(all_dx, -2 * delta_pl)
    R2_pl = 1 - np.sum((all_g - pred_pl) ** 2) / ss_tot

    # Fit BCFT: G = C · Δx^{-2Δ} · (1 + λ·η^Δ)
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
    all_improvements.append(improvement)
    better = "BCFT" if improvement > 0.0001 else "PL"
    if better == "BCFT":
        bcft_wins += 1
    else:
        pl_wins += 1

    print(f"  L{l+1:2d}H{h:2d} {delta_pl:7.4f} {R2_pl:9.6f} {delta_bcft:7.4f} "
          f"{lam_bcft:+8.4f} {R2_bcft:9.6f} {improvement:+9.6f} {better:>8}")


print()
print(f"  BCFT wins: {bcft_wins}   Power-law wins: {pl_wins}")
if all_improvements:
    print(f"  Mean ΔR²: {np.mean(all_improvements):+.6f}")
    print(f"  Median ΔR²: {np.median(all_improvements):+.6f}")

print()
print("=" * 90)
print("  INTERPRETATION")
print("=" * 90)
print()
print("  Key evidence for BCFT:")
print("    1. Spearman(G_scaled, η) significantly positive → boundary correction real")
print("    2. BCFT R² > PL R² → boundary form fits better")
print("    3. λ > 0 consistently → boundary enhances short-distance attention")
print("    4. Position bins show same Δ but different C → position-dependent amplitude")
print()
print("  If no BCFT signal:")
print("    → Boundary effects may be too weak at L=256")
print("    → Or the correction form f(η) = 1 + λ·η^Δ is too simple")
print("    → The Schwarzian boundary might manifest differently than standard BCFT")
print()
