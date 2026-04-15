"""
BCFT Lost-in-the-Middle: Training Completeness Test v2

Fixes from v1:
  - Match Δ and U-shape to the SAME heads (per-head correlation)
  - Increase inputs from 40 to 80 for cleaner statistics
  - Report per-head scatter, not just aggregates
  - Test both within-checkpoint and across-checkpoint correlations

Ariel — April 15, 2026
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from transformers import GPTNeoXForCausalLM

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 80
FIT_LOW = 3
FIT_HIGH = 50

# Fewer checkpoints, focus on the interesting range
CHECKPOINTS = [256, 512, 1000, 2000, 4000, 8000,
               16000, 32000, 64000, 143000]

MODEL_NAME = "EleutherAI/pythia-70m"


def measure_checkpoint(step):
    """Measure per-head Δ and U-shape from the same data."""
    revision = f"step{step}"
    print(f"\n  Loading {MODEL_NAME} at {revision}...")

    try:
        model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_NAME, revision=revision, torch_dtype=torch.float32,
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"  Failed: {e}")
        return None

    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    vocab_size = model.config.vocab_size

    # Full attention matrix accumulator
    attn_sum = np.zeros((n_layers, n_heads, SEQ_LEN, SEQ_LEN))

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            attn_sum[l] += outputs.attentions[l][0].numpy()
        if (inp_idx + 1) % 20 == 0:
            print(f"    {inp_idx + 1}/{N_INPUTS}")

    attn_avg = attn_sum / N_INPUTS
    del model, attn_sum
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    Q_LO = 3 * SEQ_LEN // 4
    DEEP_LO = SEQ_LEN // 2
    head_results = []

    for l in range(n_layers):
        for h in range(n_heads):
            # === Measure Δ from deep positions ===
            A_dx = np.zeros(FIT_HIGH)
            count = np.zeros(FIT_HIGH)
            for x_q in range(DEEP_LO, SEQ_LEN):
                for dx in range(FIT_HIGH):
                    x_k = x_q - dx
                    if x_k < 0:
                        continue
                    A_dx[dx] += attn_avg[l, h, x_q, x_k]
                    count[dx] += 1
            valid_c = count > 0
            A_dx[valid_c] /= count[valid_c]

            dx_arr = np.arange(FIT_LOW, FIT_HIGH)
            y = A_dx[FIT_LOW:FIT_HIGH]
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

            if R2 < 0.85 or delta < 0.05:
                continue

            # === Measure U-shape for THIS head ===
            profile = np.zeros(SEQ_LEN)
            p_count = 0
            for x_q in range(Q_LO, SEQ_LEN):
                profile[:x_q+1] += attn_avg[l, h, x_q, :x_q+1]
                p_count += 1
            if p_count > 0:
                profile /= p_count

            third = SEQ_LEN // 3
            start_attn = np.mean(profile[1:third])
            middle_attn = np.mean(profile[third:2*third])
            end_attn = np.mean(profile[2*third:Q_LO])

            if middle_attn < 1e-15 or end_attn < 1e-15 or start_attn < 1e-15:
                continue

            peak = max(start_attn, end_attn)
            valley_depth = 1 - middle_attn / peak
            u_amplitude = (start_attn + end_attn) / (2 * middle_attn)

            # Valley position: where is minimum in the profile?
            # Look only at interior positions (skip first and last 10%)
            interior_lo = max(1, SEQ_LEN // 10)
            interior_hi = Q_LO - SEQ_LEN // 10
            if interior_hi > interior_lo:
                valley_pos = interior_lo + np.argmin(profile[interior_lo:interior_hi])
                valley_frac = valley_pos / Q_LO
            else:
                valley_frac = 0.5

            head_results.append({
                'layer': l, 'head': h,
                'delta': delta, 'R2': R2,
                'valley_depth': valley_depth,
                'u_amplitude': u_amplitude,
                'start': start_attn, 'middle': middle_attn, 'end': end_attn,
                'valley_frac': valley_frac,
                'asymmetry': start_attn / end_attn,
            })

    return head_results


# ========== Run ==========
print("=" * 90)
print("  BCFT TRAINING COMPLETENESS — Per-Head Matched Analysis")
print("=" * 90)

all_results = {}
for step in CHECKPOINTS:
    heads = measure_checkpoint(step)
    if heads:
        all_results[step] = heads
        deltas = [h['delta'] for h in heads]
        valleys = [h['valley_depth'] for h in heads]
        vfracs = [h['valley_frac'] for h in heads]
        print(f"  step {step:>6d}: {len(heads)} conformal heads, "
              f"median Δ={np.median(deltas):.4f}, "
              f"median valley={np.median(valleys):.4f}, "
              f"median valley_frac={np.median(vfracs):.3f}")
    else:
        print(f"  step {step:>6d}: no data")


# ========== Within-checkpoint analysis ==========
print("\n\n" + "=" * 90)
print("  WITHIN-CHECKPOINT: Δ vs VALLEY DEPTH (per head)")
print("=" * 90)
print()
print("  BCFT predicts: across heads in the SAME model, higher Δ → deeper valley")
print()

for step in sorted(all_results.keys()):
    heads = all_results[step]
    if len(heads) < 5:
        continue
    deltas = np.array([h['delta'] for h in heads])
    valleys = np.array([h['valley_depth'] for h in heads])

    rho, p = spearmanr(deltas, valleys)
    print(f"  step {step:>6d} ({len(heads):2d} heads): "
          f"Spearman(Δ, valley) = {rho:+.3f} (p={p:.3f})  "
          f"{'✓' if rho > 0.3 and p < 0.1 else '?' if rho > 0 else '✗'}")


# ========== Across-checkpoint analysis (matched) ==========
print("\n\n" + "=" * 90)
print("  ACROSS CHECKPOINTS: AGGREGATE TRAJECTORIES")
print("=" * 90)
print()

steps_arr = []
delta_medians = []
valley_medians = []
u_medians = []
vfrac_medians = []
asym_medians = []

print(f"  {'Step':>8} {'N':>4} {'Δ_med':>8} {'Δ_iqr':>10} {'Valley':>8} "
      f"{'U-amp':>8} {'V_frac':>8} {'Asym':>8}")
print(f"  {'-'*72}")

for step in sorted(all_results.keys()):
    heads = all_results[step]
    if len(heads) < 3:
        continue
    deltas = [h['delta'] for h in heads]
    valleys = [h['valley_depth'] for h in heads]
    u_amps = [h['u_amplitude'] for h in heads]
    vfracs = [h['valley_frac'] for h in heads]
    asyms = [h['asymmetry'] for h in heads]

    d_med = np.median(deltas)
    d_iqr = np.percentile(deltas, 75) - np.percentile(deltas, 25)
    v_med = np.median(valleys)
    u_med = np.median(u_amps)
    vf_med = np.median(vfracs)
    a_med = np.median(asyms)

    steps_arr.append(step)
    delta_medians.append(d_med)
    valley_medians.append(v_med)
    u_medians.append(u_med)
    vfrac_medians.append(vf_med)
    asym_medians.append(a_med)

    print(f"  {step:>8d} {len(heads):>4d} {d_med:>8.4f} {d_iqr:>10.4f} "
          f"{v_med:>8.4f} {u_med:>8.4f} {vf_med:>8.3f} {a_med:>8.4f}")

steps_arr = np.array(steps_arr)
delta_medians = np.array(delta_medians)
valley_medians = np.array(valley_medians)
u_medians = np.array(u_medians)
vfrac_medians = np.array(vfrac_medians)

if len(steps_arr) >= 5:
    print()
    print("  CORRELATIONS:")
    for name, arr in [('valley_depth', valley_medians),
                      ('U-amplitude', u_medians),
                      ('valley_frac', vfrac_medians)]:
        rho_d, p_d = spearmanr(delta_medians, arr)
        rho_s, p_s = spearmanr(steps_arr, arr)
        rho_sd, p_sd = spearmanr(steps_arr, delta_medians)
        print(f"    {name:>15}: ρ(Δ,{name[:6]}) = {rho_d:+.3f} (p={p_d:.3f}), "
              f"ρ(step,{name[:6]}) = {rho_s:+.3f} (p={p_s:.3f})")
    rho_sd, p_sd = spearmanr(steps_arr, delta_medians)
    print(f"    {'Δ vs step':>15}: ρ(step, Δ) = {rho_sd:+.3f} (p={p_sd:.3f})")


# ========== PREDICTION 1 CHECK: Valley position ==========
print("\n\n" + "=" * 90)
print("  PREDICTION 1: VALLEY POSITION")
print("=" * 90)
print()
print("  BCFT predicts valley at ~40% of context, not 50%")
print()

all_vfracs = []
for step in sorted(all_results.keys()):
    for h in all_results[step]:
        all_vfracs.append(h['valley_frac'])

if all_vfracs:
    vf = np.array(all_vfracs)
    print(f"  Across all checkpoints and heads:")
    print(f"    Mean valley position:   {np.mean(vf):.3f} of context length")
    print(f"    Median valley position: {np.median(vf):.3f} of context length")
    print(f"    Std:                    {np.std(vf):.3f}")
    print(f"    25th-75th percentile:   [{np.percentile(vf, 25):.3f}, {np.percentile(vf, 75):.3f}]")
    print()
    if np.median(vf) < 0.45:
        print(f"  ✓ Valley is BEFORE midpoint (median = {np.median(vf):.3f})")
        if abs(np.median(vf) - 0.40) < 0.05:
            print(f"    Consistent with BCFT prediction of ~0.40")
        else:
            print(f"    But not at 0.40 — at {np.median(vf):.3f}")
    elif np.median(vf) > 0.55:
        print(f"  ✗ Valley is AFTER midpoint (median = {np.median(vf):.3f})")
    else:
        print(f"  ≈ Valley is near midpoint (median = {np.median(vf):.3f})")


# ========== Pooled per-head correlation ==========
print("\n\n" + "=" * 90)
print("  POOLED PER-HEAD: Δ vs VALLEY DEPTH (all checkpoints)")
print("=" * 90)
print()

all_d = []
all_v = []
all_u = []
all_steps_flat = []

for step in sorted(all_results.keys()):
    for h in all_results[step]:
        all_d.append(h['delta'])
        all_v.append(h['valley_depth'])
        all_u.append(h['u_amplitude'])
        all_steps_flat.append(step)

all_d = np.array(all_d)
all_v = np.array(all_v)
all_u = np.array(all_u)

if len(all_d) > 10:
    rho, p = spearmanr(all_d, all_v)
    rho_u, p_u = spearmanr(all_d, all_u)
    print(f"  N = {len(all_d)} head measurements across all checkpoints")
    print(f"  Spearman(Δ, valley_depth) = {rho:+.4f}  (p = {p:.4f})")
    print(f"  Spearman(Δ, U-amplitude)  = {rho_u:+.4f}  (p = {p_u:.4f})")
    print()

    # Bin by Δ ranges
    bins = [(0.05, 0.20, 'Δ < 0.20'),
            (0.20, 0.30, '0.20 ≤ Δ < 0.30'),
            (0.30, 0.40, '0.30 ≤ Δ < 0.40'),
            (0.40, 0.55, '0.40 ≤ Δ < 0.55'),
            (0.55, 2.00, 'Δ ≥ 0.55')]

    print(f"  {'Δ range':>20} {'N':>5} {'Median valley':>14} {'Median U-amp':>14}")
    print(f"  {'-'*58}")
    for lo, hi, label in bins:
        mask = (all_d >= lo) & (all_d < hi)
        if np.sum(mask) >= 2:
            print(f"  {label:>20} {np.sum(mask):>5} "
                  f"{np.median(all_v[mask]):>14.4f} "
                  f"{np.median(all_u[mask]):>14.4f}")

    print()
    if rho > 0.2 and p < 0.05:
        print("  ✓ Per-head: higher Δ → deeper valley (BCFT prediction holds)")
    elif rho > 0 and p < 0.1:
        print("  ~ Per-head: weak positive trend (suggestive)")
    else:
        print(f"  {'✗' if rho < 0 else '?'} Per-head: no significant "
              f"Δ-valley correlation (ρ={rho:+.3f}, p={p:.3f})")


print("\n\n" + "=" * 90)
print("  SUMMARY")
print("=" * 90)
print()
print("  Three predictions tested:")
print()
print("  1. Valley at ~40%L:")
if all_vfracs:
    print(f"     Measured: {np.median(all_vfracs):.3f}")
    print(f"     Predicted: ~0.40")
print()
print("  2. Δ controls valley depth (training completeness):")
if len(all_d) > 10:
    print(f"     Per-head ρ(Δ, valley) = {rho:+.3f} (p={p:.3f})")
    print(f"     Across checkpoints: see table above")
print()
print("  3. Valley decreases with training step:")
if len(steps_arr) >= 5:
    rho_sv, p_sv = spearmanr(steps_arr, valley_medians)
    print(f"     ρ(step, valley) = {rho_sv:+.3f} (p={p_sv:.3f})")
