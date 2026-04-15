"""
BCFT Lost-in-the-Middle: Training Completeness Prediction

The central testable claim: the lost-in-the-middle effect is controlled
by Δ, and Δ decreases with training. Therefore continued training should
reduce the U-shape, and the reduction should follow the BCFT prediction.

This script loads Pythia-70m at multiple training checkpoints, measures:
  1. Δ (conformal scaling exponent from deep-position power law)
  2. U-shape amplitude (attention profile: start/middle/end ratios)

Then checks whether they correlate as the BCFT predicts:
  Higher Δ → deeper valley → more lost-in-the-middle

Parameters are measured from random-token attention (no task data),
the same method used for the GPT-2 BCFT fits.

Ariel — April 15, 2026
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from transformers import GPTNeoXForCausalLM, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 256
N_INPUTS = 40
FIT_LOW = 3
FIT_HIGH = 50
DEEP_LO_FRAC = 0.5

# Pythia-70m checkpoints span training from step 1 to step 143000
# These are the ones available on HuggingFace
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 4000, 8000, 16000, 32000,
               64000, 143000]

MODEL_NAME = "EleutherAI/pythia-70m"

def measure_checkpoint(step):
    """Load one Pythia checkpoint, measure Δ and U-shape."""
    revision = f"step{step}"
    print(f"\n  Loading {MODEL_NAME} at {revision}...")

    try:
        model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_NAME, revision=revision, torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"  Failed to load {revision}: {e}")
        return None

    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    vocab_size = model.config.vocab_size

    # Collect averaged attention
    G_avg = {}
    for l in range(n_layers):
        G_avg[l] = np.zeros((n_heads, FIT_HIGH))
        G_count = np.zeros(FIT_HIGH)

    attn_profiles = np.zeros((n_layers, n_heads, SEQ_LEN))
    profile_counts = np.zeros((n_layers, n_heads, SEQ_LEN))
    Q_LO = 3 * SEQ_LEN // 4

    for inp_idx in range(N_INPUTS):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].numpy()  # (n_heads, seq, seq)
            for h in range(n_heads):
                head = attn[h]
                deep_lo = SEQ_LEN // 2
                for x_q in range(deep_lo, SEQ_LEN):
                    for dx in range(FIT_LOW, min(FIT_HIGH, x_q)):
                        x_k = x_q - dx
                        if x_k < 0:
                            continue
                        G_avg[l][h, dx] += head[x_q, x_k]

                for x_q in range(Q_LO, SEQ_LEN):
                    attn_profiles[l, h, :x_q+1] += head[x_q, :x_q+1]
                    profile_counts[l, h, :x_q+1] += 1

    # Normalize
    n_deep_queries = SEQ_LEN // 2
    deep_pairs_per_dx = N_INPUTS * n_deep_queries
    for l in range(n_layers):
        G_avg[l] /= max(deep_pairs_per_dx, 1)

    valid_mask = profile_counts > 0
    attn_profiles[valid_mask] /= profile_counts[valid_mask]

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Measure Δ per head
    all_deltas = []
    for l in range(n_layers):
        for h in range(n_heads):
            y = G_avg[l][h, FIT_LOW:FIT_HIGH]
            valid = y > 1e-15
            if np.sum(valid) < 10:
                continue
            dx_arr = np.arange(FIT_LOW, FIT_HIGH)
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

            if R2 > 0.85 and delta > 0.05:
                all_deltas.append(delta)

    # Measure U-shape across all layers and heads
    all_valley_depths = []
    all_u_amplitudes = []
    all_asymmetries = []

    for l in range(n_layers):
        for h in range(n_heads):
            profile = attn_profiles[l, h, :]
            third = SEQ_LEN // 3
            start_attn = np.mean(profile[1:third])
            middle_attn = np.mean(profile[third:2*third])
            end_attn = np.mean(profile[2*third:Q_LO])

            if middle_attn < 1e-15 or end_attn < 1e-15:
                continue

            peak = max(start_attn, end_attn)
            valley_depth = 1 - middle_attn / peak if peak > 1e-15 else 0
            u_amplitude = (start_attn + end_attn) / (2 * middle_attn)
            asymmetry = start_attn / end_attn

            all_valley_depths.append(valley_depth)
            all_u_amplitudes.append(u_amplitude)
            all_asymmetries.append(asymmetry)

    if not all_deltas or not all_valley_depths:
        print(f"  Insufficient data at step {step}")
        return None

    return {
        'step': step,
        'median_delta': np.median(all_deltas),
        'mean_delta': np.mean(all_deltas),
        'n_conformal_heads': len(all_deltas),
        'median_valley_depth': np.median(all_valley_depths),
        'mean_valley_depth': np.mean(all_valley_depths),
        'median_u_amplitude': np.median(all_u_amplitudes),
        'mean_u_amplitude': np.mean(all_u_amplitudes),
        'median_asymmetry': np.median(all_asymmetries),
    }


# ========== Run measurements ==========
print("=" * 90)
print("  BCFT TRAINING COMPLETENESS PREDICTION — Pythia-70m")
print("=" * 90)
print()
print("  Prediction: as training progresses, Δ → 1/4 and the")
print("  lost-in-the-middle valley gets shallower. Both should")
print("  track each other because the BCFT formula connects them.")
print()

results = []
for step in CHECKPOINTS:
    r = measure_checkpoint(step)
    if r is not None:
        results.append(r)
        print(f"  step {step:>6d}: Δ = {r['median_delta']:.4f}  "
              f"valley = {r['median_valley_depth']:.4f}  "
              f"U-amp = {r['median_u_amplitude']:.4f}  "
              f"asym = {r['median_asymmetry']:.4f}  "
              f"({r['n_conformal_heads']} conformal heads)")


# ========== Analysis ==========
if len(results) >= 4:
    print("\n\n" + "=" * 90)
    print("  RESULTS")
    print("=" * 90)
    print()

    print(f"  {'Step':>8} {'Δ':>8} {'Valley':>8} {'U-amp':>8} {'Asym':>8} {'N heads':>8}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['step']:>8d} {r['median_delta']:>8.4f} "
              f"{r['median_valley_depth']:>8.4f} "
              f"{r['median_u_amplitude']:>8.4f} "
              f"{r['median_asymmetry']:>8.4f} "
              f"{r['n_conformal_heads']:>8d}")

    # Correlation: Δ vs valley depth
    steps = np.array([r['step'] for r in results])
    deltas = np.array([r['median_delta'] for r in results])
    valleys = np.array([r['median_valley_depth'] for r in results])
    u_amps = np.array([r['median_u_amplitude'] for r in results])

    rho_dv, p_dv = spearmanr(deltas, valleys)
    rho_du, p_du = spearmanr(deltas, u_amps)
    rho_sv, p_sv = spearmanr(steps, valleys)
    rho_sd, p_sd = spearmanr(steps, deltas)

    print()
    print("  CORRELATIONS:")
    print(f"    Spearman(Δ, valley_depth) = {rho_dv:+.4f}  (p = {p_dv:.4f})")
    print(f"    Spearman(Δ, U-amplitude)  = {rho_du:+.4f}  (p = {p_du:.4f})")
    print(f"    Spearman(step, Δ)         = {rho_sd:+.4f}  (p = {p_sd:.4f})")
    print(f"    Spearman(step, valley)    = {rho_sv:+.4f}  (p = {p_sv:.4f})")
    print()

    # BCFT predicted valley depth from measured Δ
    LAMBDA = 2.0
    L = SEQ_LEN
    print("  BCFT PREDICTED vs MEASURED valley depth:")
    print(f"  {'Step':>8} {'Δ':>8} {'Measured':>10} {'Predicted':>10} {'Ratio':>8}")
    print(f"  {'-'*50}")
    for r in results:
        delta = r['median_delta']
        positions = np.arange(1, L)
        x_q = L
        attn = np.array([
            (x_q - p) ** (-2 * delta) * (1.0 + LAMBDA * ((x_q - p)**2 / (4.0 * x_q * p)) ** delta)
            for p in positions
        ])
        attn_norm = attn / attn.sum()
        fifth = max(len(positions) // 5, 1)
        start = attn_norm[:fifth].mean()
        mid = attn_norm[2*fifth:3*fifth].mean()
        end = attn_norm[-fifth:].mean()
        peak = max(start, end)
        pred_valley = 1 - mid / peak if peak > 1e-15 else 0

        ratio = r['median_valley_depth'] / pred_valley if pred_valley > 0 else 0
        print(f"  {r['step']:>8d} {delta:>8.4f} {r['median_valley_depth']:>10.4f} "
              f"{pred_valley:>10.4f} {ratio:>8.3f}")

    print()
    print("  PREDICTION ASSESSMENT:")
    if rho_dv > 0.5 and p_dv < 0.05:
        print("  ✓ CONFIRMED: Higher Δ correlates with deeper valley.")
        print("    The lost-in-the-middle effect is a training completeness issue.")
    elif rho_dv > 0 and p_dv < 0.1:
        print("  ~ SUGGESTIVE: Positive correlation but not fully significant.")
    elif abs(rho_dv) < 0.3:
        print("  ? INCONCLUSIVE: No clear correlation between Δ and valley depth.")
    else:
        print("  ✗ CONTRADICTED: Δ and valley depth do not correlate as predicted.")

    if rho_sd < -0.5 and p_sd < 0.05:
        print("  ✓ Δ decreases with training (as previously measured).")
    if rho_sv < -0.5 and p_sv < 0.05:
        print("  ✓ Valley depth decreases with training (novel confirmation).")
        print("    This is the actionable finding: more training reduces LiTM.")

    # Does the reduction track the SAME trajectory?
    if rho_dv > 0.5 and rho_sd < -0.3 and rho_sv < -0.3:
        print()
        print("  ★ THE TRAINING COMPLETENESS PREDICTION HOLDS:")
        print("    Δ decreases with training.")
        print("    Valley depth decreases with training.")
        print("    Δ and valley depth are positively correlated.")
        print("    The BCFT connects all three: the physics that determines")
        print("    conformal scaling ALSO determines the U-shape of attention,")
        print("    and both improve together as training approaches the fixed point.")

else:
    print(f"\n  Only {len(results)} valid checkpoints — insufficient for correlation.")
