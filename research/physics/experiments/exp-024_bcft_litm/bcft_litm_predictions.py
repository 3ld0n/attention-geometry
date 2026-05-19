"""
BCFT Predictions for "Lost in the Middle" — Tests 3 and 6

Prediction 3: The U-shape amplitude at each layer should correlate with
the measured Δ at that layer. Higher Δ → sharper boundary enhancement
→ deeper valley in the middle.

Prediction 6: The attention sink at position 0 should be predicted by
the BCFT parameters (Δ, λ) measured independently. Position 0 sits at
maximum boundary enhancement.

Also measures: the U-shape itself from raw attention data, and its
layer-by-layer structure.

Ariel — April 15, 2026
Prompted by Eldon connecting BCFT to the lost-in-the-middle phenomenon.
"""

import torch
import numpy as np
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
FIT_LOW = 3
FIT_HIGH = 50

# ========== Collect full attention matrices ==========
print(f"\nCollecting attention from {N_INPUTS} inputs (seq_len={SEQ_LEN})...")

# Store: per-layer, per-head average attention matrix
# Also store position-resolved two-point function for Δ measurement
attn_avg = np.zeros((n_layers, n_heads, SEQ_LEN, SEQ_LEN))

for inp_idx in range(N_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()  # (n_heads, SEQ_LEN, SEQ_LEN)
        attn_avg[l] += attn
    if (inp_idx + 1) % 15 == 0:
        print(f"  {inp_idx + 1}/{N_INPUTS} done")

attn_avg /= N_INPUTS

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ========== Measure Δ per layer (deep-position power law) ==========
print("\n" + "=" * 90)
print("  MEASURING Δ PER LAYER (deep positions, dx=[3,50])")
print("=" * 90 + "\n")

layer_deltas = {}   # layer -> median Δ across heads
layer_head_deltas = {}  # (layer, head) -> Δ

for l in range(n_layers):
    deltas_this_layer = []
    for h in range(n_heads):
        # Two-point function from deep positions (avoid boundary)
        deep_lo = SEQ_LEN // 2
        A_dx = np.zeros(FIT_HIGH)
        count = np.zeros(FIT_HIGH)
        for x_q in range(deep_lo, SEQ_LEN):
            for dx in range(FIT_HIGH):
                x_k = x_q - dx
                if x_k < 0:
                    continue
                A_dx[dx] += attn_avg[l, h, x_q, x_k]
                count[dx] += 1
        valid_count = count > 0
        A_dx[valid_count] /= count[valid_count]

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

        if R2 > 0.85 and delta > 0:
            deltas_this_layer.append(delta)
            layer_head_deltas[(l, h)] = delta

    if deltas_this_layer:
        layer_deltas[l] = np.median(deltas_this_layer)
        print(f"  Layer {l+1:2d}: median Δ = {layer_deltas[l]:.4f} "
              f"({len(deltas_this_layer)} heads with R² > 0.85)")
    else:
        print(f"  Layer {l+1:2d}: no conformal heads")

# ========== Measure U-shape per layer ==========
print("\n" + "=" * 90)
print("  MEASURING U-SHAPE PER LAYER")
print("=" * 90)
print()
print("  For a fixed query at position x_q = SEQ_LEN-1 (end of sequence),")
print("  measure the total attention received by each key position x_k.")
print("  This is the 'attention profile' that determines lost-in-the-middle.")
print()

# The U-shape: for each key position, how much attention does it get
# from a query near the end of the sequence?
# Use queries from the last quarter to average out noise.
Q_LO = 3 * SEQ_LEN // 4   # queries from positions 192-255

layer_ushapes = {}
layer_ushape_metrics = {}

for l in range(n_layers):
    # Average attention profile across all heads and queries in Q range
    attn_profile = np.zeros(SEQ_LEN)
    count = 0
    for h in range(n_heads):
        for x_q in range(Q_LO, SEQ_LEN):
            attn_profile[:x_q+1] += attn_avg[l, h, x_q, :x_q+1]
            count += 1
    attn_profile /= count

    layer_ushapes[l] = attn_profile

    # Measure U-shape metrics
    # Normalize by position to get relative attention
    # Divide into thirds: start, middle, end
    third = SEQ_LEN // 3
    start_attn = np.mean(attn_profile[1:third])       # skip pos 0 (attention sink)
    middle_attn = np.mean(attn_profile[third:2*third])
    end_attn = np.mean(attn_profile[2*third:Q_LO])

    # U-shape amplitude: (start + end) / (2 * middle)
    u_amplitude = (start_attn + end_attn) / (2 * middle_attn) if middle_attn > 1e-15 else 0
    # Valley depth: 1 - middle / max(start, end)
    peak = max(start_attn, end_attn)
    valley_depth = 1 - middle_attn / peak if peak > 1e-15 else 0
    # Asymmetry: start/end ratio
    asymmetry = start_attn / end_attn if end_attn > 1e-15 else 0

    # Attention sink: attention at position 0
    sink_attn = attn_profile[0]

    layer_ushape_metrics[l] = {
        'u_amplitude': u_amplitude,
        'valley_depth': valley_depth,
        'asymmetry': asymmetry,
        'start_attn': start_attn,
        'middle_attn': middle_attn,
        'end_attn': end_attn,
        'sink_attn': sink_attn,
    }

    print(f"  Layer {l+1:2d}: U-amp = {u_amplitude:.4f}  "
          f"valley = {valley_depth:.4f}  "
          f"asym(start/end) = {asymmetry:.4f}  "
          f"sink = {sink_attn:.6f}  "
          f"start/mid/end = {start_attn:.6f}/{middle_attn:.6f}/{end_attn:.6f}")


# ========== PREDICTION 3: U-shape amplitude vs Δ ==========
print("\n" + "=" * 90)
print("  PREDICTION 3: U-SHAPE AMPLITUDE vs Δ PER LAYER")
print("=" * 90)
print()
print("  BCFT predicts: higher Δ → sharper boundary enhancement → deeper valley")
print("  → positive correlation between Δ and U-shape amplitude")
print()

layers_both = sorted(set(layer_deltas.keys()) & set(layer_ushape_metrics.keys()))
if len(layers_both) >= 4:
    deltas_arr = np.array([layer_deltas[l] for l in layers_both])
    u_amps = np.array([layer_ushape_metrics[l]['u_amplitude'] for l in layers_both])
    valleys = np.array([layer_ushape_metrics[l]['valley_depth'] for l in layers_both])

    rho_amp, p_amp = spearmanr(deltas_arr, u_amps)
    rho_valley, p_valley = spearmanr(deltas_arr, valleys)

    print(f"  Layers with both Δ and U-shape: {len(layers_both)}")
    print()
    print(f"  {'Layer':>7} {'Δ':>8} {'U-amp':>8} {'Valley':>8}")
    print(f"  {'-'*35}")
    for l in layers_both:
        print(f"  {l+1:>7d} {layer_deltas[l]:>8.4f} "
              f"{layer_ushape_metrics[l]['u_amplitude']:>8.4f} "
              f"{layer_ushape_metrics[l]['valley_depth']:>8.4f}")

    print()
    print(f"  Spearman(Δ, U-amplitude) = {rho_amp:+.4f}  (p = {p_amp:.4f})")
    print(f"  Spearman(Δ, valley_depth) = {rho_valley:+.4f}  (p = {p_valley:.4f})")
    print()
    if rho_amp > 0 and p_amp < 0.1:
        print("  ✓ CONFIRMED: higher Δ → stronger U-shape (deeper lost-in-the-middle)")
    elif rho_amp < 0 and p_amp < 0.1:
        print("  ✗ REVERSED: higher Δ → weaker U-shape (opposite of prediction)")
    else:
        print("  ? INCONCLUSIVE: no significant correlation")
else:
    print(f"  Insufficient data: only {len(layers_both)} layers with both measurements")


# ========== PREDICTION 6: Attention sink from BCFT parameters ==========
print("\n" + "=" * 90)
print("  PREDICTION 6: ATTENTION SINK FROM BCFT PARAMETERS")
print("=" * 90)
print()
print("  Position 0 sits at the boundary. BCFT predicts it receives maximum")
print("  boundary enhancement. The attention sink should scale with x_q^{-2Δ}·(1+λ)")
print()

# Measure: how attention at position 0 varies with query position
print(f"  {'Layer':>7} {'Δ':>8} {'sink(x_q=64)':>13} {'sink(x_q=128)':>14} "
      f"{'sink(x_q=192)':>14} {'ratio 192/64':>13}")
print(f"  {'-'*75}")

for l in range(n_layers):
    sinks = []
    for x_q_test in [64, 128, 192]:
        # Average across heads
        sink_val = np.mean(attn_avg[l, :, x_q_test, 0])
        sinks.append(sink_val)

    ratio = sinks[2] / sinks[0] if sinks[0] > 1e-15 else 0

    # If sink scales as x_q^{-2Δ}, ratio should be (192/64)^{-2Δ} = 3^{-2Δ}
    if l in layer_deltas:
        predicted_ratio = (192.0 / 64.0) ** (-2 * layer_deltas[l])
        print(f"  {l+1:>7d} {layer_deltas[l]:>8.4f} "
              f"{sinks[0]:>13.6f} {sinks[1]:>14.6f} {sinks[2]:>14.6f} "
              f"{ratio:>13.4f}  (predicted: {predicted_ratio:.4f})")
    else:
        print(f"  {l+1:>7d} {'---':>8} "
              f"{sinks[0]:>13.6f} {sinks[1]:>14.6f} {sinks[2]:>14.6f} "
              f"{ratio:>13.4f}")

print()
print("  If measured ratio ≈ predicted ratio = (192/64)^{-2Δ}:")
print("  → the attention sink decays as a power law with the SAME Δ from the")
print("  conformal scaling measurement. Two independent phenomena, one exponent.")


# ========== BONUS: U-shape asymmetry ==========
print("\n" + "=" * 90)
print("  U-SHAPE ASYMMETRY (Prediction 2 preview)")
print("=" * 90)
print()
print("  BCFT on a strip: two different boundaries → asymmetric U-shape")
print("  start/end asymmetry by layer:")
print()

for l in range(n_layers):
    m = layer_ushape_metrics[l]
    bar_start = '█' * int(m['start_attn'] / max(m['start_attn'], m['end_attn']) * 20)
    bar_end = '█' * int(m['end_attn'] / max(m['start_attn'], m['end_attn']) * 20)
    print(f"  Layer {l+1:2d}: start {bar_start:<20} {m['start_attn']:.6f}"
          f"   end {bar_end:<20} {m['end_attn']:.6f}"
          f"   ratio = {m['asymmetry']:.3f}")


# ========== Summary ==========
print("\n" + "=" * 90)
print("  SUMMARY")
print("=" * 90)
print()
print("  The lost-in-the-middle U-shape measured at each of 12 GPT-2 layers.")
print("  Predictions tested:")
print("    3: U-shape correlates with Δ per layer")
print("    6: Attention sink scales as power law with independently measured Δ")
print("    2 (preview): U-shape is asymmetric (start ≠ end)")
print()
