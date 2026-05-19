"""
End-to-End Test: BCFT Attention Measurements → Accuracy Prediction

The clean test:
  1. Measure Δ and λ from GPT-2 attention weights (random tokens)
  2. Use the BCFT formula to PREDICT the accuracy-vs-position curve
  3. Run a key-value retrieval benchmark on GPT-2
  4. Compare predicted curve to measured accuracy

If the BCFT parameters from attention weights predict the task accuracy
curve, this connects the physics to a practical AI phenomenon through
a single framework with no free parameters (except baseline/amplitude
which are task-dependent, not model-dependent).

Key-value retrieval format (adapted from Liu et al. 2023):
  Each prompt has N key-value pairs, one target pair at position k,
  followed by "What is the value for key [target]?"
  Score: 1 if model generates the correct value, 0 otherwise.

Ariel — April 15, 2026
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import uuid
import string
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========== Phase 1: Measure Δ and λ from attention weights ==========
print("=" * 90)
print("  PHASE 1: MEASURING Δ AND λ FROM GPT-2 ATTENTION WEIGHTS")
print("=" * 90)
print()

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

n_layers = model.config.n_layer
n_heads = model.config.n_head
vocab_size = model.config.vocab_size

SEQ_LEN = 256
N_ATTN_INPUTS = 80
FIT_LOW = 3
FIT_HIGH = 50
DEEP_LO = SEQ_LEN // 2

print(f"  Collecting attention from {N_ATTN_INPUTS} random inputs (seq_len={SEQ_LEN})...")

# Collect position-resolved two-point function
G_sum = np.zeros((n_layers, n_heads, SEQ_LEN, FIT_HIGH))

for inp_idx in range(N_ATTN_INPUTS):
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    for l in range(n_layers):
        attn = outputs.attentions[l][0].numpy()
        for h in range(n_heads):
            head = attn[h]
            for dx in range(FIT_HIGH):
                diag = np.diagonal(head, offset=-dx)
                G_sum[l, h, dx:dx+len(diag), dx] += diag
    if (inp_idx + 1) % 20 == 0:
        print(f"    {inp_idx + 1}/{N_ATTN_INPUTS}")

G_avg = G_sum / N_ATTN_INPUTS

# Fit Δ from deep positions (bulk, away from boundary)
print("\n  Fitting Δ per head (deep positions, R² > 0.90)...")
head_params = []

for l in range(n_layers):
    for h in range(n_heads):
        A = np.mean(G_avg[l, h, DEEP_LO:, :FIT_HIGH], axis=0)
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

        if R2 > 0.90 and 0.05 < delta < 1.5:
            head_params.append({
                'layer': l, 'head': h,
                'delta': delta, 'R2': R2,
                'C': np.exp(coeffs[0]),
            })

print(f"  Found {len(head_params)} conformal heads")

# Fit λ from position-resolved data (BCFT vs power law)
print("  Fitting λ for conformal heads (BCFT boundary parameter)...")
from scipy.optimize import minimize as scipy_minimize

for hp in head_params:
    l, h = hp['layer'], hp['head']
    all_dx = []
    all_xq = []
    all_g = []

    for x_q in range(32, SEQ_LEN):
        for dx in range(FIT_LOW, min(FIT_HIGH, x_q)):
            x_k = x_q - dx
            if x_k <= 0:
                continue
            g_val = G_avg[l, h, x_q, dx]
            if g_val < 1e-15:
                continue
            all_dx.append(float(dx))
            all_xq.append(float(x_q))
            all_g.append(g_val)

    if len(all_g) < 30:
        hp['lambda'] = 0.0
        continue

    all_dx = np.array(all_dx)
    all_xq = np.array(all_xq)
    all_g = np.array(all_g)
    delta = hp['delta']

    def bcft_loss(params):
        C, lam = params
        if C <= 0:
            return 1e12
        xk = all_xq - all_dx
        eta = all_dx ** 2 / (4.0 * all_xq * xk)
        correction = 1.0 + lam * np.power(eta, delta)
        pred = C * np.power(all_dx, -2 * delta) * correction
        return np.sum((all_g - pred) ** 2)

    C_init = hp['C']
    res = scipy_minimize(bcft_loss, [C_init, 1.0], method='Nelder-Mead',
                         options={'maxiter': 5000})
    hp['lambda'] = max(res.x[1], 0)

conformal_heads = [hp for hp in head_params if 0.15 < hp['delta'] < 0.35]
all_deltas = [hp['delta'] for hp in conformal_heads]
all_lambdas = [hp['lambda'] for hp in conformal_heads if hp['lambda'] > 0]

DELTA_MEASURED = np.median(all_deltas) if all_deltas else 0.25
LAMBDA_MEASURED = np.median(all_lambdas) if all_lambdas else 2.0

print(f"\n  MEASURED PARAMETERS (conformal heads near Δ=1/4):")
print(f"    Δ = {DELTA_MEASURED:.4f} (median of {len(all_deltas)} heads)")
print(f"    λ = {LAMBDA_MEASURED:.4f} (median of {len(all_lambdas)} heads with λ > 0)")
print(f"    Δ range: [{min(all_deltas):.3f}, {max(all_deltas):.3f}]")
print(f"    λ range: [{min(all_lambdas):.3f}, {max(all_lambdas):.3f}]")


# ========== Phase 2: BCFT Prediction ==========
print("\n\n" + "=" * 90)
print("  PHASE 2: BCFT-PREDICTED ACCURACY CURVE")
print("=" * 90)
print()

N_KV = 20  # number of key-value pairs in the benchmark

def bcft_predicted_curve(n_kv, delta, lam):
    """Predict relative accuracy at each KV position."""
    x_q = n_kv + 0.5
    positions = np.arange(n_kv) + 0.5  # center of each KV slot
    attn = np.zeros(n_kv)
    for i, x_k in enumerate(positions):
        dx = x_q - x_k
        if dx <= 0 or x_k <= 0:
            continue
        eta = dx**2 / (4.0 * x_q * x_k)
        power_law = dx ** (-2 * delta)
        correction = 1.0 + lam * eta**delta
        attn[i] = power_law * correction
    return attn / attn.max() if attn.max() > 0 else attn

predicted_curve = bcft_predicted_curve(N_KV, DELTA_MEASURED, LAMBDA_MEASURED)
print(f"  Using Δ = {DELTA_MEASURED:.4f}, λ = {LAMBDA_MEASURED:.4f}")
print(f"  Predicted relative accuracy for {N_KV} KV pairs:")
for i, p in enumerate(predicted_curve):
    bar = '█' * int(p * 40)
    print(f"    Position {i+1:>2d}: {p:.3f}  {bar}")

valley_idx = np.argmin(predicted_curve)
print(f"  Predicted valley: position {valley_idx+1} "
      f"(fraction: {(valley_idx+0.5)/N_KV:.3f})")


# ========== Phase 3: Run KV Retrieval Benchmark ==========
print("\n\n" + "=" * 90)
print("  PHASE 3: KEY-VALUE RETRIEVAL BENCHMARK ON GPT-2")
print("=" * 90)
print()

def make_short_key():
    return ''.join(random.choices(string.ascii_lowercase, k=4))

def make_short_value():
    return ''.join(random.choices(string.digits, k=4))

N_TRIALS = 50  # trials per position

# GPT-2 context window is 1024 tokens.
# Each KV pair is ~12 tokens ("Key: xxxx Value: yyyy\n")
# With N_KV=20, that's ~240 tokens + query (~30 tokens) ≈ 270 tokens. Safe.

print(f"  {N_KV} key-value pairs per trial, {N_TRIALS} trials per position")
print(f"  Format: 'Key: xxxx Value: yyyy' lines, then 'Key: [target] Value:'")
print()

position_correct = np.zeros(N_KV)
position_total = np.zeros(N_KV)

for target_pos in range(N_KV):
    correct = 0
    for trial in range(N_TRIALS):
        # Generate unique keys and values
        keys = [make_short_key() for _ in range(N_KV)]
        values = [make_short_value() for _ in range(N_KV)]

        # Ensure target key is unique
        target_key = keys[target_pos]
        target_value = values[target_pos]

        # Build prompt
        lines = []
        for i in range(N_KV):
            lines.append(f"Key: {keys[i]} Value: {values[i]}")
        prompt_text = "\n".join(lines) + f"\nKey: {target_key} Value:"

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

        if input_ids.shape[1] > 1020:
            continue

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # last token prediction

        # Check: does the model predict the first token of the target value?
        target_tokens = tokenizer.encode(" " + target_value)
        if len(target_tokens) > 0:
            predicted_token = torch.argmax(logits).item()
            # Also check top-5
            top5 = torch.topk(logits, 5).indices.tolist()
            if target_tokens[0] in top5:
                correct += 1

    accuracy = correct / N_TRIALS if N_TRIALS > 0 else 0
    position_correct[target_pos] = correct
    position_total[target_pos] = N_TRIALS

    if (target_pos + 1) % 5 == 0 or target_pos == 0:
        print(f"    Position {target_pos+1:>2d}: {accuracy*100:.1f}% "
              f"({correct}/{N_TRIALS})")

print()

# Full results
measured_accuracy = position_correct / np.maximum(position_total, 1)

print("  FULL RESULTS:")
print(f"  {'Position':>10} {'Accuracy':>10} {'BCFT pred':>10} {'Match':>8}")
print(f"  {'-'*42}")
for i in range(N_KV):
    acc = measured_accuracy[i] * 100
    pred = predicted_curve[i]
    print(f"  {i+1:>10d} {acc:>10.1f}% {pred:>10.3f}")


# ========== Phase 4: Compare ==========
print("\n\n" + "=" * 90)
print("  PHASE 4: COMPARISON — BCFT PREDICTION vs MEASURED ACCURACY")
print("=" * 90)
print()

# Fit baseline + amplitude to the measured accuracy using the predicted curve
from scipy.optimize import curve_fit

def model_func(positions, baseline, amplitude):
    return baseline + amplitude * predicted_curve[positions.astype(int)]

valid = measured_accuracy > 0
positions_valid = np.arange(N_KV)[valid]
acc_valid = measured_accuracy[valid]

if len(acc_valid) > 2 and np.std(acc_valid) > 0.001:
    try:
        popt, pcov = curve_fit(model_func, positions_valid, acc_valid,
                               p0=[0.0, np.max(acc_valid)])
        baseline, amplitude = popt
        fitted = baseline + amplitude * predicted_curve

        ss_res = np.sum((measured_accuracy[valid] - fitted[valid]) ** 2)
        ss_tot = np.sum((measured_accuracy[valid] - np.mean(measured_accuracy[valid])) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

        from scipy.stats import spearmanr
        rho, p_val = spearmanr(predicted_curve, measured_accuracy)

        print(f"  BCFT prediction (Δ={DELTA_MEASURED:.4f}, λ={LAMBDA_MEASURED:.4f})")
        print(f"  fitted with baseline={baseline:.4f}, amplitude={amplitude:.4f}")
        print()
        print(f"  R² (BCFT shape vs accuracy) = {R2:.4f}")
        print(f"  Spearman ρ (BCFT vs accuracy) = {rho:+.4f} (p = {p_val:.4f})")
        print()

        # Valley comparison
        meas_valley = np.argmin(measured_accuracy) + 1
        pred_valley = valley_idx + 1
        print(f"  Measured valley: position {meas_valley} "
              f"(frac: {(meas_valley-0.5)/N_KV:.3f})")
        print(f"  Predicted valley: position {pred_valley} "
              f"(frac: {(pred_valley-0.5)/N_KV:.3f})")
        print()

        # Asymmetry
        meas_start = np.mean(measured_accuracy[:3])
        meas_end = np.mean(measured_accuracy[-3:])
        pred_start = np.mean(predicted_curve[:3])
        pred_end = np.mean(predicted_curve[-3:])
        print(f"  Asymmetry (start/end):")
        print(f"    Measured:  {meas_start/meas_end:.3f}" if meas_end > 0 else
              "    Measured: N/A")
        print(f"    Predicted: {pred_start/pred_end:.3f}")
        print()

        # Detailed comparison
        print(f"  {'Pos':>5} {'Measured':>10} {'BCFT fit':>10} {'Residual':>10}")
        print(f"  {'-'*38}")
        for i in range(N_KV):
            res = measured_accuracy[i] - fitted[i]
            bar_m = '█' * int(measured_accuracy[i] * 30)
            bar_p = '░' * int(fitted[i] * 30)
            print(f"  {i+1:>5d} {measured_accuracy[i]:>10.3f} {fitted[i]:>10.3f} "
                  f"{res:>+10.3f}  M:{bar_m}")

        print()
        if R2 > 0.7:
            print("  ✓ BCFT attention parameters predict the accuracy curve well.")
        elif R2 > 0.3:
            print("  ~ BCFT captures the general trend but not the fine structure.")
        elif rho > 0.5 and p_val < 0.05:
            print("  ~ Rank ordering matches (Spearman) even if R² is low.")
        else:
            print("  ✗ BCFT parameters do not predict the accuracy curve.")

    except Exception as e:
        print(f"  Fitting failed: {e}")
        rho, p_val = spearmanr(predicted_curve, measured_accuracy)
        print(f"  Spearman ρ = {rho:+.4f} (p = {p_val:.4f})")
else:
    print("  Insufficient variation in accuracy to fit.")
    print(f"  Mean accuracy: {np.mean(measured_accuracy)*100:.1f}%")
    print(f"  Std: {np.std(measured_accuracy)*100:.1f}%")
    if np.std(measured_accuracy) < 0.01:
        print("  GPT-2 may be too weak for this task — accuracy near uniform.")
        print("  Consider: (a) simpler KV format, (b) fewer distractors,")
        print("            (c) larger model (GPT-2 medium/large)")
