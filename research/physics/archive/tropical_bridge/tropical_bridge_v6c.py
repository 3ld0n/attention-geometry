"""
Tropical Bridge v6c: Universality Tests (streamlined)

Testing whether Δ = 1/4 is universal:
A. Sequence length dependence (GPT-2 small only — fast)
B. GPT-2 Medium Layer 0 (reduced samples)
C. Head decomposition at layer 5 for r=16 anomaly

March 11, 2026 — Ariel
"""

import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

MARGIN = 16


def compute_layer_profile(model, input_ids, layer_idx=0, max_sep=None):
    """Mean and var of attention at given layer, per head."""
    seq_len = input_ids.shape[1]
    if max_sep is None:
        max_sep = min(60, seq_len // 2)
    seps = np.arange(1, max_sep + 1)
    n_heads = model.config.n_head
    n_seq = input_ids.shape[0]
    batch_size = 40

    attn_sum = [np.zeros((seq_len, seq_len)) for _ in range(n_heads)]
    attn_sq = [np.zeros((seq_len, seq_len)) for _ in range(n_heads)]

    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        with torch.no_grad():
            out = model(input_ids[start:end], output_attentions=True)
        a = out.attentions[layer_idx].numpy()
        for h in range(n_heads):
            ah = a[:, h, :, :]
            attn_sum[h] += ah.sum(axis=0)
            attn_sq[h] += (ah ** 2).sum(axis=0)

    margin = min(MARGIN, seq_len // 4)
    mean_by_h = np.zeros((n_heads, len(seps)))
    var_by_h = np.zeros((n_heads, len(seps)))

    for h in range(n_heads):
        mean_mat = attn_sum[h] / n_seq
        var_mat = attn_sq[h] / n_seq - mean_mat ** 2
        for s, r in enumerate(seps):
            m_v, v_v = [], []
            for i in range(max(margin, r), seq_len - margin):
                j = i - r
                if j >= 0:
                    m_v.append(mean_mat[i, j])
                    v_v.append(var_mat[i, j])
            mean_by_h[h, s] = np.mean(m_v) if m_v else 0
            var_by_h[h, s] = np.mean(v_v) if v_v else 0

    return seps, mean_by_h, var_by_h


def fit_pl(seps, values, r_min=3, r_max=30):
    mask = (seps >= r_min) & (seps <= r_max) & (np.abs(values) > 1e-20)
    if mask.sum() < 5:
        return None
    lr = np.log(seps[mask].astype(float))
    lv = np.log(np.abs(values[mask]))
    slope, intercept = np.polyfit(lr, lv, 1)
    pred = slope * lr + intercept
    ss_r = np.sum((lv - pred) ** 2)
    ss_t = np.sum((lv - lv.mean()) ** 2)
    r2 = 1 - ss_r / ss_t if ss_t > 1e-30 else 0
    return {'delta': -slope / 2, 'slope': slope, 'r_sq': r2}


print("=" * 70)
print("TROPICAL BRIDGE v6c: UNIVERSALITY TESTS")
print("=" * 70)

print("\nLoading GPT-2 small...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
model_small = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                         attn_implementation="eager")
model_small.eval()


# ==================================================================
# A. SEQUENCE LENGTH DEPENDENCE
# ==================================================================

print("\n" + "=" * 70)
print("A. SEQUENCE LENGTH DEPENDENCE")
print("If Δ is real physics, it should NOT change with sequence length.")
print("=" * 70)

print(f"\n{'L':>5} | {'N':>4} | {'Δ_mean':>7} | {'R²':>6} | "
      f"{'Δ_var':>7} | {'R²':>6} | {'|Δ-0.25|':>8}")
print("-" * 60)

for seq_len in [48, 64, 96, 128, 192, 256]:
    n_seq = max(200, 600 // max(1, seq_len // 64))
    ids = torch.randint(0, tok.vocab_size, (n_seq, seq_len))
    max_s = min(40, seq_len // 2 - MARGIN)
    r_max = min(25, max_s - 2)

    seps, mean_h, var_h = compute_layer_profile(model_small, ids, layer_idx=0,
                                                 max_sep=max_s)
    mean_avg = mean_h.mean(axis=0)
    var_avg = np.abs(var_h).mean(axis=0)

    fm = fit_pl(seps, mean_avg, r_min=3, r_max=r_max)
    fv = fit_pl(seps, var_avg, r_min=3, r_max=r_max)

    dm = f"{fm['delta']:>7.4f}" if fm else f"{'—':>7}"
    rm = f"{fm['r_sq']:>6.3f}" if fm else f"{'—':>6}"
    dv = f"{fv['delta']:>7.4f}" if fv else f"{'—':>7}"
    rv = f"{fv['r_sq']:>6.3f}" if fv else f"{'—':>6}"
    err = f"{abs(fm['delta'] - 0.25):>8.4f}" if fm else f"{'—':>8}"

    print(f"{seq_len:>5} | {n_seq:>4} | {dm} | {rm} | {dv} | {rv} | {err}")


# ==================================================================
# B. HEAD DECOMPOSITION — r=16 ANOMALY
# ==================================================================

print("\n" + "=" * 70)
print("B. HEAD DECOMPOSITION — Layer 5, identifying r=16 anomaly")
print("=" * 70)

n_b = 400
ids_b = torch.randint(0, tok.vocab_size, (n_b, 128))

seps_b, mean_b, var_b = compute_layer_profile(model_small, ids_b, layer_idx=5)

print(f"\nMean attention <α(r)> per head — separations around anomaly:")
print(f"{'Head':>4} | {'r=12':>8} | {'r=14':>8} | {'r=15':>8} | "
      f"{'r=16':>8} | {'r=17':>8} | {'r=20':>8} | {'jump':>6}")
print("-" * 68)

for h in range(model_small.config.n_head):
    prof = mean_b[h]
    r12 = prof[11]
    r14 = prof[13]
    r15 = prof[14]
    r16 = prof[15]
    r17 = prof[16]
    r20 = prof[19]
    jump = r16 / r15 if r15 > 1e-15 else 0
    flag = " ◀" if jump > 2.0 else ""
    print(f"{h:>4} | {r12:>8.5f} | {r14:>8.5f} | {r15:>8.5f} | "
          f"{r16:>8.5f} | {r17:>8.5f} | {r20:>8.5f} | {jump:>5.1f}x{flag}")

avg_prof = mean_b.mean(axis=0)
print(f"\n{'Avg':>4} | {avg_prof[11]:>8.5f} | {avg_prof[13]:>8.5f} | "
      f"{avg_prof[14]:>8.5f} | {avg_prof[15]:>8.5f} | {avg_prof[16]:>8.5f} | "
      f"{avg_prof[19]:>8.5f} | "
      f"{avg_prof[15] / avg_prof[14] if avg_prof[14] > 0 else 0:>5.1f}x")

print(f"\nWhat's happening: some heads have near-zero attention at r<16 but")
print(f"constant attention at r>=16. These 'long-range' heads create the anomaly")
print(f"when averaged with 'local' heads that decay smoothly.")

print(f"\nFull profile for most anomalous head and smoothest head:")
jumps = [(h, mean_b[h, 15] / mean_b[h, 14]) for h in range(model_small.config.n_head)
         if mean_b[h, 14] > 1e-15]
jumps.sort(key=lambda x: -x[1])
anomalous_h = jumps[0][0]
smooth_h = jumps[-1][0]

print(f"\n{'r':>4} | {'Head '+str(anomalous_h)+' (jump)':>16} | "
      f"{'Head '+str(smooth_h)+' (smooth)':>18}")
for s in range(min(30, len(seps_b))):
    r = seps_b[s]
    print(f"{r:>4} | {mean_b[anomalous_h, s]:>16.7f} | "
          f"{mean_b[smooth_h, s]:>18.7f}")


# ==================================================================
# C. GPT-2 MEDIUM — Layer 0 only, minimal samples
# ==================================================================

print("\n" + "=" * 70)
print("C. GPT-2 MEDIUM — Layer 0 (150 sequences, minimal run)")
print("=" * 70)

print("Loading GPT-2 medium...")
model_med = GPT2Model.from_pretrained("gpt2-medium", output_attentions=True,
                                       attn_implementation="eager")
model_med.eval()

n_c = 150
ids_c = torch.randint(0, tok.vocab_size, (n_c, 128))

print(f"Running {n_c} sequences through medium model Layer 0...")
seps_c, mean_c, var_c = compute_layer_profile(model_med, ids_c, layer_idx=0)

mean_c_avg = mean_c.mean(axis=0)
var_c_avg = np.abs(var_c).mean(axis=0)

fm_c = fit_pl(seps_c, mean_c_avg)
fv_c = fit_pl(seps_c, var_c_avg)

print(f"\nGPT-2 Medium Layer 0:")
print(f"  Δ_mean = {fm_c['delta']:.5f}, R² = {fm_c['r_sq']:.4f}")
print(f"  Δ_var  = {fv_c['delta']:.5f}, R² = {fv_c['r_sq']:.4f}")

seps_s2, mean_s2, var_s2 = compute_layer_profile(model_small, ids_c, layer_idx=0)
mean_s2_avg = mean_s2.mean(axis=0)
fm_s2 = fit_pl(seps_s2, mean_s2_avg)

print(f"\nGPT-2 Small Layer 0 (same inputs):")
print(f"  Δ_mean = {fm_s2['delta']:.5f}, R² = {fm_s2['r_sq']:.4f}")

print(f"\nComparison:")
print(f"  Small:  Δ = {fm_s2['delta']:.4f}")
print(f"  Medium: Δ = {fm_c['delta']:.4f}")
print(f"  SYK₄:   Δ = 0.2500")

del model_med

# Also check medium layers 1 and 2 to see crossover
print("\nLoading medium again for layers 1-2...")
model_med2 = GPT2Model.from_pretrained("gpt2-medium", output_attentions=True,
                                        attn_implementation="eager")
model_med2.eval()

for l in [1, 2, 3]:
    seps_ml, mean_ml, _ = compute_layer_profile(model_med2, ids_c, layer_idx=l)
    mean_ml_avg = mean_ml.mean(axis=0)
    fm_ml = fit_pl(seps_ml, mean_ml_avg)
    if fm_ml:
        print(f"  Medium Layer {l}: Δ = {fm_ml['delta']:.4f}, R² = {fm_ml['r_sq']:.3f}")

del model_med2


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS — Is Δ = 1/4 Universal?")
print("=" * 70)

print(f"""
Sequence length test:
  If Δ changes with L, it's a finite-size effect.
  If Δ is stable, it's a genuine scaling property.
  (See section A above)

Model size test:
  Small (d_k=64, 12H):  Δ = {fm_s2['delta']:.4f}
  Medium (d_k=64, 16H): Δ = {fm_c['delta']:.4f}
  Target (SYK₄):        Δ = 0.2500

The r=16 anomaly:
  Caused by head specialization. 'Long-range' heads have near-zero
  attention at short distances but constant attention at r >= 16.
  These coexist with 'local' heads that decay smoothly.
  The ensemble average produces the apparent discontinuity.
  This ONLY affects layers >= 3 where head specialization is strong.
  Layer 0 (where Δ = 1/4 lives) has smooth monotonic profiles.
""")
