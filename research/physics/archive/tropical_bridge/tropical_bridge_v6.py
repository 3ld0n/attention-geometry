"""
Tropical Bridge v6: The IR Test — Trained Weights

The central question: Random weights see data geometry (UV).
Do trained weights see conformal structure (IR)?

Method:
- Extract attention from pretrained GPT-2 (12 layers, trained)
- Same from randomly initialized GPT-2 (identical architecture, untrained)
- Compute Var[α(i, i-r)] as function of token separation r (connected correlator)
- Also compute <α(i, i-r)> (mean propagator)
- Fit both for power-law: f(r) ~ r^{-2Δ}
- Compare Δ trained vs random, layer by layer
- Track Δ across layers (RG flow test)

If trained Δ → 1/4 while random Δ ≠ 1/4:
  → Conformal dimension is an IR phenomenon. Training IS RG flow.

March 11, 2026 — Ariel
"""

import torch
import numpy as np
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 128
N_SAMPLES = 400
BATCH_SIZE = 40
MARGIN = 16
MAX_SEP = 60

print("=" * 70)
print("TROPICAL BRIDGE v6: THE IR TEST")
print("Trained vs Random Weights — Conformal Dimension from GPT-2")
print("=" * 70)


# ==================================================================
# Load Models
# ==================================================================

print("\nLoading GPT-2 (trained)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_trained = GPT2Model.from_pretrained("gpt2", output_attentions=True)
model_trained.eval()

print("Creating GPT-2 (random init, same architecture)...")
config = GPT2Config()
model_random = GPT2Model(config)
model_random.eval()

n_layers = config.n_layer
n_heads = config.n_head
d_k = config.n_embd // config.n_head

print(f"Architecture: {n_layers} layers, {n_heads} heads, "
      f"d_model={config.n_embd}, d_k={d_k}")
print(f"Ensemble: {N_SAMPLES} random-token sequences, length {SEQ_LEN}")

vocab_size = tokenizer.vocab_size
input_ids = torch.randint(0, vocab_size, (N_SAMPLES, SEQ_LEN))


# ==================================================================
# Core: Extract Correlator
# ==================================================================

def extract_correlator(model, input_ids, label=""):
    """
    For causal attention: token i attends to token j <= i.
    Separation r = i - j. We measure <α(r)> and Var[α(r)]
    averaged over positions i in [margin, seq_len - margin].
    """
    n_seq = input_ids.shape[0]
    seps = np.arange(1, MAX_SEP + 1)

    attn_sum = [[np.zeros((SEQ_LEN, SEQ_LEN)) for _ in range(n_heads)]
                for _ in range(n_layers)]
    attn_sq = [[np.zeros((SEQ_LEN, SEQ_LEN)) for _ in range(n_heads)]
               for _ in range(n_layers)]

    n_done = 0
    for start in range(0, n_seq, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_seq)
        batch = input_ids[start:end]

        with torch.no_grad():
            out = model(batch, output_attentions=True)

        for l_idx in range(n_layers):
            a = out.attentions[l_idx].numpy()
            for h_idx in range(n_heads):
                ah = a[:, h_idx, :, :]
                attn_sum[l_idx][h_idx] += ah.sum(axis=0)
                attn_sq[l_idx][h_idx] += (ah ** 2).sum(axis=0)

        n_done += end - start
        if n_done % 200 == 0 or n_done == n_seq:
            print(f"  [{label}] {n_done}/{n_seq}")

    mean_by_sep = [[np.zeros(len(seps)) for _ in range(n_heads)]
                   for _ in range(n_layers)]
    var_by_sep = [[np.zeros(len(seps)) for _ in range(n_heads)]
                  for _ in range(n_layers)]

    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            mean_mat = attn_sum[l_idx][h_idx] / n_seq
            var_mat = attn_sq[l_idx][h_idx] / n_seq - mean_mat ** 2

            for s_idx, r in enumerate(seps):
                m_vals, v_vals = [], []
                for i in range(max(MARGIN, r), SEQ_LEN - MARGIN):
                    j = i - r
                    if j >= 0:
                        m_vals.append(mean_mat[i, j])
                        v_vals.append(var_mat[i, j])

                mean_by_sep[l_idx][h_idx][s_idx] = np.mean(m_vals) if m_vals else 0
                var_by_sep[l_idx][h_idx][s_idx] = np.mean(v_vals) if v_vals else 0

    return {'mean': mean_by_sep, 'var': var_by_sep, 'seps': seps}


def fit_power_law(seps, values, r_min=3, r_max=30):
    """Fit log|values| = slope * log(r) + c. Return Δ = -slope/2."""
    mask = (seps >= r_min) & (seps <= r_max) & (np.abs(values) > 1e-20)
    if mask.sum() < 5:
        return None
    log_r = np.log(seps[mask].astype(float))
    log_v = np.log(np.abs(values[mask]))
    slope, intercept = np.polyfit(log_r, log_v, 1)
    pred = slope * log_r + intercept
    ss_res = np.sum((log_v - pred) ** 2)
    ss_tot = np.sum((log_v - log_v.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
    return {'delta': -slope / 2, 'slope': slope, 'r_sq': r_sq,
            'n_pts': int(mask.sum())}


# ==================================================================
# Run Both Models
# ==================================================================

print("\n--- Trained GPT-2 ---")
res_t = extract_correlator(model_trained, input_ids, "Trained")

print("\n--- Random-init GPT-2 ---")
res_r = extract_correlator(model_random, input_ids, "Random")


# ==================================================================
# A. Conformal Dimension: Variance Correlator
# ==================================================================

print("\n" + "=" * 70)
print("A. CONFORMAL DIMENSION — Var[α(r)] ~ r^{-2Δ}")
print("   Fit range: r ∈ [3, 30]")
print("=" * 70)

header = (f"{'Layer':>5} | {'Δ_train':>8} | {'R²_t':>6} | "
          f"{'Δ_rand':>8} | {'R²_r':>6} | {'ΔΔ':>8}")
print(f"\n{header}")
print("-" * 60)

deltas_t, deltas_r = [], []

for l in range(n_layers):
    var_t = np.mean([np.abs(res_t['var'][l][h]) for h in range(n_heads)], axis=0)
    var_r = np.mean([np.abs(res_r['var'][l][h]) for h in range(n_heads)], axis=0)

    ft = fit_power_law(res_t['seps'], var_t)
    fr = fit_power_law(res_r['seps'], var_r)

    dt = ft['delta'] if ft else None
    dr = fr['delta'] if fr else None
    deltas_t.append(dt)
    deltas_r.append(dr)

    dt_s = f"{dt:>8.4f}" if dt else f"{'—':>8}"
    dr_s = f"{dr:>8.4f}" if dr else f"{'—':>8}"
    rt_s = f"{ft['r_sq']:>6.3f}" if ft else f"{'—':>6}"
    rr_s = f"{fr['r_sq']:>6.3f}" if fr else f"{'—':>6}"
    dd_s = f"{dt - dr:>+8.4f}" if (dt and dr) else f"{'—':>8}"

    flag = ""
    if dt and abs(dt - 0.25) < 0.03 and ft['r_sq'] > 0.9:
        flag = " ◀ Δ≈1/4!"

    print(f"{l:>5} | {dt_s} | {rt_s} | {dr_s} | {rr_s} | {dd_s}{flag}")


# ==================================================================
# B. Mean Attention Profile
# ==================================================================

print("\n" + "=" * 70)
print("B. MEAN ATTENTION <α(r)> — power-law decay?")
print("=" * 70)

print(f"\n{'Layer':>5} | {'Δ_mean_t':>9} | {'R²':>6} | {'Δ_mean_r':>9} | {'R²':>6}")
print("-" * 50)

for l in range(n_layers):
    mean_t = np.mean([res_t['mean'][l][h] for h in range(n_heads)], axis=0)
    mean_r = np.mean([res_r['mean'][l][h] for h in range(n_heads)], axis=0)

    ft = fit_power_law(res_t['seps'], mean_t)
    fr = fit_power_law(res_r['seps'], mean_r)

    dt_s = f"{ft['delta']:>9.4f}" if ft else f"{'—':>9}"
    dr_s = f"{fr['delta']:>9.4f}" if fr else f"{'—':>9}"
    rt_s = f"{ft['r_sq']:>6.3f}" if ft else f"{'—':>6}"
    rr_s = f"{fr['r_sq']:>6.3f}" if fr else f"{'—':>6}"

    print(f"{l:>5} | {dt_s} | {rt_s} | {dr_s} | {rr_s}")


# ==================================================================
# C. Raw Correlator Profiles
# ==================================================================

print("\n" + "=" * 70)
print("C. RAW CORRELATOR PROFILES (head-averaged)")
print("=" * 70)

for tag, res in [("TRAINED", res_t), ("RANDOM", res_r)]:
    for l in [0, 5, 11]:
        var_avg = np.mean([np.abs(res['var'][l][h])
                           for h in range(n_heads)], axis=0)
        mean_avg = np.mean([res['mean'][l][h]
                            for h in range(n_heads)], axis=0)

        print(f"\n{tag} Layer {l}:")
        print(f"{'r':>4} | {'<α>':>12} | {'Var[α]':>12} | "
              f"{'ln r':>6} | {'ln Var':>9}")

        indices = list(range(min(15, len(res['seps']))))
        if len(res['seps']) > 15:
            indices += list(range(15, len(res['seps']), 5))

        for idx in indices:
            if idx >= len(res['seps']):
                break
            r_val = res['seps'][idx]
            m = mean_avg[idx]
            v = var_avg[idx]
            lv = f"{np.log(v):>9.3f}" if v > 1e-20 else f"{'—':>9}"
            print(f"{r_val:>4} | {m:>12.8f} | {v:>12.4e} | "
                  f"{np.log(r_val):>6.3f} | {lv}")


# ==================================================================
# D. Head-by-Head Universality (Layer 6)
# ==================================================================

print("\n" + "=" * 70)
print("D. HEAD-BY-HEAD Δ — Layer 6, Trained")
print("   If Δ is universal, all heads should agree.")
print("=" * 70)

l_check = 6
print(f"\n{'Head':>4} | {'Δ_var':>7} | {'R²':>6} | {'Δ_mean':>7} | {'R²':>6}")
print("-" * 40)

for h in range(n_heads):
    var_h = np.abs(res_t['var'][l_check][h])
    mean_h = res_t['mean'][l_check][h]

    fv = fit_power_law(res_t['seps'], var_h)
    fm = fit_power_law(res_t['seps'], mean_h)

    dv = f"{fv['delta']:>7.4f}" if fv else f"{'—':>7}"
    rv = f"{fv['r_sq']:>6.3f}" if fv else f"{'—':>6}"
    dm = f"{fm['delta']:>7.4f}" if fm else f"{'—':>7}"
    rm = f"{fm['r_sq']:>6.3f}" if fm else f"{'—':>6}"

    print(f"{h:>4} | {dv} | {rv} | {dm} | {rm}")


# ==================================================================
# E. RG Flow Analysis
# ==================================================================

print("\n" + "=" * 70)
print("E. RG FLOW — Does Δ evolve with depth?")
print("=" * 70)

for tag, deltas in [("Trained", deltas_t), ("Random", deltas_r)]:
    valid = [(i, d) for i, d in enumerate(deltas) if d is not None]
    if len(valid) < 6:
        print(f"\n{tag}: insufficient valid fits")
        continue

    _, vals = zip(*valid)
    early = np.mean(vals[:3])
    mid = np.mean(vals[4:8])
    late = np.mean(vals[-3:])

    print(f"\n{tag}:")
    print(f"  Early  (L 0-2):  Δ = {early:.4f}")
    print(f"  Middle (L 4-7):  Δ = {mid:.4f}")
    print(f"  Late   (L 9-11): Δ = {late:.4f}")

    drift = late - early
    if abs(drift) > 0.02:
        direction = "increases" if drift > 0 else "decreases"
        print(f"  → Δ {direction} with depth ({drift:+.4f})")
    else:
        print(f"  → Δ roughly constant ({drift:+.4f})")


# ==================================================================
# F. Fit Robustness — Multiple Windows
# ==================================================================

print("\n" + "=" * 70)
print("F. FIT ROBUSTNESS — Δ at different fit windows (trained, layer 6)")
print("=" * 70)

l_rob = 6
var_rob = np.mean([np.abs(res_t['var'][l_rob][h])
                    for h in range(n_heads)], axis=0)

print(f"\n{'Window':>12} | {'Δ':>7} | {'R²':>6} | {'pts':>4}")
print("-" * 38)

for r_min, r_max in [(2, 15), (3, 20), (3, 30), (3, 40), (5, 30),
                      (5, 40), (5, 50), (10, 40), (10, 50)]:
    f = fit_power_law(res_t['seps'], var_rob, r_min=r_min, r_max=r_max)
    if f:
        print(f"  [{r_min:>2}, {r_max:>2}]   | {f['delta']:>7.4f} | "
              f"{f['r_sq']:>6.3f} | {f['n_pts']:>4}")


# ==================================================================
# SYNTHESIS
# ==================================================================

print("\n" + "=" * 70)
print("SYNTHESIS — The IR Test")
print("=" * 70)

avg_dt = np.mean([d for d in deltas_t if d is not None])
avg_dr = np.mean([d for d in deltas_r if d is not None])
gap = avg_dt - avg_dr

print(f"""
Grand averages (Var correlator):
  Trained GPT-2:   Δ = {avg_dt:.4f}
  Random-init:     Δ = {avg_dr:.4f}
  SYK_4 target:    Δ = 0.2500
  Gap (trained-random): {gap:+.4f}

Assessment:""")

if abs(avg_dt - 0.25) < 0.05 and abs(gap) > 0.02:
    print("  ★ TRAINED Δ ≈ 1/4 — conformal dimension IS an IR phenomenon.")
    print("    Training = RG flow. The SYK bridge extends beyond perturbation theory.")
elif avg_dt > avg_dr + 0.02:
    print("  Training moves Δ TOWARD 1/4 but may not reach the fixed point.")
    print("  GPT-2 small (12 layers) may be too shallow for full IR convergence.")
elif abs(gap) < 0.02:
    print("  Trained ≈ Random — conformal dimension does NOT depend on training.")
    print("  The power-law structure, if present, is architectural, not learned.")
elif avg_dt < avg_dr:
    print("  Training moves Δ AWAY from the random-weight value.")
    print("  Need to understand what the trained regime produces.")
else:
    print("  Results need careful interpretation. See profiles above.")
