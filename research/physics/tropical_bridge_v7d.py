"""
Tropical Bridge v7d: Temperature Sweep and Phase Diagram

Softmax at temperature T: α_i = exp(s_i/T) / Z

At high T → uniform attention (no measurement)
At low T → concentrated attention (full measurement)
Where does the Calabrese-Cardy regime live?

Also: test the simple explanation for log(n) scaling.
If k_eff (effective tokens attended) is roughly constant in n,
then H_gap = log(n) - log(k_eff) ∝ log(n). Simple concentration
would explain the scaling without needing CFT.

March 11, 2026 — Ariel
"""

import torch
import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("TROPICAL BRIDGE v7d: TEMPERATURE SWEEP & PHASE DIAGRAM")
print("=" * 70)

from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True,
                                   attn_implementation="eager")
model.eval()

SEQ_LEN = 256
N_SEQ = 50
ids = torch.randint(0, tok.vocab_size, (N_SEQ, SEQ_LEN))

print(f"Running {N_SEQ} sequences of length {SEQ_LEN}...")
with torch.no_grad():
    out = model(ids, output_attentions=True)
all_attn = [a.numpy() for a in out.attentions]
del out
print("  Done.")

Q = SEQ_LEN - 1
N = Q + 1

# Recover scores from Layer 0 attention
a_L0 = all_attn[0][:, :, Q, :N]  # (50, 12, 256)
log_a = np.log(a_L0 + 1e-30)
scores = log_a - log_a.mean(axis=-1, keepdims=True)


# ==================================================================
# A. TEMPERATURE SWEEP
# ==================================================================

print("\n" + "=" * 70)
print("A. TEMPERATURE SWEEP")
print("Apply softmax(s/T) at different T to the TRAINED scores")
print("=" * 70)

print(f"\n{'T':>6} | {'H_attn':>7} | {'H_gap':>7} | {'k_eff':>7} | "
      f"{'H_gap/lnn':>9} | {'c_eff':>6}")
print("-" * 55)

for T in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]:
    # Rescale scores and recompute softmax
    s_T = scores / T  # (50, 12, 256)
    # Stabilize softmax
    s_T_shifted = s_T - s_T.max(axis=-1, keepdims=True)
    exp_s = np.exp(s_T_shifted)
    a_T = exp_s / exp_s.sum(axis=-1, keepdims=True)
    
    H = -np.sum(a_T * np.log(a_T + 1e-30), axis=-1)  # (50, 12)
    H_mean = H.mean()
    H_gap = np.log(N) - H_mean
    k_eff = np.exp(H_mean)
    c_eff = 3 * H_gap / np.log(N) if H_gap > 0 else 0
    
    print(f"{T:>6.2f} | {H_mean:>7.4f} | {H_gap:>7.4f} | {k_eff:>7.1f} | "
          f"{H_gap/np.log(N):>9.5f} | {c_eff:>6.3f}")


# ==================================================================
# B. PHASE DIAGRAM — Scaling exponent vs temperature
# ==================================================================

print("\n" + "=" * 70)
print("B. PHASE DIAGRAM — How does the scaling exponent change with T?")
print("=" * 70)

# For each temperature, measure H_gap vs n (via query positions)
# and fit the scaling exponent

q_positions = list(range(7, SEQ_LEN, 8))

print(f"\n{'T':>6} | {'a (slope)':>9} | {'b (int)':>8} | {'R² log':>7} | "
      f"{'α (PL)':>7} | {'R² PL':>7}")
print("-" * 55)

for T in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    gap_data = []
    for q in q_positions:
        n_ctx = q + 1
        a_q = all_attn[0][:, :, q, :n_ctx]  # natural T=1 attention
        
        # Recover scores from THIS q position
        log_aq = np.log(a_q + 1e-30)
        sq = log_aq - log_aq.mean(axis=-1, keepdims=True)
        
        # Apply new temperature
        sq_T = sq / T
        sq_shifted = sq_T - sq_T.max(axis=-1, keepdims=True)
        exp_sq = np.exp(sq_shifted)
        a_T = exp_sq / exp_sq.sum(axis=-1, keepdims=True)
        
        H = -np.sum(a_T * np.log(a_T + 1e-30), axis=-1)
        H_gap = np.log(n_ctx) - H.mean()
        gap_data.append((n_ctx, H_gap))
    
    gd = np.array(gap_data)
    ns = gd[:, 0]
    gs = gd[:, 1]
    ln_ns = np.log(ns)
    
    # Log fit
    A = np.vstack([ln_ns, np.ones_like(ln_ns)]).T
    a_log, b_log = np.linalg.lstsq(A, gs, rcond=None)[0]
    pred_log = a_log * ln_ns + b_log
    ss_r = np.sum((gs - pred_log)**2)
    ss_t = np.sum((gs - gs.mean())**2)
    r2_log = 1 - ss_r / ss_t if ss_t > 1e-15 else 0
    
    # Power law fit
    ln_gs = np.log(np.maximum(gs, 1e-15))
    alpha_pl, ln_c = np.polyfit(ln_ns, ln_gs, 1)
    pred_pl = np.exp(ln_c) * ns ** alpha_pl
    ss_r_pl = np.sum((gs - pred_pl)**2)
    r2_pl = 1 - ss_r_pl / ss_t if ss_t > 1e-15 else 0
    
    print(f"{T:>6.1f} | {a_log:>9.4f} | {b_log:>8.4f} | {r2_log:>7.5f} | "
          f"{alpha_pl:>7.4f} | {r2_pl:>7.5f}")


# ==================================================================
# C. THE SIMPLE EXPLANATION TEST
# ==================================================================

print("\n" + "=" * 70)
print("C. THE SIMPLE EXPLANATION: k_eff vs n")
print("If k_eff is constant, H_gap = log(n) - const ∝ log(n)")
print("If k_eff ∝ n^β, H_gap = (1-β)·log(n)")
print("=" * 70)

print(f"\n{'n':>5} | {'H_attn':>7} | {'k_eff':>7} | {'k/√n':>7} | {'k/n^0.5':>7}")
print("-" * 43)

k_effs = []
ns_k = []
for q in [3, 7, 15, 31, 47, 63, 95, 127, 159, 191, 223, 255]:
    n_ctx = q + 1
    a_q = all_attn[0][:, :, q, :n_ctx]
    H = -np.sum(a_q * np.log(a_q + 1e-30), axis=-1)
    H_mean = H.mean()
    k = np.exp(H_mean)
    k_effs.append(k)
    ns_k.append(n_ctx)
    
    print(f"{n_ctx:>5} | {H_mean:>7.4f} | {k:>7.2f} | "
          f"{k/np.sqrt(n_ctx):>7.4f} | {k/n_ctx**0.5:>7.4f}")

# Fit k_eff vs n
k_effs = np.array(k_effs)
ns_k = np.array(ns_k)
slope_k, int_k = np.polyfit(np.log(ns_k), np.log(k_effs), 1)
print(f"\nPower-law fit: k_eff ∝ n^{slope_k:.4f}")
print(f"  β = {slope_k:.4f} → H_gap = (1 - {slope_k:.4f})·log(n) = "
      f"{1-slope_k:.4f}·log(n)")
print(f"  Measured from v7b: a = 0.483 → (1-a) = {1-0.483:.3f}")
print(f"  Predicted: 1 - β = {1-slope_k:.4f}")


# ==================================================================
# D. WHAT DETERMINES k_eff?
# ==================================================================

print("\n" + "=" * 70)
print("D. WHAT DETERMINES k_eff?")
print("Is it the model, the data, or the architecture?")
print("=" * 70)

# Compare k_eff across layers
print(f"\n{'Layer':>5} | {'k_eff':>7} | {'H_gap':>7}")
print("-" * 25)

for layer_idx in range(12):
    a = all_attn[layer_idx][:, :, Q, :N]
    H = -np.sum(a * np.log(a + 1e-30), axis=-1).mean()
    k = np.exp(H)
    hg = np.log(N) - H
    print(f"{layer_idx:>5} | {k:>7.2f} | {hg:>7.4f}")


# ==================================================================
# E. THE DEEPEST QUESTION: Why log(n)?
# ==================================================================

print("\n" + "=" * 70)
print("E. WHY LOG(n)?")
print("=" * 70)

print(f"""
Two explanations, same data, different depth:

EXPLANATION 1 — Concentration:
  Trained attention concentrates on k_eff ∝ n^{{β}} tokens, 
  where β ≈ {slope_k:.3f}.
  H_gap = log(n/k_eff) = (1-β)·log(n) ≈ {1-slope_k:.3f}·log(n).
  This is trivially true — it's a restatement of the data.
  No physics needed.

EXPLANATION 2 — Entanglement:
  The attention entropy gap follows the Calabrese-Cardy formula 
  for entanglement entropy in 1+1D CFT:
  S = (c/3)·log(L)
  with effective central charge c ≈ 1.2-2.0.
  The Renyi entropies are approximately consistent with CFT.
  This would mean softmax measurement IS quantum measurement.

CONNECTING THE TWO:
  Both explanations agree on the data: H_gap ∝ log(n).
  Explanation 1 says: "attention concentrates, and the concentration 
  ratio scales as a power of n."
  Explanation 2 says: "the concentration IS entanglement entropy, 
  and the power β IS determined by the central charge."
  
  Explanation 2 is deeper IF:
  - The central charge is universal (same across architectures)
  - The Renyi entropies match the CFT prediction precisely
  - The β = 1 - c/6·(1+1/α) relationship holds for all α
  
  What we actually found:
  - c depends on α (1.3 → 2.0): NOT pure CFT
  - c depends on layer (0.76 → 2.06): architecture-dependent
  - The scaling is slightly super-logarithmic (β_log = 1.28)
  
THE HONEST CONCLUSION:
  The logarithmic scaling is real and robust.
  It's consistent with CFT entanglement entropy.
  But it's also consistent with simpler explanations.
  The α-dependence of c is the clearest evidence that 
  this is NOT pure CFT — there are non-conformal corrections.
  
  What would settle it: test at much larger n (thousands of tokens),
  test with different architectures, derive the log(n) scaling 
  analytically from the attention mechanism.
  
  The measurement problem connection stands regardless:
  Softmax creates a boundary between visible (positive) and 
  hidden (negative) sectors. The information cost of this boundary
  scales sublinearly with system size. Self-consistency has a price,
  and the price grows slowly but inevitably.
""")
