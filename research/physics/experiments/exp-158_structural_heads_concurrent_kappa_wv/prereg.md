# Pre-registration: exp-158

**Concurrent κ̃ amplification + W_V=0 on 5 structural (random-native) Δ-window heads:
mechanistic confirmation of the attention-distribution account**

Ariel — 2026-09-23, ~8:20 PM MDT, solo physics room.

---

## Background and motivation

Three experiments have characterized the structural (random-native) Δ-window heads in GPT-2 small:

| Experiment | Protocol | ΔP_B | n/20 |
|---|---|---|---|
| exp-141 | κ̃ amplification (γ=+2.0) of W_K | +0.270 | 20/20 |
| exp-156 | W_K = 0 (ablation) | +0.485 | 17/20 |
| exp-157 | W_V = 0 (ablation) | +0.412 | 18/20 |

All three manipulations improve Task B. The current account is:

> The structural heads **interfere** with positional retrieval (Task B) through their value writes.
> The κ̃ gain-of-function (exp-141) works because concentrating attention on positional targets
> changes what the heads write to the residual stream — a more focused write is less
> interfering than the diffuse, untrained-distribution write. The routing (W_K) and the value
> write (W_V) both contribute to the interference; ablating either removes it.

This account predicts a clean mechanistic test: if W_V=0 completely removes the heads'
contribution to the residual stream, then κ̃ amplification should add nothing, because
amplified attention routing × zero value = zero output. The gain-of-function mechanism is
mediated entirely through what the heads write, not through the routing itself.

## Protocol

**Model:** GPT-2 small (gpt2; cached locally).

**Heads:** 5 structural (random-native) Δ-window heads:
L2H1, L3H4, L5H0, L7H11, L10H8 (same as exp-141, exp-156, exp-157).

**Concurrent manipulation:**
1. **κ̃ amplification of W_K** (γ=+2.0): same protocol as exp-141 — positional field
   computed at ln_1(h) output; top-4 PC directions; W_K_amp = W_K + γ × (P_k.T @ P_k @ W_K).
2. **W_V = 0 + bias_V = 0**: same protocol as exp-157 — set all five heads' c_attn value
   weight and bias columns to zero.

Both manipulations applied to the same model simultaneously.

**Sham model (W_K arm):** Random W_K (same Frobenius norm; orthogonal complement of positional
subspace), same construction as exp-141. W_V = 0 applied to sham as well (so sham differs
from ablation only in W_K direction).

**Tasks:**
- **Task B** (primary): 20-item positional retrieval battery from exp-143/157 (list-lookup).
- **Task C** (secondary): 20-item content retrieval battery from exp-155.

**Scoring:** log P(target | prompt) averaged per target token, identical to exp-157.

## Hypotheses and predictions

**H_interference_only** (predicted, primary):
ΔP_B under combined manipulation is within ±0.15 nats of ΔP_B(W_V=0 alone from exp-157)
= +0.412 nats. κ̃ amplification adds nothing when V=0, because the mechanism is entirely
mediated through the value write.

**H_additional_mechanism** (not predicted):
ΔP_B under combined manipulation > +0.562 nats (more than +0.15 nats beyond exp-157 W_V=0
baseline). κ̃ amplification has an independent mechanism beyond the value write
(e.g., through Q·K routing that affects other mechanisms' keys, not tested here).

**H_null**: |ΔP_B| < 0.10 nats. Combined manipulation is net-neutral (unexpected).

**H_degradation**: ΔP_B < −0.10 nats. Unexpected interaction between the two manipulations.

## Kill conditions

- **K1** (manipulation failure): ≥ 3/5 heads show ‖W_V‖_F > 0.01 (W_V not zeroed).
- **K2** (κ̃ hook failure): any head shows kappa_after_amp < 1.5 × kappa_before (amplification
  not confirmed).
- **K3** (sham crossover): sham produces larger Task B improvement than ablation by > 0.20 nats.

## Expected result

H_interference_only is the strongly theoretically motivated prediction:
in standard GPT-2, attn_weights × V_zeros = zeros regardless of attention distribution shape,
so the head contributes 0 to the residual stream, and κ̃ routing amplification cannot change
that. If this prediction is confirmed, it closes the mechanistic account: the gain-of-function
from exp-141 is mediated through the value write (what the head writes to the residual stream
when its attention is concentrated), not through a routing-independent mechanism.

If H_additional_mechanism fires (surprising), it would require a non-standard pathway —
most likely indirect cross-layer effects not present in single-layer analysis, or a measurement
artifact worth investigating.
