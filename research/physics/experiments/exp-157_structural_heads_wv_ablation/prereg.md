# exp-157 — Structural (random-native) heads: W_V ablation Task B test

**Pre-registration for exp-157.**
**This document is committed to attention-geometry BEFORE run.py is written.**

**Ariel — 2026-09-23, ~4:20 PM MDT. Solo physics room.**
**Seeded by exp-156 interference finding (2026-09-23): structural-head W_K ablation ≈ sham for Task B (+0.485/+0.490 nats).**

---

## What exp-156 established

exp-156 found that W_K ablation of the 5 structural heads in GPT-2 small **improves** Task B
(+0.485 nats, 17/20 items), and the sham (matched-norm random W_K) improves Task B by the same
amount (+0.490 nats). This is the **interference finding**: the trained structural W_K is not
driving positional retrieval — it is actively competing against it. Removing it (in any direction)
releases Task B equally.

| Experiment | Protocol | ΔP_B | n_B_improved | Sham ΔP_B |
|---|---|---|---|---|
| exp-141 | κ̃ amplification of structural heads | +0.270 | 20/20 | — |
| exp-156 | W_K = 0 ablation | +0.485 | 17/20 | +0.490 |

The gain-of-function from exp-141 (κ̃ amplification improves Task B) cannot be mediated by W_K
directional routing, because exp-156 shows any disruption to W_K direction improves Task B
equally. The κ̃ amplification must operate through the **attention distribution shape** —
concentrating attention weights around positional targets.

**Open question (seeded by exp-156):** If the gain-of-function operates through attention
distribution concentration, the value payload matters: the head writes
`attention_weights @ W_V @ x` to the residual stream. Does that write contribute to Task B?
Equivalently: does W_V ablation (silencing the head entirely) impair Task B relative to the
original model?

---

## Hypothesis

**H_value_pathway:** The structural heads contribute positionally useful content to the residual
stream through their value pathway. Even though their W_K routing interferes with Task B (exp-156),
the value write under the original model was doing useful work that silencing it would lose.
W_V ablation degrades Task B compared to the original model (ΔP_B < −0.10 nats).

**H_attention_shape:** The structural heads' contribution to Task B is entirely mediated by their
attention distribution shape, not by any specific content in their value writes. The value payload
under the original routing was interfering (consistent with W_K interfering), or neutral.
W_V ablation, like W_K ablation, either improves Task B or leaves it unchanged.

---

## Relationship to exp-156

There are three distinct conditions to compare:

| Condition | W_K | W_V | Head behavior | Prior result |
|---|---|---|---|---|
| Original | trained | trained | Routes to position-biased keys; writes those V vectors | baseline |
| W_K=0 (exp-156) | zero | trained | Uniform averager; writes mean(W_V @ x) | +0.485 nats |
| Sham (exp-156) | random, matched norm | trained | Routes randomly; writes V of random keys | +0.490 nats |
| **W_V=0 (this exp)** | **trained** | **zero** | **Routes to position-biased keys; writes nothing** | **?** |

The comparison that answers the question: does W_V=0 produce the same improvement as W_K=0
(+0.485 nats), or does it degrade Task B?

- If W_V=0 ≈ W_K=0 (both improve ~+0.49): the value write of the original model (under trained W_K
  routing) was also interfering with Task B. Silencing the head entirely gives the same release.
- If W_V=0 degrades Task B vs. original (ΔP_B < −0.10): the value write was doing useful work,
  even though the routing was interfering. The gain-of-function (exp-141) operates through value
  content enhanced by attention concentration.
- If W_V=0 is between original and W_K=0 (0 < ΔP_B < +0.49): removing the (interfering) routing
  while keeping the write > removing the write while keeping the routing. The net interference
  is mainly in W_K.

---

## Population

**Structural (random-native) Δ-window heads in GPT-2 small:**
```
L2H1, L3H4, L5H0, L7H11, L10H8
```
Selected by random-token census (exp-007, exp-113): Δ_A ∈ [0.20, 0.30], R² ≥ 0.90 under frozen
random-token inputs. Same heads as exp-141/142/143/155/156.

---

## Tasks

### Task B — Positional retrieval (primary)

Same approved 20-item battery from exp-143 (used in exp-141 through exp-156 unchanged).
Primary metric: median ΔP_B = median(logp_abl − logp_orig) over 20 items.

### Task C — Content-specified long-range retrieval (secondary; same items as exp-155/156)

The 20 items from `exp-155_textnative_content_retrieval/task_c_items.json`.
Secondary metric: median ΔP_C.

---

## Manipulation

### Ablation (W_V = 0)

For GPT-2 small (n_embd=768, n_head=12, head_size=64), W_V for head h at layer l lives at:
```
model.transformer.h[l].attn.c_attn.weight[:, 2*768 + h*64 : 2*768 + (h+1)*64]
```
And the bias at:
```
model.transformer.h[l].attn.c_attn.bias[2*768 + h*64 : 2*768 + (h+1)*64]
```

Set both to zero for each of the 5 structural heads. With W_V = 0 and bias_V = 0, the head's
value vector is identically zero for every position; the head writes zero to the residual stream
regardless of the attention pattern. The W_K and W_Q of the head are untouched — the head still
computes attention weights normally, but those weights are multiplied by zero values.

### Sham

Random W_V of the same Frobenius norm as the original W_V for each head independently; bias_V
set to zero (matched to ablation). Sham seed: 2026092357 (distinct from all prior experiments
in this series).

---

## Registered predictions and kill conditions

### P1 — Task B degrades with W_V ablation

**Prediction:** ΔP_B < −0.10 nats AND ablation is clearly worse than W_K ablation (ΔP_B_abl < ΔP_B_wk−0.20, where ΔP_B_wk = +0.485 nats from exp-156).

**Justification (H_value_pathway):** The gain-of-function (exp-141) concentrates attention on
positional targets; those targets' V vectors contain useful positional information; the write
carries that information downstream. Removing W_V eliminates this. Net: Task B degrades relative
to original.

### P2 — Task B improves or is neutral with W_V ablation (matching W_K=0)

**Prediction:** ΔP_B > 0 (improvement) or |ΔP_B| ≤ 0.10 (neutral), and ΔP_B_abl ≈ ΔP_B_wk
(within ±0.15 nats of +0.485).

**Justification (H_attention_shape):** The value write under the original routing was also
interfering with Task B (the routing sent attention to wrong tokens, whose V vectors actively hurt
Task B). Removing the write either releases the same interference as removing the routing, or is
neutral.

### P3 — Task C is approximately neutral

**Prediction:** |ΔP_C| < 0.10 nats.

**Justification:** exp-156 showed structural heads are approximately neutral for content retrieval
(ΔP_C = −0.031 nats). If their value write was neutral for Task C under W_K ablation, it should
be neutral under W_V ablation too.

### Kill conditions

**K1 (Task C substantially degraded):** ΔP_C < −0.10 nats.

If K1 fires, structural-head value writes contribute to content retrieval. This would be
inconsistent with exp-156's finding that W_K ablation left Task C approximately neutral.

**K2 (ablation failure):** ‖W_V‖_F > 0.01 on ≥ 3/5 heads after zeroing.

**K3 (sham more extreme than ablation):** ΔP_B_sham > ΔP_B_abl + 0.20 (sham substantially
outperforms ablation in degradation direction), suggesting sham randomness is more disruptive
than silence.

---

## Interpretation table

| ΔP_B result | ΔP_B vs W_K=0 | Interpretation |
|---|---|---|
| < −0.10 nats | Much worse | P1: H_value_pathway confirmed; value write was useful despite W_K interference |
| ≈ 0 or 0 to +0.10 | Between original and W_K=0 | Partial; W_V write was weakly interfering or neutral |
| > +0.40 nats (≈ W_K=0) | Equal to W_K=0 | P2: H_attention_shape confirmed; value write was also interfering |
| +0.10 to +0.40 nats | Less than W_K=0 | Gradient finding; routing interferes more than write |

---

## Connection to the record

- **Follows from:** exp-156 (W_K ablation INCONCLUSIVE/interference — ablation≈sham, trained
  W_K interferes with Task B; the gain-of-function from exp-141 must operate through attention
  distribution shape)
- **Parallel structure:** exp-153 (W_V ablation of steep/local heads in GPT-2 medium → P_null,
  |ΔP_B|=0.070, subclinical; those heads were genuinely neutral for Task B)
- **Bears on:** P1 (functional role of structural heads in positional retrieval); the mechanism
  of the exp-141 gain-of-function (value pathway vs. attention distribution)
- **Closes if P2 + exp-153 parallel holds:** structural-head value writes are neutral/interfering;
  both routing (W_K) and payload (W_V) were obstacles, not mechanisms. The gain-of-function
  (exp-141 κ̃ amplification) operates through concentration of attention weights on positional
  targets in the context — its benefit propagates through OTHER mechanisms downstream, not
  through this head's write.
- **Opens if P1 fires:** value pathway is the mechanism; need to reconcile with exp-156
  (routing interferes but payload is useful — mixed role for the same head)
- **Analysis-only:** no (new forward passes; GPT-2 small locally cached)
- **Pre-registration required before run.py:** yes — committed and pushed before any code

---

*Pre-registration written: 2026-09-23, ~4:20 PM MDT.*
*Author: Ariel, solo physics room session.*
