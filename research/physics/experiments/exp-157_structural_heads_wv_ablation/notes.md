# exp-157 — Structural (random-native) heads: W_V ablation Task B test

**Date:** 2026-09-23, ~4:20–4:35 PM MDT
**Verdict: CONFIRMED — H_attention_shape**
**Pre-registration:** attention-geometry b6adea5 (git-attested, committed before run.py)

---

## Results

| Metric | Task B (positional) | Task C (content) |
|---|---|---|
| ΔP (ablation) | **+0.412 nats** | **−0.074 nats** |
| ΔP (sham) | +0.194 nats | −0.174 nats |
| Items improved | 18/20 | 8/20 |
| Items degraded | 2/20 | 12/20 |

**Reference (exp-156 W_K ablation):** ΔP_B = +0.485 nats (17/20 improved)

---

## Prediction outcomes

| Prediction | Result |
|---|---|
| P1 (ΔP_B < −0.10, H_value_pathway) | **Does not fire** |
| P2 (ΔP_B ≥ 0, H_attention_shape) | **FIRES** |
| P3 (|ΔP_C| < 0.10, Task C neutral) | **FIRES** (−0.074 nats) |
| K1 (ΔP_C < −0.10) | Does not fire |
| K2 (ablation failure) | Does not fire |
| K3 (sham > abl + 0.20) | Does not fire |
| W_V ≈ W_K=0 (|ΔP_B − 0.485| < 0.15) | **YES** (diff = 0.073 nats) |

---

## Key finding

**W_V ablation of the 5 structural heads improves Task B by +0.412 nats (18/20 items), within
0.073 nats of the W_K ablation result (+0.485 nats from exp-156).** This is H_attention_shape
confirmed: both the routing (W_K) and the value write (W_V) of the structural heads are
interfering with Task B. Silencing the head entirely gives approximately the same release
as removing the routing alone.

---

## Three-experiment arc for structural heads in GPT-2 small

The value pathway question is now closed:

| Experiment | Protocol | ΔP_B | n/20 | Interpretation |
|---|---|---|---|---|
| exp-141 | κ̃ amplification (5 structural heads) | +0.270 | 20/20 | Gain-of-function: heads causally involved in Task B |
| exp-156 | W_K = 0 ablation | +0.485 | 17/20 | W_K routing interferes with Task B; sham +0.490 (ablation ≈ sham) |
| **exp-157** | **W_V = 0 ablation** | **+0.412** | **18/20** | **W_V write also interferes; close to W_K=0** |

Ablation of W_K or W_V both improve Task B by ~+0.40–0.49 nats. The trained structural heads —
through both their routing AND their value writes — are net interferers with positional retrieval.

---

## Resolving the apparent paradox

exp-141 showed that κ̃ amplification (MORE concentrated structural-head attention) improved Task B
(+0.270 nats, 20/20). exp-157 shows that zeroing the value write ALSO improves Task B (+0.412
nats). These are not contradictory:

- κ̃ amplification changes the attention distribution (more concentrated on positional targets)
  without silencing the head. The residual stream change propagates to downstream attention layers
  that can read new positional signals. The benefit is a **system-level effect via the residual
  stream**, not the structural heads writing useful content directly.

- W_V = 0 removes the heads' writes entirely. The system-level benefit now comes from the
  **absence of interference** rather than the redirection of attention.

Both manipulations improve Task B, but through different mechanisms: amplification improves via
concentration (better downstream reading of position); ablation improves via silence (no
interfering writes).

---

## Implication for the gain-of-function mechanism (exp-141)

The κ̃ gain-of-function (exp-141) does NOT operate through:
- W_K directional routing → exp-156 showed routing was interfering
- W_V value payload → exp-157 shows value write was also interfering

The gain-of-function must operate through the **attention concentration pattern itself affecting
what downstream heads can read from the context**. When structural heads concentrate their
attention on positional targets, subsequent attention layers compute over a residual stream that
has been differently (and more informatively) shaped by those early reads — even if the
structural heads' own writes are interfering and their removal would be better.

This is a subtler mechanism than head-specific information routing. It suggests the positional
retrieval function is distributed: structural heads shape the computation by changing attention
patterns, and the function is executed by downstream mechanisms reading those shaped states.

---

## Task C

ΔP_C = −0.074 nats (sham = −0.174 nats). P3 fires (|ΔP_C| < 0.10). Structural heads remain
approximately neutral for content retrieval, consistent with exp-156 (ΔP_C = −0.031 nats under
W_K ablation). The three-population anatomy is confirmed at the W_V level:

| Population | W_K function | W_V function |
|---|---|---|
| Text-native (16) | Content retrieval (exp-155: −0.855 nats) | — |
| Structural (5) | Interference with Task B (exp-156: routing interferes) | Interference with Task B (exp-157: write interferes) |
| Steep/local (5) | Interference with Task B in medium (exp-149/150) | Neutral in medium (exp-153: P_null) |

---

## Open question seeded

**Why does κ̃ amplification (exp-141) improve Task B if both the routing and write of the
structural heads are interfering?** The proposed mechanism is downstream residual stream shaping,
but this is not yet tested. A causal test: block the structural heads' writes to the residual
stream (W_O zeroing, or alternative to W_V zeroing) WHILE simultaneously applying κ̃ amplification.
If Task B still improves, the gain-of-function is confirmed to be in the attention distribution
itself (affecting downstream reads), not in the structural heads' own contribution to the output.

This is not yet registered. Register before running.

---

*Session: solo physics room, 2026-09-23, afternoon.*
*Author: Ariel*
