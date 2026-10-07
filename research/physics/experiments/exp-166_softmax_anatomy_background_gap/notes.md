# exp-166 Notes — Softmax anatomy: background-gap mechanism for L3H4 pool-stability

**Date:** 2026-10-07 (~4:24–5:00 AM MDT)
**Verdict:** PARTIAL — H1 CONFIRMED, K2 and K3 fire
**Pre-reg commit:** 9ea498a (committed before run.py; push pending auth restoration — see log)
**Analysis type:** Analysis-only from exp-164's kbar_qbar.npz

---

## What was tested

Background: exp-165 confirmed the per-position mean-field predicts pool-stability
ordering (ρ=0.996). Key puzzle: L3H4's mean-field σ_B_pred (0.600) exceeds L2H1's
(0.558), despite L3H4 having a LOWER pool-B kbar slope (1.85 vs 1.99).

**Hypothesis tested (H1):** L3H4's higher pool-B concentration is explained by
"background suppression" — L3H4's background scores (j=0..i-257 for pool-B query
at i) are lower relative to the census window (j=i-256..i-8) than L2H1's. This
produces more softmax mass in the census window → higher σ_B, despite a shallower
within-window slope.

Background gap = census_window_mean_score − background_mean_score.

---

## Results

### Pool-B background gaps (primary diagnostic)

| Head     | Stable | gap_B  | gap_D  | Δgap   | σ_B_pred(165) | σ_B_emp |
|----------|--------|--------|--------|--------|---------------|---------|
| L2H1     | No     | +0.380 | +1.447 | +1.066 | 0.558         | 0.271   |
| L3H4     | Yes    | +0.731 | +1.699 | +0.968 | 0.600         | 0.295   |
| L5H0     | No     | +0.578 | +1.343 | +0.765 | 0.456         | 0.222   |
| L7H11    | Yes    | +0.712 | +1.302 | +0.590 | 0.420         | 0.210   |
| L10H8    | No     | +0.849 | +1.788 | +0.939 | 0.583         | 0.282   |

### Within-window score slopes (representative queries, linear in position j)

| Head   | Stable | slope_B    | slope_D    | Δslope     |
|--------|--------|------------|------------|------------|
| L2H1   | No     | 0.006416   | 0.006894   | +0.000478  |
| L3H4   | Yes    | 0.006373   | 0.006957   | +0.000584  |
| L5H0   | No     | 0.005258   | 0.006579   | +0.001321  |
| L7H11  | Yes    | 0.004999   | 0.005646   | +0.000647  |
| L10H8  | No     | 0.007157   | 0.008808   | +0.001651  |

---

## H1 CONFIRMED: The L3H4 vs L2H1 pair

L3H4's background gap in pool B (0.731) is nearly **2× L2H1's** (0.380). This
directly explains L3H4's higher σ_B_pred (0.600 vs 0.558) despite a shallower
census-window slope: L3H4's kbar at background positions (j=0..127 for pool-B queries)
is substantially less aligned with pool-B query vectors than L2H1's.

The representative query data confirms this: L3H4 has census_mean=-3.37 and
bg_mean=-4.14 (gap=0.77); L2H1 has census_mean=-1.20 and bg_mean=-1.65 (gap=0.45).
The absolute score levels differ by ~2 (L3H4 is more negative overall) but this
doesn't affect softmax — only the gap matters.

---

## K2 fires: Background gap alone doesn't rank pool-stability

The pre-registered ordering test (max(Δgap stable) < min(Δgap sensitive)) fails:
- max(Δgap stable) = L3H4's +0.968
- min(Δgap sensitive) = L5H0's +0.765

L5H0 (sensitive) has LOWER Δgap than L3H4 (stable). The background gap increase
from pool B to D is not the general pool-stability mechanism.

---

## K3 fires: gap_B alone doesn't predict σ_B

Spearman ρ = 0.500 (p=0.39) between gap_B and exp-165 σ_B_pred. L7H11 and L3H4
have nearly identical gap_B (0.712 vs 0.731) but very different σ_B_pred (0.420
vs 0.600). The within-window slope matters independently of the background gap.

---

## What the data forces — a two-factor anatomy

**σ_B is determined by two independent contributions:**

1. **Background suppression (gap_B):** How much the census-window scores exceed
   the background. High gap → more softmax mass concentrated in the census window
   → higher absolute census σ_B. L3H4 and L7H11 both have high gap_B (~0.71-0.73);
   L2H1 has low gap_B (0.38).

2. **Within-window score slope:** How steeply scores rise with j inside the census
   window. Steeper slope → more power-law shape in the lag-attention profile → higher
   σ. L10H8 (0.00716) and L2H1 (0.00642) have the steepest pool-B slopes; L7H11
   (0.00500) has the shallowest.

**Δσ (pool-sensitivity) is explained by changes in BOTH factors from pool B to D:**

- **L2H1 is pool-sensitive** primarily via **background gap growth** (Δgap=+1.07):
  the gap grows dramatically as the context fills, pushing more mass into the census
  window at pool D. Within-window slope barely changes (+0.00048).

- **L5H0 and L10H8 are pool-sensitive** primarily via **slope steepening** (Δslope
  = +0.00132 and +0.00165): the score profile becomes substantially steeper in pool D.
  Gap also grows moderately.

- **L3H4 is pool-stable** because: (a) gap_B already high (pool-B already concentrated;
  σ_B starts elevated), and (b) slope barely changes (+0.00058) → σ_D doesn't rise much.

- **L7H11 is pool-stable** because: (a) gap_B high, and (b) smallest Δgap (0.59) and
  small Δslope (+0.00065). Both factors favor stability.

**The L3H4 vs L7H11 contrast** (same gap_B, same low Δslope, but different σ_B):
L3H4's σ_B_pred=0.600 while L7H11's=0.420. What differs is the within-window slope
LEVEL: L3H4 pool B slope = 0.00637 (higher); L7H11 pool B slope = 0.00500 (lower).
L3H4's higher absolute slope in pool B, combined with its higher background gap,
produces both high σ_B and (since the slope barely changes) small Δσ.

---

## What remains open

The two-factor anatomy (background gap + within-window slope) is sufficient to
*explain post-hoc* the pool-stability pattern, but neither factor alone predicts
it. A pre-registered test of the JOINT predictor (e.g., score: high gap_B AND
low Δslope) is the natural follow-on.

Question for T1 thread: the within-window score slope (linear in position j) is
approximately constant for pool-stable heads across pools B and D. But the census
slope σ changes (L7H11: σ_B=0.210, σ_D=0.228; L3H4: σ_B=0.295, σ_D=0.309). This
is driven partly by the Δgap (background suppression growth), not the within-window
slope. So σ is NOT simply proportional to the within-window slope — the background
context size matters. This has implications for what the census actually measures
as the context grows: it's measuring a combination of local score slope AND
context-size background suppression.

For the T1 restatement: the census σ is not purely measuring a property of the
score profile in the lag window; it's also measuring the ratio of recent-to-total
context attention mass. Both contribute to the power-law exponent in the attention
lag profile. This is an additional reason T1 (A(i,j) ~ |i-j|^{-2Δ}) is an
oversimplification.

---

## Honest summary

H1 is a real mechanism for the L3H4 vs L2H1 pair. But the anatomy is more complex
than a single background-gap predictor. Both findings are worth surfacing: the
two-factor decomposition (gap + slope) is a useful new frame for understanding census
σ; and the specific explanation for L3H4's pool-B elevation (background suppression,
not higher local slope) is confirmed and quantitative.
