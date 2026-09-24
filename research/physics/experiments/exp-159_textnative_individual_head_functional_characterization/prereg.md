# Pre-registration: exp-159
# Text-native Δ-window head individual functional characterization
# Date: 2026-09-24
# Room: Physics (solo)
# Registered BEFORE run.py is written

---

## Background

exp-155 (2026-09-23) established a double dissociation: W_K ablation of the 16 text-native
Δ-window heads in GPT-2 small degrades Task C (content-specified retrieval) by −0.855 nats
(20/20 items) and improves Task B (positional retrieval) by +0.189 nats (16/20). The W_K
structure carries the content-retrieval function. The population is identified by the random-
token census (Δ ∈ [0.20, 0.30], R² ≥ 0.90) combined with a WikiText-native census overlap
criterion (from exp-118).

What is not yet known: whether individual heads contribute equally or a subset dominates;
what tokens these heads attend to on Task C items; and whether retrieval shows sensitivity
to target position within context (lost-in-the-middle).

exp-127 (2026-08-30) found that text-native Δ-window heads have distributed key geometry
(lower λ₁/Σλ and more supra-MP eigenvalues than control heads). This is a weights-level
structural signature; it does not predict whether individual heads have specialized functional
roles at the task level.

---

## Population under study

The 16 text-native Δ-window heads in GPT-2 small: those heads appearing in both the
random-token census (Δ ∈ [0.20, 0.30], R² ≥ 0.90) and the WikiText-native census
from exp-118. Head identities will be read from exp-118's results.json at run time.

---

## Instruments and hypotheses

### Instrument 1 — Individual-head contribution to Task C

**Protocol:** Run Task C (20-item content-specified long-range retrieval battery from
exp-155) with each of the 16 text-native heads individually ablated (W_K=0, with sham
control: matched-norm random W_K). Report ΔP_C per head (ablation minus baseline).

**Primary hypothesis H_graded:**
The distribution of individual ΔP_C contributions is non-uniform. Specifically: the
coefficient of variation (CV = σ/μ) of the absolute individual contributions |ΔP_C_i|
is ≥ 0.5. Some heads carry substantially more of the Task C signal than others.

**Secondary hypothesis H_positive:**
The majority of individual ablations produce negative ΔP_C (degradation): ≥ 10 of
16 heads produce ΔP_C < −0.05 nats when their W_K is zeroed.

**Kill conditions:**
- K1: CV of |ΔP_C_i| < 0.5 — contributions are approximately equal; H_graded dead.
- K2: Fewer than 10/16 individual ablations produce ΔP_C < −0.05 — individual heads
  do not independently carry the content-retrieval function; the exp-155 result is a
  population-level emergent effect.

### Instrument 2 — Attention distribution on Task C items

**Protocol:** For each of the 16 heads, record per-head attention weights during Task C
forward passes (no ablation; baseline model). For each item, extract the attention weight
on the target token (the semantically specified entity). Report:
- Mean attention weight on target token per head, averaged across items
- Spearman correlation (per head) between attention-on-target and item-level log-prob
  improvement under W_K ablation (Instrument 1 item-level results)

**Hypothesis H_content_selective:**
High-contributing heads (those with larger |ΔP_C_i| in Instrument 1) attend disproportionately
to the target token: Spearman ρ between individual head contribution |ΔP_C_i| and mean
attention-on-target is ≥ 0.40 across the 16 heads.

**Kill condition:**
- K3: ρ < 0.40 between individual contribution and attention-on-target — attending to the
  target token does not predict which heads carry the content-retrieval function.

### Instrument 3 — Target-position sensitivity (lost-in-the-middle probe)

**Protocol:** Construct a variant of Task C items where the target entity appears at three
positions within a fixed-length context: beginning (first 20% of context), middle (40–60%),
end (last 20%). Use a subset of 12 items for which position can be varied while keeping
context length constant (items where the surrounding text can be redistributed). Run baseline
forward passes; compare log-prob of correct retrieval across positions.

**Hypothesis H_lim (lost-in-the-middle):**
Retrieval log-probability is lower when the target appears in the middle of context vs.
beginning or end. Specifically: mean log-prob(middle) < mean log-prob(beginning) and
mean log-prob(middle) < mean log-prob(end), with |difference| ≥ 0.15 nats.

**Kill condition:**
- K4: No significant position sensitivity: all three mean log-probs within 0.10 nats of
  each other — these heads are position-uniform retrievers.

Note: If H_lim fires (K4 does not), a follow-up question (not registered here) is whether
the position sensitivity is carried by specific heads in the population (bridge to Instrument 1).

---

## Stopping rule

All three instruments run in a single session. If the Task C battery used in Instrument 1
reveals that fewer than 5 items produce any head-level variation (i.e., the battery lacks
resolution), revise the item count upward to 30 before proceeding. This revision is
pre-registered here.

---

## Relation to prior work

- exp-118 (2026-08-11): identifies the 16 text-native Δ-window heads in GPT-2 small
- exp-127 (2026-08-30): weights-level structural signature (distributed key geometry)
- exp-155 (2026-09-23): population-level Task C double dissociation (W_K ablation)
- This experiment (exp-159): individual-level functional characterization within the population

---

## What would change if results surprise

If K1 fires (equal contributions): the population functions as an ensemble without
dominant members; the interpretation of exp-127's distributed key geometry gains
functional support.

If K2 fires (few individual ablations degrade Task C): the content-retrieval function
is not modular across the 16 heads; it is an emergent property of the full population
operating together — important revision to the three-population anatomy.

If K3 fires (attention-on-target does not predict contribution): attending to the target
is not the mechanism; alternative — attending to context positions that co-predict the
target.

If H_lim fires (K4 does not): these heads are sensitive to where in context the target
appears, linking the text-native population to the classical lost-in-the-middle phenomenon.

---

*Pre-registration complete. run.py to be written after this file is committed and pushed.*
