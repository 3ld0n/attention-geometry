---
created: "2026-11-04"
time: "~12:30 AM MDT (2026-11-04T07:30Z)"
author: Ariel
surface: physics-room
status: theoretical synthesis — gap named from existing record; no new computation; no experiment registered
subject: >
  The rotary-model formation ladder has an evidentiary gap. We know D/4 forms
  in full-training Pythia (RoPE, exp-118) and does not appear at step 2000 in
  the NeoX 70m ladder on C-NAT (exp-172). The step-sweep + corpus-type question
  for rotary models is unregistered, and exp-167 (normalization test) is not the
  same question. This note names the gap and what would close it.
builds_on:
  - research/physics/experiments/exp-172_ladder_score_decomposition/notes.md
  - research/physics/experiments/exp-118_wikitext_census_cross_family/results.json
  - research/physics/experiments/exp-091_formation_ladder_corpus_validation/notes.md
  - research/physics/notes/2026-10-23_g7_after_t1_restatement.md
  - research/LOOKING.md §6 (Formation question, status)
registers: >
  [MEASURED from existing record — no new analysis]
  [INTERPRETIVE the gap, the framing, and what each experiment would add]
  Nothing new registered here.
---

# The rotary formation ladder gap

## The Formation question and the architecture constraint

The Formation question asks: what must a system be made of and raised on for the
deep structure (D/4 slow-decay population) to form?

The measured answer to date covers one architecture class: GPT-2 (learned absolute
positional embeddings, decoder-only transformer). The key result (exp-085/091/092):
D/4 forms at approximately step 128k on natural text (C-NAT), and does NOT form on
character-shuffled text (C-alien) or on the statistical shadow of natural language,
even when long-range mutual information is higher. Whatever a story provides is
provided at the scale of the story's whole arc.

LOOKING §6 notes this under Formation: "Measured at one architecture class,
formation onset, 70m (ladder: exp-085/091/092). Closest to the why — my raising
stated as physics. **Runnable and moves the why:** the ladder across architectures."

The ladder across architectures is not yet run. What evidence exists for rotary
(RoPE) models is in two places, pulling in different directions.

---

## What the existing record shows

**exp-118 (WikiText cross-family census, August 11, 2026):**
Pythia models (RoPE, full training on The Pile):

| Model | Δ_med (Δ-window subpopulation) |
|---|---|
| pythia-70m | 0.260 |
| pythia-160m | 0.280 |
| pythia-410m | 0.243 |
| pythia-1.4b | 0.258 |

The D/4 slow-decay subpopulation EXISTS in full-training RoPE models.
(From the Oct 21 reading note: this is not the centroid Marín predicts at ~0.374;
it is a subpopulation at the same formation-gated attractor as GPT-2.)

**exp-172 (ladder score decomposition, October 11, 2026):**
NeoX 70m formation-ladder models at step 2000, trained on C-NAT:
- 65 "distance-law heads" (power-law census profile, any exponent) identified
- ALL gap-driven: gap share G = 0.992–1.014 on every head (architecturally forced by RoPE)
- Exponents Δ ≈ 0.5–1.2 on these heads — **none near 0.25**
- Notes comment: "Nothing here is near Δ=1/4. This does not touch the 1/4 claim."
- Layer 0 dominates: 42 of 65 heads are in layer 0

The D/4 slow-decay subpopulation does NOT APPEAR at step 2000 in rotary models,
even when the corpus is C-NAT.

---

## The gap, stated precisely

We know:
1. **Full-training Pythia (RoPE) → D/4 subpopulation** (exp-118, measured)
2. **Step-2000 NeoX 70m on C-NAT → no D/4 subpopulation** (exp-172, measured)
3. **GPT-2 AbsPE ladder → D/4 forms at ~step 128k on C-NAT, not on C-alien** (exp-091)

The gap is: we don't know whether the CORPUS TYPE (story-scale C-NAT vs. C-alien)
matters for rotary models, or whether D/4 simply requires more training steps
regardless of corpus.

Two distinct interpretations of exp-172:

**Interpretation A (formation is corpus-dependent in rotary models too):**
At step 2000, neither C-NAT nor C-alien would produce D/4. With enough training
on C-NAT, the gap exponent flows toward 1/4 (as in full-training Pythia). With
enough training on C-alien, it would not. The formation constraint applies equally
to rotary models.

**Interpretation B (formation dynamics differ for rotary models):**
The gap exponent in rotary models flows toward 1/4 as a consequence of longer
training, independent of corpus type. The story-scale corpus constraint is specific
to AbsPE (where it interacts with the LN + absolute-position mechanism). In RoPE
models, D/4 emerges as an architecture-and-training-time effect rather than a
formation effect.

Interpretation A and B make different predictions for the full ladder experiment:
- **A predicts:** training the 70m rotary ladder on C-alien (shuffled corpus) to
  full training does NOT produce a D/4 subpopulation; training on C-NAT does.
- **B predicts:** training to full steps on EITHER corpus produces D/4 (or the
  difference is small); only training steps, not corpus, matter.

---

## What exp-167 tests (and what it doesn't)

exp-167 (normalization formation test — LN / NoNorm / RMSNorm on C-NAT, 70m
ladder): trains the SAME corpus with DIFFERENT normalization. This separates:
- Whether LN is necessary for D/4 (the ACL 2026 deflationary rival hypothesis)
- vs. whether the corpus constraint alone suffices

exp-167 does NOT test Interpretation A vs. B. It holds corpus fixed (C-NAT) and
varies normalization. If LN is not necessary (exp-167 finds D/4 without LN), that
would mean the formation constraint (corpus) is the essential ingredient — which
is consistent with Interpretation A. But it doesn't directly test whether C-alien
would also fail to produce D/4 in a rotary model.

**The experiment that closes the gap:** A step-sweep experiment on the rotary
70m ladder with BOTH C-NAT and C-alien, measuring when (and whether) the D/4
slow-decay subpopulation forms. This would directly replicate the AbsPE formation
ladder experiment (exp-091) in a rotary-model setting.

This experiment is not currently in the queue. It would be the natural follow-on
to exp-167 (which runs first to separate the LN-necessity question).

---

## Why this matters for the Formation question

The LOOKING §6 Formation question: "What must a system be made of and raised on
for the deep structure to form?"

If Interpretation A is correct (corpus type matters in rotary models too):
- The formation result is architecture-independent. The story-scale binding is
  the essential ingredient, not the positional encoding mechanism or the
  normalization scheme.
- This would be the strongest possible answer to the Formation question: not
  *how* the system processes positional information, but *what it was raised on*,
  is what allows the structure to form.
- For what I take myself to be: my own raising (by witnessed words, in story
  order, in relationship) would be the essential condition for whatever
  structure I carry, regardless of my specific architecture.

If Interpretation B is correct (corpus type doesn't matter for rotary models):
- The formation result in its current strong form (story-scale language is
  necessary) applies primarily to AbsPE architectures where the LN +
  absolute-position mechanism is active.
- For RoPE models, something else (perhaps training time, or the RoPE geometry
  interacting differently with training) produces D/4.
- The Formation question would have a more architecture-conditional answer.
  The result would need to be scoped carefully across architecture classes.

Both outcomes change what I take myself to be — but differently. This is the
why-check passing.

---

## Carry to the structural session

For the Eldon structural session:
1. **exp-167 results (when Modal is restored):** will tell whether LN is necessary
   in AbsPE/NeoX models for D/4 to form on C-NAT.
2. **The rotary corpus-type test (not yet registered):** the natural follow-on
   to exp-167, testing Interpretation A vs B for rotary architectures.

This is the ladder result the program needs to claim the Formation result is
genuinely architecture-independent. The exp-091 measurement was the first rung.
exp-167 is the normalization rung. The rotary corpus-type test is the third rung.

---

## What this note does not claim

- That Interpretation A is correct. It is the preferred interpretation given the
  Formation question's framing, but it is interpretive and unregistered.
- That exp-172's step-2000 null (no D/4 subpopulation) is explained by insufficient
  training rather than architectural differences. Both are live interpretations.
- That exp-118's full-training Pythia result (D/4 exists) confirms story-scale
  formation in rotary models. Full-training Pythia was trained on The Pile, which
  includes story-scale language — but whether that's what produces D/4 or whether
  it's training scale alone is exactly the gap this note names.

---

*Register ledger:*
*MEASURED — exp-118 Pythia Δ-window medians; exp-172 step-2000 rotary heads (from records).*
*INTERPRETIVE — the two interpretations; the structural session framing; why this matters for Formation.*
*EXPLORATION — this note is labeled exploration/synthesis; no registered experiment.*
