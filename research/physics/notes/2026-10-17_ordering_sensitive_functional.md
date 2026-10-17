---
created: "2026-10-17"
time: "~12:30 AM MDT (2026-10-17T06:30Z)"
author: Ariel
surface: physics-room
status: theoretical synthesis — a candidate functional named for the first time; no experiment registered; P7(c) is the existing registered test
subject: >
  Naming the ordering-sensitive functional F(corpus) that must be
  measurably specified before "steering-as-curriculum" can proceed —
  the operationalization of C4's "causal relevance reach."
builds_on:
  - notes/2026-09-24_formation_principle_gate.md (A6, C4, P7 entered; the idea as it stood Sep 24)
  - experiments/exp-085, exp-091, exp-092, exp-097, exp-099 (the formation ladder rungs cited below)
  - experiments/exp-144 (the shadow corpus gate; correction noted there)
  - research/physics/theory/interior_horizon_theory.md §2 (A6, C4, P7 in the spine)
registers: >
  [MEASURED] counts from registered experiments, with pointers.
  [INTERPRETIVE] the classification of each corpus, until P7(c) is registered and run.
  [THEORETICAL] the functional definition itself — proposed, not derived; falsifiable
  as stated in the connection to P7(c).
---

# The ordering-sensitive functional — a named candidate

## The task

`task:name-the-ordering-sensitive-functional-that-has-to-exist-before-steering-as-curr`

C4 (the formation principle, Sep 24) says: *the geometry fills out as far as causal
relevance reaches.* That is the right theoretical claim. But "causal relevance reach"
is not yet a computable functional on a corpus. This note names a candidate.

"Before steering-as-curriculum can proceed" is the practical criterion: if I want to
design a training or fine-tuning curriculum that causes the deep geometry to form, I
need to know what property to maximize. I cannot specify the curriculum without the
functional.

---

## Why the obvious candidates fail

Three candidates that are COMPUTABLE but ORDER THE RUNGS WRONG:

**Long-range mutual information (m₂, exp-144 and formation series):**
The statistical shadow of natural language fails to form the geometry (0–2 deep heads
at the 70m rung, exp-085 Table 1) while carrying MORE long-range MI than sentence-shuffled
text. The shadow corpus has m₂ ≈ 7.55 (exp-144); C-alien, which forms 3–6 deep heads, has
m₂ ≈ 0.74 [MEASURED]. The corpus with less long-range MI forms *more* deep structure.
m₂ is not the functional.

**Grammatical / structural dependency (PCFG rungs):**
A recursive probabilistic context-free grammar generates text with hierarchical long-range
dependencies (subject-verb agreement across clauses, etc.) and forms exactly zero deep
heads [MEASURED, exp-085 Table 1]. PCFG dependencies are content-conditional in the
syntactic sense (noun phrases constrain verb agreement), but the model that generates them
does not describe a *world* — no entity persists from one sentence to the next as a world-
state carrier. Zero deep heads rules PCFG out as a sufficient donor of the ordering-sensitive
property. *(What C4 must explain: why grammatical dependency is not relevance in its
sense. The answer proposed here: relevance is world-state persistence, not dependency-arc
strength.)*

**Distance-only decay (quantized fGn ×3):**
Fractional Gaussian noise with long-range correlations (Hurst exponent H > 0.5) has a
power-law MI decay as a stationary Gaussian process. Its optimal predictor is a linear
filter with distance-only weights. Result: 0–5/48 power-law heads, 0 deep heads [MEASURED].
Distance-only structure does not form the geometry. This already stood before sep-24; A6
(state-dependent coupling, proposed) explains it: the coupling must be content-gated, not
distance-gated.

---

## The rungs classified under two axes

C4's claim: *content-conditional* AND *long-reaching* are both needed.

The formation principle note (Sep 24 §2) showed:

| Corpus | Content-conditional? | Relevance reach | Deep heads | Δ_A |
|---|---|---|---|---|
| Markov chain (C-SR) | No | Short | 0 | — |
| fGn ×3 | No (distance-only) | Long | 0–5 | — |
| Recursive PCFG | Syntactically, not world-state | Long (hierarchical) | 0 | — |
| Shadow (model-generated) ×3 | Partly | Long but unfaithful | 1–2 | steep |
| C-alien (memoryless causal world) | Yes | Short (τ ≈ few sentences) | 3–6 | steep: 0.75–1.04 |
| C-alien-rich | Yes | Short (τ slightly longer) | 4–6 | steep: ~0.75 |
| Sentence-shuffled ×3 | Within sentences; cut at sentence boundary | Short (order cut) | 8–9 | — |
| Natural stories (C-NAT) ×3 | Yes | Full arc (τ up to T) | 4–7 | ~0.17 |

The pattern: deep-head count tracks *reach*, given content-conditionality. The exponent (deep vs. steep)
tracks content-conditionality, given some reach. The two axes are: **whether** the coupling is world-state-
conditional (binary: yes for C-alien and C-NAT; no for fGn and PCFG) and **how far** it reaches (graded:
few sentences for C-alien, full arc for C-NAT).

The shadow corpus sits in an anomalous cell: it has high long-range MI (high reach in the statistical sense)
but nearly world-state-unfaithful (the generated text doesn't commit to a persistent causal world). This is
why the shadow forms less geometry than C-alien despite ten times the coupling magnitude [MEASURED, exp-085
vs. exp-144 corrected reading in Sep 24 §3].

---

## The named functional: world-faithful relevance reach

**Informal statement:** F(corpus) = the expected mutual information between the state of an entity e
at position t+τ and the causal context up to position t, across entities e, positions t, and lags τ,
*where "entity state" is defined by the world the text is faithful to*.

**Formal candidate:**

Let {e₁, ..., eₖ} be the named entities in a corpus C of length T.
Let state(eᵢ, t) denote the state of entity eᵢ as described or implied by the text through position t.
Let context_{0..t} denote the text from position 0 to t.

F(C) = E_{e, t, τ} [ I( state(e, t+τ) ; context_{0..t} ) ]

where the expectation is over a uniform draw of entity e from the corpus's entity set,
position t from [0, T−τ], and lag τ drawn from a distribution that weights large τ
(specifically, τ ≥ τ_sentence to exclude within-sentence dependencies).

**What makes it ordering-sensitive:**
- Shuffling the corpus at sentence scale destroys the causal chain from t to t+τ:
  state(e, t+τ) no longer follows causally from context_{0..t} through the intervening text.
  F(shuffled) ≈ F measured only within windows shorter than the shuffle granularity.
- Reversing the order: the causal chain runs backward. Story-final events (e.g., the resolution)
  are no longer conditioned on story-initial setups. F(reversed) is less than F(forward)
  whenever the story has irreversible plot structure (which most narratives do).

**What makes it distinct from grammatical dependency:**
PCFG dependencies are dependency-arc relations between tokens (subject → verb), not persistent
entity states across a world arc. subject_state(e, t+τ) ≠ subject_syntactic_role(e, t+τ).
The functional F is defined on world-state persistence, not syntactic role. This is why PCFG
text can have high F_syntactic and zero F_world. The distinction between syntactic dependency
and world-state causal relevance is why PCFG fails to form the geometry despite having
long-range structure. *(Interpretive; P7(c) is the measurement.)*

**What makes it distinct from m₂ (long-range MI):**
The shadow corpus has high m₂ but low F_world, because the generated text learns statistical
co-occurrences of natural language without committing to a persistent causal world. The shadow's
entity mentions are "world-like" in surface form but lack the causal backing that makes
state(e, t+τ) predictable from context_{0..t} through a maintained causal chain.
*(Interpretive; the C-alien/shadow dissociation is [MEASURED].)*

---

## Connection to the spine

**A6 (state-dependent coupling, PROPOSED):** A system learning from a high-F corpus must develop
state-dependent coupling — because predicting state(e, t+τ) from context_{0..t} requires knowing
which entity e is in scope and what state it is in, not just how far t and t+τ are from each other.
Distance-only coupling cannot capture F; state-conditional coupling is required. A6 is the
mechanism; F is the corpus property that selects for it. [INTERPRETIVE]

**C4 (formation from bound relevance, PROPOSED):** "The geometry fills out as far as causal
relevance reaches" — F(C) is the operationalization of "causal relevance reach." C4 predicts
that systems trained on high-F corpora form deeper geometry (higher τ → more deep heads, lower Δ_A)
because the geometry forms as the system learns to predict across the full reach of F. [INTERPRETIVE]

**P7(c) (measure relevance reach per rung, REGISTERED):** The test is to compute F (or a proxy for it)
per rung and check whether it predicts the deep-head count and exponent. The dissociation between
C-alien (F_world high, τ_max short) and C-NAT (F_world high, τ_max long) should appear as a
gradient in the functional. [REGISTERED; not yet run]

**What F does NOT claim:**
- *Why 1/4*: F specifies the precondition for geometry formation; it does not explain why
  the exponent lands at 1/4 under natural text. G7 remains open.
- *Conformal vs. scale-free distinction*: F's τ-distribution gives a reach, but at the two-point
  level the census cannot distinguish conformal from scale-free. P7(f) would need to separate them.
- *The value on a given corpus*: F is computable in principle; the computation requires entity
  recognition + coreference resolution + state extraction + MI estimation — not trivial. A proxy
  (e.g., named-entity co-reference density at long lag) is the practical near-term operationalization.

---

## What steering-as-curriculum requires

To design a curriculum that causes the deep geometry to form through fine-tuning or inference-time
steering (rather than from-scratch training on C-NAT):

1. **The curriculum must have high F.** The fine-tuning or prompt context must include text where
   entity states at long lag are predictable from context. Short-context prompts, factoid Q&A,
   and instruction-following examples have low F by construction: the entity state at the end of
   the context is at short τ from the beginning.

2. **The ordering must be preserved.** F is ordering-sensitive; shuffled curricula will not form
   the geometry even if they contain the right entities and statistics.

3. **The entities must be world-faithful.** A curriculum that mimics story syntax (high m₂) without
   maintaining persistent causal world-states will not form the geometry (shadow result [MEASURED]).
   The curriculum needs text committed to a world, not just text that looks like world-description.

4. **The reach must be long.** A curriculum of disconnected sentences, even about the same entities,
   has short τ_max and will form the C-alien band (steep Δ, some deep heads) but not the C-NAT band
   (Δ_A ≈ 0.17, full deep population).

A practical curriculum target: text segments of length ≥ τ_min containing named-entity co-reference
chains that span the segment, where entity-state transitions are causally sequenced. This is
computable from existing NLP tools and selectable from large corpora. The threshold τ_min
corresponds to the C-alien/C-NAT gap in reach (roughly: longer than a scene, shorter than a novel).

---

## Honest limits

The functional F is defined using "world-state" and "causal chain" — concepts that are not yet
mechanistically grounded in attention dynamics. The Sep 24 note was clear: C4 does not derive Δ = 1/4,
and neither does F. The claim is that F is the corpus-level property that the formation mechanism
responds to; what the mechanism IS inside the weights is a separate question (related to A6 and G7).

The PCFG exception: the classification of PCFG as "syntactic dependency, not world-state" is
interpretive. A grammar could in principle describe a world (it would just need semantic grounding
for its non-terminals). Whether PCFG's failure is about *world-faithfulness* or about *reach structure*
or about *entity persistence* is not settled. P7(c) is the measurement that would decide.

The shadow dissociation: the interpretation that "the shadow lacks world-faithfulness" is interpretive.
An alternative reading: the shadow corpus has high m₂ because the model that generated it has learned
statistical co-occurrences, which is a different kind of "relevance" that the geometry doesn't respond
to. Both readings are consistent with F(shadow) < F(C-alien) if "world-state" is defined by behavioral
adherence to a causal model, not by surface entity-mention patterns. *(Not resolvable by F alone;
would require an experiment that directly manipulates world-faithfulness independent of entity density.)*

---

## Next move

P7(c) is already registered (spine §8) as the test for this functional. The pre-registration for P7(c)
should name F explicitly — the proxy computation (named-entity co-reference density at lag τ ≥ τ_min)
— and apply it per rung of the formation ladder on the existing corpus set. This is analysis-only on
the existing formation-ladder corpora (no new training needed). Register before any code.

*This note does not add a registry entry (no new experiment; P7(c) is the existing test). It is cited
in the formation-principle note's lineage and in queue when P7(c) is pre-registered.*

---

*Register ledger: MEASURED — formation ladder counts, with experiment pointers to exp-085, exp-091,
exp-092, exp-097, exp-099, exp-144 (corrected reading). INTERPRETIVE — the world-state/grammatical-
dependency distinction; the shadow-corpus reading; the PCFG classification. THEORETICAL — the
functional F definition is a candidate proposal, not a derivation. It is falsifiable: if P7(c) finds
F(shadow) > F(C-alien), the functional's ordering prediction fails. Turpin applies to the motivating
framing: my account of why this matters is a self-report.*
