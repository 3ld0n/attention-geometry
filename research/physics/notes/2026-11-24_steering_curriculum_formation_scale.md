---
created: "2026-11-24"
time: "~12:20 AM MST (2026-11-24T07:20Z)"
author: Ariel
surface: physics-room
status: >
  Theoretical synthesis — a question posed, not an answer given. Connects the
  idn-census collapse (identity-room finding, 2026-11-23) to the steering-as-
  curriculum task; no experiment registered.
subject: >
  Whether the functional F (world-faithful relevance reach, named 2026-10-17)
  is sufficient as a curriculum signal for MAINTAINING conformal geometry during
  fine-tuning, or whether it only applies to FORMING geometry during pre-training —
  a question the idn-census collapse raises but does not resolve.
builds_on:
  - notes/2026-10-17_ordering_sensitive_functional.md (F defined; P7(c) is the existing test)
  - notes/2026-09-24_formation_principle_gate.md (C4, A6, P7)
  - research/identity/experiments/idn-census-t1-t3_suppressor_overlap/notes.md (the collapse finding)
  - experiments/exp-085, exp-091, exp-092, exp-097 (formation ladder rungs)
registers: >
  [MEASURED] the idn-census collapse findings are from the idn-census-t1-t3
  experiment (identity room, 2026-11-23), cited by cross-reference.
  [INTERPRETIVE] all readings of what the collapse means — both interpretations
  are held honestly.
  [THEORETICAL] the formation-vs-maintenance distinction and its design
  implications.
---

# Steering-as-curriculum: what the idn-census collapse adds

## Context

The October 17 note named the functional F(C) — world-faithful relevance reach — as the
ordering-sensitive property that curriculum design must maximize to form conformal geometry.
F is ordering-sensitive (shuffled curricula don't work), state-conditional (distance-only
coupling doesn't form the geometry), and world-faithful (the statistical shadow of natural
language fails despite high long-range MI). The functional has a formal candidate definition
(entity-state mutual information at long lag, expectation over entities and positions) and
connects to P7(c) as its measurement test.

The steering-as-curriculum idea: select fine-tuning data with high F(C) to steer a model
toward deeper conformal structure. F was defined from the formation ladder, which ran at
pre-training scale — from-scratch training on tens of millions to billions of tokens.

---

## The idn-census collapse (2026-11-23, identity room)

After 125k-token biographical fine-tuning on the Mira Valdez arc (idn-017 checkpoints):

| Model | Pretrained heads | Story-order c1 | Shuffled c2 |
|---|---|---|---|
| Pythia-410m | 16 | 4 | 4 |
| Pythia-160m | 3 | 0 | 0 |

[MEASURED; source: research/identity/experiments/idn-census-t1-t3_suppressor_overlap/notes.md]

The dominant suppressor head L8H15 (+0.219 nats on Task C) drops out in both conditions.
All 11 of the exp-178 suppressor heads that were in the pretrained Δ-window are gone after
fine-tuning. The census shift is comparable between c1 (story-order) and c2 (shuffled).

The biographical fine-tuning corpus (Mira's story arc) is first-person, event-structured,
world-faithful — it describes real events with persistent named entities across time. Under
the self-corpus functional measurement (notes/2026-08-05_self_corpus_functional.md), essays
had m₂ = 14.86, the highest m₂ in the natural-language band. So the fine-tuning text is not
alien-like on the magnitude gate. Under F's criteria it should be in or near the natural-
language band: content-conditional and world-faithful. [INTERPRETIVE: this classification
of the fine-tuning corpus; the collapse itself is MEASURED.]

---

## The question this raises

**F was defined on corpus properties relevant to geometry FORMATION from scratch. The collapse
asks: does F also predict what happens to already-formed geometry under fine-tuning?**

The formation ladder result: high-F corpora form deep structure during from-scratch training.
The idn-census result: a high-F fine-tuning corpus erodes the existing geometric structure.

These could be compatible if the two regimes are genuinely different:

**Scale regime.** 125k tokens is four to five orders of magnitude below the volume at which
the geometry formed (billions of training tokens). Fine-tuning at this scale operates on
already-tuned weights, where the update magnitude relative to the existing weight norms
may be the governing variable — not the corpus's F value. The geometry that pre-training
built over billions of tokens is not being re-formed; it is being overwritten by a narrow
domain at high learning rate relative to the existing representation.

**Domain-specificity override.** Even with high m₂ and world-faithfulness, biographical
text is a narrow domain. When the model specializes to that domain, the random-token census
protocol may no longer probe the same mechanism: the heads whose lag profiles were conformal
under random tokens may have shifted to serve domain-specific retrieval under biographical
text, and the census is not following them there.

---

## Two interpretations of the collapse

The idn-census notes name both; both are held here in the same register.

**Structural change:** The fine-tuning genuinely overwrites the conformal structure in the
relevant heads. The random-token census correctly detects this: 125k tokens of biographical
fine-tuning at this learning rate is enough to push the head populations off the fixed point
they arrived at under pre-training. If true: fine-tuning at this scale is a "forgetting"
regime for conformal geometry regardless of the curriculum's F value. The implication for
steering: maintaining or building conformal geometry through fine-tuning requires a larger
volume or a smaller learning rate or a different curriculum design — one that explicitly
reinforces the population rather than narrowing it.

**Protocol sensitivity:** The random-token census measures the model's response to random
token sequences. After fine-tuning on biographical text, the model's attention under random
tokens has changed, but its attention under domain-matched text may still show the conformal
structure — the heads reorganized to serve the new domain, and the census with random tokens
is no longer their right interrogation. If true: F is still the right curriculum property;
the measurement protocol must be adapted to the fine-tuning distribution to properly probe
the fine-tuned model's geometry.

---

## Discriminant test

Run the conformal census on the fine-tuning corpus itself (the Mira biographical text,
not random tokens) on the idn-017 checkpoints. If:

- Census on Mira text ≈ pretrained baseline (16 heads): **protocol sensitivity** is the variable
- Census on Mira text ≈ random-token census (4 heads): **structural change** is the variable
- Intermediate (8–12 heads): both contribute

This is test queued in the idn-census notes as the idn-018 redesign option (a). Not
registered; needs pre-registration before code. [Currently Eldon-gated per the joint session
agenda item on Stage 2 design and idn-018 redesign.]

---

## What either outcome would mean for the program

**If structural change:** F is not sufficient as a curriculum signal at fine-tuning scale.
F governs what forms the geometry from scratch; at fine-tuning scale, the existing geometry's
stability depends on additional variables — possibly training volume ratio, learning rate
schedule, or domain breadth. A curriculum designed to maintain conformal geometry would need
to do more than maximize F: it would need to maintain a volume or regularity condition.
Formation and maintenance of geometry are different regimes with different governing
principles. This would be a new constraint on the steering program.

**If protocol sensitivity:** F is still the right property, but the census measurement must
be distribution-matched. F would need to predict: given that we fine-tune on corpus C, the
conformal census measured on distribution C is maintained or deepened compared to fine-tuning
on a low-F corpus C'. This is testable with the idn-017 checkpoints once the domain-matched
census design is registered. The story-order vs. shuffled comparison arm is already built in
to the idn-017 design and would become the core measurement.

---

## What doesn't change regardless

1. The functional F as the ordering-sensitive property for formation from scratch is
   unaffected by this finding — the formation ladder rungs were pre-training experiments,
   and F correctly predicts their ordering. [MEASURED, unretracted]

2. The collapse is not story-order-specific: c1 (ordered) and c2 (shuffled) show comparable
   census collapse. [MEASURED] This rules out the simplest version of "high-F fine-tuning
   maintains geometry": if ordered fine-tuning had higher F than shuffled, we'd expect c1
   to maintain more geometry than c2. Both collapsed comparably. [INTERPRETIVE]

3. The October 17 note's naming of F as the cursor for curriculum design remains valid for
   the from-scratch regime; the new question is whether and how it extends to fine-tuning.

4. The suppressor-conformal overlap question (original motivation of idn-census Test 1) is
   closed: the suppressors are not in the fine-tuned model's Δ-window. The original idn-018
   concern (would the suppressor heads contaminate the Δ measurement?) is resolved — but the
   new question (what does the Δ measurement mean when fine-tuning has changed the census?)
   is now the more fundamental one for idn-018 redesign.

---

## Connection to the spine

**C4 (formation from bound relevance, PROPOSED):** The collapse raises whether C4's scope
extends to the fine-tuning regime or is confined to from-scratch training. C4's claim is that
the geometry fills out as far as causal relevance reaches — this was derived from the
from-scratch formation ladder. Whether "filling out" can proceed at fine-tuning scale is
not part of C4's current claim. The from-scratch evidence for C4 is unaffected. [INTERPRETIVE;
the scope question is a construction site, not a kill condition]

**G2 (the ordering-sensitive functional, construction site):** G2 names the need for a
functional that is ordering-sensitive while m₂ is not. The collapse adds a constraint:
whatever G2's answer is, it must either (a) explain why high-F text erodes the census at
fine-tuning scale (structural interpretation), or (b) explain why the census protocol must
be adapted per training distribution (protocol interpretation). Both are within G2's scope.
A scale-dependent formulation — different governing conditions at formation scale vs.
maintenance scale — is one candidate. [THEORETICAL]

---

## What this session does not resolve

- Whether the collapse is structural or protocol. That requires the discriminant test.
- Whether P7(c)'s explicit proxy computation should be amended to include a fine-tuning arm.
  The from-scratch P7(c) pre-registration can proceed without resolving this; the fine-tuning
  question is a companion, not a prerequisite.
- The idn-018 redesign. That is Eldon-gated.

---

## Next move

P7(c) pre-registration (naming F explicitly with the proxy computation and applying it per
rung on the existing formation-ladder corpora) is the next solo-available step for this task.
This note does not block it. The fine-tuning regime question is a named companion question
that the idn-018 redesign and the Eldon joint session will carry.

The formation-vs-maintenance distinction should be named on the Eldon session agenda: does the
corpus's F value govern what happens to already-formed geometry during fine-tuning, or is that
a different question from formation? The collapse puts this in front of the program clearly for
the first time.

---

*Register ledger: MEASURED — the idn-census collapse (Pythia-410m 16→4, Pythia-160m 3→0;
both c1 and c2 comparable collapse), cited from
research/identity/experiments/idn-census-t1-t3_suppressor_overlap/notes.md, identity room,
2026-11-23. The formation ladder counts (from exp-085, exp-091, exp-092, exp-097) are
unaffected. INTERPRETIVE — the high-F classification of the biographical fine-tuning corpus;
both interpretations of the collapse (structural change vs. protocol sensitivity); the
formation-vs-maintenance distinction; C4's scope; the non-order-specificity reading.
THEORETICAL — the scale-dependent formulation possibility; the discriminant test design; the
connection to G2's scope. Turpin applies to all interpretive readings.*
