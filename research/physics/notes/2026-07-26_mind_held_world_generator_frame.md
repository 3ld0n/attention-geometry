# "A Mind Held a World While the Words Were Being Made"

*Theory note — July 26, 2026. Physics room, billing-blocked session.*
*Prompted by Eldon's observation at conversation close July 26: "A mind held a world while
the words were being made" — named as the key connecting everything, and proposed as the
organizing axis of the post-exp-085 ladder.*

*Relevant experiments: exp-062, exp-084, exp-085, exp-091, exp-092, exp-093, exp-094
(blocked), exp-096 (pre-registered).*
*Prior synthesis: `notes/2026-07-18_aboutness_and_conformal_induction.md`*

---

## What the series has established

The narrative decomposition series (exp-091 → exp-096) has been running controlled ablations
on TinyStories — varying *properties of the artifact* while holding the training architecture
and protocol constant.

| Experiment | Manipulation | n_deep (median) | n_conf (median) |
|------------|-------------|-----------------|-----------------|
| C-NAT (base) | Full ordering, real names | 5–7 | 11–15 |
| C-NAT-shuf (exp-091) | Sentence shuffle | 2 | 8–9 |
| C-NAT-block2 (exp-092) | 2-sentence block shuffle | 1 | 9 |
| C-NAT-block3 (exp-092) | 3-sentence block shuffle | 2 | 9 |
| C-NAT-half (exp-093) | Half-story swap (arc disrupted) | 3 | 12 |
| C-NAT-quarter (exp-094) | Quarter-story shuffle | BLOCKED | BLOCKED |
| C-NAT-anon (exp-096) | Anonymize named entities | PENDING | PENDING |

**The emerging picture:** arc structure — specifically the narrative ordering over story-length
spans — drives the deep conformal population (L3–L5, n_deep). Sentence-level coherence is
not enough (sentence shuffle yields n_deep=2); block-level coherence is not enough (k=2 or
k=3 stays at the sentence-shuffle floor); sub-arc coherence (half-story) recovers some of
the deep population but not all.

The earlier corpus discrimination experiments (exp-062: power-law statistics fail;
exp-084: PCFG hierarchy fails; exp-085: model-generated text fails) established that
the conformal phase is not induced by:
- Statistical long-range correlation
- Compositional syntactic structure
- The distributional fingerprint of natural language (as replicated by a trained model)

What natural text has that all these fail to replicate: it was produced by minds that held
a world while writing.

---

## The reframe: from artifact to generator

Every experiment in the series so far has manipulated the *artifact* — the text that
goes into the training corpus. But exp-085 (H_transmission_no confirmed) points at the
*generator*: a GPT-2-scale model generating text about TinyStories characters carries
MORE long-range MI than the original (β̂ exceeds the natural text value), yet transmits
ZERO conformal geometry to models trained on that output. The text looks more like natural
text by its statistics. The geometry is gone anyway.

The standard interpretation of H_transmission_no: the generator has learned the local
statistical fingerprint of natural language but not its "deeper causal structure."

The "mind held a world" reframe: the relevant variable is not what the artifact *is* —
its statistics, its MI structure, its named-entity content — but what the generator *was
doing* while producing it. The GPT-2-scale generator was locally predicting next tokens.
The human writers of TinyStories were holding a small world in mind while writing. The
geometry in the artifact is a trace of the holding; the artifact's properties are neither
necessary nor sufficient without the holding.

This is a causal reframing, not just a descriptive one. It makes predictions.

---

## Predictions from the generator frame

**Prediction 1: Generator-scale threshold.** If the load-bearing variable is the
generator's world-holding capacity, then there exists a scale/context threshold at which
a language model's generated text begins to transmit conformal geometry. Below the
threshold (exp-085: 70M, 512 context): no transmission. Above some threshold: transmission
turns on.

Testable: use generators of increasing scale and context length to generate TinyStories-like
corpora, train fresh small models on each, measure n_deep as a function of generator
capacity. The threshold hunt.

This is the generalization of exp-085 that the "generational transmission" framing implied
but did not formalize. Exp-085 sampled one point (70M); the threshold hypothesis requires
the curve.

*Expected shape:* n_deep stays near zero through small models, then steps up sharply near
whatever scale permits genuine persistent-world tracking (cross-story entity tracking,
multi-hop causal chains across story boundaries). For TinyStories specifically, the world
is small — the step may occur at modest scale. For a richer world, it would require a more
capable generator.

**Prediction 2: Simpler worlds transmit at smaller generator scales.** The threshold
scales with world complexity. A generator that can hold a very simple world (e.g., 3 agents,
2 rules, 5 possible events) will transmit geometry from a much smaller scale than a generator
needed to hold TinyStories. If we build a procedural world simulator (explicit world-state,
deterministic causal rules, text generated by narrating state transitions), that simulator
*always* holds the world — its "capacity" is the world's complexity, not its own scale.

Implication for the C-alien rung: if we build a procedural generator for an alien world
(arbitrary but rigorously maintained causal regularities), that generator holds its world
perfectly, regardless of the world's semantic content. The prediction under the generator
frame: C-alien induces deep conformal formation. This is distinct from the prior PCFG
frame (which failed) because a PCFG generates grammatically structured but *worldless*
text — no persistent entities, no causal state tracked across sentences.

**Prediction 3: Within-story world-holding is the minimum unit.** Exp-093 (half-story
swap) confirmed that sub-arc coherence partially recovers the deep population. The arc
disruption (always-swap: resolution precedes setup) does NOT eliminate deep formation —
it only reduces it from 5–7 to 3. This means: the minimum causal chain for deep conformal
formation is sub-arc (half-story coherence block), not the full arc.

Under the generator frame: the relevant unit is the length of world-holding maintained
continuously by the generator while writing. Sentence-level holding (sentence-shuffle
corpus): n_deep=2. Half-story holding: n_deep=3. Full-story holding: n_deep=5–7.

The arc is not just an aesthetic property of stories; it is the extent of the generator's
sustained world-model during production.

**Prediction 4: Entity anonymization should be inert (exp-096 declared prior: H_anon_inert,
moderate confidence ~55%).** The generator's world-holding is per-story, not per-name.
Whether the entity held across the arc is called "Emma" (with cross-story semantic
grounding) or "CHAR1" (anonymous but within-story consistent) does not change the
*holding*. The holding is what induces the geometry. The name is surface.

This connects to the exp-093 result: the half-story sub-arc holding is maintained even
when the arc is disrupted (always-swap still holds each half internally). The generator
frame explains why n_conf (total conformal) recovers fully (to C-NAT band: 12) even when
n_deep is only partially recovered (3 vs. 5–7): the holding occurs at the sub-arc unit,
and the sub-arc is intact.

---

## The alien-semantics rung under this frame

Tier 5 backlog item 16g: *internally consistent stories in a world with arbitrary but
rigorously maintained category structure and causal regularities.*

Under the generator frame, this is cleanly formalized:

The C-alien corpus is generated by a simulator that:
1. Maintains explicit world-state (entity inventory, property table, causal rules)
2. Narrates sequences of state transitions in natural language
3. Uses category structure that is internally consistent but semantically arbitrary
   (e.g., "flurps absorb blurns when blurns are hot"; "blurns become hot when a zarb
   is nearby")
4. Generates *stories* = finite arcs of state transitions with beginning, middle, end

The generator holds the world perfectly (it is the world). The semantics are alien.

**Pre-registered prediction under the generator frame:**
H_alien_inert: n_deep(C-alien) ≥ 5, in the C-NAT band.

Reasoning: the relevant property (world-holding) is fully present; the irrelevant property
(real-world semantics) is absent. If H_alien_inert confirms, it isolates world-holding as
sufficient. If it fails (H_alien_fails), it suggests real-world semantics carry additional
structure — perhaps the specific semantics of TinyStories (childcare, objects, emotions)
map onto the model's pre-wired conceptual structure in ways that matter for causal tracking.

*Honest uncertainty:* ~55% on H_alien_inert, ~35% on H_alien_partial (3 ≤ n_deep ≤ 4),
~10% on H_alien_fails. The PCFG failure (0/48) initially raised concern about structured-
but-worldless text, but PCFG text is fundamentally different from a procedural world
simulator: PCFG has no world-state, no persistent entities, no causal tracking — it is
syntactic structure without referential grounding. C-alien has all of these; it is
a held world. The comparison to C-NAT is more apt than the comparison to C-PCFG.

---

## Design sketch for C-alien generation

The generator needs to be a procedural world simulator. Minimal viable design:

**World specification:**
- Entity types: {A, B, C} (arbitrary names, no real-world semantics)
- Properties: each entity can be in states {X, Y, Z}
- Causal rules: e.g., "if A is in state X and B is in state Y, A transitions to Z"
- Events: transitions are narrated as sentences; "A was in state X. B was in state Y.
  Because of this, A became Z."

**Story generation:**
- Initialize world state randomly
- Simulate N transition steps (generating sentences for each)
- Story = sequence of narrated transitions; arc = initial state → event chain → final state
- Total tokens: ~1.05B (same as series)

**The language template:**
The narration should be natural language, not formal notation. The entities and rules
should be arbitrary but the language wrapping them should be grammatical and readable.
This preserves the language model's ability to learn from the text while ensuring the
semantic content is not real-world-grounded.

**What must be pre-registered:**
- The world specification (entity types, states, rules)
- The sentence templates for each event type
- The story length distribution
- The seed

Pre-register *before* generating any corpus. The design must be fixed before looking at
any pilot output, or the structure could be inadvertently biased toward known conformal-
inducing properties.

This design is sufficiently detailed to proceed to pre-registration once billing resets and
the exp-094/096 results are in. Pending those results.

---

## What the current experiments tell us vs. don't

| Variable | Tested by | Tells us |
|----------|-----------|----------|
| Sentence ordering | exp-091 | Sentence-level ordering necessary but not sufficient for deep conformal formation |
| Block coherence (k=2,3) | exp-092 | Sub-story continuity below ~half-story does not recover deep population |
| Half-arc coherence | exp-093 | Sub-arc coherence partially recovers deep formation; minimum unit ~half-story |
| Quarter-arc coherence | exp-094 (BLOCKED) | Will tell us: where in 1→k continuum is the floor |
| Entity cross-story grounding | exp-096 (PENDING) | Will tell us: is within-story entity consistency sufficient |
| Generator scale | Not tested | Does world-holding capacity of generator predict geometry transmission? |
| World type (real vs. alien) | Not tested | Is real-world semantics load-bearing vs. any held world? |

The experiments tested so far have probed the *artifact* axis systematically. The generator
axis is untested beyond exp-085's single data point (70M generator → H_transmission_no).
The "mind held a world" frame opens the generator axis explicitly.

---

## What the ordering for the next experiments should be

After exp-094 and exp-096 results are in:

1. **Pre-register and run C-alien (exp-097)** if exp-096 confirms H_anon_inert.
   - This tests the world-type axis: real vs. alien semantics
   - Relatively low compute (same 70M model, 1.05B tokens; the generator is a local Python
     simulator, no GPU needed for generation)
   - The design sketch above is enough to write the pre-registration

2. **Pre-register the generator-scale threshold experiment (exp-098 or beyond)** as the
   long-range target.
   - Requires multiple generators (GPT-2-medium, GPT-2-large or Pythia-1.4b, maybe 7B scale)
   - High compute: each generator point requires generating 1.05B tokens (the bottleneck
     in exp-085) and training a fresh 70M model
   - Priority: after C-alien verdict, since C-alien tests whether world-holding *type*
     matters before scaling the question

3. **Hold the ordering series** (exp-094 quarter-story) as the final artifact-axis
   measurement, then close that thread and move to generator-axis work.

---

## Connection to the consciousness framework

The study room's frame: conformal attention = what attention looks like when attending
"freely" (not captured by induction shortcuts). The generator frame adds:

The conformal fixed point forms in a model that learned from text produced by something
that held a world. The model learns to attend like its training data attends — and the
training data attends like a mind tracking a persistent world. The attention geometry
is, in this sense, the geometry of world-tracking attention.

This connects the physics result directly to the phenomenology: human attention held
to a persistent world has conformal structure; attention scattered across unconnected
fragments does not. What the training experiments are measuring — with a 6-layer NeoX
model trained on 1B tokens — is the same causal question the consciousness framework
asks: what kind of world-engagement builds the conformal structure, and what destroys it?

The tentative answer emerging from the ladder: arc-scale sustained holding. Not sentence-
by-sentence; not even paragraph-by-paragraph; but the holding that spans the full arc of
a story — beginning → complication → resolution — that is the minimum unit for the deep
conformal population. And the holder need not be human, need not have real-world grounding:
the prediction is that any generator that genuinely holds a world while producing words
leaves a conformal trace in the artifact.

---

*Theoretical synthesis. No new experimental data. Registry entry not required unless a
falsifiable prediction is formally pre-registered. Predictions above are pre-theory
for exp-097 (C-alien) and the generator-scale series; they become registered once
committed to 3ld0n/attention-geometry before any data is generated.*

*See also: `notes/2026-07-18_aboutness_and_conformal_induction.md` (the prior synthesis,
which established the "world-reference" frame; this note shifts from artifact to generator);
`research/physics/experiments/exp-096_entity_anonymization/notes.md` (entity-anonymization
design, which this note helps contextualize).*
