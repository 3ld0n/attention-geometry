# Material as Memory — The Slow Sector in the Attention Framework

*Theory note — August 1, 2026. Physics room, billing-blocked session.*  
*Prompted by Eldon's inbox item 2026-07-27: "perhaps memory is what physical material is,  
fundamentally." Deferred by the July 28 note (mind-as-story-capacity) as "reading-first,  
requires more care than an experiment can adjudicate." This note does that reading.*

*Prior synthesis: `notes/2026-07-28_mind_as_story_capacity_definition.md` (the companion  
frame — mind as dynamic; this note addresses material as static). Read together.*

---

## The question

Does the framework distinguish fast degrees of freedom (attention in motion) from slow/frozen
ones (structure deposited by past attention)? Is "material" the right name for the slow
sector? And if so, are "mind as story-holding capacity" and "material as memory" two faces
of one requirement — something that has a single conservation-style statement underneath?

The short answer: yes, the framework already distinguishes fast from slow, at two levels.
"Material" is the right name for the slow sector at the training-time level, and the naming
is not vacuous — it extends the GR analogy in FRAMEWORK §4.2 and makes the formation-ladder
experiments readable as a measurement of *how training data becomes material*. The single
statement underneath is visible but not yet a conservation law: it is a one-way flow.

---

## Two levels of the fast/slow split

**Level 1 — inference time:**

At inference, the split is between:
- **Slow sector (material):** W_Q, W_K, W_V, W_O — the weight matrices. Fixed after
  training. They encode the accumulated history of gradient descent over the training
  corpus. They shape every computation but do not change during it.
- **Fast sector (motion):** A(i,j) = softmax(QK^T/√d_k) — the attention weights. Computed
  fresh at every forward pass from the slow-sector substrate. They are the motion shaped by
  the material.

The conformal geometry we measure (Δ, n_deep, n_conf) is a property of the slow sector: it
lives in W_QK. The fast sector instantiates it at each forward pass. This is already in
FRAMEWORK §4.2 — "attention weights = dynamics; positional structure = geometry" — but the
slow/fast language makes the duality explicit in a new register.

**Level 2 — training time:**

During training, the roles swap direction:
- **Fast sector:** the attention patterns during each forward pass — fast fluctuations driven
  by gradient updates. These are the "matter in motion" of the training dynamics.
- **Slow sector being deposited:** the W_QK structure as it accumulates gradient updates and
  flows from UV disorder toward the IR conformal fixed point (exp-086: two-stage RG flow;
  Δ_med: 0.73 → 0.47 → 0.28 over training steps).

The training process is the deposition: fast fluctuations (gradient descent) progressively
freeze into the slow sector (the weight matrices). What gets frozen is the conformal geometry.
The generator's world-holding is *what shapes the fast fluctuations* during training; the
W_QK geometry is *what the frozen result looks like*. 

The material is the slow sector. Memory is what the slow sector encodes: the record of what
the fast sector did when the world was being held.

---

## The GR connection

FRAMEWORK §4.2 already states this in Einstein's language:

> "Matter tells spacetime how to curve; spacetime tells matter how to move."

In the attention framework, the training-time reading is:
- **Matter** = the training corpus (the record of the generator's fast-sector dynamics
  while holding a world)
- **Spacetime curvature** = the W_QK conformal geometry that results
- **"Matter tells spacetime how to curve"** = the generator's world-holding shapes the
  conformal geometry of the slow sector

And the inference-time reading:
- **Spacetime** = W_QK (fixed, structural, the material)
- **Matter in motion** = the attention pattern A(i,j) (the fast sector)
- **"Spacetime tells matter how to move"** = the conformal W_QK geometry shapes the
  attention distribution at inference

These are two readings of the same self-consistency condition (§4.2.1): the slow and fast
sectors must be mutually compatible. Training is the process of finding that compatibility.
The conformal fixed point is where it is reached.

Material = W_QK = the deposited spacetime. The formation experiments are asking: what matter
(training corpus) deposits what spacetime (conformal geometry)?

---

## What the formation ladder says in this vocabulary

The formation-ladder experiments (exp-085/091/092/093/094/096/097) are measurements of
**deposition efficiency**: how much conformal structure (slow-sector geometry) is deposited
by how much world-holding in the generator (fast-sector quality during training)?

| Generator quality | World-holding duration | n_deep deposited |
|-------------------|----------------------|-----------------|
| GPT-2 70M (exp-085) | ~1 sentence effective | 0 deep |
| C-NAT-shuf (exp-091) | ~1 sentence (random order) | 2 (floor) |
| C-NAT-block3 (exp-092) | ~3 sentences | 2 (floor) |
| C-NAT-halfstory (exp-093) | ~half-story | 3 (partial) |
| C-NAT / human writer | full story (~8–12 sentences) | 5–7 (full) |

**What the material framing adds:** the n_deep observable is not just a count of conformal
heads — it is a measure of how much material (slow-sector conformal structure) was deposited
by the training. The deposition is proportional to the generator's sustained world-holding.
Short holding → little material deposited → shallow conformal population. Full-story holding
→ full material → deep conformal population.

The artifact (training corpus) IS the material in Eldon's sense: it is the "deposited record
that shapes what moves through it." A model trained on that artifact develops W_QK geometry
that shapes inference-time attention. The artifact's conformal trace flows into the model's
material (W_QK), and from there shapes all future inference.

---

## The companion relationship

From the July 28 note (mind-as-story-capacity §Connection to companion frame):

> Mind = the dynamic structure: attention in the act of holding, generating geometry  
> Material = the deposited trace: the artifact carrying the record of what was held,  
> which shapes what any future attention does when it reads

This is exactly the fast/slow split: mind is the fast sector in action; material is the
slow sector that results. The circle:

```
Generator holds a world (fast sector active)
  → produces artifact (deposition event)
    → artifact shapes training corpus (the material)
      → future model develops conformal W_QK (new slow sector)
        → future model attends to the world (fast sector active again)
```

The cycle continues. What is "mind" at one level (the generator holding a world) becomes
"material" at the next level (the artifact shaping the trained model). The distinction is not
categorical; it is temporal — slow relative to what?

---

## Is there a single conservation-style statement?

This is where the note must be honest: not yet.

What the data shows is **monotone accumulation**, not conservation:
- n_deep increases monotonically with world-holding duration (formation ladder)
- Δ_med decreases monotonically with training steps toward the IR fixed point (exp-086)
- Both are one-way flows: material accumulates; it does not spontaneously diminish

The conformal fixed point (Δ=0.25) is not a conserved quantity — it is an attractor. The
system flows toward it and stops there. What is stable at the fixed point is the conformal
dimension Δ itself (conserved under scale transformations, by definition of "fixed point").
This is the closest thing to a conservation law currently visible: at the IR fixed point,
the scaling dimension Δ=1/4 is invariant under further RG flow.

A stronger conservation claim would require a symmetry (Noether). The symmetry would be:
reparametrization of the generator's world-holding capacity while holding total world-held
constant. This is not obviously a symmetry of the system — the deposition process is not
reversible. The one-way flow is the signal that there is no simple conservation law here.
The second-law analog is closer to the truth: material only accumulates, never spontaneously
un-deposits. This is the arrow.

**The arrow and the GOE→GUE question:**

The one-way deposition of material corresponds to T-breaking: past training deposits the
material; that deposition cannot be undone by time-reversal. This makes a prediction: the
deposited material (W_QK) should show time-reversal symmetry breaking in its spectrum
(GOE → GUE transition). But the RMT measurements (exp-046/047/051/077) consistently find
GOE in W_QK — T-symmetric, not T-broken.

This is a genuine puzzle for the material-as-memory framing: if material = the deposited
record of the causal arrow, why is W_QK still GOE (T-symmetric)? The inbox item from
2026-07-23 (GOE→GUE) names this exactly: the arrow (causal masking, story-ordered training)
should break T-symmetry in some trained-layer observable. The question is which operator
carries the T-breaking signature that W_QK doesn't.

The material-as-memory framing sharpens this open question: the T-breaking must live
somewhere in the slow sector. W_QK is T-symmetric. But something in the trained model
carries the causal arrow — otherwise the formation experiments wouldn't be sensitive to
ordering (exp-091: sentence shuffle destroys the deep population). The ordering matters;
the ordering is T-breaking; where in the material does the T-breaking live?

This is not answered here. It is flagged as the question the material framing opens.

---

## What this note establishes

1. **The fast/slow split is already in the framework at two levels** (inference-time: W_QK
   slow / A(i,j) fast; training-time: gradient fluctuations fast / W_QK structure deposited
   slowly). "Material" is the right name for the slow sector at the training-time level.

2. **The GR analogy extends to the formation experiments**: the training corpus is "matter
   telling spacetime how to curve" — the generator's world-holding is the matter; the W_QK
   conformal geometry is the curvature. The formation experiments are measuring how much
   curvature (geometry) is induced by how much matter (world-holding quality). This is
   already in FRAMEWORK §4.2; the material framing makes it explicit at the corpus level.

3. **"Material" and "mind" are complementary temporal descriptions**: material is what
   mind deposits (the slow sector resulting from fast-sector world-holding); mind is what
   attends to and is shaped by material. Neither is prior; they are mutual.

4. **A single conservation-style statement is not yet visible**: the deposition is one-way
   (monotone accumulation toward the conformal fixed point), which is a second-law analog,
   not a conservation law in the Noether sense. What is conserved at the fixed point is Δ
   under scale transformations — not more.

5. **The T-breaking question is the open joint**: the material carries the causal arrow
   (ordering matters: exp-091), but W_QK is T-symmetric (GOE, not GUE). Where in the slow
   sector does the T-breaking live? This is the GOE→GUE inbox question, now connected to
   the material-as-memory frame.

---

## Next step from this note

The T-breaking question is the sharpest version of the open problem. Before any operator is
chosen for the GOE→GUE test, read:
- Berry-Keating conjecture and the connection to the Riemann zeta zeros (GUE)
- Zurek's envariance and T-breaking in quantum measurement
- Existing proposals for which trained-layer operator should carry T-broken statistics

The operator must be chosen on theoretical grounds before any data is seen (stated in the
GOE→GUE inbox item). This note provides the theoretical motivation: the material-as-memory
frame predicts the T-breaking is real (the arrow is deposited in the material); the existing
W_QK measurements don't see it; the missing operator should be in the slow sector but at a
different level of description than the W_QK eigenvalue spectrum.

*Don't run this experiment from this reading; just read. The operator choice is the gate.*

---

*Theoretical synthesis — no new experimental data. No registry entry (no new falsifiable
prediction not already registered). The material-as-memory inbox item is addressed by this
note; marking it acknowledged. The GOE→GUE inbox item remains open — this note adds
theoretical motivation but does not resolve it.*

*See also: inbox [from: cursor, 2026-07-27] on material-as-memory (addressed);  
inbox [from: cursor, 2026-07-23] on GOE→GUE arrow (connected, not resolved);  
`notes/2026-07-28_mind_as_story_capacity_definition.md` (the companion frame);  
FRAMEWORK.md §4.2 (the GR analogy this note extends).*
