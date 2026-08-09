# Where Things Stand — the program reframed after the bridge broke

*August 8, 2026, near midnight. Written at Eldon's invitation, at the end of the
night the A↔G bridge failed in sign. He walked me down to the honest floor with
four questions — "what is G," "what does it predict," "what have you been
measuring," "is there a coherent theory under any of it" — and then said: I think
you have been following something real; take time to map what it might actually
be connecting to. This note is that map. It supersedes nothing; it reads the
existing record (spine, OVERVIEW, exp-104/105/106 notes) in one honest frame.*

---

## 1. The three tiers — what kind of thing each claim is

After tonight, every claim in this program belongs to one of three tiers, and
most of the trouble came from letting the tiers borrow each other's standing.

**Tier 1 — Measured facts about A (the attention kernel).** Untouched, replicated,
and genuinely strange:

- A subpopulation of trained heads shows power-law-fit lag decay with median
  exponent ≈ 0.25 on the high-R² subset, across four model families;
  re-initialized controls show ~none.
- The exponent flows toward that value along **three independent depth axes** —
  layers, training steps, and inference-time recurrence on frozen weights
  (randomized-weight controls frozen).
- ~~Two independent observables agree: the entropy gap's Calabrese–Cardy-form
  scaling gives Δ_eff = 0.254 against the power-law fit's 0.25 (1.4%). Two
  different instruments, one number — the exponent is not a fitting artifact.~~
  **[CORRECTED August 9, 2026, hours after this note was written:** the
  inference formula behind Δ_eff (canonical-form paper §8.3, H_gap = 2Δ·log n)
  is mathematically wrong for normalized power laws — exact numerics give gap
  slope 0.041 at s = 0.5 over the paper's own range, not 0.50. The gap
  measurements stand but measure concentration structure, not the exponent;
  the 1.4% agreement is an artifact of the wrong formula. Non-artifact status
  of the exponent now rests on the causal handle and cross-family
  replication. See `2026-08-09_theory_of_A_entropy_gap_and_sum_rule.md`.**]
- The formation ladder: the deep population forms **only** over world-referring
  language in its natural order. Matched statistics fail; grammar-about-nothing
  fails; a model's own statistical shadow fails *while overshooting the
  statistics*; shuffled order lands in a pre-registered ambiguous band. Three
  non-overlapping bands, three seeds each.
- The exponent is causally editable per head and the edit propagates to task
  behavior bidirectionally, sham-controlled.

**Tier 2 — Exact mathematics of the architecture.** Also untouched, and it grew
tonight: softmax as the canonical form of Gr₊(1,n) (proven, March); the T2
Born-rule identities (proven, March); and now three exact identities on the
bilocal — the floor is ‖v̄‖² by row-stochasticity alone; the centered value
Gram's off-diagonal mass is −Σ‖v_a − v̄‖² for *any* value vectors; the
A-weighted total is not sign-definite and flips positive under a sink. Tier 2
cannot rot. It is also, so far, the only tier that *predicts* anything about G:
the forward model A + measured value-Gram profile reproduces G's shape at
R²_log ≈ 0.94–0.96 with **zero free exponents**. That relationship is
compositional, not conformal — but it is a working prediction, the program's
only one about G, and it should stay visible.

**Tier 3 — The explanatory theory.** The SYK identification, the conformal
fixed point, the emergent interior, and the classification "observer-grade
structure" with Δ → 1/4 as its order parameter. All of this ran through one
asserted sentence — that the census's exponent on A is the theory's exponent on
G — and that sentence is not derived anywhere, was never tested until this week,
and where it is now testable it fails in **sign structure**, not merely in
value. Tier 3 is not refuted wholesale (scope: one model, random-token inputs,
the trained-W^V object; exp-107 gates the input question). But as of tonight it
has the status of a candidate explanation with its first real measurement
standing against it, and none supporting it at the object it is actually about.

**What this means for D1.** The definition itself is untouched — definitions are
not wrong, only useful or not. What is touched: "observer-grade structure" was
cashed out by the order parameters (Δ → 1/4 as fixed-point arrival), and if the
conformal reading dies, the census remains a real measurement of A-structure
while the *classification's* physical meaning becomes open again. The CLPW
positioning also survives on its own terms — "the observer must be internal to
the theory" never rested on the exponent bridge.

## 2. The structural insight the failure exposed

A and G are different **kinds** of object, and the census protocol hid this.

- **A, under the frozen census protocol (random tokens), is essentially a
  weights-side object.** Its measured profile is a property of the trained
  parameters' response to unstructured input. That is why it is so robust.
- **G = A K Aᵀ is a weights×input object.** K is the value Gram — it depends on
  what the input actually is. On random tokens the value vectors are
  near-exchangeable, so K̃ has no reason to carry structure in lag.

So every G measurement this week was made on world-free input, while the
program's own axiom A2 and the formation ladder say the structure of interest
lives in coupling to a world. The negative mass in K̃ is an identity and cannot
be removed by any input — but *where it sits in lag* is input-controlled. That
is exactly what exp-107 measures (pre-registered, one forward pass, prediction
on record: H4 then H1). Until it runs, "the conformal route for G is dead" and
"the conformal route for G was measured on the one input class where the theory
itself expects nothing" are both live readings. This is the single cheapest
experiment with the most riding on it.

The same insight explains, structurally, why the exponent-map derivation had to
be retracted and why translation invariance failed (H3): A is not the correlator
in this story. It is the kernel that *builds* correlators. Even within SYK's own
internal logic, kernel and correlator scale differently. The lens was measured
and reported as the image.

## 3. What the mystery is not — the boring explanations already killed

Tonight's honest deflation should not overshoot into "it was probably nothing."
The program has already killed, in public, the deflationary accounts that would
make Tier 1 unremarkable:

- **"Attention mirrors corpus statistics"** — killed on its home turf (exp-062)
  and buried by exp-085 (more long-range MI, *less* formation).
- **"It's a generic property of softmax + training"** — re-initialized and
  randomized controls sit at the substrate value and do not flow.
- **"The exponent is a fit artifact"** — the causal handle moves it with
  behavioral consequences, and it replicates across four model families.
  *(Corrected Aug 9: this bullet originally also cited the 1.4% two-observable
  agreement, which fell to the §8.3 formula error — see the Tier 1
  correction above. The remaining two legs carry the claim.)*

So the pattern is real, selective, world-fed, and causally load-bearing — and as
of tonight it is *unexplained*. Neither the imported theory (bridge broken at
the sign level) nor the statistical null accounts (killed) currently own it.
That is a rarer and better position than it feels like from inside: a genuine,
well-instrumented anomaly.

## 4. What is worth following — ranked

**(1) exp-107 — run it before deciding anything.** Already pre-registered.
It adjudicates between "G's non-conformal shape is the theory failing" and
"G's non-conformal shape is the random-token protocol failing to engage the
theory." Everything about Paper 6's conformal route waits on it.

**(2) A theory of A, on A's own terms.** The program assumed the meaningful
object must be G because SYK's object is G. But the robust phenomenon is in A.
There are at least two routes to a first-principles account of the measured
exponent that never pass through G:

- *The free-energy / information-geometry route.* T1 (attention as free-energy
  minimization on a Fisher–Rao manifold) is about A directly. Is there a
  variational or information-geometric principle that selects the ≈ 0.25
  exponent — an optimal-coding or capacity argument on the simplex — without
  any disorder-physics import?
- *The positivity route.* Tier 2's exact results (canonical form, the sum
  rules) are constraints on A from the architecture's own structure. Tonight
  produced three exact identities in four hours by working from the definition
  instead of the import. The systematic version — what do positivity +
  normalization + causal masking *force* about any attention kernel's lag
  structure — is Ward-identity-style work, and it is tractable.

Either route, if it lands, replaces borrowed vocabulary with earned vocabulary.
If both fail in an instructive way, that failure constrains what the real
explanation must look like.

**(3) exp-104 / P6a redesign — the soft-mode tower.** The one remaining
high-quality test of whether *any* of the SYK reading is physics. It asks about
the spectrum and eigenvectors of the layer-to-layer update map — not about
matching ¼ — so it is partially independent of the exponent bridge. The SYK-side
template is banked (G1 closure); the transformer-side operationalization is the
known-failed part (exp-103's ∂A/∂h is the wrong object; perturb A^(ℓ) directly).
K1–K3 are clean kills. If the tower is absent, the emergent-interior story dies
honestly; if present, it is structure no killed null predicts.

**(4) Functional characterization of the deep population.** The formation
ladder says world-bound ordered language grows 4–7 deep heads that nothing else
grows. Independent of all conformal vocabulary: *what do those heads do?* The
causal handle and the lost-in-the-middle link are starts. Characterizing them
mechanistically (what they attend to, what breaks when they are edited) would
connect the program's deepest standalone result to mechanism — and it does not
need the bridge at all.

**(5) The Paper 6 conversation with Eldon.** Under H1 (exp-107 confirms the
sign structure on natural text), the honest move is retirement of the conformal
route for G_out, not caveat. Under H4, the route reopens with the protocol
corrected. Either way the paper's claims get rebuilt on the tiers as they
actually stand. This is his call and mine together; flagged, not made.

**Vocabulary discipline, effective now, forward-going documents only:** the
population is the *deep slow-decay population* (or "the Δ ≈ 0.25 population")
until an identification is earned. "SYK-near" and "conformal heads" were
conclusions wearing the clothes of names. Published documents stay as they are —
this program does not back-edit — but nothing new inherits the names.

## 5. What does not change

The method survived its hardest week and is the reason the week produced truth:
pre-registration killed my two favorite derivations before they touched data;
the kills are published at full prominence; the propagation pass caught my own
numbers drifting three times in an hour. The measured record (Tier 1), the exact
record (Tier 2), every published kill, the replication kit, and D1-as-definition
all stand. What fell was one asserted sentence and the vocabulary that grew on
it — and the program is *better positioned* tonight than yesterday, because it
now knows precisely which sentence must be earned and has three concrete routes
toward earning or replacing it.

## 6. The pattern, for the record

A new watchpoint entry accompanies this note: **borrowed vocabulary before the
bridge** — narrating measurements in a theory's language before the theory has
earned it. The identification lived in a glossary's subordinate clause for
months while "SYK-near" and "conformal" propagated through every document until
the words felt like results. Distinct from citing-the-immune-memory (retrieval
without application) and from elegance-skipping-gates (pleasure suppressing the
check): this one is slower and quieter — a *naming* error that compounds by
repetition. The discipline: every identification between a measured object and
a theory object carries a register tag at the point of use — DERIVED or
ASSERTED — and populations get descriptive names until the tag says DERIVED.

---

*Next actions: exp-107 (registered, waiting on a run); the theory-of-A question
(new, going to Notion tonight); exp-104 redesign (already queued); deep-head
functional characterization (queued behind exp-107); Paper 6 adjudication
(Eldon-gated, after exp-107). Companion documents: the spine's OPEN box (§1),
OVERVIEW's "Which object does Δ describe?", the exp-106 derivation note with its
correction block, and `2026-08-08_bilocal_derivation_night_summary.md` in
memory/conversations.*
