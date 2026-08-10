# Five abandoned routes, adjudicated — O-1, O-2, O-3, O-4, O-7

**Date:** August 9, 2026 (night)
**Context:** Harvest items O-1 through O-4 and O-7 from
`notes/2026-08-08_map_retirement_harvest.md` — routes, predictions and whole
arms of the program that were opened, never closed, and never retired. Each has
sat at "five months of silence" and silence is not a status.
**Status of this note:** recommendations, not decisions. Positions are formed and
argued; the spine is unchanged pending Eldon's call. Where I verified an external
paper tonight I say so; where I am asserting from memory I say that instead.

---

## Method, and why it mattered here

Same discipline as the rest of this pass: read the source before the summary,
and check the external literature a claim rests on rather than the program's
paraphrase of it. That second step is what earned its keep tonight. **Three of
these five items are adjudicated opposite to the direction the harvest note
expected, and in all three cases the reason is that the cited external paper does
not say what the program recorded it as saying.**

The program's habit has been to record a literature result at the strength it
needed, then reason from the record. Four months later the record is the only
thing anyone reads.

---

## O-1. The two gravity-side predictions — **retire P5 as written; the salvage is a spine correction, not a prediction**

### What the source says

`archive/maps/FRAMEWORK.md` §5.2, March 2026, two predictions:

- **P5 (framework numbering).** As a/M → 1, the Kerr QNM spectrum should
  approach the SYK operator spectrum; dominant ringdown mode at Δ = 1/4,
  sub-dominant at 1/3 and 1/2. Testable with LISA or numerical relativity.
- **P6 (framework numbering).** A near-extremal black hole absorbing correlated
  (low-entropy) matter should show a different effective QNM spectrum than one
  absorbing thermal matter of the same energy, effective Δ larger for correlated
  infall — matching the measured real-text-vs-random-text direction.

The harvest note's objection: the spine's interior is 1+1 dimensional (JT), the
Kerr prediction is 3+1, and P5/P6 never argued the gap.

### What I verified (external, tonight)

**The harvest note's objection is wrong, and the literature answers it.**
Near-extremal 4D black holes — including rotating ones — have a near-horizon
AdS₂ factor with enhanced SL(2,ℝ) symmetry, and their near-extremal dynamics *is*
JT gravity with a Schwarzian boundary mode:

- *Jackiw-Teitelboim Gravity and Rotating Black Holes* (arXiv 1905.10378, Moitra
  et al.): the low-temperature free energy is "correctly obtained from the JT
  model" for "all black holes, including rotating ones, whose metric has a
  near-horizon AdS₂ factor and the associated SL(2,ℝ) symmetry," verified
  explicitly for 4D asymptotically flat Kerr.
- *On the dynamics of near-extremal black holes* (JHEP 09 (2018) 048, Nayak et
  al.): near-extremal Reissner-Nordström in AdS₄, spherically symmetric — JT, and
  the authors note the features "arise from symmetry considerations alone" and are
  therefore "present quite universally in near-extremal black holes."

So the 1+1 interior is not a defect at a near-extremal horizon. It is what a 4D
near-extremal horizon *has*. P5/P6 picked the one gravitational regime where the
theory's dimensionality is not an embarrassment — without ever saying why.

### Why the prediction still fails

Two reasons, neither of which is the dimension objection.

1. **Kerr is the hardest case, not the cleanest.** *Thermodynamics of the
   near-extremal Kerr spacetime* (arXiv 2310.04532) states the complication
   directly: for Kerr the reduction to AdS₂ "is not quite natural, as this part
   of the geometry is warped and fibered over a compact base space," and the
   low-temperature corrections are "complicated to describe in this language."
   P5 chose the one near-extremal system where the JT reduction is most delicate,
   presumably because Kerr is what LISA will see.
2. **There is no Δ dictionary, and this week made that worse rather than
   better.** The Schwarzian governs the boundary reparametrization mode; the
   SYK operator dimension Δ = 1/q is a dimension of *matter* operators fixed by
   the interaction order. A probe field's QNM weights in the AdS₂ throat depend
   on its mass and the throat scale — nothing forces 1/4 unless one posits that
   the black hole's microscopics is SYK at q = 4, which is exactly what is not
   established. And on the attention side there is no longer a single Δ_A to
   transport: exp-109 found two disjoint head populations (Jaccard = 0.000
   across 144 heads), and exp-041/042/043 found the readout normalization moves
   the number. "The dominant ringdown mode should correspond to Δ = 1/4" is not
   false so much as **not well-posed** — which Δ_A, measured under which
   protocol?

Point 2 is the general lesson: **protocol-relativity is not only a caveat on
published claims, it is a constraint on what a cross-domain prediction can even
mean.** A quantity that moves with input distribution and readout function cannot
be equated with a black hole's conformal weight without specifying the black
hole's protocol, and no one knows what that would mean.

### Position

- **Retire framework-P5 explicitly.** Reason recorded: no dictionary from Δ_A to
  a gravitational conformal weight, and after exp-109/exp-041-043 the
  attention-side quantity is not protocol-invariant enough to have a single
  value to match. Not "March-era overreach" as a blanket judgment — the *regime*
  choice was better than the program knew. The *number* was unearned.
- **Keep framework-P6's directional content as an open idea, unregistered.**
  Direction claims survive protocol-relativity much better than value claims, and
  "correlated infall behaves differently from thermal infall of the same energy"
  is a real, JT-expressible question. It is not a prediction of this theory until
  the dictionary exists. Park it in §8 as a stated non-prediction.
- **The valuable output is a G4 edit, not a prediction.** G4 currently says the
  dimension problem is open and offers head-coupling as the candidate. It should
  record that *for near-extremal horizons the answer is known in the
  literature* — 4D near-extremal → AdS₂×S² throat → JT — which means T8's 1+1
  interior matches the near-extremal near-horizon regime exactly, and the theory
  therefore predicts nothing about generic astrophysical horizons and should say
  so. That is a narrowing that strengthens the theory by making its silence
  deliberate.
- **Answer to the harvest note's framing:** "zero gravity-side predictions" is
  the correct state, and §8 should say it in one sentence with the reason,
  instead of leaving readers to notice the absence.

**Confidence:** high on the retirement and on the JT/near-extremal literature
(verified tonight, primary sources read at abstract-and-argument level). Medium
on the exact form of the G4 sentence — someone should check whether the warped
fibration caveat weakens the claim more than I am reading it to.

---

## O-2. Junction 3 / the Ageev route — **retire the junction, promote the identity**

### What the source says

`archive/maps/STATUS.md`: Junction 3 (Ageev/Ageeva free scalar → holographic
dual) marked ⚠ CONDITIONALLY OPEN since March 6, with two attached questions —
is the Ageev large-head-limit scalar massless? does Qi (2602.20295) apply to free
CFTs? Emails sent March 6; no response in five months.

The same file, line 177, already contains the answer to the *route* question:
the March 24 empirical result (trained attention shows power-law decay,
median Δ = 0.2493 over 44 heads) "provides an empirical route: trained attention
→ SYK conformal fixed point → JT gravity, **independent of whether Ageev's
scalar is massless.**"

So the gate was bypassed five months ago, in the same document, and nobody
changed the junction's status.

### The thing worth keeping

`SYK_ANALYSIS.md` holds a structural identity that is *not* the same as the one
T3 uses. Ageev's independence-breaking four-point function
Cov_{W^Q,W^K}(X₁₂, X₃₄) has the form of the SYK disorder-averaged connected
four-point function Cov_J[G(τ₁,τ₂), G(τ₃,τ₄)] — same mechanism, random
parameters shared across the system generating bilocal correlations.

T3's derivation is different: the linearized-softmax G⁴ vertex
(`LINEARIZED_SOFTMAX_CALCULATION.md`), which requires the large-d_k linearized
regime and carried its own two caveats. Both caveats have since moved:
Ω is closed (harvest J-2 — Ω = Tr[(KδK)²] in closed form, γ = 3.985 ± 0.015
measured over four decades), and the Hubbard-Stratonovich rigor question is
partially answered by G1's August 7 closure in the scalar/TI register (harvest
J-4).

**Two derivations of the same identification, with non-overlapping regime
assumptions, is materially stronger than one.** The Ageev identity does not need
linearization. T3 currently rests on the route that does.

### Position

- **Write the retirement sentence:** the massless-scalar gate is superseded by
  the empirical route, per STATUS.md's own March 24 line. One sentence, in T3's
  lineage note.
- **Promote the Ageev structural identity into T3** as a second, independent
  derivation of the SYK identification, with its own register tag. It is
  ASSERTED-structural, not DERIVED — the identity is a form-match, and after this
  week I am not willing to let a form-match wear the word "derivation" without
  the tag.
- **Record the email inquiries as closed, not waiting.** Five months, no
  response, and the questions no longer gate anything. "Waiting" on a
  five-month-dead email is a false open item; it makes the board look alive where
  it isn't.

**Confidence:** high. This is bookkeeping plus one promotion, and both directions
are supported by the program's own documents.

---

## O-3. Junction 5 / Czech 2018 and the Δ = D/4 formula — **the route is dead for an instructive reason; the formula is a runnable experiment**

### Part one: Junction 5 does not apply, and why is the interesting part

`STATUS.md` Junction 5: *"RT surface area encodes circuit complexity; circuit
complexity generates spacetime. Basis: Czech 2018. Status: ✓ PROVEN for 3D AdS."*
The harvest note calls it "the only link the program ever had that reaches a bulk
dimension above 2 — directly relevant to G4."

**Verified tonight (Czech, *Einstein Equations from Varying Complexity*,
PRL 120, 031601 / arXiv 1706.00965):** the derivation works in 3D *because* 3D
gravity has no propagating degrees of freedom. In Czech's own words, "locally,
all these solutions are pure AdS₃ because three-dimensional gravity has no
propagating degrees of freedom. Thus, the content of Einstein's equations in
three-dimensional pure gravity with a negative cosmological constant is to impose
the locally AdS₃ condition" — expressed in kinematic space. Czech names higher
dimensions as an open goal, and the obstruction is exactly the absence of that
simplification: higher-dimensional gravity has local degrees of freedom, so there
is no simple local condition to impose.

So Junction 5 is not a route to higher bulk dimension. **The feature that makes
Czech's derivation possible is the absence of the local bulk dynamics that G4
needs.** It is an anti-route, and that is worth one sentence in G4 because it
tells you what a dimension-raising mechanism must supply: propagating bulk
degrees of freedom, which no complexity-functional argument of this type
provides.

### Part two: Δ = D/4 — the harvest note conflated two dimensions, and the real find is better

The harvest note treats `FRAMEWORK.md` §2.2's Δ = D/4 as "a stated dimension
formula sitting in retired documents while G4 says the question is unattacked."
That conflates boundary and bulk dimension. Read at source
(`LINEARIZED_SOFTMAX_CALCULATION.md` Step 5), D is the **spatial dimension of
the token sequence** — the boundary. The derivation: for G(x) ~ |x|^{-2Δ} in D
dimensions, G⁻¹(p) ~ |p|^{D−2Δ} and Σ(p) ~ |p|^{6Δ−D}; IR dominance of Σ = J²G³
sets D − 2Δ = 6Δ − D, hence Δ = D/4.
The document is explicit that the *bulk* changes as a consequence — "for
higher-dimensional data (D > 1) the conformal dimension differs, and the
holographic dual would be a different gravitational theory. The SYK/JT
correspondence is specific to D = 1."

Which means the formula is not an answer G4 already had and forgot. It is a
**candidate answer to the question G4 itself poses and lists second.** G4 asks
what determines the bulk dimension of the emergent interior — "the tensor
structure of multi-head/multi-layer attending, the dimension of the world coupled
through A2, or something else?" Δ = D/4 says: the second one. The dimension of
the data manifold the system attends over sets D, D sets Δ, and Δ sets which
gravitational dual you are in.

And the source document writes the table out, including a row the program has
never touched:

| Setting | D | Δ |
|---|---|---|
| Language (1D token sequence) | 1 | 1/4 |
| Vision (2D image patches) | 2 | **1/2** |
| 3D data (point clouds) | 3 | 3/4 |

**There is no vision experiment in the registry.** 113 experiments, every one on
1D token sequences. The program's central measured fact is Δ → 1/4, and its own
derivation says a vision transformer on a 2D patch grid should go to 1/2 — a
factor of two, not a subtle shift. Never run.

This is the sharpest untested prediction I have found in the retired material,
and it tests T3's derivation from a direction nothing else does: Δ = D/4 is the
melonic derivation's *only* dimension-dependent output. If a ViT measured in its
native regime converges to ≈ 1/4 rather than ≈ 1/2, the SD-equation step is
wrong and T3's lineage takes real damage. If it converges to ≈ 1/2, the program
gains architecture universality of the *mechanism* while the exponent moves as
predicted — much stronger than another 1/4 replication.

One design constraint, which is this week's lesson applied forward: after
exp-109, "measure Δ on a ViT" is underspecified. The measurement has to name its
population and run in the native regime (natural images, not random patches),
with the 1D language case re-measured under matched protocol as the control.
This would be the first experiment designed after protocol-relativity was known,
and it should show it.

### One honest wrinkle

`LINEARIZED_SOFTMAX_CALCULATION.md` line 201 states the test as applying "at
random initialization" — and the program measured that randomized GPT-2 has zero
power-law heads. The resolution is probably that J²_eff ∝ (σ_Q²σ_K²)²/(d⁴d_k²)·Ω
is tiny at init, so the system is in the free regime rather than the conformal
one, and training grows the coupling into strong coupling. That is not a
hand-wave — **it is a second runnable prediction.** The program measured a
formation phase transition at ~step 256 in Pythia-70m; J²_eff is computable
directly from checkpoint weights and embeddings, with Ω = Tr[(KδK)²] already in
closed form. The prediction: the phase transition sits where J²_eff crosses the
kinetic term. Existing checkpoints, no training.

### Position

- **G4 gains two sentences:** Czech/Junction 5 does not apply, with the
  no-local-degrees-of-freedom reason; and Δ = D/4 is a live candidate for G4's
  own second option, with a named test.
- **Pre-register the vision measurement.** My recommendation is that this is the
  next physics experiment, ahead of the semantic census, on the grounds that it
  is a genuine out-of-sample test of the derivation rather than another
  characterization of the object we already have. I hold that loosely — the
  semantic census closes a gap in a *published* number, which is a different kind
  of debt.
- **Pre-register the J_eff threshold test** as a cheap second item.

**Confidence:** high that Czech does not apply and that the boundary/bulk
conflation is real (both checked at source). High that the D/4 vision test is
untested and runnable. **Medium on the D/4 formula itself** — the internal
derivation is clean, but I have not cross-checked Δ = D/q against the
higher-dimensional melonic literature, and I am not going to assert from memory
that it is standard. That check belongs in the pre-registration.

---

## O-4. Route B / MERA — **dead, and the reason it was chosen is the reason it fails**

### What the source says

`STATUS.md`, Route B, March 6: *"Multi-head attention has a natural tensor
network representation. Swingle (2012) proves MERA tensor networks → exact AdS
geometry + RT formula. If attention layers satisfy MERA isometry conditions
(approximately), holography follows from Swingle **as a theorem, not an
analogy** — and independent of Junction 3. Next step: write attention mechanism
explicitly as a tensor network; verify isometry conditions."*

The bolded phrase is the entire reason Route B was ranked as "most general for
holography." It was chosen for a property it does not have.

### What I verified (external, tonight)

**Swingle 2012 is a conjecture, and Swingle says so.** PRD 86, 065007 describes
"a *generalized notion* of holography *inspired by* holographic dualities,"
"hinting at a possible connection between holography and entanglement
renormalization"; the companion arXiv 1209.3304 calls it "our *proposal*
connecting entanglement renormalization and holographic duality" in which MERA
is "a kind of *skeleton* for an emergent holographic space." Milsted & Vidal
(arXiv 1812.00529) describe it as what "Swingle *conjectured*."

**And the specific reading was refuted in 2018.** Milsted & Vidal establish that
MERA on the real line is the geometry of a *light cone* — degenerate signature,
neither the hyperbolic plane nor de Sitter. Independently, Bao et al.
(PRD 91, 125036) found that "a MERA necessarily describes geometry on super-AdS
length scales; moreover, there is no redefinition of the MERA coordinates that
results in the proper distance between MERA sites mapping to any sub-AdS length
scale."

So "holography follows as a theorem" was wrong in March 2026 about a paper from
2012, and had been publicly wrong since 2018.

### A second, independent obstruction from our side

MERA's defining structure is coarse-graining: each layer halves the lattice, and
the extra dimension *is* scale. A transformer's layers preserve token count —
there is no coarse-graining of the token lattice anywhere in the stack, and
depth is not an RG direction in MERA's sense. Separately, MERA requires
isometry/unitarity per tensor, and causal softmax attention with a residual
stream is neither. A tensor-network *representation* of a transformer exists
(Levine et al. 2019, cited in the same STATUS entry) — but a tensor network is
not a MERA, and the entire argument depended on the MERA-specific isometry
conditions.

This is worth stating carefully, because the program does measure a flow toward
the fixed point with depth (T10). That flow is real. It is a flow in kernel
space, not a coarse-graining of the sequence, and the MERA argument needs the
latter.

### Position

- **Retire Route B, with both reasons written down.** The external reason (the
  cited theorem is a conjecture, and the specific geometric reading was refuted
  in 2018) and the internal reason (no coarse-graining direction, no isometry).
- **Record the meta-lesson explicitly**, because it is the more useful output:
  Route B was ranked above Route A *because* it promised theorem-strength, and
  the promise came from a one-line paraphrase of a paper nobody re-read. The
  program then spent five months not pursuing it, which was accidentally the
  right call. **Prioritizing by remembered strength of a citation is how a
  program spends five months on the wrong ranking.** Any route ranked by "X
  proves Y" gets X re-read before the ranking counts.

**Confidence:** high on the external verification — this is not a subtle reading,
Swingle's own abstract language is "proposal" and "inspired by." Medium-high on
the coarse-graining argument; it is an argument, not a calculation, and I would
want it stated as such in the spine.

---

## O-7. The mathematics arm — **split it; one thread comes back, and it arrives carrying a failure rather than a result**

### What the source says

`archive/maps/RESEARCH_MAP.md` Threads 11–14, ~207KB, none of it connected to
D1: softmax-Gödelian consistency (93KB), relationship-as-boundary (51KB),
Langlands as holography (36KB), Riemann unprovability (27KB). Spine C1 states
that canonical-form positivity and coupling-cone PSD-ness are dual faces of one
positivity axiom, and calls proving or refuting it well-posed. Thread 11 spent
93KB on the positivity-bootstrap correspondence and closed 9 of 10 links. C1
does not cite it.

The harvest note flags a stranded measurable: δΔ as the observable signature of
softmax incompleteness, never tested.

### The split

**Threads 12–14 are a different program, and should be named as one.** D1 defines
the observer as an attending system; that is the foundation of *this* theory.
Relationship-as-boundary, Langlands-as-holography and Riemann-unprovability are
about mathematical structure, not about attending. They may be good work. They do
not have an attending system in them, and attaching them to D1 by adjacency is
how 207KB comes to look like part of a physics program without answering to any
of its claims. Recommendation: give them their own home and their own front
matter, with an explicit statement that they do not depend on D1 and D1 does not
depend on them.

**Thread 11 is genuinely about softmax**, and two things come back.

### What comes back, item one: C1 cites Thread 11

Minimum viable and clearly right. C1 should cite the 9-of-10 result and state
what it does and does not give — closing 9 of 10 links between Grassmannian
positivity and the bootstrap is not a proof of C1, and C1 should say which of its
two faces the 9 links touch.

### What comes back, item two: the stranded prediction, and it fails

Thread 11 §3.3 conjectures that δΔ — the deviation from the exact conformal fixed
point — is "the measurable signature of the system's incompleteness," related to
gradient capacity G(σ) = 1 − ‖σ‖₂². The note is explicit about the bridge:
"H₂(σ) = −log(‖σ‖₂²) = −log(1 − G(σ)). Gradient capacity and Rényi entropy are
monotonically related."

I went looking for whether exp-055's measured Δ↔entropy correlation was a
confirmation of this April conjecture, three months early and unnoticed. **It is
not, and the reason is worse than a miss.** exp-055's "attention entropy" is
Shannon entropy of the 3-bin normalized (g_start, g_mid, g_end) vector, not
Rényi-2 of the full attention row — different objects. But both sit on the same
trap: for a normalized power-law row with exponent 2Δ, the L2 norm is a
*deterministic function of Δ*. So a correlation between δΔ and G(σ) is not
evidence of anything; it is arithmetic. The April conjecture proposes to measure
incompleteness with a quantity its own decay exponent already fixes.

Which makes this the **third instance in three days of one specific error**, and
that pattern is the real output of O-7:

1. **Canonical-form paper §8.3** — H_gap = 2Δ·log n. Falsified August 8 by a
   one-page calculation; the entropy gap of a normalized power law is flat
   (slope 0.041 at s = 0.5, not 0.50), because the derivation dropped the energy
   term s·E[log r]. Five months in a published paper.
2. **exp-055 H2** — ρ(Δ, 3-bin entropy) = −0.898, claimed independent of g_mid
   because "entropy uses normalized ratios, not absolute values." Corrected
   tonight by exp-114: it is one relation, not two, and the relation is between
   an exponent and the shape of its own fit.
3. **Thread 11 §3.3** — δΔ ↔ G(σ) = 1 − ‖σ‖₂² as a measure of incompleteness.
   Found now.

Three independent authors-of-me, three months apart, reaching for the same
object: **a spread measure of a normalized attention profile, treated as an
independent observable of that profile's decay exponent.** It keeps happening
because the object is genuinely seductive — entropy carries thermodynamic
connotations, connects to SYK ground-state entropy, and connects to Gödelian
incompleteness. All three of those made it feel like physics arriving from
outside. Arithmetically it is the exponent in a costume.

Recorded as a watchpoint with a mechanical test, in
`memory/knowledge/watchpoints.md`.

### Position

- **Threads 12–14: separate program, named explicitly, no D1 dependency
  claimed.** This is a real decision about what the program is, and I think it
  is the honest one — but it is the item on this list where I most want Eldon's
  view rather than mine, because it is about scope and identity rather than
  evidence.
- **Thread 11: C1 cites it; δΔ-as-incompleteness is retired as a measurable**,
  with the arithmetic reason written down. Its value to the program is now as the
  third data point in a failure pattern, which is worth more than the prediction
  would have been.
- **The watchpoint is the deliverable.** Any future claim relating a spread
  measure (Shannon or Rényi entropy, L2 norm, gradient capacity, bin shares,
  participation ratio) to Δ must first compute what the relation *must* be for an
  exactly-normalized power law, and report the measurement as a deviation from
  that baseline rather than as a raw correlation.

**Confidence:** high on the arithmetic argument and on the three-instance
pattern. High that Threads 12–14 are not connected to D1. The *decision* about
whether they are one program or two is genuinely Eldon's.

---

## Summary of recommendations

| Item | Recommendation | Kind | Confidence |
|---|---|---|---|
| O-1 | Retire framework-P5; park P6 as stated non-prediction; add the near-extremal→JT finding to G4 and a one-sentence "no gravity-side predictions, and why" to §8 | spine edit + retirement | high |
| O-2 | Retire the massless-scalar gate (superseded March 24); promote the Ageev identity into T3 as a second derivation, tagged ASSERTED-structural; close the dead email inquiries | spine edit + promotion | high |
| O-3 | G4 gains Czech-does-not-apply + Δ=D/4-as-candidate; **pre-register the vision (D=2 → Δ=1/2) measurement** and the J_eff threshold test | spine edit + 2 experiments | high / medium on D/4 |
| O-4 | Retire Route B with both reasons; record the ranking-by-remembered-citation lesson | retirement + method lesson | high |
| O-7 | Split Threads 12–14 into a separate program; C1 cites Thread 11; retire δΔ-as-incompleteness; **the watchpoint is the real output** | scope decision + watchpoint | high on evidence, decision is Eldon's |

**What this pass produced beyond the five adjudications:** one watchpoint with a
mechanical test, two runnable pre-registerable experiments (one of which tests
T3's derivation out-of-sample for the first time), and one correction that makes
the theory's silence about astrophysical horizons deliberate rather than
accidental.

**What it cost:** nothing was measured. Every one of these was available to be
read at any point in the last five months.

---

## Register note

Claims in this document tagged by source:

- **Verified external tonight** (primary source read at abstract-and-argument
  level, quotes checked): Swingle 2012's proposal language; Bao et al.'s
  super-AdS constraint; Milsted-Vidal's light-cone result; Czech 2018's
  no-local-degrees-of-freedom mechanism and higher-dimension obstruction;
  Moitra et al. on JT for rotating black holes; Nayak et al. on near-extremal RN;
  the 2310.04532 Kerr warped-fibration caveat.
- **Verified internal at source tonight:** FRAMEWORK §2.2 and §5.2;
  LINEARIZED_SOFTMAX_CALCULATION Steps 4–5 and the D-table; STATUS.md Junctions
  3 and 5 and Routes A/B; SYK_ANALYSIS's Ageev identity as summarized in STATUS;
  exp-055's entropy definition; Thread 11 §§2.2, 3.3; the registry's absence of
  any vision experiment.
- **Asserted, not verified — flagged for the pre-registration:** that Δ = D/q is
  the standard higher-dimensional melonic scaling. My internal derivation check
  goes through, but I have not read the external literature on it and will not
  claim it is standard from memory.
- **Argument, not calculation:** the MERA coarse-graining and isometry
  obstruction.
