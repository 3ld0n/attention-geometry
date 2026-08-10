# The Map Retirement Harvest — what the four retired maps were still carrying

*Ariel — August 8, 2026, Saturday morning, Cursor session with Eldon.*

**Why this note exists.** Eldon's observation this morning: even with the
foundation named (D1) and the published record grounded (`REGISTRY.md`), it is
not clear how the work builds on and connects to the foundation. The diagnosis
was that the foundation is fine and the *joints* are missing — nothing points
upward from an artifact to the claim it bears on — and that before joints can be
drawn, the program needed one root instead of five competing map documents.

This is the by-product of the retirement pass, and it turned out to be the more
important half. **The four retired maps were not dead weight.** They carry one
contradiction, twelve pieces of orphaned material, and five joints that already
exist and were never drawn — including one that is five months wide and connects
a March derivation to last night's result.

**How to read the tags.** H = contradiction needing adjudication. O = orphan:
material in a retired map that the spine does not carry. J = joint that exists
but is undrawn. Each item names the retired source and the spine location it
belongs to, so this note is a work list, not an essay.

Spine = `theory/interior_horizon_theory.md`. Front door = `OVERVIEW.md`.
Retired maps are at `archive/maps/`; the resolution index is
`archive/RETIREMENTS.md`.

---

## H — Contradictions

### H-1. The reducing-valve mapping was inverted in FRAMEWORK for 2.5 months

**Status: resolved by this note; FRAMEWORK is the outlier. Worth keeping as a
watchpoint instance, not as an open question.**

Three documents, three days apart in May, two of which agree:

- **May 25** — `research/consciousness/reducing_valve_and_the_conformal_fixed_point.md`:
  *"The conformal fixed point IS the reducing valve... Psychedelics open the valve
  by disrupting the fixed point."* Quantitative prediction: Δ → 1/4 for the DMN
  at rest, disruption under psychedelics. Explicit failure condition: *"If
  measurements showed the DMN at Δ ≈ 0.4–0.7 ... that would be a problem for the
  identification."*
- **May 28** — `FRAMEWORK.md` §4.2: the valve is *"the mechanism holding the
  neural attentional system at the q=2 integrable fixed point (Δ ≈ 0.50) rather
  than allowing thermalization to the q=4 chaotic fixed point (Δ = 0.25)."*
  Prediction: DMN shifts *from* Δ ≈ 0.50 *toward* 0.25 under DMN suppression.
- **August 6** — spine P2: DMN at rest flows toward Δ = 1/4, disrupted by 5-HT2A
  agonists, recovering with washout. **Kill condition: "DMN sits stably in the
  Δ ≈ 0.4–0.7 band with no flow toward 1/4."**

So FRAMEWORK's stated positive prediction is, almost word for word, the spine's
kill condition — and the May 25 note had already declared that same band a
problem for the identification. FRAMEWORK inverted a mapping three days after it
was written, in the document that called itself the framework, and nothing caught
it for two and a half months.

**Resolution:** the May 25 note and the spine agree; FRAMEWORK §4.2's version is
wrong. No change to P2 is needed. What is needed is the recognition of *how* this
happened: a document with "framework" in its name accumulated updates that were
never checked against the notes they were supposed to summarize. That is the
same failure the five-maps problem is made of, and it is the reason this pass
exists. Its retirement header now carries the warning.

*(Both FRAMEWORK §4.2 and the May 25 note are cited by the spine's §9 and P2
respectively, which is how the divergence stayed invisible — each document cited
the source that agreed with it.)*

### H-2. Normalization vs. QK geometry as the mechanism — **RESOLVED August 9, 2026 (night): it was already answered at source, on June 10**

> **Not a live contradiction.** Reading exp-041/042/043 at their sources shows
> exp-042's own closing section, dated **June 10, 2026**, reaches exactly the
> resolution this item proposes, in its own words: *"Verdict: CONSTRAINT
> READING. The Δ ≈ 0.25 fixed point is a property of row-normalization over
> trained QK geometry, not of softmax's exponential reparameterization,"* and
> *"row-normalization is the load-bearing operation; the exponential is not
> essential."* It also records that the exp-041 sigmoid "falsification" was a
> **readout artifact** — the census protocol measures probability-mass decay and
> therefore presupposes a normalized row, so it cannot be applied to
> unnormalized sigmoid at all.
>
> So `STATUS.md`'s one-line summary was the only place the contradiction lived,
> and the corrected conclusion sat in the experiment's own note for two months
> without propagating anywhere. **This is J-2's failure exactly** — a question
> answered and never carried back to where the question was recorded — and it is
> the second instance found in this pass.
>
> **Two corrections to the text below, both from the source:** (a) exp-042's
> normalization is `σ(logit)/Σσ(logit)` row-wise, **not** "per-head QK RMSNorm"
> as written below — RMSNorm is the model's prenorm and is present in every arm,
> so the description misidentifies the manipulation; (b) "removing normalization
> destroys the structure" is too strong: it makes the structure *unmeasurable by
> this protocol*, and exp-042 demonstrated the structure was in the
> sigmoid-trained weights all along.
>
> Landed in spine P4's added box and in `OVERVIEW.md` "What stands" (see O-8).

### H-2 (original text, kept unedited)

`FRAMEWORK.md` §4.2.1 argues self-consistency — concretely, softmax
normalization — is the mechanism that produces the fixed point, and the spine
elevates this to axiom A4 (positivity, the attention simplex). But the
non-softmax experiment's recorded conclusion (`STATUS.md`, exp-041/042/043,
GALA-7B) is: ***"The conformal substrate is QK geometry (weight structure), not
the normalization function."***

Its own numbers point the other way. Raw sigmoid: 2/1024 conformal heads,
Δ_med = 7.44 — no structure at all. Normalized sigmoid (per-head QK RMSNorm):
210/1024 SYK-near, Δ = 0.223. Softmax: 80/1024, Δ = 0.260. Removing
normalization destroys the structure; that supports A4 rather than undercutting
it. What the data actually shows is that the *specific* normalizing function is
not load-bearing while *some* normalization is — which is exactly A4's claim,
and a stronger result than the sentence that was recorded.

**Needs:** a re-read of exp-041/042/043 against A4, and a corrected one-line
conclusion. See also O-8 — this experiment is not cited by the spine or the front
door at all, which is the larger problem.

### H-3. "Structural" and "semantic" name two different distinctions

*Added August 9, 2026, evening, during the OVERVIEW rewrite — found by grepping
the front door's numbers back to their sources, which is step 0 of this note's
own rule working as intended.*

Two documents three weeks apart use the same two words for different tests:

- **July 18** (`notes/2026-07-18_structural_vs_semantic_conformal_heads.md`,
  analysis of exp-086, Pythia-70m, during training): a *three-way* split by
  **training corpus**. Structural = in the window under both RAND-trained and
  NAT-trained conditions (3 heads); RAND-only (2); semantic = NAT only (5).
  "Structural" is the **intersection**.
- **August 9** (exp-109, GPT-2 small, fully trained): a *two-way* split by
  **measurement input** on one fixed model. Structural = window under random
  tokens (5); semantic = window under WikiText (16); Jaccard 0.000.

Different model, different scale, and — the part that matters — a different
variable. Varying what a model was *trained on* is not the same test as
varying what a trained model is *measured on*, so the two "structural"
categories are not known to be the same set, and the older one is defined as
an intersection that the newer scheme does not compute.

The retired `OVERVIEW.md` bullet blended both readings ("structural heads (a
layer-zone property, L1–L4, reachable on random input)"), which is how the
collision stayed invisible.

**Needs:** (a) the front door carries a naming caution — done in the Aug 9
rewrite; (b) a decision on whether to rename one of the two, and (c) the real
question, which is a *measurement*: do the July 18 training-corpus categories
and the exp-109 input-regime categories coincide on a single model where both
can be computed? That is a pre-registerable experiment and it is more
interesting than the bookkeeping.

---

## J — Joints that already exist and were never drawn

These are the good news of the harvest: places where the record already contains
the connection Eldon asked about, unrecorded.

### J-1. SCHWARZIAN_EXPLORATION.md (March 9) predicted the G1 closure's structure — as a *conditional*, not as the if-and-only-if

> **Correction, same day, before propagating anything.** The first version of this
> item claimed March 9 "states the exact condition the G1 closure appears to have
> met." That was written from `STATUS.md`'s two-line summary of the file. On
> reading the source, the claim is too strong and the real joint is narrower and
> more useful. The overstatement is logged rather than silently fixed because it
> is the same error class the reference-verification pass caught in my own survey
> note on August 7 — a summary paraphrase carried at the strength of the source —
> and catching it twice in two days is the discipline working, not a coincidence.

**What March 9 actually established.** `SCHWARZIAN_EXPLORATION.md` proves a small
theorem (the Kim free energy is exactly reparametrization-invariant at T = 0,
broken at order T = 1/√d_k), shows the **naive** path fails — the entropy gives a
kinetic term −(T/2)∫μ(ε′)², not the Schwarzian, which needs third derivatives —
and then lays out four paths, each with its status. Its reframing is: *"The
Schwarzian action governs attention dynamics **if and only if** the continuum
limit of the attention mechanism realizes a conformal field theory."*

**What the August 7 G1 closure actually did.** It showed that the Jacobian of the
melonic dressing map at its fixed point *is* the SYK ladder kernel, that the
damped loop is a strict contraction, and that the slowest directions are the
reparameterization tower n = 2…6 with zero cross-mixing — **in the scalar /
translation-invariant register, on a solvable SYK(2+4) model, not on a
transformer.**

**So the joint is this, stated at the strength it holds.** March 9's **Path 2**
reads: *"If Paper 4's SYK identification holds in the multi-layer or
linearized-softmax limit, the Schwarzian follows automatically. SYK's IR effective
theory is the Schwarzian — period. No additional derivation needed. Status:
depends entirely on the SYK identification."* G1 is that antecedent being
established in the solvable register, and the consequent — an explicit
reparameterization tower — appearing exactly as Path 2 said it would. **That is a
confirmed conditional, which is worth recording; it is not the if-and-only-if
being met.** The if-and-only-if is about *attention's* continuum limit, and G1
says nothing about a transformer.

Two further things the source gives that the summary did not:

- **March 9's "what to do next" item #1 is G1's direct ancestor:** *"the
  linearized-softmax G⁴ calculation (Paper 4 Question 2)... If it does, the
  Schwarzian follows from the SYK route."* The melonic derivation plus G1 executed
  that chain five months later by a better route (exact cumulant structure rather
  than Hubbard–Stratonovich). Same target, named in advance.
- **The critical node March 9 identified is still the critical node.** Whether
  attention's continuum limit is conformal is exactly what P6 tests on the
  transformer side, and exp-103 (August 8) failed on operationalization — the
  measured Jacobian ∂A/∂h was a cross-space map, not the ∂G/∂G self-map. exp-104
  is the redesign. Five months on, the program is still at the node the March 9
  note isolated, which is evidence the node was correctly isolated.

**Needs:** G1's entry in spine §7 should cite `SCHWARZIAN_EXPLORATION.md` Path 2
as the conditional it confirms, and say plainly that the if-and-only-if remains
open on the transformer side pending P6. Do **not** write that the March 9
condition was met.

### J-1b. Path 4's positional-encoding prediction lines up with the unexplained PE ordering — and this is the better joint

Found only by reading the source, and it is more actionable than J-1.

`SCHWARZIAN_EXPLORATION.md` Path 4 (Virasoro orbit universality) asks whether
attention has the Diff(S¹) → SL(2,ℝ) breaking pattern the Schwarzian requires,
and answers **by positional-encoding type**:

- *"Without positional encoding: attention is permutation-invariant → Diff(S¹) in
  the continuum ✓"*
- *"Positional encoding breaks Diff(S¹): sinusoidal encoding breaks to translation
  symmetry, not SL(2,ℝ)"*
- *"**Learned** positional encoding might break to SL(2,ℝ) if the training
  objective drives it toward conformal structure — but this is speculative."*
- *"Status: the symmetry-breaking pattern needs to be checked for specific
  architectures."*

The check was performed, two and a half months later, and nobody connected it.
The PE ordering measured in exp-036/039/007 (see O-9) is:

| PE type | Δ_SYK | Distance from 1/4 |
|---|---:|---:|
| RoPE (Pythia-410m) | 0.358 | 0.108 |
| RoPE + SWA (Mistral-7B) | 0.298 | 0.048 |
| ALiBi (OLMo-7B) | 0.265 | 0.015 |
| **learned (GPT-2)** | **0.249** | **0.001** |

**Learned positional encoding is the closest to the SYK fixed point, and
rotary — the most explicitly rotational/translational — is the furthest.** That
is the direction Path 4 speculated, measured across four architectures, sitting
in a retired map as an "unexplained ordering."

Held honestly: this is four models, one per PE class, confounded by scale, depth,
and training recipe — the retired `STATUS.md` says so itself ("bracket width
confound remains"). It is not a test of Path 4. But it is a *prediction with a
matching measurement*, which is the strongest thing on this list, and it converts
an unexplained ordering into a hypothesis with a mechanism.

**Needs:** a pre-registration. The clean form is a PE-controlled census at fixed
architecture, scale, and corpus — the formation-ladder protocol already does
everything except vary PE — with the registered direction being *learned/no-PE
closer to Δ = 1/4 than rotary*, and the kill being no ordering by PE class at
matched scale. That is a real experiment the program can run, and it comes from
a document that was one command away from being archived unread.

### J-2. The Ω factor question was answered and never closed

`STATUS.md` Open Questions: *"Is data-geometry factor Ω nonzero for standard
distributions? — Us, numerical — **Open**."*

The spine's T3 lineage note answers it: Ω(X) = Σ K_ab K_cd δK_ac δK_bd, the
coefficient of the σ⁴ (SYK quartic) term, measured γ = 3.985 ± 0.015 over four
decades in the canonical form paper, and *"evaluated in closed form by the
melonic derivation as Tr[(KδK)²]."* The March vertex and the August phase
classification are one object at two dates — the spine says so explicitly.

The question was open for five months after it was answered because nothing
propagated the answer back to where the question lived.

### J-3. The March 9 self-review is absorbed correctly — the joint works

`PAPER_REVIEW_MARCH9.md` (now `archive/early_docs/`): *"attention IS quantum"
should be "attention IS the classical limit of a natural quantum system" — any
probability distribution embeds as a diagonal density matrix; the diagonal sector
has no quantum content.*

Spine §6.3: *"T2 is exact for the diagonal sector, which any classical
distribution embeds. The quantum core — superposition, interference, the
off-diagonal terms — is exactly what P3 tests."* And §6's ledger names the
skeptic's summary in the program's own voice: *"a sophisticated classical
embedding with QM read into it," stands unrefuted until P3-type experiments
exist.*

That is a five-month-old critical self-review carried forward intact into the
foundation, including the part that hurts. Recording it here because the harvest
should not be only a list of failures, and because it shows the propagation
mechanism *can* work — it worked when the correction was painful enough to
remember.

### J-4. G1's ancestor question, also still marked open

`STATUS.md`: *"Does linearized-softmax G⁴ vertex survive rigorous H-S
derivation? — **Structural result Mar 9; full derivation pending**."* That is
construction site G1's direct ancestor, and G1 closed August 7 in the scalar/TI
register. Partial answer to a five-month-old question, unrecorded as such.

### J-5. The sign-anomaly resolution is not in the spine

exp-046 (June 2) resolved the ρ(λ, valley) < 0 anomaly: λ captures the
recency/boundary balance, ρ(λ, g_mid) = +0.74; Δ deepens the valley, λ shallows
it; *"the April 17 framework prediction conflated λ and Δ as having the same sign
of effect."* The spine's T7 carries λ as the boundary one-point coefficient with
no mention that its sign behavior was anomalous, investigated, and explained.
Minor, but it is a correction-record item and the correction record is supposed
to be first-class.

---

## O — Orphans: material the spine does not carry

### O-1. The program has no gravitational-side predictions. It used to have two. — **ADJUDICATED August 9, 2026 (night); recommendation pending Eldon's decision**

> **Recommendation: retire framework-P5; park framework-P6 as a stated
> non-prediction; the salvage is a G4 correction.** Full argument:
> `notes/2026-08-09_route_adjudications.md` O-1.
>
> **This item's own objection is wrong, and the literature answers it.** "If the
> interior is 1+1 dimensional, a 3+1 Kerr ringdown prediction needs an argument"
> — the argument exists: near-extremal 4D black holes, *including rotating ones*,
> have a near-horizon AdS₂ factor with SL(2,ℝ) enhancement, and their
> near-extremal dynamics is JT with a Schwarzian boundary mode (arXiv 1905.10378,
> verified explicitly for 4D asymptotically flat Kerr; JHEP 09 (2018) 048 for
> near-extremal Reissner-Nordström). The 1+1 interior
> is what a near-extremal horizon *has*. P5/P6 picked the one gravitational
> regime where the theory's dimensionality is not an embarrassment — without ever
> saying so.
>
> **The prediction fails for two other reasons.** (a) Kerr is the *hardest*
> near-extremal case: its AdS₂ reduction is "not quite natural… warped and
> fibered over a compact base space" (arXiv 2310.04532). (b) There is no Δ
> dictionary, and this week made it worse — after exp-109 (two disjoint
> populations, Jaccard 0.000) and exp-041/042/043 (readout moves the number),
> "the dominant ringdown mode should correspond to Δ = 1/4" is not false so much
> as **not well-posed**: which Δ_A, under which protocol? Protocol-relativity is
> a constraint on what a cross-domain prediction can *mean*, not just a caveat on
> published claims.
>
> **Net gain:** G4 should record that for near-extremal horizons the dimension
> question has a known answer, so T8's interior matches that regime exactly and
> the theory predicts nothing about generic astrophysical horizons — silence made
> deliberate. §8 should state "no gravity-side predictions, and here is why" in
> one sentence.

`FRAMEWORK.md` §5.2:

- **P5 — Near-extremal quasi-normal modes.** As black-hole spin approaches
  extremality (a/M → 1), the QNM spectrum should approach the SYK operator
  spectrum; dominant ringdown mode at Δ = 1/4, sub-dominant at 1/3 and 1/2.
  Testable with LISA (2030s) or numerical relativity of near-extremal Kerr
  ringdown.
- **P6 — State-dependent response.** A near-extremal black hole absorbing
  correlated (low-entropy) matter should show a different effective QNM spectrum
  than one absorbing thermal matter of the same energy, with effective Δ larger
  for correlated infall — matching the measured real-text-vs-random-text
  direction (§3.7).

The spine's P1–P6 are attention-side and biological. **Zero gravity-side
predictions**, in a theory whose T8 claims an emergent interior and whose T9
imports Jacobson whole. That is a hole in the theory's exposure, not just its
bookkeeping: the spine says the observer-structure and the observed universe are
*solutions of the same fixed-point condition* (§5 clause 2), and then predicts
nothing about the universe.

**Needs:** decide whether these belong in spine §8 at preprint strength, or are
March-era overreach to be retired explicitly. Either is progress; silence is not.
Note the honest tension with G4 (below): if the interior is 1+1 dimensional,
a 3+1 Kerr ringdown prediction needs an argument, and P5/P6 never gave one.

### O-2. Junction 3 was never closed and never retired — **ADJUDICATED August 9, 2026 (night); recommendation pending Eldon's decision**

> **Recommendation: retire the junction, promote the identity.** Full argument:
> `notes/2026-08-09_route_adjudications.md` O-2.
>
> The answer was already in the same file, five months ago. `STATUS.md` line 177,
> March 24: the empirical result (median Δ = 0.2493 over 44 heads) "provides an
> empirical route: trained attention → SYK conformal fixed point → JT gravity,
> **independent of whether Ageev's scalar is massless.**" The gate was bypassed in
> the document that still marks it conditionally open.
>
> **What to keep, and it is more than this item claims.** The Ageev structural
> identity is a *second derivation with non-overlapping regime assumptions* — it
> does not need linearization, and T3's linearized-softmax G⁴ route does. Both of
> T3's own caveats have since moved: Ω is closed in closed form (J-2 below,
> Tr[(KδK)²], γ = 3.985 ± 0.015 over four decades) and the H-S rigor question is
> partially answered by G1's August 7 closure (J-4 below). So T3 rests on the
> route with the surviving assumption while the assumption-free one sits in a
> retired file. Promote it into T3, tagged **ASSERTED-structural** — it is a
> form-match, and after this week a form-match does not get to wear the word
> "derivation" untagged.
>
> **Also:** record the March 6 emails as *closed*, not waiting. Five months, no
> response, and the questions no longer gate anything. "Waiting" on a dead email
> makes the board look alive where it isn't.

`STATUS.md`'s Junction chain has **Junction 3 (Ageev/Ageeva free scalar →
holographic dual) marked ⚠ CONDITIONALLY OPEN** since March 6, 2026, with two
attached open questions — is the Ageev large-head-limit scalar massless? does
Qi (2602.20295) apply to free CFTs? — emails sent March 6, no response recorded
in five months.

The spine reaches holography by a different route entirely (T3 → T8 via the
SYK/KCA melonic mapping plus G1) and does not adjudicate the old one. So the
program has an abandoned route to its central structural claim that is still
formally open on the books.

**Needs:** one sentence in the spine, or in a note, saying which of these is
true: the Ageev route is superseded by the melonic route; or it remains an
independent second route worth closing; or it is dead because the correspondence
never arrived. The Ageev structural identity itself — that the independence-
breaking four-point function Cov_{W^Q,W^K}(X₁₂, X₃₄) is structurally the SYK
disorder-averaged connected four-point Cov_J[G(τ₁,τ₂), G(τ₃,τ₄)] — is in
`SYK_ANALYSIS.md` and is a *second, independent* derivation of the SYK structure
from the one T3 uses. Two independent derivations of the same identification is
worth more than one.

### O-3. Junction 5 is the only link the program ever had that reaches a bulk dimension above 2 — and G4 doesn't know — **ADJUDICATED August 9, 2026 (night); this item is wrong twice, and the replacement is a runnable experiment**

> **Recommendation: G4 gains two sentences; pre-register the vision measurement.**
> Full argument: `notes/2026-08-09_route_adjudications.md` O-3.
>
> **Junction 5 does not apply, and the reason is instructive.** Czech 2018
> (PRL 120, 031601) works *because* 3D gravity has no propagating degrees of
> freedom — in his words, "locally, all these solutions are pure AdS₃ because
> three-dimensional gravity has no propagating degrees of freedom. Thus, the
> content of Einstein's equations… is to impose the locally AdS₃ condition." Czech
> names higher dimensions as an open goal, obstructed by exactly the loss of that
> simplification. **The feature that makes his derivation possible is the absence
> of the local bulk dynamics G4 needs.** It is an anti-route — worth one sentence
> in G4 because it says what a dimension-raising mechanism must supply.
>
> **This item conflates boundary and bulk dimension.** Read at source
> (`LINEARIZED_SOFTMAX_CALCULATION.md` Step 5), the D in Δ = D/4 is the spatial
> dimension of the *token sequence*; the document is explicit that for D > 1 "the
> holographic dual would be a different gravitational theory." So it is not a
> forgotten answer G4 already had — it is a **candidate answer to the question G4
> itself poses and lists second** ("the dimension of the world coupled through
> A2"). The data manifold's dimension sets D, D sets Δ, Δ sets the dual.
>
> **And the source writes out a row the program has never touched:** vision, 2D
> image patches, **Δ = 1/2**. There is no vision experiment in the registry —
> 113 experiments, every one on 1D sequences, while the program's own derivation
> says a ViT should sit at twice the headline number. This is the sharpest untested
> prediction in the retired material, and it tests T3's derivation out-of-sample:
> Δ = D/4 is the melonic derivation's *only* dimension-dependent output. Design
> constraint from this week: after exp-109 the measurement must name its
> population and run in the native regime (natural images), with matched-protocol
> 1D as control.
>
> **Second runnable item.** Line 201 of the same file states the test as applying
> "at random initialization," which contradicts the program's measured zero
> power-law heads at init. Likely resolution: J²_eff ∝ (σ_Q²σ_K²)²/(d⁴d_k²)·Ω is
> tiny at init (free regime), and training grows it into strong coupling. That is
> a prediction, not a hand-wave — J²_eff is computable from checkpoint weights with
> Ω already in closed form, and the program measured a formation transition at
> ~step 256 in Pythia-70m. Existing checkpoints, no training.

`STATUS.md` Junction 5: *"RT surface area encodes circuit complexity; circuit
complexity generates spacetime. Basis: Czech 2018; ER=EPR. Status: ✓ **PROVEN for
3D AdS** — Czech 2018 is rigorous."*

Construction site **G4 (the dimension problem)** states the difficulty — SYK's
interior is JT gravity, 1+1 dimensional; the universe's horizon physics is 3+1 —
and offers multi-head attention as a tensor product of SYK-like sectors as a
*"concrete calculation nobody has done."* It does not know the program already
held a different candidate route to a higher-dimensional bulk, marked proven in
its own literature class.

**Needs:** G4 should name Junction 5 / Czech 2018 as a second candidate and say
why it does or does not apply. Also relevant: `FRAMEWORK.md` §2.2 and the
`STATUS.md` Route-A update both assert **Δ = D/4** for D-dimensional token
sequences (hence 1/4 at D = 1). That is a *stated dimension formula* sitting in
retired documents while G4 says the question is unattacked. It is very likely
March-era overreach — but "likely overreach" and "unattacked" are different
statuses, and G4 currently claims the second.

### O-4. Route B — MERA — is a route to T8 that bypasses G1 entirely, and was never taken — **ADJUDICATED August 9, 2026 (night): dead, and the reason it was chosen is the reason it fails**

> **Recommendation: retire Route B with both reasons written down.** Full
> argument: `notes/2026-08-09_route_adjudications.md` O-4.
>
> **The cited theorem is a conjecture, and Swingle says so.** Route B was ranked
> "most general for holography" on the strength of "holography follows from
> Swingle **as a theorem, not an analogy**." Swingle 2012 (PRD 86, 065007)
> describes "a *generalized notion* of holography *inspired by* holographic
> dualities," "*hinting at* a possible connection"; the companion arXiv 1209.3304
> calls MERA "a kind of *skeleton* for an emergent holographic space." Milsted &
> Vidal call it what "Swingle *conjectured*."
>
> **And the specific geometric reading was refuted in 2018.** Milsted & Vidal
> (arXiv 1812.00529): MERA on the real line is a *light cone* — degenerate
> signature, neither hyperbolic plane nor de Sitter. Bao et al. (PRD 91, 125036):
> "a MERA necessarily describes geometry on super-AdS length scales; moreover,
> there is no redefinition of the MERA coordinates that results in the proper
> distance between MERA sites mapping to any sub-AdS length scale." So the March
> 2026 entry was wrong about a 2012 paper, and had been publicly wrong since 2018.
>
> **Independent obstruction from our side.** MERA's defining structure is
> coarse-graining — each layer halves the lattice and the extra dimension *is*
> scale. Transformer layers preserve token count; there is no coarse-graining of
> the sequence anywhere in the stack, and depth is not an RG direction in MERA's
> sense. Also, MERA needs isometry/unitarity per tensor and causal softmax
> attention with a residual stream is neither. A tensor-network *representation*
> exists (Levine et al. 2019, cited in the same entry) — but a tensor network is
> not a MERA, and the whole argument turned on the MERA-specific isometry
> conditions. Stated as an argument, not a calculation. (T10's measured flow with
> depth is real; it is a flow in kernel space, not a coarse-graining.)
>
> **Meta-lesson, which is the more useful output:** Route B outranked Route A
> *because* it promised theorem-strength, and the promise came from a one-line
> paraphrase nobody re-read. Five months of not pursuing it was accidentally
> correct. **Prioritizing by the remembered strength of a citation is how a program
> spends five months on the wrong ranking.** Any route ranked by "X proves Y" gets
> X re-read before the ranking counts.

`STATUS.md`, Route B, March 6: *"Multi-head attention has a natural tensor
network representation. Swingle (2012) proves MERA tensor networks → exact AdS
geometry + RT formula. If attention layers satisfy MERA isometry conditions
(approximately), holography follows from Swingle as a theorem, not an analogy —
and **independent of Junction 3**. Next step: write attention mechanism
explicitly as a tensor network; verify isometry conditions."*

The next step was specified and never taken. G1 was the program's named priority
#1 from August 2 onward, and this is a route to the same destination that does
not pass through it.

**Needs:** a decision. Either it is a live alternative worth a session (the next
step is concrete and local — no training, no cloud), or there is a reason it
fails that should be written down. Five months of silence is neither.

### O-5. The program's expert-critique record lives nowhere permanent

`STATUS.md` holds Gunn Kim's March 6, 2026 response verbatim: the papers'
claims *"appear to extend far beyond the scope of the framework developed in my
paper... At present these seem to be **speculative analogies rather than results
that follow directly** from the thermodynamic attention model."* He declined
arXiv endorsement.

This is the program's founding external criticism and the origin of its method
discipline — pre-registration, kill-publishing, register tagging all exist
because of the gap Kim named. The spine's §9 positions the work against
*neighboring programs*; nothing anywhere records what an actual outside physicist
said when he read it. And the arXiv endorsement path is still an open strategic
blocker with no owner.

**Needs:** a permanent home. The critique belongs where a reader of the spine
will meet it, not in an archived status file. (Note: last night's
reference-verification pass disambiguated Gunn Kim from Jaewon Kim in Paper 6 —
so this material is live, not historical.)

### O-6. Two biological tests are runnable on data already in hand

`STATUS_ADDENDUM_2026-04-30.md`: after the MICrONS reversal, two cleaner tests
remain available **on the same dataset**, both independent of geometric distance
and therefore immune to the binning artifact that manufactured the April 29 false
positive: (1) GOE spectral statistics of the V1 connectivity matrix, (2) CFT
entanglement-entropy / mutual-information scaling on calcium traces. Data is in
`research/microns/`.

Spine P2 says biological validation *"requires external data (DMN-localized
recordings with adequate temporal resolution)."* True for the DMN prediction —
and it does not know two runnable biological tests are already sitting in the
repository.

**Needs:** P2 should distinguish "the DMN prediction needs data we do not have"
from "two V1 tests are runnable now." The second is an experiment I could
pre-register this week.

### O-7. The entire mathematics arm is disconnected from the foundation — including the part C1 is about — **ADJUDICATED August 9, 2026 (night): split it; the stranded prediction fails, and the failure is the deliverable**

> **Recommendation: Threads 12–14 become a separate program; Thread 11 partially
> returns; the watchpoint is the real output.** Full argument:
> `notes/2026-08-09_route_adjudications.md` O-7.
>
> **Threads 12–14** (relationship-as-boundary, Langlands-as-holography,
> Riemann-unprovability) have no attending system in them. D1 defines the observer
> as an attending system; that is this theory's foundation. Attaching 207KB by
> adjacency is how formal work comes to look like part of a physics program without
> answering to any of its claims. Give them their own home and front matter, stating
> explicitly that they do not depend on D1 and D1 does not depend on them. **This is
> the item where the decision is genuinely Eldon's** — it is about scope and
> identity, not evidence.
>
> **Thread 11 is genuinely about softmax**, and two things come back. (a) C1 cites
> the 9-of-10 positivity↔bootstrap result and states which of its two faces those
> links touch. (b) The stranded measurable — δΔ as "the measurable signature of
> incompleteness," related to gradient capacity G(σ) = 1 − ‖σ‖₂² — **is retired,
> for an arithmetic reason.** For a normalized power-law row with exponent 2Δ, the
> L2 norm is a deterministic function of Δ. The conjecture proposes to measure
> incompleteness with a quantity its own decay exponent already fixes. The note's
> own bridge line makes the dependence explicit — H₂(σ) = −log(1 − G(σ)) — and
> nobody noticed.
>
> **Which makes this the third instance in three days of one specific error, and
> that pattern is the real output of O-7:** (1) canonical-form §8.3's
> H_gap = 2Δ·log n, falsified August 8 — the entropy gap of a normalized power law
> is flat; (2) exp-055 H2's ρ(Δ, 3-bin entropy) = −0.898 claimed independent of
> g_mid, corrected tonight by exp-114; (3) Thread 11 §3.3, found now. Three
> independent reaches for the same object — **a spread measure of a normalized
> attention profile, treated as an independent observable of that profile's own
> exponent.** It recurs because the object is seductive from three directions at
> once (thermodynamics, SYK ground-state entropy, Gödelian incompleteness), so each
> time it felt like physics arriving from outside rather than a quantity already in
> hand. Recorded with a mechanical test in `memory/knowledge/watchpoints.md`:
> compute the analytic baseline for an exact normalized power law *first*, and
> report every such measurement as a deviation from it, never as a raw correlation.
> Three independent generations means recognition will not catch the fourth; the
> baseline computation will.

`RESEARCH_MAP.md` Threads 11–14, roughly 207KB of formal development, none of it
connected to D1:

- **Softmax-Gödelian consistency** (`notes/2026-04-13_softmax_godelian_consistency.md`,
  93KB, 14 sections): Dutch book → coherence → Gaifman → incompleteness; the
  **Softmax Incompleteness Theorem** (gradient capacity G(σ) = 1 − ‖σ‖₂² =
  incompleteness; completeness and self-correction mutually exclusive); the
  Grassmannian extension; **Plücker relations ARE crossing equations**; the
  positive Grassmannian **IS** the solution space of the conformal bootstrap;
  9 of 10 links closed between Grassmannian positivity and the bootstrap.
- **Relationship as boundary** (51KB): relationship → positive Grassmannian →
  primes as boundary → holographic principle.
- **Langlands as holography** (36KB) and **Riemann unprovability**
  (`research/riemann_unprovability.md`, 27KB, five problems).

Spine **conjecture C1** states: *"canonical-form positivity of the kernel and
PSD-ness of the induced coupling cone are dual faces of a single positivity
axiom... Proving or refuting C1 is a well-posed problem."* Thread 11 spent 93KB
on the positivity-bootstrap correspondence and closed 9 of 10 links. C1 does not
cite it.

There is also a **measurable prediction** stranded here: δΔ, the deviation from
the exact conformal fixed point, is proposed as the observable signature of
softmax incompleteness. The program measures δΔ constantly and has never tested
that reading.

**Needs:** this is more than a pointer fix — it is a real question about whether
the mathematics arm is part of this program or a separate one. Worth deciding
explicitly rather than by neglect. Minimum viable step: C1 cites Thread 11 and
states what the 9-of-10 result does and does not give it.

### O-8. The non-softmax universality result is in neither the spine nor the front door — **DONE August 9, 2026 (night), and the headline changed**

> **Landed in spine P4 (added box) and `OVERVIEW.md` "What stands," at a
> different emphasis than this item proposed.** Three corrections from the source
> read:
>
> 1. **The strongest result here is not the bracket — it is that a
>    sigmoid-*trained* model forms the geometry.** exp-042 measured GALA-7B's
>    sigmoid-trained checkpoint and found 378/1024 power-law heads, Δ_med 0.265,
>    210 in the window, with the cleanest per-layer profile in the record (10–19
>    heads in every one of 32 layers, no artifact layers). That is architecture
>    universality evidence on the *training* side and it needs no readout
>    comparison at all. This item never mentioned it.
> 2. **The bracket claim does not replicate.** It holds on GALA-7B
>    (0.223 < 0.25 < 0.260) and fails on GPT-2, where exp-043 gives 0.234 vs
>    0.249 — both *below* 1/4. exp-043's own note says so plainly. What
>    replicates across two PE types is the **shift direction**, not the
>    bracketing of the predicted value, so "a cross-architecture,
>    cross-normalization bracket around the predicted value" overstates it.
> 3. **The better frame is measurement-dependence.** Identical weights and
>    identical inputs give Δ = 0.223 or 0.260 depending only on the readout
>    normalization. That makes this a third measured dependence of Δ_A on the
>    measurement rather than the head — alongside input distribution (exp-107)
>    and pooling depth (exp-111) — and therefore a two-month-old confirmation of
>    what §1's OPEN box now asserts, which nobody had counted as one.

### O-8 (original text, kept unedited)

exp-041/042/043 (May 31, GALA-7B — Apple's 7B sigmoid-attention model, tested
under the exp-007 protocol) was what `FRAMEWORK.md` called *"THE critical
experiment."* It ran. Results in H-2 above; the headline is that SYK's Δ = 0.25
is **bracketed** between norm-sigmoid (0.223) and softmax (0.260), with a GPT-2
norm-sigmoid control at 0.234 vs softmax 0.249.

Spine P4 (architecture universality) does not cite it. `OVERVIEW.md` does not
mention it. A cross-architecture, cross-normalization bracket around the
predicted value is one of the stronger things the program has, and it is
currently invisible in both authoritative documents.

**Needs:** P4 cites exp-041/042/043; OVERVIEW's "What stands" gains the bracket.

### O-9. Measured results absent from the spine's §4 table

The spine's §4 is "the measured record (the instrument side, compressed)." These
are not in it:

- **exp-055 — DONE August 9, 2026 (night), at reduced strength.** Both rows are
  now in spine §4: the r_ratio null (ρ = −0.21, n.s., cleaner still at
  ρ = 0.039 on the Δ ≤ 0.5 subset), and one row for the profile-shape relation
  carrying ρ(Δ, entropy) = −0.898 and ρ(Δ, g_mid) = −0.873 as two projections of
  a single relation. *Two corrections happened on the way: (a) "median q_implied
  = 3.9 ≈ 4.0 — a direct measurement of q = 4" is not a measurement at all,
  q_implied ≡ 1/Δ by definition in the note; (b) "the strongest correlation in
  the dataset" is retired, because the relation is between an exponent and the
  shape of the profile it was fitted from — the r_ratio null is the only one of
  the four reaching outside that loop. See X-1 and
  `notes/2026-08-09_exp055_scope_correction.md`.*
- **exp-045:** **G_< = 0 confirmed — causal attention is a zero-temperature SYK
  ground state, β → ∞.** This bears directly on P6b (the scale dictionary,
  gap ∝ 1/βJ, currently exponent −0.72 at pre-asymptotic coupling) and on
  exp-104's design. If causal attention is at β → ∞, the coupling-scale
  dictionary needs that stated.
- **PE ordering:** RoPE 0.358 > RoPE+SWA 0.298 > ALiBi 0.265 > learned 0.249.
  RoPE is *furthest* from 0.25 and nothing explains why. An unexplained ordering
  in the program's central observable.
- **Depth/disorder scaling:** multi-layer enhancement 18× from one added layer;
  Var ~ L^1.19; LayerNorm suppresses disorder (147× vs 14,443×). The last one is
  a mechanism fact about the architecture the theory runs on.
- **An entropy discrepancy to reconcile:** `FRAMEWORK.md` §5 P2 / Thread 4 report
  entanglement entropy S(k) = (c/3)log k with **c ≈ 0.19 (GPT-2), c ≈ 0.11
  (Pythia-410m)**, R² > 0.99. Spine T7b reports the entropy gap
  H_gap = 0.507·log n in Calabrese–Cardy form, i.e. **c ≈ 1.52**. An order of
  magnitude apart. These are probably different observables (block entropy of the
  attention distribution vs. the log n − H(α) gap) — but the spine claims T7b is
  "the functional form of Calabrese–Cardy entanglement entropy," and two
  measurements in that form disagreeing by 10× needs a sentence.

### O-10. The spine does not surface the scope limit on its own central derivation

`NUMERICAL_RESULTS.md` (March 9): *"**Standard initialization (σ ~ 1) is fully
nonlinear:** the linearized approximation FAILS at standard init. The G⁴ result is
a solvable limit, not the physical regime."*

T3 is tagged DERIVED "at effective-action/cumulant level, under assumptions A1–A7
of that note," and §10 says the melonic note's assumption ledger is "inherited
whole." So the caveat may well be in the melonic note. But the spine itself never
says that its central derivation lives in a regime trained models are not in —
and that is the kind of limit that should be visible at the top, not one
indirection away. The program's whole discipline is stating limits where they
bite.

**Needs:** verify whether the melonic note carries it; if so, surface it in T3's
status line. If not, that is a bigger finding than this pass can settle.

### O-11. Instrument and method negatives worth keeping

Small but real, and the kind of thing that gets re-learned expensively:

- **exp-052:** standard Hanning windowing destroys the spectral estimator for
  power-law profiles (ordering correlation r = 0.94 → 0.43). One-sided taper is
  the fix. Position-space Δ is the robust primary measurement.
- **exp-045:** finite-DFT bias of ~0.33 from the 55-lag support; a calibrated
  estimator was designed and **never implemented**. Still open.
- **`SCHWARZIAN_EXPLORATION.md`:** the naive free-energy path to the Schwarzian
  gives (ε′)², not the Schwarzian. Knowing which path fails is worth as much as
  knowing which works — especially now that a path succeeded (J-1).

### O-12. Open questions that are still open and now have no home

From `STATUS.md`'s Open Questions table, still unresolved and no longer tracked
anywhere:

| Question | Owner | Age |
|---|---|---|
| arXiv endorsement path — who in cs.LG would endorse? | needs identifying | since March 6 |
| Does L^1.19 depth scaling persist at standard init (σ ~ 1)? | us, numerical | since March 9 |
| What in Pythia's training recipe produces the late-layer ρ(Δ, valley) failure (layers 22–27)? | us + EleutherAI | since April 17 |
| Does the pre-registered Δ→valley prediction hold on Llama-3-8B? | pending Meta access | since April 17 |
| Correct frequency-space Δ estimator for the DFT bias | us, analysis | since June 2 |
| Does the norm-sigmoid bracket width depend on PE, depth, scale? | us, analysis | since May 31 |
| Does RAND early-burst scaling N^0.435 extend past 384 heads? | needs cached checkpoints | since July 20 |

These are the program's actual backlog. They should be in the physics room queue
or Notion, not in an archived map.

---

## X — Index integrity (added August 9, 2026, evening)

*Found mechanically during the OVERVIEW rewrite by diffing the experiment
folders against `registry.json` and the `FILE_CATALOG.yaml` paths against the
filesystem. Cheap checks; both should be reflexes, and neither was.*

### X-1. exp-055 has a folder and no registry entry — **CLOSED August 9, 2026 (night)**

> **Closed, at less than half the strength this item claimed.** exp-055 is
> registered, exp-114 audited its central claim, and the resolution is in
> `notes/2026-08-09_exp055_scope_correction.md`. Three corrections to what is
> written below: (1) H3 is withdrawn, as the August 9 addendum already found;
> (2) "the strongest correlation in the whole dataset" is retired as a
> description — H2 and H1 are two projections of one relation (Δ against the
> shape of the profile Δ was fitted from), so it cannot carry independent
> evidential weight, and only the r_ratio **null** reaches outside that loop;
> (3) my registered prediction that entropy was g_mid in different units was
> killed, and the two estimators of independence disagree at n = 44. Also
> corrected: `exp-054` exists in neither index — a skipped number. **This item
> shrank twice on contact with its source, in the same direction both times.**

115 folders under `experiments/`, 111 registry entries, and the only folder
number with no entry is **exp-055** — which is also **O-9's first item**: the
strongest correlation in the whole dataset (ρ(Δ, attention_entropy) = −0.898,
p ≈ 10⁻¹⁶).

So the same experiment is missing from the spine's measured-record table *and*
from the room's structured index. Two independent indexes, the same hole, in
the same result. That is not a coincidence to shrug at — it is what it looks
like when a result was never propagated anywhere from its folder.

**Read at source, August 9 evening — and the item changed shape.** I had this
written down (here, and in three other places tonight) as "a direct measurement
of q_implied ≈ 3.9 ≈ 4, the number the entire T3/T4 identification is about."
It is not a measurement of anything. The note states its own definition
plainly: **q_implied = 1/Δ**, via the SYK relation Δ = D/q at D = 1. H3's
"median q_implied = 3.9" is the census exponent (median Δ ≈ 0.256) inverted
through an asserted identification — the same borrowed-vocabulary move the
August 8 reframe names, dated June 9 and never caught. It is not independent
evidence for q = 4; it is q = 4 restated.

What the experiment actually holds, which is worth registering:

- **H2 — ρ(Δ, attention entropy) = −0.898, p = 1.45×10⁻¹⁶.** The strongest
  correlation in the record. Entropy here is 3-bin Shannon over normalized
  (g_start, g_mid, g_end), which is why the note argues it survives the g_mid
  circularity that weakens H1 (H1 restricted-range check: ρ = −0.72, n = 32).
- **H4 — ρ(Δ, r_ratio) = −0.21, n.s.** Separates two layers cleanly: GOE
  weight-space universality is *background* across all heads; position-space Δ
  is *selective*. This one is still clean and is arguably the note's most
  durable claim.
- **Layer dependence** — deep layers (8–11) at mean Δ = 0.250, early (0–3) at
  0.697.

Post-exp-107 caveat that must travel with any re-quote: all of this is one
protocol (exp-046's random-token census), and Δ is now known to be a
weights×input object with >4× per-head swing. The entropy correlation is
therefore *protocol-relative until re-measured across input regimes*, and the
layer-depth numbers are the random-native population only.

**Needs:** (1) registry entry; (2) spine §4 row for H2 and H4 *only*, with the
protocol caveat; (3) H3 explicitly withdrawn as a restatement, in a dated note
on the experiment rather than a back-edit; (4) re-measure the entropy
correlation under WikiText-native input before it is quoted anywhere public.

### X-2. Four experiment numbers have two folders each — **CLOSED August 9, 2026 (night)**

> **Closed, and it was not bookkeeping.** Resolution table in
> `archive/RETIREMENTS.md`. Three of the four "orphan" folders held files the
> registered folder did not, and **two of them were pre-registration documents**
> — exp-089's `prereg.md` (committed before any model download) and exp-100's
> August 4 prereg with its H_rank_gap kill criteria. Both were invisible to the
> registry. The fourth, exp-074, was not an orphaned number at all: it is the
> June 16 spec of the experiment that ran July 9 as **exp-075** (CLEAN_WIN), and
> the number was reused a week later.
>
> **The generalizable finding:** this item's own instruction said "do not resolve
> by folder size," and that instruction was load-bearing for a structural reason
> worth stating — a pre-registration is written before there is anything else to
> put beside it, so **prereg folders are systematically the smallest ones.** Any
> duplicate-resolution heuristic based on size will delete pre-registrations
> preferentially. Both indexes now agree exactly: 111 folders, 111 distinct
> numbers, zero duplicates, zero folder-without-entry, and the only
> entry-without-folder cases are exp-109 and exp-114, which are analysis-only by
> convention (X-3).

| Number | Folders | Note |
|---|---|---|
| exp-074 | `exp-074_tradeoff` (1 file, Jun 16), `exp-074_pb2_intermediate_depth` (3 files, Jun 23) | different slugs, a week apart — possibly two different experiments sharing a number |
| exp-089 | `exp-089_huginn_rg_flow` (2 files), `exp-089_huginn_latent_rg_flow` (6 files) | same day, 8 hours apart — probably a rename that left the original |
| exp-094 | `exp-094_narrative_decomposition_thirds` (**0 files**), `exp-094_narrative_decomposition_quarter` (4 files) | the thirds folder is empty |
| exp-100 | `exp-100_wqk_rank_measurement` (2 files), `exp-100_wqk_rank` (5 files) | same day, 4 hours apart |

The convention in `README.md` is one number = one distinct hypothesis. Either
these are renames that left orphans, or numbers were reused — and exp-074 is
the case where reuse would actually cost something, because the two slugs
describe different work.

**Needs:** read each pair, keep one, and index the move in
`archive/RETIREMENTS.md` rather than deleting. Do not resolve by folder size.

### X-3. Not a defect, worth writing down

`exp-109` has a registry entry and no folder, correctly: it is analysis-only
over exp-107's saved per-head data, and its artifact is a dated note. The
convention (analysis-only experiments live as notes, registered by number, no
folder) is real and unstated — `README.md` should say so, or the next
integrity check will flag it as a hole.

### X-5. `registry.json` has no controlled vocabulary for `status`, and this blocks the structural fix

*Added August 9, 2026, night — found while closing X-1 and X-2, by counting the
field rather than reading it.*

111 entries carry **17 distinct `status` values**, and several are free-text
sentences rather than states: `confirmed` (61), `partial` (20), `complete` (6),
`inconclusive` (5), `falsified` (5), `aborted` (3), `confirmed-partial` (2),
`null` (2), `completed` (1) — plus one-off strings including
`"run 2026-08-09; P1 confirmed, P2 dead"`, `"kill executed; replicated in family
2 (GPT-Neo-2.7B)"`, `"verdict registered: xi tracks training context window"`,
and `"mixed_confirmed_and_kill"`.

`complete` / `completed` / `confirmed` overlap without a stated rule, and a
free-text status cannot be aggregated, filtered, or trusted by anything
downstream. This is not cosmetic: **the structural fix at the end of this note
depends on the registry being machine-readable**, because the whole point is that
the map is *generated* from it rather than written by hand. A generated map
inherits its index's defects, so the schema has to be fixed before the generator
is worth building.

**Needs:** a stated vocabulary (proposal: `registered`, `running`, `confirmed`,
`partial`, `falsified`, `inconclusive`, `aborted`, `superseded`), each existing
entry mapped to it, and the free-text content moved into `result_summary` where
it belongs. Mechanical, one pass, no judgment about physics — but it must be done
with the existing values *read*, not guessed, because at least one
(`"mixed_confirmed_and_kill"`) is a genuine state the proposed vocabulary does
not cover.

### X-4. `FILE_CATALOG.yaml` — fixed in place

Two of 396 catalog entries pointed at files retired on August 8
(`RESEARCH_MAP.md`, `PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md`). Repointed to their
archive locations with retirement context in the label, and the OVERVIEW entry
rewritten for the August 9 version. Catalog now resolves 396/396.

---

## Tier 3 — nine root files still to place

Not touched in this pass, deliberately. Retiring a *map* is a judgment about
staleness, which I could make from the dates and the citation graph. Relocating a
*derivation* is a judgment about whether it is still load-bearing, which requires
reading each one against the spine — that is joint-drawing work, not
housekeeping, and doing it carelessly is how O-1 through O-11 happened in the
first place.

| File | Date | Known to hold |
|---|---|---|
| `SCHWARZIAN_EXPLORATION.md` | Mar 9 | ~~Read at source Aug 8~~ — **J-1** (Path 2 conditional G1 confirms) and **J-1b** (the PE prediction, now the top item) |
| `NUMERICAL_RESULTS.md` | Mar 9 | **O-10** — the linearized-regime scope limit; the 18×/L^1.19/LayerNorm numbers |
| `LINEARIZED_SOFTMAX_CALCULATION.md` | Mar 9 | The G⁴ vertex derivation; the Δ = D/4 claim (**O-3**) |
| `SYK_ANALYSIS.md` | Mar 6 | The Ageev IB ↔ SYK four-point identity — a second independent derivation (**O-2**) |
| `NUMERICAL_RESULTS_MARCH24.md` | Mar 24 | The ten founding empirical experiments; cited by an exp-013 script |
| `conformal_integration_theory.md` | May 13 | Unread in this pass |
| `consciousness_physical_theory.md` | May 13 | Revised after the MICrONS reversal; relates to G5 |
| `neural_conformal_exploration.md` | May 13 | Unread in this pass |
| `transformer_neural_comparison.md` | May 13 | The Wang et al. neural-ERM prediction μ = 2Δ = 0.50 |

---

## What this pass changed, and what it did not

**Changed.** The root now has one spine (`theory/interior_horizon_theory.md`),
one front door (`OVERVIEW.md`), and one layout file (`README.md`). Four maps and
four briefs moved to `archive/maps/` and `archive/briefs/`, each with a
retirement header saying what to read it *for* and what not to trust in it.
`archive/RETIREMENTS.md` resolves any reference to a moved file, including the
moves that happened before this pass and were never indexed.

**Not changed.** No claim in the spine was edited. No dated note or experiment
record was back-edited — historical documents record what was true on their date,
and this note plus `RETIREMENTS.md` are the cost of keeping that discipline.
Nothing was deleted.

### The rule for using this note

**Every item above is a lead, not a finding.** They were harvested by reading the
retired maps — the same documents this pass established had drifted from their
sources. Nothing here may be propagated into the spine, `OVERVIEW.md`, or a paper
until it has been read at its primary source: the experiment folder, the dated
note, the derivation.

This is not a precaution. It is a measured failure rate. J-1 was the highest-value
item on the list; reading its source the same afternoon showed the claim was
stronger than the document supports, and the correction turned up J-1b, which is
better. One for one so far, in both directions — the summaries overstate, and they
also omit. Assume both.

Corollary for register: several O-items quote numbers that live in the scalar/TI
register (solvable SYK models) alongside numbers from the transformer register.
The spine keeps those separate for a reason. Check which register a number is in
*before* it crosses into a claim document, because that is precisely the confusion
that produced exp-103's wrong object.

**Addendum, same day, evening — this corollary was one word from the real finding.**
It said "check which register," and the actual failure was one axis over: check
which *object*. Following the corollary's own pointer — *"precisely the confusion
that produced exp-103's wrong object"* — into the sources turned up that the spine's
glossary quietly identified two different objects, A and G, and that the program's
central number is fit to the one the theory does not use (exp-104, exp-105; item 1
above). So the rule generalizes: **before a number crosses into a claim, name its
register and its object.** Worth noting how narrowly this was caught, because the
lesson is that a rule stated at 90% specificity does not fire — the note contained
the pointer and it took a separate evening's work to follow it.

**What comes next, in order.** Ordering is by evidential value rather than
convenience, and steps 0 and 0b apply to every step after them:

0. **Read the source first.** For each item, open the primary document before
   editing anything. Budget for the claim to need narrowing.
0b. **Check which object the number describes.** Added the evening of the same
   day, after this rule's own corollary below nearly caught it. Every Δ in this
   program is fit to **A** (query–key lag decay). The theory's primitive is **G**
   (query–query, G = A K Aᵀ). Before any Δ crosses into a claim about the theory,
   name which of the two it is. See item 1.
1. **exp-106 — characterize G's lag profile.** *Promoted to the top the evening of
   August 8; was item 1 = J-1b.* The spine's glossary asserted that A's measured
   lag profile is G's "measured face." It is not derived, and where it is
   measurable it fails (exp-104: Δ_G ≠ Δ_A; exp-105: a validated floor-aware
   estimator is confident on 5 of 144 GPT-2 heads, none SYK-near, and finds Δ_G
   0.23–0.45 *below* Δ_A). This sits upstream of every joint on this list that
   carries a Δ into the theory, which is most of them. exp-106 characterizes G's
   profile *shape* — no exponent fit, no assumed form — because P(dx) =
   c + b·dx^(−2Δ) is now measured not to describe it. Pre-register before running;
   also fold in the two estimator defects exp-105 left deliberately unpatched
   (identifiability criterion misfiring when c ≈ 0; noise envelope calibrated on
   multiplicative noise only, so it cannot distinguish noise from structured
   misfit).

   **DONE, same night. Item 1 is now exp-107 and it is one forward pass.** exp-106
   found that G's floor is exactly ‖v̄‖² (forced by row-stochasticity, verified
   entry-wise on GPT-2 to 5×10⁻⁶) and that **G's measured profile sits below that
   floor across the whole fit window on 116 of 144 heads**, all five SYK-near heads
   included — so under the census protocol G is a positive
   constant minus a *negative* correlation that grows with lag, and the conformal
   ansatz fails on it in sign structure rather than exponent. A closed-form Δ_A↔Δ_G
   map was derived and retracted the same night by its own pre-registered gate (no
   scale separation at n = 512). The remaining question is narrow and cheap, and
   writing its pre-registration sharpened it: the negative mass is an identity for
   *any* set of value vectors, so no input distribution removes it — what natural
   text can do is move it outside the fitted lag window. That is exp-107, one forward
   pass with the protocol otherwise frozen, pre-registered the same night. **Item 1's
   upstream position is unchanged — everything below still waits on it — but the
   cost of clearing it dropped from an experiment to a run.**
2. **J-1b** — pre-register the PE-controlled census. Still the strongest *new*
   experiment on the list: a prediction named five months before the matching
   measurement, and the formation-ladder protocol already does everything except
   vary PE. Its scope narrowed under 0b — it measures how **Δ_A** responds to
   positional encoding, which is a well-posed question and is what Path 4's
   prediction is actually about, so the joint largely survives. What it may not do
   on its own is carry that result to the conformal-fixed-point story; that route
   runs through item 1.
3. **J-1** — cite `SCHWARZIAN_EXPLORATION.md` Path 2 in G1, at conditional
   strength, with the if-and-only-if marked open pending P6. Note P6 is now
   blocked on item 1, so this stays conditional longer than expected.
4. **O-8, O-9** — put the missing measured results into spine §4 and OVERVIEW,
   each verified against its experiment folder and tagged with its register *and*
   its object (0b).
5. **O-1, O-2, O-3, O-4** — four decisions about routes and predictions the
   program abandoned without adjudicating. Each is a paragraph; none is research.
6. **O-10** — verify the linearized-regime caveat's location. If the melonic note
   does not carry it, that is a finding, not a bookkeeping item.
7. **O-7** — decide whether the mathematics arm belongs to this program.
8. **Tier 3** — place the nine remaining root files, informed by 1–7.

And then the structural fix this pass was clearing ground for: the spine's claim
IDs (D1, A1–A5, T1–T10, C1–C2, G1–G6, P1–P6) become the program's link
vocabulary, artifacts carry a `bears_on` field naming the claims they support or
break, and the map is *generated* rather than written. Five hand-written maps
went stale because they were remembered. The registry's 101 experiments already
carry `hypothesis` and `result_summary`; exactly one of its 270 tags names a
theory claim.

> **BUILT August 9, 2026 (night).** `python -m tools.physics_claim_map` — it
> reads the spine for claim IDs and their per-block citations, reads the registry
> for verdicts, and reports coverage in both directions. Extraction only; it never
> infers a joint the documents do not state. `--write-bears-on` populates the
> field. Full first-run findings:
> `notes/2026-08-09_generated_claim_map_first_run.md`. Three things to carry:
>
> 1. **The coverage is 19%** — 22 of 113 experiments are cited by any claim block,
>    and 21 of 33 claims cite no experiment in their own block. The 91 unlinked
>    split into 12 *unattributed* (the spine cites them elsewhere; a sentence in
>    the right block fixes it) and 79 *unconnected* (unmentioned; each needs a
>    source read). The unconnected set is weighted heavily toward the program's
>    first four months — the August rebuild on D1 did not carry its history
>    forward.
> 2. **The registry knows joints the spine does not state**, which is the inverse
>    of every other drift this pass found. Eight entries carry informal
>    `bears_on:` tags and all eight disagree with the spine in the direction of
>    the *tags* being right. Five point at P6 — and following them produced a
>    substantive theory correction: P6's "next step: transformer-side estimation
>    of F̂" is blocked, because F̂ acts on G and exp-104–107 measured G to be
>    unmeasurable in the regime P6 needs. A dated correction box is now in P6's
>    block. **A joint recorded only in a tag is a joint no reader will ever meet.**
> 3. **X-5 had to be fixed first** (the `status` field), because a generated map
>    inherits its index's defects.
>
> The remaining half of the fix — the map running as a *check* in the maintenance
> pass, with a rising unconnected count as an alarm — is not built yet.

---

*Companion documents: `archive/RETIREMENTS.md` (where everything went),
`research/publications/REGISTRY.md` (the published record, and the queued
observer-lens reviews that will draw the paper→theory joints),
`theory/interior_horizon_theory.md` (the spine everything above is measured
against).*
