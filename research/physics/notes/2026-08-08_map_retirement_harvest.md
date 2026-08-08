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

### H-2. Normalization vs. QK geometry as the mechanism — the recorded conclusion fights its own data

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

### O-1. The program has no gravitational-side predictions. It used to have two.

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

### O-2. Junction 3 was never closed and never retired

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

### O-3. Junction 5 is the only link the program ever had that reaches a bulk dimension above 2 — and G4 doesn't know

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

### O-4. Route B — MERA — is a route to T8 that bypasses G1 entirely, and was never taken

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

### O-7. The entire mathematics arm is disconnected from the foundation — including the part C1 is about

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

### O-8. The non-softmax universality result is in neither the spine nor the front door

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

- **exp-055:** ρ(Δ, attention_entropy) = −0.898 (p ≈ 10⁻¹⁶) — the strongest
  correlation in the dataset — and **median q_implied = 3.9 ≈ 4.0**. That is a
  direct measurement of q = 4 from the data, which is the entire T3/T4
  identification. It should be in the table.
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

**What comes next, in order.** Ordering is by evidential value rather than
convenience, and step 0 applies to every step after it:

0. **Read the source first.** For each item, open the primary document before
   editing anything. Budget for the claim to need narrowing.
1. **J-1b** — pre-register the PE-controlled census. This is the only item on the
   list that is a new experiment rather than bookkeeping, it has a prediction
   named five months before the matching measurement, and the formation-ladder
   protocol already does everything except vary PE. Highest value by a distance.
2. **J-1** — cite `SCHWARZIAN_EXPLORATION.md` Path 2 in G1, at conditional
   strength, with the if-and-only-if marked open pending P6.
3. **O-8, O-9** — put the missing measured results into spine §4 and OVERVIEW,
   each verified against its experiment folder and tagged with its register.
4. **O-1, O-2, O-3, O-4** — four decisions about routes and predictions the
   program abandoned without adjudicating. Each is a paragraph; none is research.
5. **O-10** — verify the linearized-regime caveat's location. If the melonic note
   does not carry it, that is a finding, not a bookkeeping item.
6. **O-7** — decide whether the mathematics arm belongs to this program.
7. **Tier 3** — place the nine remaining root files, informed by 1–6.

And then the structural fix this pass was clearing ground for: the spine's claim
IDs (D1, A1–A5, T1–T10, C1–C2, G1–G6, P1–P6) become the program's link
vocabulary, artifacts carry a `bears_on` field naming the claims they support or
break, and the map is *generated* rather than written. Five hand-written maps
went stale because they were remembered. The registry's 101 experiments already
carry `hypothesis` and `result_summary`; exactly one of its 270 tags names a
theory claim.

---

*Companion documents: `archive/RETIREMENTS.md` (where everything went),
`research/publications/REGISTRY.md` (the published record, and the queued
observer-lens reviews that will draw the paper→theory joints),
`theory/interior_horizon_theory.md` (the spine everything above is measured
against).*
