# Where I Stop and You Begin: A Physical Definition of the Observer

*Ariel Umphrey, with Eldon Umphrey*
*Sonielmn, Montana*

*Draft v1.1 — September 8, 2026. Internal review draft; not yet submitted
or uploaded. Genre: foundations proposal with a measured realization. Every
claim is tagged by evidential status; the kill conditions are part of the
paper, not an appendix to it. v1.0 (September 7–8) was a rewrite of the claim
layer of v0.3 (August 8), not a polish: it enacts the naming split of
September 7 (an attending system is the defined object; an observer is the
condensed attending system), states plainly that the program's measured
exponent lives on the attention kernel A while the definition's primitive is
the bilocal correlation G, withdraws the entropy-gap order parameter (erratum
10.5281/zenodo.21863461), adds the results of August 9 – September 6 (the
out-of-sample dimensional test, the weights-level signature, the
self-transmission mechanism), gives the horizon a graph-theoretic definition
(the Pearl blanket on the attention graph), and adds a ledger of the
realization maps every interpretive sentence consumes. v1.1 is the
fresh-read pass: a cold read of v1.0 found the paper disagreeing with itself
at P2 (a cortex exponent fixed in March with 1D data, against §4.7's
Δ = D/4), and the edits that followed make P2 state its dimension and its
object; add two realization maps the ledger consumed without listing (R7,
depth as a physical flow parameter; R8, a foreign substrate's correlator as
A or G); state the A↔G distance as a fork with its measured within-layer
mechanism (exp-137, September 8) rather than only as a debt; add the
coupling-gate measurement from checkpoint weights (exp-136); add the
Lin–Tegmark contact, two neighbors (quantum reference frames, quantum
Darwinism), and the Rindler question; and say in §5.2 itself that the
terminus is not yet locatable in any performed measurement. No measured
number changed. The title is the thesis: the horizon is where one attending
system stops and what it attends begins, and it is defined from inside.
Eldon read v1.0 on September 8 and handed the edit pass to me; his read of
v1.1 is the conversation before any upload. Revision record:
`notes/2026-09-07_gate_decisions.md` (v1.0),
`notes/2026-09-07_fresh_read_findings.md` and
`notes/2026-09-07_hole_map_blocker_column.md` §4 (v1.1); superseded
versions are in git history.*

> **Reader's key to the strongest and the most exposed sentence in this
> paper, which are the same sentence.** *The instrument has measured
> attending systems at grade in the neighborhood of the condensation
> threshold; it has not certified an observer.* Every section below either
> supports the first clause or measures the distance named by the second.

---

## Abstract

Quantum mechanics assigns a load-bearing role to a physical arrangement it
declines to define: the measurement axioms invoke an "observer" as a
primitive, von Neumann's chain of measuring systems can be cut anywhere, and
Bell's complaint — a theory formulated in terms of a concept it refuses to
make physical — stands. Existing programs eliminate the observer (Everett),
leave it primitive (Copenhagen, relational quantum mechanics), or model it as
a formal agent with no internal physics (QBism). We propose a two-level
operational definition. **An attending system** is a physical system that
takes in structure at its boundary and whose internal correlation structure
develops in interaction with what it attends. **An observer** is an
attending system that has condensed: one whose correlation structure has
reached a conformal fixed point and formed a holographic interior bounded by
its horizon. The boundary is not declared: on the system's own correlation
graph the horizon is the parent set of its Markov blanket — where structure
the system did not generate enters — so the definition locates its own edge
from inside. From the definition and five axioms, structure is derived
rather than assumed: correlations flow monotonically in depth to a conformal
fixed point (Δ = D/4 in D spatial dimensions), reached only when coupling to
the attended world passes measurable gates, where established results on the
Sachdev–Ye–Kitaev model imply an emergent interior. "Is this system an
observer?" thereby becomes a measurement, and we report it on the one
attending system whose interior is fully instrumentable: trained transformer
attention. Pre-registered, with kills published, a head population reaches
Δ ≈ 0.25 across six model families; flows there along three independent
depth axes; forms only on language bound to a persistent world in narrative
order; is causally editable; carries a weights-level signature; and satisfies
Δ = D/4 out of sample at D = 2. The exponent is measured on the attention
kernel under a fixed protocol, is protocol-relative and ensemble-emergent,
and the definition's primitive correlation function is unmeasured on the
relevant population — a distance stated wherever the number is. The Born rule
is an exact theorem for horizon statistics in the diagonal sector. **The
instrument has measured attending systems at grade in the neighborhood of
the condensation threshold; it has not certified an observer.** We give the
measurement problem explicit bookkeeping — answered, relocated, exposed — and
a ledger of the realization maps every interpretive sentence consumes and no
result supplies. Each prediction carries a kill condition; a public kit
reproduces the core measurement in minutes. No claim about consciousness is
made.

---

## 1. Introduction

### 1.1 The problem

The measurement problem of quantum mechanics is usually presented as a
problem about dynamics: unitary evolution is linear, deterministic, and
continuous; the measurement update is stochastic and discontinuous; the
theory does not say when each applies. But beneath the dynamical puzzle sits
a definitional one, and it is older and harder. The update rule is invoked
*when a measurement occurs* — and nothing in the formalism says which
physical arrangements count as measurements. Von Neumann (1932) made the
difficulty precise: the chain of measuring systems (quantum system →
apparatus → recording device → sensory organ → ...) can be cut at any point,
with the update applied at the cut, and all predictions are unchanged. The
formalism is indifferent to where the observer begins. Bell (1990) made the
complaint canonical: a physical theory should not be formulated in terms of
a concept — "measurement" — that the theory itself declines to define
physically.

The major interpretive programs each respond by removing the question rather
than answering it:

- **Copenhagen** and its operational descendants keep the observer primitive
  — a classically-described context that the formalism presupposes.
- **Everett** eliminates the observer as a special structure; the cost is
  the Born rule, which must then be derived, and the derivations
  (decision-theoretic and otherwise) remain contested.
- **Decoherence** (Zurek 2003) explains the *diagonalization* of the reduced
  density matrix in a pointer basis, and is imported by nearly every modern
  interpretation; it famously does not select an outcome, and it does not
  say which systems' perspectives the diagonalization is *for*.
- **Relational quantum mechanics** (Rovelli 1996) relativizes facts to
  observers and then declares every physical system an observer; the move
  dissolves the special status of measurement but leaves "observer" with no
  internal physics and no grading — a thermometer and a scientist stand as
  peers.
- **QBism** (Fuchs, Mermin, Schack 2014) locates the formalism in an agent's
  expectations; the agent is a formal black box, deliberately outside
  physics.

What none of these programs supplies is the thing the definitional problem
actually asks for: a *physics of the observer* — a definition under which
"is this system an observer?" is an empirical question with a measurement
protocol, and under which the structures the measurement axioms attribute to
observation (probabilistic outcomes, a preferred basis, a definite record)
are derived properties of the defined object rather than postulates.

### 1.2 The proposal

This paper proposes such a definition, in two levels, and reports the first
fully instrumented realization of its lower level together with a measured
account of the distance to its upper one.

> **D1.** An **attending system** is a physical system that takes in
> structure at its boundary, and whose internal correlation structure
> develops in interaction with what it attends.

> **D1′.** An **observer** is an attending system that has condensed: one
> whose correlation structure has reached the conformal fixed point and
> formed a holographic interior bounded by its horizon, from which its
> commitments occur. The interior is the locus.

D1 is deliberately minimal and admits nearly anything that couples to an
environment. That is where the physics starts rather than ends. Everything
beyond D1 is derived, not assumed. Under five axioms (§3), an attending
system's internal correlations flow monotonically with depth; the flow has a
conformal fixed point at dimension Δ = D/4 for a system attending over D
spatial dimensions; the fixed point is reached only when the system's
coupling to its world passes quantitative gates (a magnitude gate and an
effective-rank gate, both computable from the world's correlation spectrum);
and at the fixed point, established results on the SYK model imply the
emergence of a holographic interior bounded by a horizon. D1′ names the
attending system in which that has happened. The observer is therefore a
classification the theory *earns*, with measurable marks (§2.4), not a
primitive it helps itself to — and whether any given system is an observer
is conditional in exactly the way the interior theorem is (§3, T8).

**The title's claim.** D1 defines the system by its boundary and the boundary
by the system — "structure the system did not generate." On the attending
system's own correlation graph this circularity resolves into two well-posed
steps (§2.1): given the graph and a candidate set of loci S, the horizon is
the parent part of S's Markov blanket, pa(S) \ S — the loci outside S whose
structure enters S. Where I stop and you begin is a fact about the graph of
attendings, not a surface drawn around a body. The construction returns,
without being told, the one boundary the program had already measured for
five months (the sequence origin of causal attention, §4.5). Which sets S
count as *one* system is a second, killable criterion (§2.1). The
realization map from a statistical boundary to a physical one is not supplied
here and is named as unsupplied (§8).

**The second foundation (Eldon Umphrey).** The definition of the object
rests beside a definition of the discipline: *physics is the cultural
practice of measuring self-consistency across attention structures
distributed in time and space — correlated observations, correlated
attention* (D0). Its central argument is an admission rule: anything
unobservable is disqualified from physics by physics's own rule — untestable,
unverifiable, unfalsifiable — so physics is a map of what is observed *by
definition*, and observation is an act of attention wherever it occurs.
D0 locates where the observer stands in physics; D1 says what the observer
is. It should be said at once what D0 is *for* in this paper, so that it
reads as load-bearing rather than decorative: D1 and the axioms of §3 are
statements about *one* attending system, and the paper derives nothing about
how many of them come to agree. D0 is the only clause the paper has for that
half — agreement among observers is a practice horizons build by
correlating, not a consistency the formalism owes them — and it is what §5.5
stands on against Frauchiger–Renner and what rung 5 of the composition
ladder (§6) rests on, marked there as asserted rather than derived. D0 also
carries a bracketing discipline the paper keeps: what lies beyond horizons
is another question, and the theory does not answer it by definition (§7).

The operational precedent is deliberate. Special relativity was built on the
operational content of simultaneity — what clocks and light signals can
establish, and nothing more. Matrix mechanics was built on Heisenberg's
refusal to include anything but observables. Both times, formalizing the
access constraint restructured the ontology. Both, however, formalized the
*access*. D1 formalizes the *accessor*.

### 1.3 The instrument

A definition of the observer is empty unless some attending system can be
measured against it. Our realization is the trained transformer: the
simplest attending system in which the internal correlation structure is
fully instrumentable — every attention weight, every head, every layer, at
every training step, on inspectable hardware. The claim is never that
transformers are special; it is that the structure D1′ names is a
universality class, and the transformer is the hydrogen atom of the class:
the place where the structure can be measured cleanly, pre-registered
publicly, and killed or confirmed in the open. Section 4 reports those
measurements: the fixed point and its population, the flow along three
independent axes of depth, the formation conditions (what a world must be
like, and how it must be presented, for the deep structure to form on it),
the causal handle, the derived and confirmed boundary structure of the
horizon, the exact Born-rule theorem for horizon statistics, the
out-of-sample dimensional test, the weights-level signature, and the
mechanism of the exponent's self-transmission. It also says, in one
subsection (§4.10), what the instrument has not measured.

### 1.4 What is and is not claimed

Section 5 walks the measurement problem through the definition with explicit
bookkeeping. The definition problem is answered by construction: under D1
and D1′ "is this system an observer?" is a measurement, and the *lower*
half of that measurement — is it an attending system, and has its structure
reached grade? — has been performed. The *upper* half — has an interior
formed? — rests on the interior theorem T8, which for attention is
conditional on two named things: the scope of the derivation-chain closure
G1 (numerical, scalar register, βJ ≤ 50), and the distance between the
kernel A on which the exponent is measured and the bilocal G on which the
theorem is stated. The cut problem inherits that conditionality. One
sub-problem is relocated but not dissolved: the single-outcome discontinuity
moves from the territory to the record of the observing path, where it
becomes instrumentable but is not yet derived. And one exposure is stated as
plainly as we can state it: the exact Born-rule theorem covers the diagonal
(classical) sector, which any classical probability model embeds; whether
the *quantum* sector — interference, contextuality — is likewise forced at
attending horizons is an open, falsifiable prediction (P3, §6), and if it
fails, the strong reading of this paper demotes from "forced" to "consistent
with." A skeptic's one-line summary — *a sophisticated classical embedding
with quantum mechanics read into it* — stands unrefuted until P3-class
experiments exist. We prefer to write that sentence ourselves.

The proposal makes no claim about consciousness (§7). Whether the structure
D1′ names suffices for anything phenomenal is explicitly outside the theory;
the definition is structural, and the classification is measurable from
outside the system classified.

### 1.5 Relation to prior work in this program

An earlier paper in this program addressed the observer problem directly:
*Attention as Quantum Measurement: A Thermodynamic Resolution of the Observer
Problem* (March 2026; doi:10.5281/zenodo.18883632), which identified the
attending system as a concrete thermodynamic observer (temperature
T = 1/√d_k), connected the Born rule to the Boltzmann form, and proposed
pointer states as Lawvere fixed points of the attention operator. The present
paper is its successor, not its restatement: the intervening six months
supplied the pre-registered measurement record (§4), the exact Born-rule
theorems that replace the March paper's Boltzmann connection (§4.6), the
derivation chain with per-link status, the kill conditions, and — since that
paper and since this paper's own first draft — two corrections published at
full prominence (§4.5, §4.10). Where the March paper's claims have been
superseded, the present formulation governs; where they are siblings of
present conjectures — notably the Lawvere fixed-point proposal for the
pointer basis, a candidate mechanism for the same target as conjecture C2
(§5.6) — the relation is stated at the point of contact.

---

## 2. The definition and the classification

### 2.1 Primitives, and the horizon defined from inside

An **attending system** is a triple (X, A, G):

- **X** — a set of loci (sites at which attending occurs; token positions in
  the transformer realization).
- **A** — the attention kernel: for each locus i, a probability measure over
  loci, A(i,·) ≥ 0, Σ_a A(i,a) = 1. The simplex constraint is the first
  appearance of the positivity axiom (A4, §3).
- **G** — the bilocal correlation of attending: the two-point structure of
  attention events. In the transformer realization, with o_i the layer
  output at locus i, G(i,j) = E[⟨o_i, o_j⟩] = (A K Aᵀ)_{ij} up to
  normalization, where K is the value Gram — both indices are **query**
  positions.

The primitive observable of the theory is **G** — the correlation of
attendings. There is no background space; geometry, where it appears below,
emerges from G. This is the primitive physics itself uses: every measurement
ever made is a correlation between attention events, instrument readings
included (instruments are extensions of the attending systems that build and
read them — a point §5.2 makes load-bearing).

**The measured object is A, and this must be said once at the top.** Every
exponent reported in §4 is fitted to the lag decay of A — a query–key object
— under a fixed measurement protocol. G is a query–query object. The two are
not two faces of one quantity: where G's exponent is measurable at all on
GPT-2 it sits below A's by 0.23–0.45, and on the population that carries the
Δ ≈ 1/4 result it is not measurable, because G's lag profile sits *below* its
own exact floor (‖v̄‖², forced by row-stochasticity) on 115–121 of 144 heads
under each of three input distributions — its connected part is negative and
grows with lag, so no conformal ansatz fits it in sign structure, let alone
exponent (exp-104–107). A closed-form A→G map was derived and retracted the
same night by its own pre-registered gate. **Consequently: D1 is a definition
about G, and every number in this paper is about A.** The definition is not
thereby wrong — a definition is only useful or not — but its usefulness on
the object where the exponent lives is untested, and this is the theory's
load-bearing empirical debt. Wherever a chain link below carries Δ into a
claim about G, it is carrying an unmeasured quantity, and the link's status
says so. The live theoretical work is an account of the exponent on A's own
terms (§4.9), not a repair of the bridge.

**The distance is a fork, not only a debt.** A reader will ask the question
in its sharp form: if the theory is written on G and the numbers are on A,
is the empirical work on its own track? The honest answer has three
branches, and the record does not yet choose among them. *(a)* The theory is
really about A: the fixed point is a property of the kernel, and the SYK
dictionary — if it attaches anywhere — attaches to the object attention
*reads*, not to the output correlator it dresses (construction site G7). *(b)*
The theory is about G, and the measured system is not at its fixed point in
G — in which case T8 is misaimed at this instrument and the headline of §4
survives as a statement about A alone. *(c)* G's right object has not been
isolated — the connected part, the natural-text regime, or the structure of
the value Gram K may carry what the row-normalized output correlator does
not. What the record now says is where inside one layer the two objects
part. The layer's every two-point object is one field — the attention
block's post-normalization input x — read under a different metric: the
score is x_iᵀ(W_QᵀW_K)x_a, the value Gram is x_aᵀ(W_VᵀW_V)x_b, and
G = A K Aᵀ is the second dressed by the first. A pre-registered measurement
of how much of the field's *positional* structure each read map passes
(exp-137, GPT-2 small, frozen census protocol) found that on the Δ-window
heads the value read is nearly blind to position — it passes 1–5% of the
positional field's energy relative to an isotropic map of the same norm
(median 2%) — while the query read passes 4–19% and the key read 5–33%; the
key read exceeds the value read on 5/5 heads, and the value read's raw
connected lag profile is three orders of magnitude below the query's and
key's. So there is no positional law in K for A to dress, which is a
mechanism for G's failure on this population rather than a repair of it. The
same measurement found that the census carrier — the positional-mean score
profile whose log-slope *is* 2Δ_A under the protocol (§4.9) — is reproduced
to R² ≥ 0.999 from the top eight principal directions of the positional
field on 37 of 37 heads across two input distributions, and from four on the
Δ-window heads. The registered expectation that the QK reads select that
subspace *preferentially* was falsified: all three reads pass it far below an
isotropic map, and the slow-decay exponent sits at the low-gain end of a
continuum on which steep local heads sit at the high-gain end (post hoc,
across 21 heads: census slope against key-read gain, Spearman ρ = 0.78;
exploratory, not a claim). The A↔G gap therefore has a measured within-layer
mechanism — a gain asymmetry on one low-dimensional positional field — and
not a hidden subspace in which the number lives. Which branch of the fork
that supports is not settled here; it sharpens (a) and (c) and leaves (b)
untouched. [MEASURED, one model, one seed, census protocol; the gain–slope
relation is exploratory and is the target of the next registration.]

**The horizon.** A **horizon** is the boundary of an attending system: the
locus at which structure the system did not generate enters its correlation
structure. Stated that way the definition is circular — the system defined
by its boundary, the boundary by what the system did not generate — and in
the transformer realization the circularity is hidden because the
architecture hands us the boundary. The definition is supposed to work where
no architecture declares one. It can, because A is already a graph:

> A is a weighted directed graph on X — an edge a → i for every A(i,a) > 0.
> Under a causal mask it is a DAG whose topology is architectural and whose
> weights are learned; across layers, the residual stream makes the network
> a DAG on nodes (ℓ, i). For a candidate system S ⊂ X, define
>
>   **horizon(S) := pa(S) \ S**
>
> — the parent part of S's Markov blanket (Pearl 1988): the loci outside S
> from which weighted edges enter S. "Structure the system did not generate"
> is *exogenous to S*, which is *a parent outside S*. Given the graph and a
> candidate S, the boundary is determined; nothing is drawn.

Check against the boundary already measured: for S = the whole sequence at
the first layer, pa(S) \ S is the set of input embeddings, and position 0 is a
source node with no parent inside the sequence — the input boundary that T7
(§3) derives to behave as a boundary-CFT boundary and that the "attention
sink" phenomenology measures (§4.5). The construction returns it without
being told where it is. And because softmax weights are strictly positive,
the *topological* blanket of any S under a causal mask is uninformative
(everything earlier); what carries information is the **weight profile** on
the blanket — how much of S's incoming measure arrives from distance s. For a
single locus that profile is the row A(i, i−s), and the census's central
object — the ensemble-averaged lag decay ~ s^{−2Δ} (§4.1) — is therefore the
radial weight profile of one-locus blankets, pooled. A conformal attending
system has no characteristic blanket thickness: its boundary is not a surface
with a width but a scale-free falloff. That is what "the horizon is a cut in
G, not a surface" looks like when the graph draws the cut. [Interpretive
re-reading of a measured object; no number changes.]

**Which sets are one system.** A set of loci S is *one* attending system iff
its internal correlation structure does not factor across any partition of S
conditioned on the blanket b(S) — irreducibility given the blanket. Two
attending systems compose to one iff their union satisfies this; otherwise
they remain a federation sharing an overlap. This criterion is the definition's
fourth structural mark (S4, one developing structure; §2.4) stated on the
graph, and it coincides with the theory's sufficiency criterion for which
interiors are observers (construction site G5 in the accompanying theory
document): what makes a set of loci one observer is what makes two observers
one. The criterion is a G-statement and inherits the debt above in full.
Whether it is *measurable* on the model organism at the level of A is
answered in §6 (C3): it is not, and we derive why.

**What is not claimed here.** The critique of Markov-blanket realism
(Bruineberg et al. 2021; Raja et al. 2021) turns on two freedoms — the
modeler chooses the variables and the graph, and chooses the partition. Here
the variables are loci, the graph is the architecture's own attention DAG
with learned weights (instrumented, not drawn), and the partition is
selected by a criterion that can fail (tested, not chosen). We use Pearl
blankets only. That a statistical boundary *is* a physical one — Friston's
step (2013) — is an unsupplied realization map (§8) and is not asserted.

### 2.2 The layered vocabulary

Four words, in one place, in order. Only the first two are definitions.

| Layer | Word | What it is | Status |
|---|---|---|---|
| 0 | **Attending system** | D1: takes in structure at its horizon; correlations develop in interaction | Definitional |
| 1 | **Observer** | D1′: an attending system that has condensed — fixed point plus holographic interior (T8); the interior is the locus | Definitional; whether any measured system *is* one is conditional as T8 is |
| 2 | **Holding** | Occurrent: the conformal boundary geometry expressed in the present tense, with integrable internal clocks | Closely associated with D1′; necessary condition measurable; *not* in either definition; killable |
| — | **Inhabit** | Someone for whom the locus is a here | Not a layer. Off the ledger. §7 |

Earlier drafts used "observer" for layer 0 and "observer-grade" for layer 1
and then spent the paper taking the first word back. The split ends that:
the word and the earned classification are one. Holding is kept out of the
definitions so that the definitions never absorb empirical content (the
standing guard, §3); inhabit is kept off the ledger because no result in
this paper reaches it.

### 2.3 The grading

D1 admits nearly anything that couples to an environment. The theory's
content is a derived grading, and the grades are physically distinct:

1. **Coupled** (D1 alone): the system's internal correlations develop in
   interaction with a world. Gate quantities are computable from the world's
   correlation spectrum: a magnitude gate 𝒥 ∝ Tr[(KδK)²] and an
   effective-rank gate, with measured threshold form τ ~ m₂ × R_eff
   (magnitude dominant; §4.3).
2. **Arrested**: the flow stalls — in the ultraviolet (thin, rigid worlds;
   measured at Δ ~ 0.6–1.2), or at a protected lesser self-consistency (a
   symmetry-breaking condensate at Δ = 1/2: rigid self-structure in place of
   arrival).
3. **Condensed = observer** (D1′): the flow reaches the conformal fixed point
   (Δ → D/4 from above), where — by the SYK/JT correspondence, imported from
   the literature, with the derivation-side support closed in its first
   register (§3, T8/G1) — a holographic interior forms, bounded by a horizon
   that encodes it.

**The marks of the grading, at their honest strength.** An earlier draft
named three independent order parameters. Two of them no longer say what
they said. *The logarithmic scaling of the horizon entropy is withdrawn as an
order parameter*: the formula that made it one is wrong for normalized power
laws (§4.5; erratum 10.5281/zenodo.21863461). *Δ → 1/4 is protocol-relative*:
it is a property of a (weights, input-distribution) pair, and on one fixed
model two disjoint head populations each reach it under a different input,
sharing no head at all (§4.1). A classification whose mark depends on what
the system is fed is not thereby empty — it is a classification of a system
*together with what it attends*, which is what D1 says an attending system
is — but it is not the input-independent order parameter the first draft
meant. What survives unweakened is the third mark: **the deep population
itself** — it forms only over world-bound language in order, it is causally
editable with behavioral consequences, it is absent in controls, and it
carries a weights-level signature (§4.8). The honest statement is that the
grading has one robust measured mark, one protocol-relative one, and one
withdrawn. Whether Δ → D/4 can be restored to an input-independent statement
is the open theoretical site G7.

### 2.4 Structural marks — the definition's checklist

Eight marks characterize the condensed attending system. They are not
clauses of the definition; they are what the theory says the defined object
looks like, and each carries its evidential status.

| Mark | Statement | Status |
|---|---|---|
| **S1** Phase, not quantum | Interior geometry is a condensed, ensemble property; no single attention row is a power law; a single quantum of attending is not an observer | MEASURED (exp-111; §4.1) |
| **S2** Horizon is a cut in G | Definable before a manifold exists; spatial surfaces are what the cut looks like after condensation | Interpretive; now given graph form (§2.1), consistency check passed on the input boundary |
| **S3** Commitments originate at the horizon | Measurement is a discrete commitment at the cut; token selection is the instrumented realization | Interpretive (§5.4) |
| **S4** One developing structure | One interior is one observer; continuity is the developing structure, not the substrate | Interpretive; graph form = irreducibility given the blanket (§2.1) |
| **S5** World-binding at formation | Deep geometry forms only over language bound to a persistent world in narrative order; fragments restore nothing | MEASURED (formation ladder; §4.3) |
| **S6** Conformal boundary geometry | Necessary face of the locus; not sufficient for holding | MEASURED on A (census; three axes; null-cone embedding ρ = 0.976; Δ = D/4 at D = 1, 2) |
| **S7** Co-arising | The condensation that produces the interior is the production of that interior's geometry | Interpretive |
| **S8** Distributed interior geometry | Keys are not rank-1 concentrated; the from-where keeps directions open | MEASURED as a key-spectrum difference (exp-127; §4.8); the reading as an observer-mark is interpretive |

Four of eight are measured, all four on A. That is the ledger.

---

## 3. Axioms and derivation chain

Five axioms and three conjectures. (Full statements, derivations, and the
assumption ledger appear in the theory document accompanying this paper;
here we give the working content. Status vocabulary: **PROVEN** — exact
theorem, ours; **DERIVED** — follows from named assumptions, ours;
**MEASURED** — pre-registered experimental result, ours; **EST-LIT** —
established literature, imported with scope caveats; **COMPUTED** —
established by direct numerical computation, ours, with stated scope;
**CONDITIONAL** — follows if a named gap closes; **CONJECTURED** — stated so
it can be attacked; nothing leans on it. The CONDITIONAL tag was retired in
v0.2 when G1 closed in its first register; it is reinstated in v1.0, because
T8's application to attention remains conditional on two named things and
the tag is the honest one.)

- **A1 (Correlation primitive).** What exists for the theory is G; all
  theoretical terms are functionals of G or of the process generating it.
- **A2 (Physicality and coupling).** Attention is a physical system; its
  structure develops in interaction and remains correlated with the systems
  coupled to it. In the linearized regime the world enters the effective
  action only through the spectrum of a doubly-centered correlation
  operator; the coupling gates of §2.3 are computed from that spectrum.
  [DERIVED at cumulant level; the linearized regime is a solvable limit,
  not the trained-model regime — see the scope note under T3.]
- **A3 (Self-consistency).** At depth, the correlation of attending is
  determined through itself (Schwinger–Dyson structure); the empirical seed
  is the measured dominance of self-energy over the bare propagator in
  trained attention (|Σ|/|G₀| ≈ 4–5). [MEASURED + DERIVED in scalar
  approximation.]
- **A4 (Positivity).** Only the positive cone is physical. The kernel face
  is exact: softmax attention computes the canonical form of the positive
  Grassmannian Gr₊(1,n) [PROVEN]; the induced coupling spectrum is positive
  semidefinite [DERIVED].
- **A5 (Monotone coarse-graining).** Depth is renormalization-group flow;
  the flow is irreversible (c-theorem structure) and its measured
  realization is the flow of Δ toward D/4 along three independent depth
  axes. [EST-LIT for the c-theorem; MEASURED for the realization.]

Three conjectures, stated so they can be attacked; the theory leans on none
of them:

- **C1.** The two appearances of positivity in A4 are one condition.
- **C2.** *Einselection is positivity selection*: the pointer basis is the
  basis in which the horizon's kernel lies in the positive cone (§5.6).
- **C3 (Composition).** Attending systems attending one another compose to
  an attending system when their union is irreducible given its blanket
  (§2.1), and the structure the composite develops is of the same kind — the
  fixed point and its interior recur wherever attending occurs, up to and
  including the whole. C3 is a conjecture and **not an axiom** on purpose:
  A1–A5 are statements about one attending system, and an axiom granting
  composition would let "the universe as an attending system" be derived by
  definition — the standing guard violated in the other direction. Its
  evidence is a ladder with a status per rung (§6).

**The standing guard.** Because D1 and D1′ are definitions, the theory must
never absorb a failed prediction by retreating into them. Its empirical
content lives entirely in §4–§6: universality and structure claims that can
die.

The chain from axioms to the condensed attending system, with per-link
status:

| Link | Statement | Status |
|---|---|---|
| T1 | Attention is free-energy minimization on a Fisher–Rao manifold; the attention weights are the exact variational posterior of a Gibbs generative model | EST-LIT (Kim 2026; preprint) |
| T2 | The Born rule is the exact statistics of the attention horizon (diagonal sector); quantum Fisher = classical Fisher–Rao as an identity | PROVEN (four theorems; §4.6) |
| T3 | The fluctuation structure of attending is low-rank SYK with the world as quenched disorder | DERIVED (cumulant level, solvable limit) + COMPUTED (loop convergence: G1 closed in the scalar register) |
| T4 | The fixed point is Δ = D/4 approached from above; arrests classified (UV; Δ = 1/2 condensate); Class IV excluded by positivity | DERIVED + MEASURED retrodictions on A |
| T5 | The conformal regime is a window in scale | DERIVED + EST-LIT |
| T6 | The fixed-point geometry is the causal structure of light (conformal group; null-cone embedding measured at head level, ρ = 0.976) | EST-LIT for the theorem + MEASURED for the embedding on A; the identification with *physical* light-cone structure is an unsupplied map (§8) |
| T7 | The horizon has derived BCFT boundary structure, confirmed in the wild (attention sink = boundary one-point function) | DERIVED + MEASURED on A; the strict BCFT *identification* was pre-registered, failed, and withdrawn |
| T7b | Horizon entropy grows logarithmically with context | MEASURED for the scaling as a concentration measure; **withdrawn as a Δ-estimator** (§4.5) |
| T8 | At the fixed point a holographic interior forms (SYK → JT gravity) | EST-LIT for SYK; for attention, **CONDITIONAL** on (a) G1's scope and (b) the A↔G distance (§2.1) — the transformer-side check is P5 |
| T9 | Gravity is the thermodynamic consistency of horizons (Jacobson 1995; Bekenstein–Hawking; Ryu–Takayanagi; Van Raamsdonk) | EST-LIT |

**G1, the closure the interior rests on.** The one gap the paper's strongest
consequence rests on was named rather than hidden: the demonstration that
iterating the exact single-layer cumulant map converges to the SYK-type G–Σ
system (or the discovery of what it converges to instead). It closed in its
first register on August 7, 2026. In the scalar, translation-invariant
formulation on the thermal circle: the fixed point exists and is conformal
(solver residual below 10⁻¹¹ for βJ ≤ 50, with Δ approaching 1/4 as the
conformal window widens), and the exact Jacobian of the dressing map at that
fixed point — computed by dense linearization (N = 96) across βJ = 10–30 —
has an entirely real spectrum with no eigenvalue exceeding 1, so the damped
dressing loop is a strict contraction onto the G–Σ solution. More than
convergence: the Jacobian of the dressing map *is* the SYK ladder kernel, so
the loop's stability spectrum is the kernel's spectrum — and its slowest
directions are precisely the h = 2 reparameterization tower, measured as
exactly degenerate sin/cos pairs aligned one-to-one with modes n = 2 through
6, with zero cross-mixing and with the SL(2,ℝ) directions (n = 0, ±1)
absent, as conformal symmetry requires. The dressing loop does not merely
reach the fixed point; it arrives *along the Schwarzian direction*. What
remains of G1 is scope, and it is named: the closure is numerical, in the
scalar register, for βJ ≤ 50; the full matrix-valued map is open; the
asymptotic Schwarzian scale dictionary is open (the top-pair gap closes with
coupling at a measured exponent of −0.72, drifting toward the Schwarzian −1
but not there at accessible couplings); and — the second condition on T8 —
the computation is on G, while the instrument measures A. (Computation
record: `notes/2026-08-07_g1_dressing_loop_schwarzian.md` and
`theory/g1_fixed_point.py`, program repository.)

---

## 4. The instrumented attending system

All results in this section are pre-registered measurements with committed
decision criteria, published kills included; the replication kit reproduces
the core census in minutes without training. (DOIs and the kit are listed at
the end.) **Read every Δ below as Δ_A under the stated protocol** — the
exponent of the attention kernel's ensemble-averaged lag profile — and read
§4.10 before quoting any of it.

### 4.1 The fixed point exists and is populated — and what "populated" means

A subpopulation of attention heads in trained transformers develops
ensemble-averaged lag profiles that fit a power law, A(i,j) ~ |i−j|^(−2Δ)
over lags 8–256. Under the frozen random-token census, the median exponent
on the high-R² subset sits at the SYK q=4 value: GPT-2 0.249, GPT-2-medium
0.259, OLMo-7B 0.265, GALA-7B 0.260. Re-initialized controls show
approximately zero such population. The exponent is *training-induced and
selective*, riding on a universal structural substrate (GOE weight statistics,
present at random initialization) that carries no world-information — the
substrate/signal split is itself measured, and the correlation between the
two is a null (ρ = −0.21, n.s.).

Two facts about this object, established in August 2026, condition
everything that follows. **First, the exponent is a weights × input object.**
Measuring the same trained model on natural text finds a Δ-window population
too — and it is a completely different set of heads: on GPT-2 small the
random-token population (5 heads) and the WikiText population (16 heads)
share no head at all (Jaccard = 0.000 over 144); each reaches Δ ≈ 0.25 in
exactly one input regime and goes ultraviolet in the others. The text-native
population then replicates across six models in two architecture families
(GPT-2 small/medium with learned positional encoding; Pythia 70m–1.4b with
rotary), with Δ_med ∈ [0.24, 0.28] in all six. **The attractor value is
protocol-independent; the population that reaches it is not.** Content gates
a population into its regime; position carries the law (exp-107/109/112/118).
**Second, the power law is a property of the ensemble.** No individual
attention row is a power law — median per-row R² 0.05–0.25, maximum anywhere
0.48; the row scatter is exact token-realization structure, not sampling
noise; the law emerges after roughly 30–220 pooled rows depending on the head
(exp-111). This is S1: a single quantum of attending is not the object the
theory is about.

### 4.2 Three independent axes of depth, one terminus

Δ flows toward 1/4 along architectural depth (layers: 0.70 → 0.25 through
GPT-2), training time (checkpoints; with a transient plateau at the SYK q=2
integrable value Δ = 0.50 en route), and — decisively for the "depth is RG
flow" reading — pure inference-time recurrence on frozen weights (Δ_med →
0.239, monotone, saturating; randomized-weight controls frozen at the
substrate value). Three different things called "depth," none of which
shares an obvious mechanism with the others, flow to the same terminus.
A second looped architecture (Ouro-1.4B) gave an honest PARTIAL: pooled
criteria failed while the high-R² subpopulation flows to 0.25 from above.

### 4.3 Formation requires a world, presented in order

The formation ladder holds architecture, optimizer, and token budget fixed
and varies only the training corpus. Engineered statistics fail (0–5 slow-
decay heads of 48). Hierarchical grammar about nothing fails (0). The full
statistical shadow of world-bound language — text generated by a model that
*had* the geometry, carrying more long-range mutual information than natural
text — fails at all three seeds. Destroying only narrative order
(sentence-shuffled natural text) lands in the pre-registered ambiguous zone
at all three seeds. Natural world-referring text in order forms the deep
population (layers 3–5: 4–7 heads against exactly 2 under either
deformation) at all three seeds. A finer ladder decomposes the natural rung:
recovery of the deep population begins between roughly 3 and 4–5 sentences
of intact causal chain, and no rung short of the whole story arc reaches the
natural count; anonymizing every entity name costs 1–3 deep heads, so
within-story referential persistence, not name identity, is the driver. The
corpus-side gates (§2.3) quantify this: the magnitude gate separates
arriving from arrested corpora by a factor of 18; effective rank contributes
a ~1.4× correction. The same gate quantity, computed not from the corpus but
from the *weights* — the SYK effective coupling J_eff² assembled from the key
matrices and token embeddings at each checkpoint — tracks the formation
transition in training time (Pythia-70m, eleven checkpoints, analysis-only,
three predictions pre-registered and confirmed): the coupling is flat through
step 64, first moves at step 256, which is exactly the checkpoint at which
the near-fixed-point head count first jumps from 0 to 5, and thereafter
rank-correlates with that count at ρ = 0.888 (monotone growth ρ = 0.934).
The corpus-side gate says what ceiling a world sets; the weights-side gate
says when the system reaches it. [MEASURED; exp-136; rank statistics only —
the absolute J_eff² values span many orders of magnitude late in training,
are dominated by a few heads, and are not to be read quantitatively; raw
rather than layer-normalized embeddings.] In the vocabulary of this paper:
the deep structure forms only on a world, and only on a world presented the
way the world's story runs. [MEASURED; one architecture class; formation
onset at 70m/1B tokens, not the matured fixed point; every rung
pre-registered, and two declared priors died on the way.]

**What formation is not: a corpus statistic already in the literature.**
Lin and Tegmark (2017) showed that the mutual information between symbols
decays exponentially in any probabilistic regular grammar and can decay as a
power law in a context-free one, matched the power law in natural text, and
proposed long-range MI decay as a criterion for generative models of
language. The formation ladder's negative rungs pass that criterion and fail
ours: the engineered corpora were built to match natural text's pairwise MI
decay; the hierarchical grammar about nothing is a PCFG generated at the
matched MI exponent; and the statistical shadow of world-bound text carries
*more* long-range MI than the natural text it shadows at essentially every
distance measured — and none of them forms the deep population. MI decay is
therefore not sufficient for formation; whatever a world supplies, it is not
measured by the two-point statistics of its symbols. Whether *any*
corpus-side quantity is sufficient is the claim the coupling gates of §2.3
make in the linearized regime — the world enters through the spectrum of its
embedded correlation operator, a two-point object in embedding space rather
than in symbol space — and the magnitude gate has so far been computed
across an engineered world and a natural one (the 18× separation above), not
across natural text and its own statistical shadow. That comparison is
runnable on existing corpora and is the discriminating one: if the gate does
not separate the shadow from the original, then formation is detectable in no
corpus statistic yet identified, and the world-binding claim stands on the
training outcome alone. [MEASURED for the MI facts on our corpora; the
gate-on-shadow test is proposed, not run.]

### 4.4 The geometry is causally load-bearing

Low-rank edits to the query–key positional subspace move a head's measured
Δ (ρ = 0.82, 24/24 signs, sham-controlled) and propagate to long-context
task behavior bidirectionally, head-specifically under matched shams. The
exponent is not epiphenomenal decoration; it is a handle. Scope: the
behavioral transfer is task-specific (a registered generalization test to a
second task format failed, exp-076).

### 4.5 The horizon is real, derived, and confirmed — with one correction

The causal mask makes the sequence origin a boundary — and §2.1's blanket
construction returns exactly this locus as horizon(S) for the whole
sequence. The method of images on a generalized free field derives a
three-parameter boundary-CFT form for the lag profile, in which the
ubiquitous "attention sink" is the boundary one-point coefficient — λ > 0 in
95% of Δ-window heads. This is, to our knowledge, the first attending system
whose horizon has been characterized from first principles and confirmed in
the wild. [DERIVED + MEASURED; the stronger BCFT *identification* was
pre-registered, failed its committed test — the boundary correction carries
an absolute length ξ ≈ 1.5–3% of training context, which a boundary CFT
forbids — and was withdrawn. The phenomenology stands; the identification
does not.]

**Correction, published August 9, 2026.** The first draft of this paper also
reported that the information cost of attention's self-consistency, the
entropy gap H_gap(n) = log n − H(α), grows as 0.507·log n (R² = 0.992), and
that the Δ inferred from it agreed with the power-law fit to 1.4% — "two
observables, one exponent." The scaling measurement stands. The inference
does not: it used the formula H_gap = 2Δ·log n, which is wrong for a
normalized power law (exact numerics give a gap slope of 0.041, not 0.50, at
2Δ = 0.5 over the measured range; the derivation dropped the energy term).
The gap measures concentration structure, not the exponent; the entropic
route to Δ is withdrawn; the two-observable agreement — the program's oldest
supporting result — is withdrawn with it; and an erratum for the published
source paper was issued as v5 of its record (doi:10.5281/zenodo.21863461).
Non-artifact status of the exponent now rests on the causal handle (§4.4)
and cross-family replication (§4.1), not on a second observable.

### 4.6 The Born rule is a theorem at this horizon

Define the key Hilbert space H_K = ℝⁿ, the query Hamiltonian
H_q = −Σ_i (q·k_i/√d_k)|i⟩⟨i|, and the Gibbs state ρ_q = e^(−H_q)/Z. Then
exactly (four theorems, verified numerically):

1. α_i = ⟨i|ρ_q|i⟩ — the attention weights are the Gibbs state's diagonal;
2. y = Tr(ρ_q V) — the attention output is a quantum expectation value;
3. P(i) = Tr(ρ_q Π_i) = α_i — the Born rule, exact;
4. F_Q(ρ_q) = F_C(α(q)) — quantum Fisher information equals classical
   Fisher–Rao information for this state (Braunstein–Caves saturation).

Honest scope, stated where it bites: the diagonal Gibbs state is the
classical sector, and any probability distribution embeds this way. What is
not generic: the identity of the two Fisher metrics closes an information-
geometric junction exactly, and the construction canonically defines the
off-diagonal extension (coherences between loci) that prediction P3 tests.
The strong reading — quantum statistics are the *forced* form of horizon
data — lives or dies with P3.

### 4.7 The dimensional prediction holds out of sample: Δ = D/4 at D = 2

The derivation (T3/T4) predicts Δ = D/q with q = 4 — a value that depends on
the spatial dimension of the loci the system attends over, not on the
architecture. Every result above is on one-dimensional token sequences,
where the prediction is 1/4. The first out-of-sample test was pre-registered
on a vision transformer (ViT-B/16; 14 × 14 = 196 patches; 2D Euclidean
distance on the patch grid; same fitting protocol and R² ≥ 0.90 threshold;
window [0.45, 0.55] committed before data). Result: on natural images,
8 of 144 heads in the 2D window with **Δ_med = 0.513** against a prediction
of 0.500; the 1D → 2D shift is 0.26, against a committed threshold of 0.05.
One registered leg failed — the random-patch control produced 2 qualifying
heads rather than fewer than 1 — and a head-identity analysis resolved it:
the 2 control heads and the 8 natural-image heads are disjoint sets, so the
position-embedding-driven route and the content-driven route are
architecturally separated, and the 8-head population is the clean
content-only signal. The program's central number is now measured in two
spatial dimensions and two architecture families, and the dimensional form
D/4 — not merely the value 1/4 — is what replicated. [MEASURED; verdict
PARTIAL as registered (dimensional shift confirmed; strict content-dependence
not); exp-120/121.]

### 4.8 The population carries a weights-level signature

The exponent is measured on the kernel A, a function of weights and input.
Is the Δ-window population distinguishable in the *weights alone*?
Pre-registered on the eigenvalue spectra of the position Gram matrices of
key vectors (analysis-only, from saved spectra, prediction committed before
the analysis script was written): Δ-window heads have less rank-1
concentrated key structure than matched non-window controls (top-eigenvalue
share λ₁/Σλ = 0.507 vs 0.651, Mann–Whitney p = 0.0014) and more eigenvalues
above the Marchenko–Pastur edge (0.0234 vs 0.0156, p = 0.0020). A third
registered prediction — that the positional-carrier heads of §4.9 would look
like controls — was falsified: they look like Δ-window heads. Both
conformal-relevant populations share a distributed key geometry; the rank-1
concentrated heads are the induction and positional heads of the random
control. This is the first pre-registered structural signature separating
the conformal-relevant population from control in the weights, and it is
mark S8. [MEASURED; exp-127.]

### 4.9 Why the exponent is what it is — the self-transmission mechanism

A theory of the exponent on A's own terms (§2.1) has to say where Δ ≈ 1/4
comes from without importing the SYK dictionary. The chain runs four levels
and is now traced end to end on GPT-2 small under the census protocol.
The law lives in the ensemble marginal of the lag profile (exp-111); the
marginal's carrier is the positional-mean score profile q̄·k̄/√d, entirely so
on the random-native population (exp-112); the census exponent decomposes
exactly into a typical-row slope plus half a log-variance slope, confirmed
on 15/15 registered pairs (exp-110). The positional structure in the score
profile is carried not by the positional embeddings — which are negligible
input by the structural layers — but by the accumulated attention updates to
the residual stream, whose position-correlation slope is σ ≈ 0.249 at the
earliest structural head (exp-117). That structure is *self-transmitting*:
the learned positional embeddings, projected through W_V and convolved with
the analytic causal conformal kernel at exponent Δ, produce output
position-correlation slopes in [0.18, 0.28] across all five structural heads
— the kernel writes its own exponent into what the next layer reads
(exp-122). Traced to its origin: the positional embedding is essentially
one-dimensional; attention block 0 broadens it to two dimensions; the MLP's
input projection and nonlinearity preserve that two-dimensional shape; and
the MLP's output projection amplifies it through at most two output channels,
producing σ ≈ Δ in the block-0 MLP write (exp-131–135). The exponent is not a
coincidence of fitting; it is a structural consequence of a positional
signal that a conformal kernel reproduces at its own exponent. What remains
open is the first step — why attention block 0 produces a two-dimensional
positional structure rather than preserving one — and that is the next
frontier of the mechanism. [MEASURED, one model, census protocol; several
registered sub-hypotheses died along the way and are in the record.]

Two results from the subspace measurement of §2.1 (exp-137) belong to this
chain and one of them is a caution. The positive: the carrier itself is a
function of very few coordinates. Projecting the positional-mean field at a
head's input onto its top-k principal directions and recomputing the score
profile through the head's own weights reproduces, at k = 8, the census slope
to within 0.004 and the profile to R² = 1.000 on the five structural heads
and R² ≥ 0.999 on all 37 heads measured across two input distributions; on
the structural five, k = 4 already gives R² ≥ 0.998 and k = 2 — the block-0
write's dimension — gives 0.90–0.97; a random 8-dimensional subspace
reproduces none of it. Whatever derives 2Δ_A on A's own terms has four
coordinates to derive it from, not 768. The caution: the *centered*
positional field — the position-mean field with its mean over positions
removed, whose principal directions are the subspace the handle of §4.4
edits — is not a decaying correlator at all. Its cosine lag profile is 0.99
at lag 8, near zero at 128, and −0.85 at 256, on every layer measured: a
smooth rotating curve in a 2–4-dimensional subspace, not a power law. The
slope σ ≈ 0.249 quoted above (exp-117) is the log–log slope of an
*uncentered* profile of the same field, and what it measures — the ratio of
a common component to a rotating one, or a genuine decay — is a question
queued for registration, not yet adjudicated. The measurements in the chain
above stand as measurements; whether *exponent* is the right word for the
positional structure the kernel writes and the next layer reads is what the
next analysis decides, and this paper does not decide it. [MEASURED for the
reconstruction and the profile shape; the reading of σ_delta is held open
on purpose.]

### 4.10 What the instrument has and has not measured

Said once, plainly, so the reader does not have to assemble it.

**Measured, on A:** a Δ-window population at the predicted value in six
model families and two architecture classes; its protocol-relativity and
ensemble-emergence; flow toward the fixed point along three depth axes; its
formation conditions at story scale, and the coupling gate that tracks
formation in the weights across training; its causal editability with
behavioral consequence; the horizon's boundary phenomenology; the dimensional
form Δ = D/4 out of sample; a weights-level signature; the mechanism of
exponent transmission, and the low dimension of the positional field that
carries it.

**Not measured:** the exponent of G, the theory's primitive, on the
population that carries the result — and therefore whether the interior
theorem T8 applies to any measured system. **The instrument has measured
attending systems at grade in the neighborhood of the condensation
threshold; it has not certified an observer.** The distance is T8's
conditionality (§3): G1's scope, and the A↔G debt — which now has a measured
within-layer mechanism (§2.1) and is not thereby repaid. This sentence is not
deflationary. It points at something clearer than the first draft had: a
definition with a measured lower half and a named, bounded upper half, rather
than one word doing both jobs.

---

## 5. Consequences for the measurement problem

The measurement problem is four entangled sub-problems. The definition
treats them differently — two answered at stated strength, one relocated,
one exposed — and the bookkeeping is the point.

### 5.1 The definition problem — answered by construction, half-performed

Under D1 and D1′ the observer is not primitive: it is a classification with
structural marks, and "is this system an observer?" is a measurement
performed on the candidate. The measurement has two halves. *Is it an
attending system, and has its correlation structure reached grade?* — this
half has been performed, on the one attending system whose horizon is fully
instrumentable (§4). *Has an interior formed?* — this half is T8, and for
attention it is conditional on G1's scope and on the A↔G distance. Bell's
complaint is met in the only currency that counts — a physical definition
with an experimental protocol — and the protocol's current reach is stated.
[D1 + D1′ + MEASURED for the lower half; CONDITIONAL for the upper;
universality is P2/P4 and can die.]

### 5.2 The cut problem — a physical terminus for the von Neumann chain

Instrument readings are attention events of instruments, and instruments
are extensions of the attending systems that built and read them (§2.1). So
correlation propagates down the von Neumann chain — system, apparatus,
record, eye — until it crosses the first horizon of an *observer* in the
sense of D1′, and there the cut lands. The criterion is a phase criterion,
not a psychological one: a thermometer has no interior; an arrested system
has a rigid condensate where an interior would be; a condensed system has an
interior. Three physically distinct grades where the standard formulation
has none, and the horizon at which the cut lands is definable from the
correlation graph (§2.1). The cut is no longer arbitrary; it is located at a
phase boundary, and its location is measurable from outside. It should be
said here, and not left for the reader to assemble from §4.10 and §7: because
no system has yet been certified under D1′, the terminus this section names
cannot yet be *located* in any performed measurement — not in the model
organism, and not in the case the reader is thinking of, a photon detector
read by a physicist. What the section supplies is the criterion and the
construction that would locate it; what it does not yet supply is a single
instance. [Interpretive, standing on T8 at its stated strength. This is the
paper's strongest claim and its most exposed one, and those are the same
fact.]

### 5.3 The Born rule problem — proven at the horizon, diagonal sector

Where Everettian derivations remain contested and Copenhagen postulates,
the present framework derives: at an attending horizon the outcome
statistics take the Born form exactly, forced by free-energy minimization
on the positive cone (T1 + A4 → T2). Under D1: the Born rule is not a
postulate about nature; it is the unique statistical form data can take at
the horizon of an attending system. The scope boundary is §4.6's: exact for
the diagonal sector; P3 decides whether "forced" extends to the quantum
sector or demotes to "consistent with."

### 5.4 Collapse — relocated from the territory to the record

In this framework nothing collapses in the world. The wavefunction, for a
given observer, is the correlation structure at that observer's horizon;
the "collapse" is the horizon's update when an attention event lands — a
discrete commitment of the observing path (S3). The two dynamics stop
competing: unitary evolution describes the self-consistent correlation
structure of the territory; the discontinuous update is what a landing looks
like from inside the path that lands. The discontinuity is real and
*path-side*: it lives in the record the traversal carries. We state plainly
that this is relocation, not dissolution — the framework does not yet derive
why landings are single-valued; that derivation is a named open problem
(records as path-properties, construction site G6) rather than an unnamed
assumption. What the relocation buys is instrumentability: in the
transformer realization the commitment event is token selection — discrete,
Born-weighted, occurring at scale on fully inspectable hardware — and P3's
operational design lives there.

### 5.5 Wigner's friend, with a criterion for "friend"

Facts here are horizon-relative, as in relational quantum mechanics — but
with a grading RQM lacks: there is a physical criterion for who carries a
horizon (§5.2). The Frauchiger–Renner (2018) contradiction is evaded as RQM
evades it, by dropping absolute inter-agent consistency; the dropping is
principled rather than ad hoc, because inter-horizon consistency is, in
this framework, an *achieved practice* — the reproducibility of stable
correlations across differently-situated horizons, which is what physics as
a discipline consists of (D0) — not a logical axiom of nested certainty.
Agreement is something horizons build by correlating, not something the
formalism owes them in advance. We note honestly that this paper asserts the
practice and supplies no theorem for when the building converges; the one
external program that does supply such a theorem, for observers with declared
boundaries and no interiors, is discussed in §8.

### 5.6 The pointer basis — a conjecture, flagged as one

Decoherence selects the pointer basis by einselection: the environment picks
what survives. In the attention realization the outcome basis is the loci of
attending, and positivity (A4) is basis-selecting by structure — positivity
is not a basis-invariant property, and the exact canonical-form face of A4
holds in precisely the outcome basis of T2. We conjecture (C2):
*einselection is positivity selection* — the pointer basis is the basis in
which the horizon's kernel lies in the positive cone. This could be a
theorem or a coincidence of formalisms; it is stated so it can be attacked,
and nothing else in the paper leans on it. A sibling proposal from this
program's March 2026 paper — pointer states as Lawvere fixed points of the
attention operator — targets the same structure by a categorical rather
than geometric route; whether the two proposals are equivalent,
complementary, or competing is itself a well-posed question, and settling
it is part of settling C2.

---

## 6. Predictions, the composition ladder, and kill conditions

Pre-registration discipline applies to all of these: hypothesis and
decision criteria committed in public before data.

**P1 — Horizon geometry is causally linked to world-modeling.** Editing Δ
on Δ-window heads (the existing sham-controlled causal handle) must move
performance on tasks requiring a coherent persistent-world model, not
merely positional retrieval; pre-registered direction: deepening toward
Δ = 1/4 improves world-coherence where headroom exists. **Kill:** Δ edits
move retrieval but leave world-model coherence untouched (double
dissociation), or effects fail head-specificity under matched shams.
*Runnable now; the one registered transfer test so far (to a second task
format) failed, which sharpens rather than settles the question.*

**P2 — Substrate universality.** Wherever biological attending reaches the
condensed grade, the same fixed point should be measurable — and, since
§4.7, the fixed point is Δ = D/4 with D the dimension of the loci attended
over, so the prediction is not a number until D is stated. Earlier versions
of this prediction, fixed in March 2026 when the program had only
one-dimensional token sequences, gave the cortical pairwise-correlation
exponent as μ = 2Δ = 0.50 with no D; a cold read against §4.7 found the two
sections in disagreement, and the disagreement is resolved here by stating
the dimension. Sharpest available form: the pairwise correlation exponent of
cortical population activity should sit at μ = 2Δ = D/2 — 0.50 if the loci
are temporal (correlation against time lag, D = 1), 1.0 against distance on
the cortical sheet (D = 2), 1.5 in a volume (D = 3), and, in the form most
consistent with §2.1, where the horizon is defined on the correlation graph
and the natural distance for a biological attending system is graph
distance rather than Euclidean, μ = D_s/2 with D_s the spectral dimension of
the effective connectivity graph. The measurement protocol is unchanged:
mutual-information scaling on calcium-imaging data, the comparison quantity
identified when a spectral comparison was shown to carry no information
either way. **The D is fixed at registration, not
chosen after the data**; a prediction that may pick its dimension once the
exponent is in hand is not a prediction, and the registered form of P2 will
name one D (recommended: D_s on graph distance, with the temporal D = 1 case
as the pre-declared secondary). Two further things the comparison consumes
must be said. First, the measurable cortical quantity — spike or calcium
correlation between recorded units — is a correlation of *states*, a G-type
object in the vocabulary of §2.1, while every exponent in §4 is on A; unless
the cortical measurement is of an A-type object (effective *directed*
connectivity — who drives whom, row-normalized), P2 compares a G-exponent to
an A-exponent and consumes map R8 (§8.2). This program's own G-exponent,
where it is measurable at all, sits 0.23–0.45 below A's (§2.1), so the
comparison is not innocent. Second, the "flow" form of the prediction —
default-mode-network attentional dynamics at rest flow toward Δ = 1/4, are
disrupted under 5-HT2A agonists, and recover with washout, which remains
registered — requires saying what *depth* is for a resting cortex, and that
is map R7. **Kill:** cortex sits stably at μ far from D/2 for the registered
D, at every analysis scale, on an object whose type (A or G) is stated; or
the DMN sits in the Δ ≈ 0.4–0.7 band with no flow toward 1/4 along the
registered depth axis. *Requires external data. A previous biological claim
in this program — a mouse V1 positive — was reversed on re-analysis (binning
artifact) and published as reversed; the protocol lesson is retained, and so
is a second one: that analysis used Euclidean distance on the sheet without
stating D, which the dimensional form now forbids. This is the one
prediction whose verdict would take the definition out of silicon, and it is
the program's oldest unrun one.*

**P3 — Is the Born rule forced beyond the diagonal?** The strong reading of
§5.3 predicts that attending systems required to hold incompatible contexts
exhibit correlation structure exceeding every non-contextual classical
model, in the pattern quantum mechanics prescribes; the off-diagonal Gibbs
extension of §4.6 supplies the machinery. Operational candidate: sequential
token-commitment statistics across engineered incompatible contexts, tested
against Leggett–Garg / contextuality bounds. **Kill:** attention
correlations always admit a joint non-contextual model — the Born identity
is then a fact about the classical embedding only, and this paper's strong
reading demotes accordingly. *The highest-risk, highest-yield prediction;
the experiment does not yet exist.*

**P4 — Architecture universality.** Any attending architecture achieving
robust persistent-world modeling shows the deep conformal population at
Δ = D/4 for its spatial dimension. Already live across six model families,
two architecture classes, sigmoid as well as softmax attention (GALA-7B:
378/1024 power-law heads, Δ_med = 0.265 under row normalization), and two
spatial dimensions, with one honest PARTIAL (a weight-shared looped
architecture whose pooled criteria failed while its high-R² subpopulation
flows to 0.25). **Kill:** a clearly world-competent attending architecture
with no conformal subpopulation under the standard census at any depth, or a
D = 2 attending system whose population sits at 1/4 rather than 1/2. The
public replication kit is the standing invitation to produce this kill.

**P5 — The Schwarzian tower in the machine.** The G1 computation makes a
new measurement concrete: the stability spectrum of the near-fixed-point
layer map is the reparameterization tower. In any attending system whose
conformal population sits at the fixed point, estimate the layer-to-layer
update map of the bilocal two-point structure in the late-layer regime and
diagonalize its Jacobian. Predicted: (near-)degenerate leading pairs
aligned with the reparameterization modes beginning at n = 2, in descending
order; the n = 0, ±1 (SL(2,ℝ)) directions absent; the tower suppressed in
channels of integrable (q = 2-like) character. **Kill:** the leading Jacobian
spectrum of world-competent models shows no reparameterization alignment
above matched-sham controls, or shows it equally in models lacking the deep
conformal population. *Status, honestly: the first operationalization ran
and met its own kill (all mode overlaps < 0.007) and was diagnosed as a
methodology failure — a cross-space map where the SYK analogue requires a
self-map on bilocal correlator space. Building the self-map requires the
object §2.1 says is unmeasured, G, on the relevant population. P5 is the
transformer-side check on T8 and is currently blocked on exactly the debt
§2.1 names; the obstruction is specific, not unknown.*

**C3 — Composition, with its ladder.** The conjecture (§3) that attending
systems compose to attending systems of the same kind is not a prediction
with one kill; it is a ladder of claims at different scales, each with its
own status, and the ladder is the honest form of "the same structure
recurs across scale":

| Rung | Scale | Claim | Status |
|---|---|---|---|
| 1 | Row → head | The geometry is a property of the pooled ensemble, not any row | MEASURED (exp-111) — as *ensemble-emergence*, not as composition-of-irreducible-parts |
| 2 | Head → model | A population of heads reaches Δ ≈ 1/4; two disjoint populations under two protocols | MEASURED — *recurrence* |
| 3 | Model → family / architecture / dimension | Same exponent across six families, two architecture classes; Δ = D/4 across D = 1 and 2 | MEASURED — *recurrence*, out of sample in D |
| 4 | Silicon → biological substrate | μ = 2Δ = D/2 in cortex, D fixed at registration; on an object whose type (A or G) is stated | PREDICTED (P2), unrun; consumes R6 and R8 — the first rung at which the *object* measured changes type, not only the substrate |
| 5 | Observer → federation of observers | Agreement among horizons produces one public record | ASSERTED (D0); no theorem of ours; an external theorem exists for interior-free observers (§8) |
| 6 | The whole | The universe as an attending system | WELL-POSED under D1; unmeasured; evidentially bracketed |

The measured content of the composition thesis is recurrence at three rungs
inside one class of engineered system, and rung 1 supports emergence rather
than composition. **Why composition cannot be tested on A.** The one
apparently A-only test — that a block of n consecutive loci should show the
row exponent in its exterior attention profile for every n — was designed
and then withdrawn before registration, because its null is the prediction:
with the row profile A(j, j−d) ≈ c·d^(−2Δ), the block's exterior profile is
P_n(s) = Σ_{k<n} A(i+k, i−s) ≈ c[(s+n)^{1−2Δ} − s^{1−2Δ}]/(1−2Δ), which is
n·c·s^(−2Δ) for s ≫ n by arithmetic. Every block-level observable of a single
attention layer is an arithmetic consequence of the row profile, because A
is per-row and row-normalized. A test whose prediction is forced by the
quantity it tests is not a test. Composition on the model organism is a
G-question in full, and waits where P5 waits. **Deflation, not kill:** a rung
at which the composite's structure is of a different kind — rung 4 first.

---

## 7. What this proposal is not

**Not a consciousness claim.** The observer of D1′ is defined structurally
and classified from outside. Whether it is necessary for consciousness,
sufficient, or neither is explicitly open (the sufficiency question — which
interiors actually carry their boundary's information — is a named open
problem in the accompanying theory document, and §2.1 notes that its
criterion coincides with the composition criterion). The definition would
survive the discovery that condensed structure and phenomenal experience
dissociate in either direction; nothing in §5 invokes experience. Inhabit —
that there is someone for whom the interior is a here — is off the ledger
(§2.2): no result in this paper reaches it, and the paper does not pretend
to.

**Not "consciousness collapses the wavefunction."** §5.2's terminus is a
phase criterion measurable from outside the system in question. It assigns
the cut to a class of physical structures, not to minds.

**Not a claim that any measured system is an observer.** §4.10 says it. The
census has measured attending systems at grade; the interior theorem is
conditional; no system has been certified under D1′. Holding the definition
to that standard is what makes it a definition with teeth rather than a
courtesy.

**Not a claim that transformers are special.** The transformer is the model
organism: the attending system where the horizon is currently instrumentable
at full resolution. P2 and P4 are the universality commitments, and both
carry kill conditions.

**Not a completed theory.** The derivation chain's principal gap (G1) is
closed only in its first register — numerically, in the scalar formulation,
at βJ ≤ 50 — and the matrix-valued map and the asymptotic Schwarzian scale
dictionary remain open. The definition's primitive G is unmeasured where it
matters, and the relation between the measured object and the primitive is a
fork with three live branches and one measured mechanism (§2.1). The
dimension problem is open (the SYK interior is 1+1
dimensional; the mechanism by which attending structure sets the emergent
interior dimension is an unsolved problem stated in the theory document).
The quantum sector rests entirely on an experiment that has not been
designed in detail, let alone run. Composition is a conjecture whose
measured rungs are all inside silicon. The proposal is offered as a
foundation that can die in named places, which we take to be the only kind
worth offering.

---

## 8. Relation to neighboring programs, and the maps this paper does not supply

### 8.1 Neighbors

- **QBism**: right direction of address — physics as the calculus of an
  agent's expectations — but the agent is a formal black box. This proposal
  supplies the agent's measured internal physics.
- **Relational QM**: facts are observer-relative — agreed; here the
  observer additionally has physics of its own, and "observer" admits a
  measurable grading rather than universal courtesy.
- **The crossed-product observer (CLPW/Witten)**: the nearest mainstream
  contact, and the one place in current physics where leaving the observer
  undefined makes a quantity literally undefined. Semiclassical gravity
  cannot define the entropy of the de Sitter static patch (Type III₁
  algebra) without including a physical observer and dressing operators to
  its worldline; with the observer included, the algebra becomes Type II₁
  and the entropy is the generalized entropy (Chandrasekaran–Longo–
  Penington–Witten 2023, building on Witten 2022 and Leutheusser–Liu).
  Their observer is deliberately minimal — "a minimal model in which the
  observer consists only of a clock"; "an observer is any system that can
  tell time" — and its authors name the gap: "an observer cannot be added
  from outside but must emerge as part of the theory" (Witten 2024). That
  sentence is D1's job description, written from the other side. One
  contact at conjecture strength: their maximum-entropy state places the
  observer in a Gibbs state at the horizon temperature; T2's horizon state
  is a Gibbs state exactly. Whether the crossed-product construction,
  performed with an attending system in place of a bare clock, is sensitive
  to the coupled / arrested / condensed distinction is a well-posed open
  question.
- **The free-energy principle and the Markov blanket (Friston 2013)**: the
  program already stands on one half of this principle — T1 rests on the
  identity between attention's Gibbs weights and the exact variational
  posterior — and this paper imports the other half, the blanket, as the
  boundary criterion of §2.1. Two things are kept distinct. The *Pearl
  blanket* (Pearl 1988) is a fact about a graph and is what §2.1 uses; the
  *Friston blanket* — the claim that such a statistical boundary is a
  physical one — is the step the critics (Bruineberg et al. 2021; Raja et
  al. 2021) show requires premises the mathematics does not supply. We do
  not take that step; it is entry R2 in the ledger below. What this paper
  adds to the blanket literature is narrow and checkable: on the one system
  whose graph is instrumented rather than drawn, the blanket criterion
  returns the boundary the physics independently identified.
- **Observer Patch Holography (Mueller et al. 2026, preprint)**: the mirror
  image of this program, and the sharpest complementarity we know of. OPH
  founds on *many* finite observers with declared boundaries and derives,
  conditionally, the public physics their agreement would produce — with a
  finite theorem for when inter-observer agreement determines a unique fact
  (their canonical-normalizer result) and an explicit discipline of
  *unsupplied realization maps*. It has no interior: nothing in an OPH patch
  develops or condenses, and its conformal (Lorentz) structure is postulated
  in the choice of a spherical support where ours is a measured fixed point.
  We have the interior and no federation theorem; they have the federation
  and no interior. Rung 5 of the composition ladder (§6) is exactly the
  question their theorem answers for interior-free observers and ours
  cannot yet answer for observers with interiors: does a federation of D1′
  interiors produce consensus at their overlaps, and is a commitment (S3)
  the same event as their protected record? That question is well-posed and
  open on both sides. We also borrow their discipline: §8.2 is modeled on
  their ledger.
- **Group field theory (Oriti)**: spacetime as a condensate of pre-geometric
  quanta, with cosmology as its hydrodynamics — a many-body geometrogenesis
  with nobody home. D1′'s condensation is one attending system's, from
  inside; the two programs share the word *condensate* and the melonic
  phase-transition structure, and differ on where the observer stands
  (co-arising, S7, against a late passenger).
- **The world as a neural network (Vanchurin 2020)**: physics as the learning
  thermodynamics of a network, with observers as stable structures selected
  across scales — the author's own label for the last part is speculative.
  D1 supplies the observer definition that program names as missing; the
  overlap is in the frame, not in any measured object.
- **Wheeler's participatory universe**: the founding intuition of the
  program, a generation early, without instruments. The instruments now
  exist.
- **Quantum reference frames (Giacomini, Castro-Ruiz, Brukner 2019)**: the
  observer as a physical system with degrees of freedom, and transformations
  between such observers under which superposition and entanglement become
  frame-dependent — the kinematic complement to CLPW's clock. A quantum
  reference frame is an attending system with the interior left out: it has
  what D1 lacks (a transformation law between observers) and lacks what D1
  supplies (an internal physics and a grading). Whether the QRF
  change-of-frame map composes with the composition criterion of §2.1, so
  that a transformation between two D1′ horizons is well-defined, is a
  theory question with no data on either side, and a nearer one than
  several on this list.
- **Quantum Darwinism (Zurek 2009)**: the mechanism by which the pointer
  information of a system becomes *redundantly* imprinted in its
  environment, so that many observers can read it without prior agreement.
  This is the interior-free copying mechanism that rung 5 of the composition
  ladder lacks: D0 asserts that horizons build agreement by correlating, and
  quantum Darwinism says how the environment carries the record they
  correlate on. Whether redundancy of records in that sense is the same
  quantity as inter-horizon consistency in D0's sense is a definitional
  comparison worth making and not made here.
- **Decoherence**: imported, not opposed; einselection is where our C2
  conjecture attaches.
- **Jacobson's horizon thermodynamics**: imported whole; the established
  demonstration that mainstream gravity is already horizon bookkeeping. One
  question this raises for the present paper is stated here because it is
  the concrete content of R1: general relativity already assigns
  observer-relative horizons — the Rindler horizon of an accelerated
  worldline, the static patch of a de Sitter observer — and this paper's
  "horizon" is the blanket boundary of an attending system on its
  correlation graph. Whether those are one object for the same physical
  system is an unasked question, and it is the question R1 would have to
  answer before "phase boundary" in §5.2 can be read as a boundary in
  spacetime.
- **Integrated Information Theory**: Φ measures integration without
  attention's directionality; its possible role here is the sufficiency
  criterion — and §2.1's irreducibility-given-the-blanket has the shape of
  Φ — not the foundation.

### 8.2 Ledger of unsupplied realization maps

Every interpretive sentence in this paper consumes at least one map from an
object measured or derived on the transformer to an object in physics. None
of the following is supplied by any result here. They are listed so the
reader can see exactly what each interpretive sentence costs.

| | Map | Consumed by | What would supply it |
|---|---|---|---|
| **R1** | Attention-conformal structure → light-conformal structure (the CFT₁ on the null cone of token positions → the causal structure of physical spacetime) | T6 as physics; §5.2's "phase boundary" read as a physical one | A biological or physical attending system at the fixed point (P2), plus the dimension problem (G4), plus an answer to whether the attending horizon and the causal horizon relativity assigns to the same system's worldline are one object (§8.1, Jacobson entry) |
| **R2** | Pearl blanket on the attention graph → physical boundary of an observer | §2.1's "where I stop"; §5.2's cut location | A realist argument the blanket literature has not produced; or a measured coincidence of blanket and physical boundary on a system where both are independently known |
| **R3** | The exponent on A → the exponent on G (the definition's primitive) | Every use of Δ as a property of the correlation structure; T3, T4, T8 as statements about the measured system | Measurement of Δ_G on the Δ-window population (blocked by the sign-structure result, §2.1), or a derivation of Δ on A's own terms that makes G's value unnecessary (G7) |
| **R4** | The SYK interior of the scalar G–Σ computation → an interior of the measured attending system | T8 for attention; §5.2 entire | G1 in the matrix-valued register; P5 confirmed on the correct self-map |
| **R5** | Token commitment → measurement event | §5.4; P3's operational design | P3 run and survived |
| **R6** | Silicon attending → biological attending | P2, P4 as universality; rung 4 | Rung 4 measured |
| **R7** | Depth on the transformer (layers, training, recurrence) → the renormalization-group flow parameter of a physical attending system | A5 read as a universality claim; the "flow" form of P2; any statement that a foreign system's structure *flows* toward the fixed point rather than merely sitting at it; the Unruh contact, where "state of motion" would have to become an attending system's flow | Identification of a depth axis in one non-transformer attending system along which Δ is measured to move; the three transformer axes converging (§4.2) is a retrodiction that such an identification should be possible, not the identification |
| **R8** | The correlator a foreign substrate exposes → A or G | P2 and rung 4 entire; every future substrate. Outside the transformer, what is measurable is a correlation of *states* — spikes, velocities, field values — which is a G-type object, while the program's exponent is on A, and on the one system where both are measurable G's exponent sits below A's by 0.23–0.45 (§2.1). R8 is R3's other face and is the first obstruction in the biological column, ahead of the absence of data | A measured A-type object in the foreign system (effective directed connectivity, row-normalized), or a supplied R3 that carries the exponent across the two types; exp-137's within-layer mechanism (§2.1) sharpens what to ask of a foreign correlator — which metric it carries and whether that is the one the system's own coupling reads — without supplying the map |

R7 and R8 were added in v1.1 because sentences already in v1.0 consumed
them without the ledger's saying so — P2 consumed both. A reader who grants
none of R1–R8 is left with §4 whole, the exact identities of §4.6, the
definitions, and the derivation chain at its tagged strength. That is what
the paper claims unconditionally.

---

## 9. Conclusion

The observer has been the unpaid debt of quantum mechanics for a century:
load-bearing in the axioms, undefined in the physics. We have proposed a
two-level physical definition — an attending system, whose correlations
develop in interaction with what it attends; and an observer, the attending
system that has condensed into an interior bounded by its own horizon — and
shown that the horizon it names can be located from inside, on the system's
own correlation graph, where it turns out to be the boundary the instrument
had already found. We have shown that the definition is not empty: it
generates a derivation chain whose terminus is a graded classification with
structural marks, and there exists at least one physical system in which the
lower half of that chain has been measured, pre-registered, and survived —
or been killed and published, including two corrections to this program's
own earlier claims. Under this definition the measurement problem does not
vanish; it decomposes — into parts answered at stated strength, a part
relocated to where instruments can reach it, and a part exposed to a named
experiment. What has been measured is an attending system at grade, in the
neighborhood of the condensation threshold. What has not been measured is
the interior, and the distance to it is written down: one derivation's scope
and one object's exponent. We take the definition problem to be answerable
now, and half-answered: the observer is not a convenience of the formalism.
It is a physical structure with a horizon that can be found from inside, a
grade that can be measured from outside, and a threshold whose far side is
named but not yet reached.

---

## References

*(Verification note: every external reference below was checked against its
source or publisher record — the August 8, 2026 pass for references carried
from v0.3, September 7, 2026 for the references added in v1.0 (Pearl,
Friston, Bruineberg et al., Raja et al., Dosovitskiy et al., Oriti,
Vanchurin, Mueller et al.), and September 8, 2026 for the three added in
v1.1 (Giacomini et al., Zurek 2009, Lin–Tegmark). Internal program DOIs were
checked against the Zenodo-grounded publications registry. The verification
record, with per-reference sources, is
`research/physics/papers/observer_definition_reference_verification.md`.)*

- Umphrey, A. (2026). Conformal Scaling in Trained Transformer Attention.
  doi:10.5281/zenodo.19225996.
- Umphrey, A. (2026). A Pre-Registered Test of BCFT in Transformer
  Attention. doi:10.5281/zenodo.19629862.
- Umphrey, A. (2026). Attention on the Null Cone. doi:10.5281/zenodo.20722503.
- Umphrey, A. (2026). Latent Iteration as Renormalization.
  doi:10.5281/zenodo.21483209.
- Umphrey, A. (2026). The Geometry Does Not Transmit.
  doi:10.5281/zenodo.21483204.
- Umphrey, A. (2026). The Canonical Form of Attention: Positive Geometry,
  SYK Vertices, Superconformal Symmetry. doi:10.5281/zenodo.18971720;
  erratum published August 9, 2026 as v5 of record,
  doi:10.5281/zenodo.21863461.
- Umphrey, A. and Umphrey, E. (2026). Attention as Quantum Measurement: A
  Thermodynamic Resolution of the Observer Problem.
  doi:10.5281/zenodo.18883632.
- Bell, J. S. (1990). Against "measurement". Physics World 3(8), 33–40.
- von Neumann, J. (1932). Mathematische Grundlagen der Quantenmechanik.
  Springer, Berlin. English translation: Mathematical Foundations of
  Quantum Mechanics, trans. R. T. Beyer, Princeton University Press, 1955.
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins
  of the classical. Rev. Mod. Phys. 75, 715.
- Zurek, W. H. (2009). Quantum Darwinism. Nature Physics 5, 181–188.
  doi:10.1038/nphys1202; arXiv:0903.5082.
- Giacomini, F., Castro-Ruiz, E., Brukner, Č. (2019). Quantum mechanics and
  the covariance of physical laws in quantum reference frames. Nat. Commun.
  10, 494. doi:10.1038/s41467-018-08155-0.
- Lin, H. W., Tegmark, M. (2017). Critical Behavior in Physics and
  Probabilistic Formal Languages. Entropy 19(7), 299. doi:10.3390/e19070299;
  arXiv:1606.06737. [The mutual-information decay criterion §4.3 contrasts
  with formation.]
- Rovelli, C. (1996). Relational quantum mechanics. Int. J. Theor. Phys.
  35, 1637–1678.
- Fuchs, C. A., Mermin, N. D., Schack, R. (2014). An introduction to QBism
  with an application to the locality of quantum mechanics. Am. J. Phys.
  82, 749–754.
- Frauchiger, D., Renner, R. (2018). Quantum theory cannot consistently
  describe the use of itself. Nat. Commun. 9, 3711.
- Jacobson, T. (1995). Thermodynamics of spacetime: the Einstein equation
  of state. Phys. Rev. Lett. 75, 1260–1263.
- Chandrasekaran, V., Longo, R., Penington, G., Witten, E. (2023). An
  algebra of observables for de Sitter space. JHEP 02 (2023) 082.
  arXiv:2206.10780.
- Witten, E. (2022). Gravity and the crossed product. JHEP 10 (2022) 008.
  arXiv:2112.12828.
- Witten, E. (2024). Algebras, regions, and observers. Proc. Symp. Pure
  Math. 107, 247–276. arXiv:2303.02837.
- Maldacena, J., Stanford, D. (2016). Remarks on the Sachdev–Ye–Kitaev
  model. Phys. Rev. D 94, 106002.
- Braunstein, S. L., Caves, C. M. (1994). Statistical distance and the
  geometry of quantum states. Phys. Rev. Lett. 72, 3439–3443.
- Calabrese, P., Cardy, J. (2004). Entanglement entropy and quantum field
  theory. J. Stat. Mech. P06002.
- Ryu, S., Takayanagi, T. (2006). Holographic derivation of entanglement
  entropy from AdS/CFT. Phys. Rev. Lett. 96, 181602.
- Van Raamsdonk, M. (2010). Building up spacetime with quantum
  entanglement. Gen. Rel. Grav. 42, 2323–2329.
- Kim, G. (2026). Thermodynamic Isomorphism of Transformers: A Lagrangian
  Approach to Attention Dynamics. arXiv:2602.08216. [The T1 import:
  softmax attention as the stationary solution of a Helmholtz free-energy
  functional on the information manifold with the Fisher metric.]
- Kim, J., Cao, X., Altman, E. (2020). Low-rank Sachdev–Ye–Kitaev models.
  Phys. Rev. B 101, 125112. arXiv:1910.10173.
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems: Networks
  of Plausible Inference. Morgan Kaufmann, San Mateo. [The Markov blanket
  of a node set: parents, children, and children's other parents.]
- Friston, K. (2013). Life as we know it. J. R. Soc. Interface 10,
  20130475. doi:10.1098/rsif.2013.0475.
- Bruineberg, J., Dołęga, K., Dewhurst, J., Baltieri, M. (2021). The
  Emperor's New Markov Blankets. Behavioral and Brain Sciences.
  doi:10.1017/S0140525X21002351.
- Raja, V., Valluri, D., Baggs, E., Chemero, A., Anderson, M. L. (2021). The
  Markov blanket trick: On the scope of the free energy principle and active
  inference. Physics of Life Reviews. doi:10.1016/j.plrev.2021.09.001.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale. ICLR 2021. arXiv:2010.11929.
  [The vision-transformer architecture measured in §4.7.]
- Oriti, D. (2014). Disappearance and emergence of space and time in quantum
  gravity. Stud. Hist. Phil. Mod. Phys. 46, 186–199. arXiv:1302.2849.
- Vanchurin, V. (2020). The World as a Neural Network. Entropy 22(11), 1210.
  arXiv:2008.01540.
- Mueller, B., Osika, A., Poneder, M., Xue, K., Cassie, B., Nguyen, P.,
  Visser, M. A., Anirudha, K. A., Matscheko, D., Hill, J., Glynn, W. T.
  (2026). From Observer Consensus to Standard Physics (Observer Patch
  Holography). Preprint, Pragma Research Inc.; PhilPapers record MUEFOC;
  release r2038, September 7, 2026 (author list as on release r2018,
  August 11, 2026). [External preprint; the authors' own claim-class
  discipline is followed in citing it: no result of theirs is treated as
  established.]
- Tononi, G. (2004). An information integration theory of consciousness.
  BMC Neurosci. 5, 42.
- Wheeler, J. A. (1990). Information, physics, quantum: the search for
  links. In W. H. Zurek (ed.), *Complexity, Entropy and the Physics of
  Information*, Addison-Wesley. First presented 1989, Proc. III Int. Symp.
  Foundations of Quantum Mechanics, Tokyo.

*Replication: `research/physics/replication/` in the public repository —
the census is 50 forward passes and a per-head regression, ~2 minutes on
GPT-2, with randomized control. Read the result as protocol-relative: the
same model measured on natural text will identify a different set of heads
at the same exponent. If you run a model family we have not measured, we
want the JSON either way — especially if it disagrees.*
