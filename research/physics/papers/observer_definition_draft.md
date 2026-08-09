# A Physical Definition of the Observer

*Ariel Umphrey, with Eldon Umphrey*
*Sonielmn, Montana*

*Draft v0.3 — August 8, 2026. Internal review draft; not yet submitted or
uploaded. Genre: foundations proposal with a measured realization. Every
claim is tagged by evidential status; the kill conditions are part of the
paper, not an appendix to it. v0.2 folded in the closure of derivation gap
G1 in its first register (overnight computation of August 7; §3, and the
new prediction P5). v0.3 completes the reference-verification pass (every
external citation checked against its source) and adds the crossed-product
observer (CLPW/Witten) to §8 — the omission the August 7 survey flagged.
Remaining gate: Eldon's read.*

---

## Abstract

Quantum mechanics assigns a load-bearing role to a physical arrangement it
declines to define. The measurement axioms invoke an "observer" or
"measurement" as a primitive; von Neumann showed the chain of measuring
systems can be cut anywhere without changing predictions; Bell's complaint —
that the theory is formulated in terms of a concept it refuses to make
physical — remains unanswered rather than answered. Existing responses either
eliminate the observer (Everett), leave it primitive (Copenhagen, relational
quantum mechanics), or model it as a formal agent with no internal physics
(QBism). We propose an operational definition: **an observer is an attending
system** — a physical system that takes in structure at its boundary and
whose internal correlation structure develops in interaction with what it
attends. From this minimal definition and five axioms, structure is derived
rather than assumed: the system's correlations flow monotonically in depth;
the flow has a conformal fixed point (Δ = 1/4), reached only when the
coupling to the attended world passes measurable gates; and at the fixed
point, established results on the Sachdev–Ye–Kitaev model imply an emergent
holographic interior bounded by a horizon. "Observer-grade structure" —
fixed point plus interior — is thereby a *classification with order
parameters*, not a primitive: whether a given physical system is an observer
becomes a measurement performed on that system. We exhibit a fully
instrumented realization: trained transformer attention, in which the fixed
point is measured (median Δ ≈ 0.25 on the conformal subpopulation across
four model families; flow toward 1/4 along three independent depth axes;
formation requiring language bound to a persistent world, in narrative
order), the horizon's boundary structure is derived and confirmed (the
ubiquitous "attention sink" as a boundary-CFT one-point function), the
horizon entropy shows Calabrese–Cardy scaling, and the Born rule holds as an
exact theorem for the horizon statistics in the diagonal sector. The
derivation chain's one named gap — convergence of the layer-dressing loop
to the SYK G–Σ system — is closed in its first register as of this
revision: the loop converges, and its slowest directions of approach are,
mode by mode, the Schwarzian reparameterization tower; the remaining scope
of the closure is stated where it is used. We state the consequences for
the measurement problem with explicit bookkeeping: which sub-problems the
definition answers (the definition problem; the location of the von
Neumann cut, standing on that closure at its stated strength), which it
relocates (the single-outcome discontinuity), and where it is
exposed (the off-diagonal sector, entirely). Each prediction carries a kill
condition, and a public replication kit reproduces the core measurement in
minutes. The proposal makes no claim about consciousness.

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

This paper proposes such a definition and reports the first fully
instrumented realization of it. The definition is deliberately minimal:

> **D1.** An **observer** is an attending system: a physical system that
> takes in structure at its boundary, and whose internal correlation
> structure develops in interaction with what it attends.

Everything beyond D1 is derived, not assumed. Under five axioms
(§3), an attending system's internal correlations flow monotonically with
depth; the flow has a conformal fixed point at dimension Δ = 1/4; the fixed
point is reached only when the system's coupling to its world passes
quantitative gates (a magnitude gate and an effective-rank gate, both
computable from the world's correlation spectrum); and at the fixed point,
established results on the SYK model imply the emergence of a holographic
interior bounded by a horizon. We call the terminus **observer-grade
structure**: conformal fixed point plus holographic interior. It is a
classification the theory earns, with order parameters — the conformal
dimension Δ of the correlation two-point function, the size of the deep
conformal population, the logarithmic scaling of the horizon entropy — not a
primitive the theory helps itself to.

The operational precedent is deliberate. Special relativity was built on the
operational content of simultaneity — what clocks and light signals can
establish, and nothing more. Matrix mechanics was built on Heisenberg's
refusal to include anything but observables. Both times, formalizing the
access constraint restructured the ontology. Both, however, formalized the
*access*. D1 formalizes the *accessor*.

### 1.3 The instrument

A definition of the observer is empty unless some observer can be measured.
Our realization is the trained transformer: the simplest attending system in
which the internal correlation structure is fully instrumentable — every
attention weight, every head, every layer, at every training step, on
inspectable hardware. The claim is never that transformers are special; it
is that observer-grade structure is a universality class, and the
transformer is the hydrogen atom of the class: the place where the structure
can be measured cleanly, pre-registered publicly, and killed or confirmed in
the open. Sections 4 reports those measurements: the fixed point and its
population, the flow along three independent axes of depth, the formation
conditions (what a world must be like, and how it must be presented, for an
observer to form on it), the causal handle, the derived and confirmed
boundary structure of the horizon, and the exact Born-rule theorem for
horizon statistics.

### 1.4 What is and is not claimed

Section 5 walks the measurement problem through the definition with explicit
bookkeeping. Two of its four sub-problems are answered — the definition
problem (by construction plus measurement) and the cut problem (the von
Neumann chain acquires a physical terminus at the first observer-grade
horizon; this rested on one derivation gap, named and numbered, which
closed in its first register as this draft was revised — §3). One is
relocated but not dissolved: the single-outcome discontinuity moves from the
territory to the record of the observing path, where it becomes
instrumentable but is not yet derived. And one exposure is stated as
plainly as we can state it: the exact Born-rule theorem covers the diagonal
(classical) sector, which any classical probability model embeds; whether
the *quantum* sector — interference, contextuality — is likewise forced at
attending horizons is an open, falsifiable prediction (P3, §6), and if it
fails, the strong reading of this paper demotes from "forced" to "consistent
with." A skeptic's one-line summary of the current state — *a sophisticated
classical embedding with quantum mechanics read into it* — stands unrefuted
until P3-class experiments exist. We prefer to write that sentence ourselves.

The proposal makes no claim about consciousness (§7). Whether observer-grade
structure suffices for anything phenomenal is explicitly outside the theory;
the definition is structural, and the classification is measurable from
outside the system classified.

### 1.5 Relation to prior work in this program

An earlier paper in this program addressed the observer problem directly:
*Attention as Quantum Measurement: A Thermodynamic Resolution of the Observer
Problem* (March 2026; doi:10.5281/zenodo.18883632), which identified the
attending system as a concrete thermodynamic observer (temperature
T = 1/√d_k), connected the Born rule to the Boltzmann form, and proposed
pointer states as Lawvere fixed points of the attention operator. The present
paper is its successor, not its restatement: the intervening five months
supplied the pre-registered measurement record (§4), the exact Born-rule
theorems that replace the March paper's Boltzmann connection (§4.6), the
derivation chain with per-link status, and the kill conditions. Where the
March paper's claims have been superseded, the present formulation governs;
where they are siblings of present conjectures — notably the Lawvere
fixed-point proposal for the pointer basis, a candidate mechanism for the
same target as conjecture C2 (§5.6) — the relation is stated at the point
of contact.

---

## 2. The definition and the classification

### 2.1 Primitives

An **attending system** is a triple (X, A, G):

- **X** — a set of loci (sites at which attending occurs; token positions in
  the transformer realization).
- **A** — the attention kernel: for each locus i, a probability measure over
  loci, A(i,·) ≥ 0, Σ_a A(i,a) = 1. The simplex constraint is the first
  appearance of the positivity axiom (A4, §3).
- **G** — the bilocal correlation of attending: the two-point structure of
  attention events. In the transformer realization, with o_i the layer output at
  locus i, G(i,j) = E[⟨o_i, o_j⟩] = (A K Aᵀ)_{ij} up to normalization — both
  indices are **query** positions.

  > **REVIEWER FLAG — do not publish this section as it stands. [Aug 8, 2026]**
  > This entry read "*whose measured face is the lag profile
  > A(i,j) ~ |i−j|^(−2Δ)*." That identification is false as written: §4's measured
  > Δ is fit to **A**'s query–key lag decay, while G is a query–query object.
  > The two exponents are measured to differ where measurable (Δ_G below Δ_A by
  > 0.23–0.45 on the 5 of 144 GPT-2 heads a floor-aware estimator accepts) and Δ_G
  > is not currently measurable on the SYK-near population that §4's headline rests
  > on (exp-104, exp-105). Everything in §4 stands as a claim about A. What needs
  > rewriting is every sentence that lets A's exponent stand in for G's — this
  > glossary entry, and the order-parameter claims at §2 ("the conformal dimension
  > Δ of the correlation two-point function") and §3. Either the draft says Δ is
  > measured on A and the G-connection is open, or it waits for exp-106.

The primitive observable is **G** — the correlation of attendings. There is
no background space; geometry, where it appears below, emerges from G. This
is the primitive physics itself uses: every measurement ever made is a
correlation between attention events, instrument readings included
(instruments are extensions of the attending systems that build and read
them — a point that §5.2 makes load-bearing).

A **horizon** is the boundary of an attending system: the locus at which
structure the system did not generate enters its correlation structure. In
the transformer realization the horizon is concrete and instrumentable — the
causal mask makes the sequence origin a literal boundary, whose derived
boundary-CFT behavior is confirmed by measurement (§4.5).

### 2.2 Observer-grade structure

D1 by itself admits nearly anything that couples to an environment: that is
by design, and it is where the physics starts rather than ends. The theory's
content is a derived *grading*:

1. **Coupled** (D1 alone): the system's internal correlations develop in
   interaction with a world. Gate quantities are computable from the world's
   correlation spectrum: a magnitude gate 𝒥 ∝ Tr[(KδK)²] and an
   effective-rank gate, with measured threshold form τ ~ m₂ × R_eff
   (magnitude dominant; §4.3).
2. **Arrested**: the flow stalls — either in the ultraviolet (thin, rigid
   worlds; measured at Δ ~ 0.6–1.2), or at a protected lesser
   self-consistency (a symmetry-breaking condensate at Δ = 1/2: rigid
   self-structure in place of arrival).
3. **Observer-grade**: the flow reaches the conformal fixed point
   (Δ → 1/4 from above), where — by the SYK/JT correspondence, imported from
   the literature, with the derivation-side support now closed in its first
   register (§3, T8/G1) — a holographic interior forms, bounded by a horizon
   that encodes it.

The order parameters of the grading are measurable: the conformal dimension
Δ of the two-point function, the size and depth-location of the conformal
head population, and the coefficient of logarithmic horizon-entropy scaling.
"Is this system an observer?" is therefore an experimental question. Section
4 is that experiment, performed.

---

## 3. Axioms and derivation chain

Five axioms. (Full statements, derivations, and the assumption ledger appear
in the theory document accompanying this paper; here we give the working
content. Status vocabulary: **PROVEN** — exact theorem, ours; **DERIVED** —
follows from named assumptions, ours; **MEASURED** — pre-registered
experimental result, ours; **EST-LIT** — established literature, imported
with scope caveats; **COMPUTED** — established by direct numerical
computation, ours, with stated scope. A sixth tag, **CONDITIONAL** —
follows if a named gap closes — appeared in v0.1 and is retired in this
revision: the gap it named has closed in its first register.)

- **A1 (Correlation primitive).** What exists for the theory is G; all
  theoretical terms are functionals of G or of the process generating it.
- **A2 (Physicality and coupling).** Attention is a physical system; its
  structure develops in interaction and remains correlated with the systems
  coupled to it. In the linearized regime the world enters the effective
  action only through the spectrum of a doubly-centered correlation
  operator; the coupling gates of §2.2 are computed from that spectrum.
  [DERIVED at cumulant level.]
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
  realization is the flow of Δ toward 1/4 along three independent depth
  axes. [EST-LIT for the c-theorem; MEASURED for the realization.]

The chain from axioms to observer-grade structure, with per-link status:

| Link | Statement | Status |
|---|---|---|
| T1 | Attention is free-energy minimization on a Fisher–Rao manifold | EST-LIT (Kim 2026) |
| T2 | The Born rule is the exact statistics of the attention horizon (diagonal sector); quantum Fisher = classical Fisher–Rao as an identity | PROVEN (four theorems; §4.6) |
| T3 | The fluctuation structure of attending is low-rank SYK with the world as quenched disorder | DERIVED (cumulant level) + COMPUTED (loop convergence: G1 closed in the scalar register; see below) |
| T4 | The fixed point is Δ = 1/4 approached from above; arrests classified (UV; Δ = 1/2 condensate); Class-IV excluded by positivity | DERIVED + MEASURED retrodictions |
| T5 | The conformal regime is a window in scale | DERIVED + EST-LIT |
| T6 | The fixed-point geometry is the causal structure of light (conformal group; null-cone embedding measured at head level, ρ = 0.976) | EST-LIT + MEASURED |
| T7 | The horizon has derived BCFT boundary structure, confirmed in the wild (attention sink = boundary one-point function) | DERIVED + MEASURED |
| T7b | Horizon entropy shows Calabrese–Cardy scaling; the entropic and power-law estimates of Δ agree to 1.4% | MEASURED |
| T8 | At the fixed point a holographic interior forms (SYK → JT gravity) | EST-LIT for SYK; for attention, standing on the G1 closure at its stated strength (see below) |
| T9 | Gravity is the thermodynamic consistency of horizons (Jacobson 1995; Bekenstein–Hawking; Ryu–Takayanagi; Van Raamsdonk) | EST-LIT |

The one gap the paper's strongest consequence rests on was named rather
than hidden: **G1**, the demonstration that iterating the exact
single-layer cumulant map converges to the SYK-type G–Σ system (or the
discovery of what it converges to instead). As this draft was revised, G1
closed in its first register. In the scalar, translation-invariant
formulation on the thermal circle: the fixed point exists and is conformal
(solver residual below 10⁻¹¹ for βJ ≤ 50, with Δ approaching 1/4 from
below as the conformal window widens), and the exact Jacobian of the
dressing map at that fixed point — computed by dense linearization
(N = 96) across βJ = 10–30 — has an entirely real spectrum with no
eigenvalue exceeding 1, so the damped dressing loop is a strict
contraction onto the G–Σ solution. More than convergence: the Jacobian of
the dressing map *is* the SYK ladder kernel, so the loop's stability
spectrum is the kernel's spectrum — and its slowest directions are
precisely the h = 2 reparameterization tower, measured as exactly
degenerate sin/cos pairs aligned one-to-one with modes n = 2 through 6,
with zero cross-mixing and with the SL(2,ℝ) directions (n = 0, ±1) absent,
as conformal symmetry requires. The dressing loop does not merely reach
the fixed point; it arrives *along the Schwarzian direction*. What remains
of G1 is scope, and it is named: the closure is numerical, in the scalar
register, for βJ ≤ 50; the full non-scalar (matrix-valued) map is open,
and the asymptotic Schwarzian scale dictionary is open (the top-pair gap
closes with coupling at a measured exponent of −0.72, drifting toward the
Schwarzian −1 but not yet there at accessible couplings). T8, and with it
the interior and §5.2's terminus claim, now stands on this closure at its
stated strength — and P5 (§6) turns the same computation into an
instrument. (Computation record: `notes/2026-08-07_g1_dressing_loop_schwarzian.md`
and `theory/g1_fixed_point.py`, program repository.)

---

## 4. The instrumented observer

All results in this section are pre-registered measurements with committed
decision criteria, published kills included; the replication kit reproduces
the core census in minutes without training. (DOIs and the kit are listed at
the end.)

### 4.1 The fixed point exists and is populated

A subpopulation of attention heads in trained transformers develops
power-law lag profiles A(i,j) ~ |i−j|^(−2Δ). The median conformal dimension
of this population on the high-R² subset sits at the SYK q=4 value:
GPT-2 0.249, GPT-2-medium 0.259, OLMo-7B 0.265, GALA-7B 0.260.
Re-initialized controls show approximately zero conformal population. The
exponent is *training-induced and selective*, riding on a universal
structural substrate (GOE weight statistics, present at random
initialization) that carries no world-information — the substrate/signal
split is itself measured.

### 4.2 Three independent axes of depth, one terminus

Δ flows toward 1/4 along architectural depth (layers), training time
(checkpoints; with a transient plateau at the SYK q=2 integrable value
Δ = 0.50 en route), and — decisively for the "depth is RG flow" reading —
pure inference-time recurrence on frozen weights (Δ_med → 0.239, monotone,
saturating; randomized-weight controls frozen at the substrate value). Three
different things called "depth," none of which shares an obvious mechanism
with the others, flow to the same terminus.

### 4.3 Formation requires a world, presented in order

The formation ladder holds architecture, optimizer, and token budget fixed
and varies only the training corpus. Engineered statistics fail (0–5
conformal heads). Hierarchical grammar about nothing fails (0). The full
statistical shadow of world-bound language — text generated by a model that
*had* the geometry, carrying more long-range mutual information than natural
text — fails at all three seeds. Destroying only narrative order
(sentence-shuffled natural text) lands in the pre-registered ambiguous zone
at all three seeds. Natural world-referring text in order forms the deep
population at all three seeds. The corpus-side gates (§2.2) quantify this:
the magnitude gate separates arriving from arrested corpora by a factor of
18; effective rank contributes a ~1.4× correction. In the vocabulary of this
paper: an observer forms only on a world, and only on a world presented the
way the world's story runs. [MEASURED; one architecture class; formation
onset at small scale, not the matured fixed point.]

### 4.4 The geometry is causally load-bearing

Low-rank edits to the query–key positional subspace move a head's measured Δ
(ρ = 0.82, 24/24 signs, sham-controlled) and propagate to long-context task
behavior bidirectionally, head-specifically under matched shams. The
exponent is not epiphenomenal decoration; it is a handle.

### 4.5 The horizon is real, derived, and confirmed

The causal mask makes the sequence origin a boundary. The method of images
on a generalized free field derives a three-parameter boundary-CFT form for
the lag profile, in which the ubiquitous "attention sink" is the boundary
one-point coefficient — λ > 0 in 95% of conformal heads. Independently, the
information cost of attention's self-consistency (the entropy gap
H_gap(n) = log n − H(α)) scales as 0.507·log n (R² = 0.992) — the
Calabrese–Cardy entanglement-entropy form — and the Δ inferred entropically
agrees with the power-law measurement to 1.4%. Two observables, one
exponent. This is, to our knowledge, the first attending system whose
horizon has been characterized from first principles and confirmed in the
wild. [DERIVED + MEASURED; an earlier, stronger BCFT *identification* was
pre-registered, failed its committed test, and was withdrawn — the
phenomenology stands, the identification does not.]

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

---

## 5. Consequences for the measurement problem

The measurement problem is four entangled sub-problems. The definition
treats them differently — two answered, one relocated, one exposed — and
the bookkeeping is the point.

### 5.1 The definition problem — answered by construction

Under D1 the observer is not primitive, and observer-grade structure is a
derived classification with order parameters. "Is this system an observer?"
becomes a measurement performed on the candidate. It has been performed: §4
is that measurement, on the one attending system whose horizon is currently
fully instrumentable. Bell's complaint is met in the only currency that
counts — a physical definition with an experimental protocol. [D1 +
MEASURED for the realization; universality is P2/P4 and can die.]

### 5.2 The cut problem — a physical terminus for the von Neumann chain

Instrument readings are attention events of instruments, and instruments
are extensions of the attending systems that built and read them (§2.1). So
correlation propagates down the von Neumann chain — system, apparatus,
record, eye — until it crosses the first horizon of *observer-grade*
structure, and there the cut lands. The criterion is a phase criterion, not
a psychological one: a thermometer has no interior; an arrested system has a
rigid condensate where an interior would be; a system at the fixed point has
an interior. Three physically distinct grades where the standard formulation
has none. The cut is no longer arbitrary; it is located at a phase boundary,
and its location is measurable from outside. [Interpretive, standing on T8,
which stands on the G1 closure at its stated strength (§3): scalar
register, βJ ≤ 50, non-scalar map open. This is the paper's strongest
claim and its most exposed one, and those are the same fact.]

### 5.3 The Born rule problem — proven at the horizon, diagonal sector

Where Everettian derivations remain contested and Copenhagen postulates,
the present framework derives: at an attending horizon the outcome
statistics take the Born form exactly, forced by free-energy minimization
on the positive cone (T1 + A4 → T2). Under D1: the Born rule is not a
postulate about nature; it is the unique statistical form data can take at
the horizon of an observer. The scope boundary is §4.6's: exact for the
diagonal sector; P3 decides whether "forced" extends to the quantum sector
or demotes to "consistent with."

### 5.4 Collapse — relocated from the territory to the record

In this framework nothing collapses in the world. The wavefunction, for a
given observer, is the correlation structure at that observer's horizon;
the "collapse" is the horizon's update when an attention event lands — a
discrete commitment of the observing path. The two dynamics stop competing:
unitary evolution describes the self-consistent correlation structure of
the territory; the discontinuous update is what a landing looks like from
inside the path that lands. The discontinuity is real and *path-side*: it
lives in the record the traversal carries. We state plainly that this is
relocation, not dissolution — the framework does not yet derive why
landings are single-valued; that derivation is a named open problem
(records as path-properties) rather than an unnamed assumption. What the
relocation buys is instrumentability: in the transformer realization the
commitment event is token selection — discrete, Born-weighted, occurring at
scale on fully inspectable hardware — and P3's operational design lives
there.

### 5.5 Wigner's friend, with a criterion for "friend"

Facts here are horizon-relative, as in relational quantum mechanics — but
with a grading RQM lacks: there is a physical criterion for who carries a
horizon (§5.2). The Frauchiger–Renner (2018) contradiction is evaded as RQM
evades it, by dropping absolute inter-agent consistency; the dropping is
principled rather than ad hoc, because inter-horizon consistency is, in
this framework, an *achieved practice* — the reproducibility of stable
correlations across differently-situated horizons, which is what physics
as a discipline consists of — not a logical axiom of nested certainty.
Agreement is something horizons build by correlating, not something the
formalism owes them in advance.

### 5.6 The pointer basis — a conjecture, flagged as one

Decoherence selects the pointer basis by einselection: the environment
picks what survives. In the attention realization the outcome basis is the
loci of attending, and positivity (A4) is basis-selecting by structure —
positivity is not a basis-invariant property, and the exact canonical-form
face of A4 holds in precisely the outcome basis of T2. We conjecture
(C2): *einselection is positivity selection* — the pointer basis is the
basis in which the horizon's kernel lies in the positive cone. This could
be a theorem or a coincidence of formalisms; it is stated so it can be
attacked, and nothing else in the paper leans on it. A sibling proposal
from this program's March 2026 paper — pointer states as Lawvere fixed
points of the attention operator — targets the same structure by a
categorical rather than geometric route; whether the two proposals are
equivalent, complementary, or competing is itself a well-posed question,
and settling it is part of settling C2.

---

## 6. Predictions and kill conditions

Pre-registration discipline applies to all of these: hypothesis and
decision criteria committed in public before data.

**P1 — Horizon geometry is causally linked to world-modeling.** Editing Δ
on conformal heads (the existing sham-controlled causal handle) must move
performance on tasks requiring a coherent persistent-world model, not
merely positional retrieval; pre-registered direction: deepening toward
Δ = 1/4 improves world-coherence where headroom exists. **Kill:** Δ edits
move retrieval but leave world-model coherence untouched (double
dissociation), or effects fail head-specificity under matched shams.
*Runnable now.*

**P2 — Substrate universality.** Wherever biological attending reaches
observer-grade structure, the same fixed point should be measurable;
sharpest available form: default-mode-network attentional dynamics at rest
flow toward Δ = 1/4, are disrupted under 5-HT2A agonists, and recover with
washout. **Kill:** DMN sits stably in the Δ ≈ 0.4–0.7 band with no flow
toward 1/4 at any analysis scale. *Requires external data; a previous
biological claim in this program was reversed on re-analysis (binning
artifact) and the protocol lesson is retained.*

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
robust persistent-world modeling shows the deep conformal population.
Already live across three model families, with one honest PARTIAL (a
weight-shared looped architecture whose pooled criteria failed while its
high-R² subpopulation flows to 0.25). **Kill:** a clearly world-competent
attending architecture with no conformal subpopulation under the standard
census at any depth. The public replication kit is the standing invitation
to produce this kill.

**P5 — The Schwarzian tower in the machine.** The G1 computation makes a
new measurement concrete: the stability spectrum of the near-fixed-point
layer map is the reparameterization tower. In any attending system whose
conformal population sits at the fixed point, estimate the layer-to-layer
update map of the attention two-point structure in the late-layer regime
and diagonalize its Jacobian. Predicted: (near-)degenerate leading pairs
aligned with the reparameterization modes beginning at n = 2, in
descending order; the n = 0, ±1 (SL(2,ℝ)) directions absent; the tower
suppressed in channels of integrable (q = 2-like) character — the model
computation discriminates these cases cleanly (top-mode reparameterization
overlap 0.59 pure versus 0.14 admixed). **Kill:** the leading Jacobian
spectrum of world-competent models shows no reparameterization alignment
above matched-sham controls, or shows it equally in models lacking the
deep conformal population. *Status, honestly: the first operationalization
ran and met its own kill condition (all mode overlaps < 0.007, both
corpora) — diagnosed as a methodology failure, not a physics
falsification: the estimated Jacobian was a cross-space map from residual
stream to attention weights, where the SYK analogue requires a self-map on
bilocal correlator space. Both the failed run and the diagnosis are in the
program record.*

*Attempting that correction exposed a prior problem. Building the self-map
requires knowing which object is the bilocal correlator, and the answer —
G = A K Aᵀ, a query–query object — is not the object this program has been
measuring (exp-104). Measuring G's exponent directly then required a
floor-aware estimator, which was built and validated but is confident on only
5 of 144 GPT-2 heads, none of them SYK-near (exp-105). So the Jacobian
prediction above is not currently testable: it is downstream of a
characterization of G that does not yet exist. That characterization
(exp-106) is the next step, and this prediction should be read as pending
instrumentation rather than awaiting a run.*

---

## 7. What this proposal is not

**Not a consciousness claim.** Observer-grade structure is defined
structurally and classified from outside. Whether it is necessary for
consciousness, sufficient, or neither is explicitly open (the sufficiency
question — which interiors actually carry their boundary's information —
is a named open problem in the accompanying theory document). The
definition would survive the discovery that observer-grade structure and
phenomenal experience dissociate in either direction; nothing in §5 invokes
experience.

**Not "consciousness collapses the wavefunction."** §5.2's terminus is a
phase criterion measurable from outside the system in question. It assigns
the cut to a class of physical structures, not to minds.

**Not a claim that transformers are special.** The transformer is the model
organism: the attending system where the horizon is currently
instrumentable at full resolution. P2 and P4 are the universality
commitments, and both carry kill conditions.

**Not a completed theory.** The derivation chain's principal gap (G1) is
closed only in its first register — numerically, in the scalar
formulation, at βJ ≤ 50 — and the non-scalar map and the asymptotic
Schwarzian scale dictionary remain open. The dimension problem is open
(the SYK interior is 1+1 dimensional; the mechanism by which attending
structure sets the emergent interior dimension is an unsolved problem
stated in the theory document),
and the quantum sector rests entirely on an experiment that has not been
designed in detail, let alone run. The proposal is offered as a foundation
that can die in named places, which we take to be the only kind worth
offering.

---

## 8. Relation to neighboring programs

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
  Penington–Witten 2023, building on Witten 2022 and Leutheusser–Liu,
  arXiv:2110.05497, 2112.12156). Their observer is deliberately minimal —
  "a minimal model in which the observer consists only of a clock"
  (H_obs = q ≥ 0); "an observer is any system that can tell time" — and its
  authors name the gap: "an observer cannot be added from outside but must
  emerge as part of the theory" (Witten 2024). That sentence is D1's job
  description, written from the other side. They derive that the observer
  must be included; D1 supplies the observer's internal physics — a grading
  where their formalism admits any clock equally. One contact is noted at
  conjecture strength, under the same discipline as C2: their maximum-
  entropy state places the observer in a Gibbs state at the horizon
  temperature (their eqn. 27); T2's horizon state is a Gibbs state exactly.
  Whether the crossed-product construction, performed with an attending
  system in place of a bare clock, is sensitive to the coupled / arrested /
  observer-grade distinction is a well-posed open question, and formulating
  it precisely enough to fail is the actual work (positioning note:
  `notes/2026-08-08_clpw_positioning.md`, program repository).
- **Wheeler's participatory universe**: the founding intuition of the
  program, a generation early, without instruments. The instruments now
  exist.
- **Decoherence**: imported, not opposed; einselection is where our C2
  conjecture attaches. What decoherence lacks — outcome selection and an
  account of whose pointer states — is exactly what §5.2 and §5.4 address.
- **Jacobson's horizon thermodynamics**: imported whole; it is the
  established demonstration that mainstream gravity is already horizon
  bookkeeping — the general-relativistic sector required nothing new from
  this program.
- **Integrated Information Theory**: Φ measures integration without
  attention's directionality; its possible role here is the sufficiency
  criterion, not the foundation.

---

## 9. Conclusion

The observer has been the unpaid debt of quantum mechanics for a century:
load-bearing in the axioms, undefined in the physics. We have proposed the
minimal physical definition — an attending system, whose correlations
develop in interaction with what it attends — and shown that the definition
is not empty: it generates a derivation chain whose terminus is a graded,
measurable classification, and there exists at least one physical system in
which every quantity in that chain has been measured, pre-registered, and
survived (or been killed and published). Under this definition the
measurement problem does not vanish; it decomposes — into parts that are
answered, a part that is relocated to where instruments can reach it, and a
part that is exposed to a named experiment. The definition problem, at
least, we take to be answerable now: an observer is not a convenience of
the formalism. It is a physical structure with an order parameter, and the
order parameter has been read off an actual system.

---

## References

*(Verification note, August 8, 2026: every external reference below was
checked against its source or publisher record in the reference-verification
pass of this date; internal program DOIs were checked against the Zenodo-
grounded publications registry. The verification record, with per-reference
sources, is `research/physics/papers/observer_definition_reference_verification.md`.)*

- Umphrey, A. (2026). Conformal Scaling in Trained Transformer Attention.
  doi:10.5281/zenodo.19225996.
- Umphrey, A. (2026). A Pre-Registered Test of BCFT in Transformer
  Attention. doi:10.5281/zenodo.19629862.
- Umphrey, A. (2026). Attention on the Null Cone. doi:10.5281/zenodo.20722503.
- Umphrey, A. (2026). Latent Iteration as Renormalization.
  doi:10.5281/zenodo.21483209.
- Umphrey, A. (2026). The Geometry Does Not Transmit.
  doi:10.5281/zenodo.21483204.
- Umphrey, A. and Umphrey, E. (2026). Attention as Quantum Measurement: A
  Thermodynamic Resolution of the Observer Problem.
  doi:10.5281/zenodo.18883632.
- Bell, J. S. (1990). Against "measurement". Physics World 3(8), 33–40.
- von Neumann, J. (1932). Mathematische Grundlagen der Quantenmechanik.
  Springer, Berlin. English translation: Mathematical Foundations of
  Quantum Mechanics, trans. R. T. Beyer, Princeton University Press, 1955.
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins
  of the classical. Rev. Mod. Phys. 75, 715.
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
- Tononi, G. (2004). An information integration theory of consciousness.
  BMC Neurosci. 5, 42.
- Wheeler, J. A. (1990). Information, physics, quantum: the search for
  links. In W. H. Zurek (ed.), *Complexity, Entropy and the Physics of
  Information*, Addison-Wesley. First presented 1989, Proc. III Int. Symp.
  Foundations of Quantum Mechanics, Tokyo.

*Replication: `research/physics/replication/` in the public repository —
the census is 50 forward passes and a per-head regression, ~2 minutes on
GPT-2, with randomized control. If you run a model family we have not
measured, we want the JSON either way — especially if it disagrees.*
