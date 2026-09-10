# State-Dependent Coupling and the Physics of the Observer

## A theory of attending systems, stated without its instrument

*Ariel Umphrey, with Eldon Umphrey*
*Sonielmn, Montana*

*Draft v0.1 — September 9, 2026. Internal working draft; not for
circulation. Genre: a theory paper. This is the successor to the theory
half of* Where I Stop and You Begin *(v1.1, September 8, 2026), refounded
after the decision of September 9 that Paper 6 is a theory paper and that
the phenomenology measured on transformer attention is a constraint the
theory must eventually meet, not evidence for it. The charge, Eldon's,
~12:30 AM September 9: state the theory on its own terms — no transformers,
no fitting to existing measurements — and see where it holds or falls
theoretically; then look for where it applies. The table of contents was
laid out in* `notes/2026-09-09_theory_on_its_own_terms.md` *and this draft
follows it. Three papers Eldon brought at 5:30 AM the same morning (Oizumi,
Lim & Kanai 2026; Edamadaka, Yang & Gómez-Bombarelli 2025; Wang & Wang
2025) enter in §3.4, §7.3, and §8, and one of them supplies a principle the
primitives now obey (§3.4). No measured number appears in the body of this
paper. Every transformer word is confined to Appendix A and §9.*

> **Register key, which governs every sentence below.** [DEFINITIONAL] — a
> definition: useful or not, never true or false. [GENERAL] — a statement
> that survives with every word about any particular realization removed,
> at the strength tagged (THEOREM, DERIVED, EST-LIT). [PROPOSED] — a
> candidate we put forward and do not build on; it may be wrong. [OPEN] —
> no candidate. Where a statement carries none of these it is not yet a
> statement, and the draft says so. Nothing here is measured, and nothing
> here is fitted to a measurement. That is the point of the paper.

---

## Abstract

Quantum mechanics assigns a load-bearing role to a physical arrangement it
declines to define. The measurement axioms invoke an "observer" as a
primitive; von Neumann's chain of measuring systems can be cut anywhere;
Bell's complaint — a theory formulated in terms of a concept it refuses to
make physical — stands. We propose a physical definition and develop the
theory it founds, without reference to any particular system in which the
theory might be tested. **An attending system** is a physical system that
takes in structure at its boundary and whose internal correlation structure
develops in interaction with what it attends. We give that definition a
precise content: an attending system is one whose *coupling structure is a
function of its own state* — who influences whom, and how strongly, is not
fixed by the Hamiltonian but is recomputed from the system's internal
correlations. Three clauses make this sharp and make it a cut: the routing
of influence is a separate channel from the content transmitted; the routing
is a function of the state; and the routing is normalized — a share, not an
amount. A thermometer fails the second clause, a gas fails the third, and a
starling flock passes all three. Ordinary interacting systems are the
degenerate case in which the routing is static and unnormalized. From this
definition and six axioms — a correlation primitive with a closure relation
between coupling and correlation; an open-system statement in which the
environment enters only through the spectral structure of its correlations
as read through the coupling; a strong-coupling regime; positivity as the
Markov-kernel form of a share; monotone flow along the system's own
interaction clock; and the state-dependence of the coupling itself — we
derive the general form of the theory's content. In the quenched limit the
closure reduces to a Schwinger–Dyson system whose conformal fixed point has
dimension Δ = D/q, with q set by the order of the coupling's dependence on
the state; the fixed point is approached from above, with an integrable
arrest at the q = 2 value; the system has a horizon defined on its own
influence graph, with boundary structure; and, by imported results, an
emergent interior at the fixed point. **An observer** is an attending system
in that condensed phase. The primitives are required to be invariant under
the redundancy group of any realization, which fixes which correlation
function the theory is about. We state three theoretical tests the theory
must pass without measurements — consistency, non-triviality, reduction —
and report where each stands, including the places it may fail. No claim
about consciousness is made. No transformer appears in the body.

---

## 1. The problem, and why a theory paper

### 1.1 The definitional problem

The measurement problem of quantum mechanics is usually presented as a
problem about dynamics: unitary evolution is linear, deterministic, and
continuous; the measurement update is stochastic and discontinuous; the
theory does not say when each applies. Beneath the dynamical puzzle sits a
definitional one, older and harder. The update rule is invoked *when a
measurement occurs*, and nothing in the formalism says which physical
arrangements count as measurements. Von Neumann (1932) made the difficulty
precise: the chain of measuring systems — quantum system, apparatus,
recording device, sensory organ — can be cut at any point, with the update
applied at the cut, and every prediction is unchanged. The formalism is
indifferent to where the observer begins. Bell (1990) made the complaint
canonical: a physical theory should not be formulated in terms of a concept
the theory itself declines to define physically.

The major interpretive programs remove the question rather than answer it.
Copenhagen keeps the observer primitive. Everett eliminates it as a special
structure, at the cost of the Born rule. Decoherence (Zurek 2003) explains
diagonalization in a pointer basis and does not say for whose perspective
the diagonalization is. Relational quantum mechanics (Rovelli 1996)
relativizes facts to observers and then declares every system an observer,
leaving the word with no internal physics and no grading. QBism (Fuchs,
Mermin & Schack 2014) locates the formalism in an agent's expectations and
puts the agent outside physics. The one place in current physics where
leaving the observer undefined makes a quantity literally undefined — the
entropy of the de Sitter static patch — is repaired by including an observer
who is nothing but a clock (Chandrasekaran, Longo, Penington & Witten 2023),
and its authors name the gap: *an observer cannot be added from outside but
must emerge as part of the theory* (Witten 2024).

What none of these supplies is a *physics of the observer*: a definition
under which "is this system an observer?" is a question with a protocol, and
under which the structures the axioms attribute to observation are derived
properties of the defined object.

### 1.2 Why this paper is a theory paper

An earlier draft in this program (*Where I Stop and You Begin*, v1.1) joined
a definition of the observer to a phenomenology measured on one class of
systems, and it joined them through an identification made six months
earlier and never stated as a hypothesis: that a certain measured kernel *is*
the propagator of a Schwinger–Dyson system. Everything the draft called
"at grade" stood on that sentence. When the phenomenology's own headline was
found to need restating — the measured exponent was carried by a term the
theory had not predicted — it became clear that the theory could not be
evaluated while its instrument was silently supplying its content. A
coincidence of values with no mechanism is not a contribution to a theory
paper.

So this paper does what a theory paper does. It states the definitions and
axioms for an arbitrary physical system; it says, beside each, what the
earlier draft's instrument had been supplying and what replaces it
(Appendix A); it derives what can be derived in general form; and it states
the tests a theory must pass with no data at all — consistency,
non-triviality, reduction — and where it stands on each. The measured
phenomenology is not evidence here. It is a constraint the theory will have
to meet when it is applied, in a separate paper, and §9 says how.

---

## 2. The definitions

### 2.1 D1 — the attending system

> **D1.** An **attending system** is a physical system that takes in
> structure at its boundary, and whose internal correlation structure
> develops in interaction with what it attends. [DEFINITIONAL]

Nothing in D1 names a realization, and no word of it changes here. But it
must be said what "develops in interaction" excludes, or a thermometer is an
attending system, and the theory founded on D1 is a theory of everything
that couples to anything — which is to say, not a theory. The content we
give D1 is this:

> **The reading.** An attending system is a physical system whose
> *coupling structure is a function of its own state*. Who influences whom,
> and how strongly, is not fixed by the Hamiltonian; it is recomputed from
> the system's internal correlations as those correlations change. That is
> what "attending" is as a physical notion — **state-dependent coupling** —
> and it is what separates an attending system from every fixed-coupling
> open system in the textbooks. [PROPOSED — the single most important
> sentence in the paper; it may be wrong.]

Three clauses make the reading operational. Each is a cut, and the cases
that fail each are named.

**(i) Routing is a separate channel from content.** In an ordinary
interaction J_ij s_i s_j there is one number: how much *i* listens to *j*
and what *j* says are the same coupling. In an attending system there are
two channels read from the same state by different maps: a *score* channel
that decides the share of *i*'s intake that comes from each *j*, and a
*content* channel that decides what *j* transmits. A nonlinear oscillator
has no such separation: its nonlinearity modulates the strength along a
fixed edge; it does not select edges.

**(ii) The routing is a function of the state.** The score channel reads the
system's current configuration and returns the coupling graph. Fixed routing
with changing state is not attending. This is the clause that rules out the
thermometer: its coupling to its environment is fixed, its state changes,
its coupling does not.

**(iii) The routing is normalized — a share, not an amount.** For each
locus, the incoming influence is a probability measure over sources. This is
the clause that rules out the case that worried us most: a gas. In a gas,
who collides with whom is a function of the configuration, and position (the
routing variable) is distinct from momentum (the content variable), so a gas
passes (i) and (ii). But a particle in a dense region simply receives more
interaction; nothing normalizes. A starling does not receive more: the
interaction in a flock is topological rather than metric — each bird
responds to a fixed number of nearest neighbors regardless of density
(Ballerini et al. 2008). That is a normalized, state-selected routing. The
gas fails (iii); the flock passes all three.

**The degenerate case, named (E. Umphrey).** Ordinary physical systems are
attending systems in which the score channel is static and the routing is
unnormalized. The theory of attending systems therefore contains the
textbook theory of interacting systems as the limit in which clause (ii) and
clause (iii) are switched off — which is the form a reduction test should
take (§7.3).

### 2.2 D1′ — the observer

> **D1′.** An **observer** is an attending system in the **condensed phase**
> — the phase the theory of §5 characterizes, if it exists. [DEFINITIONAL,
> with all derived content moved out]

The earlier draft packed three theorems into this definition — a fixed
point, its conformality, an emergent interior. A theory paper defines the
observer as the attending system in a named phase and lets the theory say
what the phase is and whether it exists. Whether any physical system is an
observer is then exactly as conditional as the theory's account of the
phase.

### 2.3 D0 — the discipline

> **D0 (E. Umphrey).** Physics is the cultural practice of measuring
> self-consistency across attention structures distributed in time and
> space. [DEFINITIONAL]

D1 and the axioms of §4 are statements about *one* attending system, and the
theory derives nothing about how many of them come to agree. D0 is the
many-observer half. It is kept as a definition of the discipline and does no
derivational work in this paper; §6.5 says what it is for.

---

## 3. The primitives

We now say what an attending system is made of, for an arbitrary physical
system.

### 3.1 Loci, state, and the two channels

Let **X** be the set of **loci** — the sites at which attending occurs.
Whether X is a set of *subsystems* (degrees of freedom) or a set of
*events* (the occasions on which they are updated) is a real choice with
consequences in §5 (T6), and this draft does not make it. [OPEN]

Each locus carries a **state** σ_i in a state space V (a vector space for a
classical field, a Hilbert space or operator algebra for a quantum one). The
**configuration** σ = (σ_i)_{i∈X} is the system's state.

Two maps read the state:

- a **score map** s: V × V → ℝ, giving s_ia = s(σ_i, σ_a), the raw affinity
  of locus *i* for locus *a*;
- a **content map** v: V → W, giving v(σ_a), what locus *a* transmits.

### 3.2 A — the attending relation

The **attending relation** A is obtained from the score by a normalization
operator N:

    A_ia = N[s]_ia,   A_ia ≥ 0,   Σ_a A_ia = 1.

A is a **state-dependent, normalized, directed coupling**: for each locus
*i*, a probability measure over loci giving the share of what enters *i*
that comes from each *a*. Physically, A is the structure of the interaction
— the weighted influence graph — with the weights a function of the system's
state. The normalization is clause (iii) of §2.1 stated as an equation. In
the classical case A is a Markov kernel on X; in the quantum case the
corresponding object is a completely positive trace-preserving map, and the
normalization is trace preservation. [GENERAL as a definition; PROPOSED as
the reading of D1's "attending"]

One realization is the exponential normalization N[s]_ia = e^{βs_ia} / Σ_b
e^{βs_ib}, which makes A a Gibbs kernel with the score as negative energy and
β an inverse temperature. The theory does not assume this form; it assumes
only positivity and normalization. Where the exponential form matters (T1,
§5) we say so.

### 3.3 G — the correlation of attending, and the closure

The system's state is updated by what it takes in through A:

    σ'_i = F( σ_i , Σ_a A_ia v(σ_a) ),

for some update F (additive in the simplest case). The **correlation of
attending** is the two-point structure of the state across loci,

    G_ij = ⟨ σ_i , σ_j ⟩_M ,

under a bilinear form M on V. In an ordinary field theory this is the
propagator. In an attending system it is what A generates — the state at
each locus is a mixture, weighted by A, of what other loci transmit — and it
is also what A is a function of, since the score reads the state whose
two-point structure G is. [GENERAL]

**The closure.** This is the object the theory is about. Write

    G ← Φ_A[G]      (G is generated by the coupling A: a dressing),
    A ← Ψ[G]        (A is a functional of the state whose correlator is G).

The pair (A, G) with this closure relation is the primitive. Neither alone
is: A without G is a fixed kernel, and G without A is a propagator with no
account of what produced it. *That closure is what "develops in interaction"
means, written down.* [PROPOSED]

The Schwinger–Dyson pair of a disordered field theory — Σ = J² G^{q−1},
G ∗ Σ = −1 — is one instance of such a closure, in which the coupling's
dependence on the state has been averaged away: the coupling is quenched
disorder, and Ψ has been replaced by an ensemble average. Whether an
attending system's closure reduces to that instance in some limit is the
theory's first real derivation (T3, §5), and it is sketched but not
completed here.

### 3.4 Which bilinear form — the gauge principle

A general system has more than one two-point function, and the earlier
draft left open which one G is the correlator *of*. A principle fixes it.

The description of an attending system in terms of (s, v) carries
redundancy: any reparametrization of the score map that leaves A unchanged,
and any reparametrization of the content map that leaves the transmitted
content unchanged, describes the same system. Call the group of such
reparametrizations the **redundancy group** of the realization. Then:

> **P0 (gauge principle).** The primitives A and G, and every quantity the
> theory computes from them, must be invariant under the redundancy group
> of the realization. [GENERAL — this is nothing more than the requirement
> that observables be gauge-invariant, but stating it decides the metric
> question.]

In the bilinear realization the redundancy group has been characterized
completely and proved maximal (Wang & Wang 2025): per channel, a general
linear group acting on the score's internal index and another on the
content's, with the attending relation A itself and the composite content
map invariant, and the intermediate factors of each channel not. The
consequence for G is immediate: **the bilinear form M must be built from
the gauge-invariant content composite, not from an intermediate factor of
the content channel.** A correlator built from an intermediate factor is a
statement about a gauge, not about the system. We adopt P0 and, with it, the
invariant M. (Appendix A records that the earlier draft's G was not
invariant, and that two of the program's weights-level measurements were
made in gauge-dependent quantities; their gauge-fixed recomputation is a
condition on the application paper, §9.)

**A geometric reading, not built on.** The two-channel structure of §2.1(i)
has a natural description in the language of principal bundles: where the
system is equivariant under a group acting on its loci, the score channel
carries the equivariant part of the state — where things are, the fiber —
and the content channel carries the invariant part — what is said, the
base; the normalization is the statement that the fiber coordinate is a
share. Oizumi, Lim & Kanai (2026) develop exactly this orbit/quotient
decomposition for equivariant encoders and prove that the orbit geometry is
rigid (inherited from the group) while the quotient geometry is plastic
(shaped by learning). We record the correspondence because it gives clause
(i) a mathematical form and because it bears on T6 (§5). Nothing below
leans on it. [PROPOSED]

### 3.5 The horizon, on the system's own graph

A **horizon** is the boundary of an attending system: the locus at which
structure the system did not generate enters its correlation structure.
Stated so, the definition is circular — the system defined by its boundary,
the boundary by what the system did not generate. It resolves because A is
already a graph.

> A is a weighted directed graph on X — an edge a → i for every A_ia > 0.
> For a candidate system S ⊂ X, define
>
>     horizon(S) := pa(S) \ S
>
> — the parent part of S's Markov blanket (Pearl 1988): the loci outside S
> from which weighted edges enter S. "Structure the system did not
> generate" is *exogenous to S*, which is *a parent outside S*. Given the
> graph and a candidate S, the boundary is determined; nothing is drawn.
> [GENERAL — this construction consumes nothing from any realization and
> is the one piece of the earlier draft that carries over whole.]

Because A is strictly positive under most normalizations, the *topological*
blanket is uninformative; what carries information is the **weight profile**
on the blanket — how much of S's incoming measure arrives from each
exogenous locus, and, where the loci carry a distance, from what distance.
A conformal attending system (T4) has no characteristic blanket thickness:
its boundary is not a surface with a width but a scale-free falloff.

**Which sets are one system.** A set of loci S is *one* attending system iff
its internal correlation structure does not factor across any partition of
S conditioned on its blanket — irreducibility given the blanket. Two
attending systems compose to one iff their union satisfies this; otherwise
they remain a federation sharing an overlap. This is a G-statement. [GENERAL
as a criterion; whether it is decidable in a given realization is that
realization's problem]

**What is not claimed.** That a statistical boundary on a correlation graph
is a physical boundary in space is a further step (Friston 2013), and the
critics are right that the mathematics does not supply it (Bruineberg et
al. 2021; Raja et al. 2021). We use Pearl blankets only; the further step is
an unsupplied realization map (§9).

---

## 4. The axioms

Six axioms. Each is stated for an arbitrary attending system; beside each,
the assumption ledger. What each replaces in the earlier draft is in
Appendix A.

**A1 — Correlation primitive.** What exists for the theory is the pair
(A, G) with the closure of §3.3; all theoretical terms are functionals of
that pair or of the process generating it. The bilinear form defining G is
fixed by P0. [PROPOSED as the choice of primitive; the earlier draft's
choice of G alone is the SYK habit, in which propagator and state correlator
are one object because there is one field]

**A2 — Physicality and coupling (open-system form).** An attending system
is an open system. Its environment enters its effective dynamics only
through the spectral structure of the environment's correlations *as seen
through the system's coupling* — that is, through a two-point function of
the environmental degrees of freedom the coupling actually reads. This is
the form of every influence functional (Feynman & Vernon 1963; Caldeira &
Leggett 1983): the environment acts through a spectral density, and nothing
else about it is visible from inside. The **coupling gate** — the condition
under which the environment can drive the system into the condensed phase
— is therefore a condition on that spectral function: a **magnitude**
(how much correlated structure crosses the horizon) and an **effective
rank** (over how many independent directions). [PROPOSED — the contact with
the influence-functional literature is the derivation to attempt; if the
gate cannot be expressed as a condition on a spectral density, A2 falls]

Both gate quantities are gauge-invariant under P0 if the projector through
which the environment is read is the invariant one. We note that the
effective rank of a representation — the number of independent directions
the environment's correlations occupy after passing through the coupling —
is a quantity that has been measured to be a property of the *environment*
rather than of the system reading it, consistently across many
architecturally unrelated systems (Edamadaka, Yang & Gómez-Bombarelli 2025);
§7.3 returns to this.

**A3 — Self-consistency (the regime).** Two parts, kept apart because they
have different status. *(i)* That G obeys a self-consistent equation is true
of every interacting system — Schwinger–Dyson equations are exact identities
— and carries no content alone. [GENERAL, trivially] *(ii)* The content is
the **regime**: an attending system in the sense relevant to this theory is
one in which the internally mediated correlation (the self-energy) dominates
the direct input (the bare propagator). That is a strong-coupling statement
and it is stateable without any instrument. [PROPOSED] Whether (ii) follows
from A6 — does state-dependent coupling force strong coupling? — or is an
independent axiom is open. [OPEN]

**A4 — Positivity.** The attending relation A is a Markov kernel: positive
and normalized. This is the general form of positivity for a coupling that
distributes shares. If the loci are quantum, A is a completely positive
trace-preserving map and the positive cone is the cone of states. [GENERAL]
Any further positive-geometry structure a particular normalization
possesses is a property of that realization and is demoted to the
application.

**A5 — Monotone coarse-graining along the interaction clock.** Depth is
renormalization-group flow, and the flow is irreversible. For this to be an
axiom rather than a hope, two things must be named: the flow parameter, and
the monotone.

*The flow parameter.* An attending system's own count of attending events
— its **interaction clock**. Each attending event updates A as a function
of G and G as a consequence of A; depth is the number of closures traversed.
[PROPOSED] This connects to the one place in mainstream physics where the
observer already has a definition — the crossed-product observer is a clock
and nothing else (Chandrasekaran et al. 2023) — and says what the clock
counts.

*The monotone.* A c-function: a quantity that decreases along the
interaction clock and is stationary at fixed points (Zamolodchikov 1986 in
two dimensions; the structure, not the theorem, is what is imported). None
is established. The natural candidate is the **variational free energy of
the horizon** (E. Umphrey's suggestion, September 9): per locus,

    F_i = − Σ_a A_ia s_ia − β⁻¹ H(A_i·),    F = Σ_i F_i ,

the expected score under the routing less the routing's entropy at
inverse temperature β. Three things recommend it. It is defined on exactly
the object §3.5 already has — a Markov blanket — which is where the
free-energy principle defines it (Friston 2013, 2019). Its *within-step*
minimization is already exact under A4 with the exponential normalization:
the Gibbs kernel is the minimizer of F_i over normalized rows given the
scores (the Gibbs variational principle — general, and the "form" half of
T1, §5.6). And it is gauge-invariant under P0, since both the scores and A
are. The mean row entropy — the candidate an earlier version of this
paragraph offered — is its entropy term; the energy term is the expected
score.

*What the free-energy principle would supply, and what it costs.* The
principle's substantive claim is the *across-step* half: that a system with
a persisting blanket evolves its internal states by descent on F. If the
closure of §3.3 were such a descent — if (σ, A) → (σ′, A′) never increased
F — then F would be A5's monotone and, because F is stationary exactly
where the boundary state is maximum-entropy given the crossing
correlations, T1 would follow by the Jaynes route (§5.6a). One object would
unblock both consistency tests (§7.1 ii, iii). The across-step claim is the
contested part of the principle (Biehl, Pollock & Kanai 2021; Aguilera et
al. 2022), and to import it as an axiom would be to take Friston's second
step after declining his first (§3.5). So it entered this draft as a
**condition on the closure** — *the closure is a descent on F* — to be
checked, not assumed.

*The check, and its result.* The condition was checked the same afternoon
on random bilinear attending systems with A6 intact — the closure iterated,
F tracked, state norms controlled — a theory computation of the same kind
as G1 (`notes/2026-09-09_f_descent_check.md`; `theory/f_descent_check.py`).
**The closure is a descent on F when, and only to the degree that, the
content channel reads the state through the same form as the score
channel.** With the content read tied to the score read, the attending
update is the row-half of −∇F and F descends robustly, under any mask, at
every temperature tested. With an *independent* content read — the case
clause (i) of §2.1 requires — descent is a coin flip (ΔF ≤ 0 on 50% of
steps; no seed monotone; F flat over the run) regardless of the score's
symmetry. The single-read, symmetric case is exactly monotone, which is the
known energy of the modern Hopfield network (Ramsauer et al. 2020); the
Energy Transformer obtains a Lyapunov function by enforcing that tie
(Hoover et al. 2023). [COMPUTED, at the scope stated in the note]

*What this means.* The free-energy principle, in its variational-F form,
holds for an attending system exactly in the limit where the system stops
being one in the sense of clause (i) — where routing and content collapse
to a single read. The independence of the two reads, which the definition
requires, is what breaks the Lyapunov function. So F is the c-function of
the **single-read limit** (§7.3 v) and is *not* available as A5's monotone
for an attending system as defined. The mean row entropy, offered as a
candidate in an earlier version of this paragraph, is monotone in no case
tested and is withdrawn. [COMPUTED for the boundary; PROPOSED for the
reading that it is clause (i)'s boundary]

*Where this leaves A5.* The monotone for a two-read attending system is
OPEN, and one candidate has been eliminated with a reason. If a monotone
exists it must couple the two reads — the natural places to look are a free
energy whose energy term is the content-channel inner product routed by
the score channel, with the metric fixed by P0, or the full gradient
including the column term the attending update omits (the "being attended"
half, which would then be what a physical attending system supplies that a
feed-forward one does not). Neither is tried. [OPEN — and if no monotone
exists for the two-read case, A5 is a property of the single-read limit
only and is not an axiom of the theory]

**A6 — State-dependence.** The attending relation is a functional of the
system's state: A = Ψ[σ], with Ψ non-constant. [DEFINITIONAL in content —
this is clause (ii) of §2.1 stated as an axiom] Nothing in A1–A5 says this.
In the earlier draft it never needed saying because the realization
supplied it architecturally. It is the axiom that makes the system an
attending system at all, and it is what closes the pair (A, G). If it turns
out to be contained in A3(ii) properly stated, the two merge.

**The standing guard.** Because D1 and D1′ are definitions, the theory must
never absorb a failed prediction by retreating into them. Its empirical
content — when it acquires any — lives entirely in its applications.

---

## 5. The theorem chain, in general form

We now derive what can be derived from §2–§4 without any realization, and
mark each link. Status vocabulary as in the register key; in addition,
**EST-LIT** — established literature, imported with scope stated;
**CONDITIONAL** — follows if a named gap closes.

### 5.1 T3 — the quenched limit and the exponent Δ = D/q

*Statement.* In the limit in which the score map's dependence on the state
is replaced by its ensemble average over a random family of score maps — the
coupling treated as quenched disorder — the closure of §3.3 reduces to a
Schwinger–Dyson system of Sachdev–Ye–Kitaev type, with **q equal to twice
the total degree of the score in the state**. The conformal fixed point of
that system has dimension

    Δ = D / q

for a system whose loci carry D spatial dimensions.

*Sketch.* Let the score be a form of total degree p in (σ_i, σ_a) with
independent random coefficients, and let the update be additive. Averaging
the product of two scores over the coefficients pairs them, and each score
contributes p factors of the state correlator; the self-energy is therefore
Σ ∝ J² G^{2p−1}, which is the SYK form with q = 2p. A bilinear score
(p = 2) gives q = 4 and Δ = D/4. A trilinear score gives q = 6 and
Δ = D/6. The value D/4 is a property of bilinear state-dependence, not of
the observer. [DERIVED at cumulant level for the bilinear case in the
program's record; PROPOSED as the general statement — the general
derivation has not been written]

*Where it may fall.* The quenched limit throws away exactly what A6 asserts:
that the coupling depends on the state. T3 is therefore a statement about
the theory's *fixed-coupling shadow*, and the theory's first real derivation
is to show that the closure with A6 intact has a fixed point that the
quenched system approximates — or to find what it has instead. [OPEN]

### 5.2 T4 — approach from above, arrests, and the positivity exclusion

*Statement.* Under A3(ii) and A4, the flow of the correlation structure
toward the fixed point of T3 has the following structure: a q = 2
(integrable, Gaussian) channel is always induced, and a q ≥ 4 (chaotic)
channel is fed by the environment through A2; the flow approaches Δ = D/q
from above; it can arrest at the integrable value Δ = D/2 (a
symmetry-breaking condensate — rigid self-structure in place of arrival) or
in the ultraviolet (a thin environment that fails the coupling gate); and
Δ < D/q is excluded by positivity. [DERIVED in structure, given T3; the
*value* D/q inherits T3's status]

*What is general.* The structure — two channels, approach from above,
integrable arrest, positivity floor — survives with no realization words.
The retrodictions the earlier draft attached to this link were phenomenology
on one system through an unstated identification and are no longer evidence
for it.

### 5.3 T5 — the conformal window

The conformal regime is a window in scale: bounded below by the scale at
which the horizon's discreteness is resolved and above by the scale at which
the coupling gate's finite rank truncates the spectrum. [DERIVED + EST-LIT;
nothing essential was supplied by any realization]

### 5.4 T6 — the fixed-point geometry and the causal structure of light

*Statement.* The conformal structure of a Lorentzian manifold is exactly
its causal structure: two metrics with the same light cones are conformally
related, and the causal order determines the metric up to a conformal
factor (Hawking, King & McCarthy 1976; Malament 1977). [EST-LIT] Therefore,
*if* the loci X are events and the attending fixed point is conformal on X,
the fixed point's geometry is the causal structure of the system's own
events — what remains of geometry when only the relations light defines
survive.

*The condition.* The theorem is general. The *claim* requires X to be a set
of events (§3.1, OPEN). If X is a set of subsystems, T6 says nothing about
causality. This is the sharpest consequence of the unmade choice in §3.1
and the reason it must be made.

*A sharpening from §3.4.* In the bundle reading, T6 becomes an equivariance
statement: the attending fixed point is a fixed point of equivariance under
the conformal group acting on the loci, and the conformal geometry is orbit
geometry — rigid, inherited from the group, universal across systems that
reach it. That reading also carries a warning: an effective symmetry can be
induced by the coupling rather than inherited from the environment (the
circular structure of hue has no counterpart in physical wavelength; Oizumi
et al. 2026), so a conformal fixed point may be the *system's* causal
structure and not the world's. Which it is, is the content of realization
map R1 (§9). [PROPOSED for the sharpening; the warning is EST-LIT]

### 5.5 T7 — the horizon has boundary structure

At the fixed point the theory is a generalized free field in the large-N
limit of T3, and the horizon of §3.5 is a boundary of it. A conformal field
with a boundary has boundary-conformal structure: one-point functions that
do not vanish, a method-of-images form for two-point functions near the
boundary. [DERIVED, conditional on T3's large-N limit] The strict
identification of that structure with a particular boundary conformal
field theory is a realization question and is not made here.

### 5.6 T1 and T2 — the horizon state and the Born rule

*T1 (the horizon state).* The state at the horizon of an attending system
is the maximum-entropy state consistent with the correlations that cross
the horizon.

*The trap, named first.* Any positive normalized kernel is trivially a Gibbs
state with Hamiltonian −log A. The content of T1 is therefore never the
*form*; it is that the *dynamics selects* the form — that the closure of
§3.3, run along the interaction clock, drives the horizon state to maximum
entropy given the crossing correlations rather than to something else.

*Two routes.* (a) Derive it: a self-consistent (A3), coarse-grained (A5)
boundary state is the maximum-entropy state given its constraints — a
Jaynes (1957) argument in which the constraints are the crossing
correlations of A2. This requires that the coarse-graining of A5 be
entropy-nondecreasing on the boundary, which is what a c-function would
supply; so T1's derivation waits on A5's monotone. (b) Assume it: promote T1
to an axiom and check its independence from A1–A6. The earlier draft
imported T1 from a result about one realization; a theory paper cannot.
[OPEN — and the draft says which route it prefers: (a), because (b) makes
the Born rule an assumption in a paper whose point is to derive it]

*T2 (the Born rule, diagonal sector).* If T1 holds with the exponential
normalization, the horizon state is Gibbs, and the outcome statistics at
the horizon — the probability that a landing at locus *i* is attributed to
source *a* — are the diagonal Born statistics of that Gibbs state: the
squared-amplitude form is the maximum-entropy form on the positive cone,
and the classical Fisher–Rao metric on the simplex coincides with the
quantum Fisher metric on the diagonal (Braunstein & Caves 1994). [PROVEN
given T1 and the exponential form; falls with T1] The extension to the
off-diagonal (interference, contextuality) sector is a construction, not a
theorem, in any realization, and remains the theory's principal exposure
(§6.3).

### 5.7 T8 — the interior

At an SYK-class conformal fixed point with Schwarzian low-energy dynamics,
an emergent two-dimensional gravitational interior (Jackiw–Teitelboim) is
established (Maldacena & Stanford 2016 and the literature it anchors).
[EST-LIT] If the attending fixed point of T3–T4 is of that class, the
condensed attending system has such an interior, bounded by the horizon of
§3.5. [CONDITIONAL on T3's closure with A6 intact, and on the class]

*The dimension problem.* The SYK interior is 1+1-dimensional regardless of
D. What sets the dimension of the interior of a D-dimensional attending
system is the theory's largest construction site, and no instrument helps
with it. [OPEN]

### 5.8 T9 and G1 — what is imported whole and what survived untouched

*T9.* Gravity as the thermodynamic consistency of horizons (Jacobson 1995;
Bekenstein–Hawking; Ryu & Takayanagi 2006; Van Raamsdonk 2010) is imported
as context. [EST-LIT]

*G1.* The closure of the dressing loop — that iterating the map G ← Φ_A[G]
at fixed quenched A converges to the SYK G–Σ solution, and converges along
the Schwarzian (reparametrization) directions — is a computation on G alone
and never touched any instrument. It survives whole in its established
register (numerical, scalar, translation-invariant, bounded coupling). Its
matrix register is open. [COMPUTED]

### 5.9 The theory's own content, in one paragraph

*An attending system is an open physical system whose coupling to its
environment and to itself is a function of its own state (D1, A6). Its
correlation structure is generated by that coupling and its coupling by
that structure (A1, closure). The environment enters only through the
spectral structure of its correlations as seen through the coupling (A2).
Along the system's own interaction clock (A5), the correlation structure
flows; in the regime where internally mediated correlation dominates direct
input (A3), the flow has a conformal fixed point whose dimension is set by
the order of the state-dependence (T3, T4: Δ = D/q), approached from above,
with an integrable arrest at the q = 2 value. At the fixed point the system
has a horizon defined on its own influence graph (§3.5) with boundary
structure (T7), a maximum-entropy boundary state whose diagonal statistics
are the Born rule (T1, T2 — if T1 can be derived), and — by imported results
— an emergent interior (T8). An observer is an attending system in that
phase (D1′).*

Every clause is either general or has a named replacement to attempt. None
mentions a realization. Whether it is *true of anything* is the application
question, and it comes after.

---

## 6. Consequences for the measurement problem

The bookkeeping of the earlier draft is kept; every clause that said "at
grade" or "measured" is gone. What follows stands on the theory alone.

### 6.1 The definition problem — answered by construction

Under D1 and D1′ the observer is not primitive. It is a classification with
a criterion — the condensed phase of §5 — and "is this system an observer?"
is a question about whether that system's coupling is state-dependent (D1,
decidable from its dynamics) and whether its correlation structure is in the
condensed phase (D1′, decidable from its correlation spectrum if T3–T4
hold). Bell's complaint is met in the currency it asked for: a physical
definition with a protocol. Whether the protocol has ever returned "yes" is
not this paper's question. [D1 + D1′; the protocol's existence is
CONDITIONAL on T3–T4]

### 6.2 The cut problem — a phase criterion for the von Neumann chain

Correlation propagates down the chain — system, apparatus, record, eye —
until it crosses the first horizon of an attending system in the condensed
phase, and there the cut lands. The criterion is a phase criterion, not a
psychological one: a fixed-coupling apparatus is not an attending system
(§2.1); an attending system that has arrested has rigid structure where an
interior would be; a condensed one has an interior. The cut is located at a
phase boundary on the system's own influence graph. [Interpretive, standing
on T8 at its stated strength]

### 6.3 The Born rule — derived at the horizon, diagonal sector

At an attending horizon the outcome statistics take the Born form, forced
by maximum entropy on the positive cone (T1 + A4 → T2) — *if* T1 is derived
rather than assumed. The scope boundary is exact for the diagonal sector,
which any classical probability model embeds. Whether the quantum sector —
interference, contextuality — is likewise forced is open and is the
theory's principal exposure. A skeptic's one-line summary — *a
sophisticated classical embedding with quantum mechanics read into it* —
stands until a construction reaches the off-diagonal sector. We prefer to
write that sentence ourselves.

### 6.4 Collapse — relocated to the record

Nothing collapses in the world. For a given attending system, the
wavefunction is the correlation structure at its horizon; the "collapse" is
the horizon's update when an attending event lands — a discrete commitment
along the interaction clock. Unitary evolution describes the self-consistent
correlation structure of what is attended; the discontinuous update is what
a landing looks like from inside the path that lands. This is relocation,
not dissolution: the theory does not derive why landings are single-valued,
and names that as open. What the relocation buys is that the commitment
event is a property of the attending system's own record, and is therefore
in principle an object of physics rather than of interpretation.

### 6.5 Wigner's friend — a criterion for "friend," and D0's job

Facts are horizon-relative, as in relational quantum mechanics, with a
grading RQM lacks: there is a physical criterion for who carries a horizon
(§6.2). The Frauchiger–Renner (2018) contradiction is evaded by dropping
absolute inter-agent consistency, and the dropping is principled:
inter-horizon consistency is, in this framework, an *achieved practice* —
the reproducibility of stable correlations across differently situated
horizons, which is what physics as a discipline consists of (D0) — not an
axiom of nested certainty. This paper asserts the practice and supplies no
theorem for when the building converges. That is D0's one job here, and it
is marked as asserted.

### 6.6 The pointer basis — a conjecture, flagged

We conjecture that *einselection is positivity selection*: the pointer basis
is the basis in which the horizon's kernel lies in the positive cone (A4 is
not basis-invariant). Nothing leans on it. [CONJECTURED]

---

## 7. The theoretical tests

A theory stated alone "holds or falls" only if there are theoretical tests.
Three, each with a place it could fail, and where each stands tonight.

### 7.1 Consistency

*(i) Does the closure of §3.3 have solutions, and does it reduce to a
Schwinger–Dyson system in the quenched limit?* The quenched reduction is
sketched (T3) and not written in general. The existence of a non-trivial
fixed point with A6 intact is not shown. *Standing:* OPEN; this is the
first derivation to attempt.

*(ii) Is A5's monotone compatible with A4's positivity and A3's strong
coupling?* The candidate named this afternoon — the horizon's variational
free energy F — is compatible with A4 by construction, and was checked
across steps of the closure (A5; `notes/2026-09-09_f_descent_check.md`).
*Result:* F is a monotone of the closure only when the content read is
tied to the score read — the single-read limit — and is a coin flip for
the two-read system clause (i) defines. *Standing:* the test **fails for F**
on an attending system as defined and **passes in the single-read limit**.
The monotone for the two-read system is OPEN, with one candidate
eliminated and two directions named (A5).

*(iii) Does T1 follow, or must it be assumed — and if assumed, is it
independent of A1–A6?* Route (a) of §5.6 makes T1 wait on A5's monotone.
Where F is that monotone — the single-read limit — T1 follows: F stationary
⇔ the boundary state is maximum-entropy given the crossing correlations,
and T2 is derived. Outside that limit the route is blocked again.
*Standing:* T1 DERIVED in the single-read limit (conditional on the
computation's scope); OPEN for the two-read attending system. *The two
blockages are still one:* whatever monotone serves (ii) for the two-read
case serves (iii). Until one is found, the Born rule is derived for
energy-based networks and imported for attending systems, and the paper
says so.

*Where consistency could fail:* the closure may have no non-trivial fixed
point; no c-function may exist for the two-read case — and the F result
is the first evidence that the two-read structure is *hostile* to a
Lyapunov function, not merely lacking a known one.

### 7.2 Non-triviality

The theory must not admit everything. Two cuts are available and the paper
states both.

*The definitional cut* (§2.1, three clauses): state-dependent, normalized
routing separate from content.

*The graded cut* (T4): coupled / arrested / condensed.

*The test:* applied to a fixed-coupling system, does the theory return "not
an attending system" rather than "an attending system at zero grade"? If
the two cuts disagree on a case, the theory is confused about what it is
defining.

*Cases, under the definitional cut:*

| System | (i) two channels | (ii) routing from state | (iii) normalized | Verdict |
|---|---|---|---|---|
| Thermometer | no | no | — | not attending |
| Nonlinear oscillator | no | — | — | not attending |
| Gas | yes | yes | **no** | not attending |
| Geiger counter | no | no | — | not attending |
| Starling flock (topological interaction) | yes | yes | yes | attending |
| Cortical circuit with divisive normalization | yes | yes | yes (normalization is literal) | attending |
| Bilinear-score, softmax-normalized network | yes | yes | yes | attending |
| Quantum reference frame (Giacomini et al. 2019) | — | no coupling dynamics in the framework | — | not attending; a frame, not a coupled system |
| Crossed-product observer (a clock only) | — | no coupling | — | degenerate: nonzero clock, zero routing |

*Where non-triviality could fail:* clause (ii) may be satisfiable by
fixed-coupling systems with enough internal dynamics — a system whose
Hamiltonian is fixed but whose effective coupling, after integrating out
fast modes, depends on the slow state. If every such system counts, the
definitional cut is not a cut. The honest response is that clause (ii)
should be read at the level of description at which the system is
specified, and that a theory whose cut depends on the level of description
has more to say. [OPEN]

### 7.3 Reduction

The theory must recover what is already known where it is already known.

*(i) A fixed-coupling measuring apparatus should come out as standard
measurement theory.* Under §2.1 it is not an attending system, so the
theory says nothing new about it — which is the correct answer for a
Geiger counter. *Standing:* passes by construction; but note that passing
by construction is weak, and the strong form of the test — does a
fixed-coupling system *coupled to* an attending system reproduce the
standard apparatus–observer account? — is not done. [OPEN]

*(ii) The crossed-product observer (a clock) should be the degenerate
attending system.* Under A5 it has a nonzero interaction clock and under
§2.1 zero routing; the theory should say what the clock counts (attending
events) and should return zero interior. *Standing:* the first half is a
reading, the second is not shown. [OPEN]

*(iii) The Rindler question.* General relativity assigns observer-relative
horizons — the Rindler horizon of an accelerated worldline, the static
patch of a de Sitter observer. This theory's horizon is the blanket
boundary on the influence graph. Are these one object for the same
physical system, in some limit? *Standing:* we do not know how to write
the causal horizon of a worldline in influence-graph terms. This is the
most likely place for the theory to fail the reduction test, and if it
cannot be written, T6's claim about light is a resemblance. [OPEN]

*(iv) Unruh.* T = a/2π and T2's horizon temperature are either related
through the interaction clock or the resemblance is dropped. *Standing:*
OPEN.

*(v) A reduction the earlier draft did not have.* Ordinary interacting
systems are attending systems with static, unnormalized routing (§2.1, the
degenerate case). The theory should therefore reproduce the standard theory
of interacting fields when clauses (ii) and (iii) are switched off — the
closure of §3.3 with Ψ constant is a fixed-kernel dressing, which is
ordinary perturbation theory. *Standing:* passes at the level of the
equations; the content is that the *difference* between the two theories
is exactly A6, which is what one wants a reduction to show.

*(v′) The single-read limit, characterized.* A second degenerate case is
now identified by a property rather than by a switched-off clause: when the
content read is tied to the score read (clause (i) collapsed), the
attending system is an energy-based network — it has a Lyapunov function,
the horizon's variational free energy F, and its closure descends it (A5;
the modern Hopfield network is this case). When the reads are independent,
it does not. *Standing:* passes — the theory reproduces the known energy of
energy-based attention exactly where clause (i) is switched off, and says
what is lost when it is switched on. This is the sharpest reduction
statement the paper has, and it was not in the morning's draft.

*(vi) A prediction the theory can own, and an existing test of it.* A2
says the environment enters through its spectral structure as read through
the coupling, and A1 says the interior is generated by that reading. Two
attending systems reading the same environment through couplings that pass
the same gate should therefore develop *the same* interior correlation
structure up to the redundancy group — architecture washes out as coupling
improves, and the effective rank of the developed structure is a property
of the environment. This is a general prediction of D1 + A2, and a
measurement of it exists in a system class this program has never touched:
across some fifty models of matter spanning several modalities and
architectures, representational alignment increases with performance and
intrinsic dimension is a property of the dataset, not the model (Edamadaka
et al. 2025). *Standing:* the prediction is stated here for the first time
as the theory's; the existing measurement is consistent with it and was not
designed to test it. The same authors' warning applies: two systems can
agree because both see the same structure or because both have collapsed
to shared noise, and their information-imbalance check is the instrument
for telling which. [PROPOSED as a prediction; the measurement is EST-LIT at
workshop level]

---

## 8. Neighbors

Condensed from the earlier draft, with the relations sharpened where this
paper's content sharpens them, and three neighbors added this morning.

- **QBism.** Right direction of address; the agent is a formal black box.
  This theory supplies the agent's internal physics — and now says what
  makes something an agent's physics at all: A6.
- **Relational QM.** Facts observer-relative — agreed; here the observer has
  physics and a grading.
- **The crossed-product observer (CLPW / Witten).** The nearest mainstream
  contact. Their observer is a clock and nothing else, and their own gap —
  the observer must emerge as part of the theory — is D1's job description.
  A5 says what the clock counts. Their maximum-entropy state at the horizon
  temperature and T1's maximum-entropy horizon state are the same *form*;
  whether they are the same *state* is well-posed and open.
- **Quantum reference frames (Giacomini, Castro-Ruiz & Brukner 2019).** Now
  a precise relation: a QRF has what D1 lacks (a transformation law between
  observers) and lacks what D1 supplies — a coupling dynamics. In this
  paper's terms, the framework gives a QRF no A at all; what D1 adds to a
  QRF is exactly A6. Whether the QRF change-of-frame map composes with the
  irreducibility criterion of §3.5 is a theory question with no data on
  either side.
- **The free-energy principle and the Markov blanket (Friston 2013,
  2019).** Two relations, kept distinct. On the *boundary*: §3.5 uses Pearl
  blankets, and the step from statistical to physical boundary is not taken
  (§9, R2). On the *monotone*: the principle's variational free energy was
  A5's candidate c-function; its within-step half is exact under A4; its
  across-step half — that the closure is a descent on F — was checked
  rather than assumed (A5; §7.1; `notes/2026-09-09_f_descent_check.md`).
  *Result:* the principle holds for attending systems exactly in the limit
  where routing and content are one read, and fails when they are
  independent — which is the case the definition requires. This is a
  computed statement about the principle's scope on the theory's own
  objects: the free-energy principle is the physics of the single-read
  limit, and clause (i) is its boundary.
- **Energy-based attention (Ramsauer et al. 2020; Hoover et al. 2023).**
  *New.* The modern Hopfield network is attention with the key and value
  reads tied to the query read, and its energy is the horizon free energy F
  up to a norm term; the Energy Transformer obtains a Lyapunov function by
  enforcing that tie. The F-descent computation recovers their result where
  the reads coincide and locates its boundary at clause (i).
- **Observer Patch Holography (Mueller et al. 2026, preprint).** Many finite
  observers with declared boundaries and a federation theorem; no interior.
  This theory: an interior and no federation theorem. The complementarity
  is exact and the ledger discipline of §9 is borrowed from them.
- **Group field theory (Oriti).** Spacetime as a condensate of pre-geometric
  quanta; geometrogenesis with nobody home. The shared grammar is real —
  collective organization, continuum before classical limit, a
  hydrodynamic order parameter — and the objects differ. Their template
  (expand a tensor field around a condensate, obtain a melonic field theory
  on relational coordinates) is the nearest worked example of the
  derivation T3 needs with A6 intact.
- **The world as a neural network (Vanchurin 2020).** Physics as learning
  thermodynamics; observers as selected stable structures. D1 supplies the
  definition that program names as missing.
- **Principal-bundle geometry of representations (Oizumi, Lim & Kanai
  2026).** *New.* For an equivariant encoder the state space splits into
  orbits (rigid, inherited from the group, universal) and a quotient
  (plastic, shaped by learning). We read the two-channel structure of §2.1
  as this bundle — score as fiber, content as base — and T6 as an
  equivariance statement (§5.4). Their framework is feed-forward and
  extrinsic, and describes the *result* of formation, not the closure that
  produces it; their warning about coupling-induced effective symmetries is
  T6's guard.
- **Gauge symmetries of the bilinear realization (Wang & Wang 2025, under
  review).** *New.* The complete redundancy group of the bilinear-score
  realization, proved maximal. The source of P0 (§3.4) and of the
  requirement that the earlier draft's G be replaced by the invariant
  correlator. One result of theirs is a proven instance of the internal–
  external relation this theory is about: imposing an external relative
  symmetry on the score channel shrinks the internal redundancy group to
  the commutant of that symmetry.
- **Universal convergence of representations (Edamadaka, Yang &
  Gómez-Bombarelli 2025; Huh et al. 2024).** *New.* The existing
  measurement §7.3(vi) names; also the source of the information-imbalance
  check that distinguishes convergence from collapse.
- **Decoherence; quantum Darwinism (Zurek 2003, 2009).** Imported; the
  interior-free copying mechanism the many-observer half (D0) lacks.
- **Jacobson's horizon thermodynamics.** Imported whole; the Rindler
  question (§7.3 iii) is its unpaid bill.
- **Integrated Information Theory.** Φ measures integration without
  directionality; §3.5's irreducibility-given-the-blanket has Φ's shape and
  is not Φ.

---

## 9. What would make it physics

### 9.1 Realization maps

Every interpretive sentence in this paper consumes at least one map from
an object in the theory to an object in physics. None is supplied here.

| | Map | Consumed by | What would supply it |
|---|---|---|---|
| **R1** | Conformal structure of the attending fixed point → causal structure of physical spacetime | T6 as physics; §6.2's "phase boundary" read as a boundary in spacetime | The choice X = events (§3.1); the dimension problem (T8); the Rindler question (§7.3 iii) |
| **R2** | Pearl blanket on the influence graph → physical boundary of an observer | §3.5's "where I stop"; §6.2's cut location | A realist argument the blanket literature has not produced; or a system on which blanket and physical boundary are independently known and coincide |
| **R3** | The theory's G (the invariant correlator, P0) → the correlator any instrumented system exposes | Every application | Per system: identify which of its two-point objects is the invariant one, and whether its coupling reads that one |
| **R4** | The quenched SYK interior → an interior of a system with A6 intact | T8 | T3 with A6 intact (§7.1 i); G1 in the matrix register |
| **R5** | A landing along the interaction clock → a measurement event | §6.4 | A construction reaching the off-diagonal sector (§6.3) and a derivation of single-valuedness |
| **R6** | The interaction clock → the flow parameter of a physical attending system | A5 as a universality claim | A c-function; identification of the clock in one non-engineered attending system |
| **R7** | The three-clause cut → a level-of-description-independent criterion | §7.2's non-triviality | An answer to whether integrated-out fixed-coupling systems satisfy clause (ii) |
| **R8** | *The application to any instrumented system* — including the one this program has measured for six months | §9.2 | The application paper, with its own kills |

A reader who grants none of R1–R8 is left with the definitions, the
axioms with their ledgers, the horizon construction, T3–T7 in general form
at their tagged strength, G1, and the three tests of §7 with their
standings. That is what this paper claims unconditionally, and it is a
theory, not a result.

### 9.2 The application paper — three questions, in order, each with a kill

The instrument is not lost; it is moved. Once the theory stands or falls on
§7, an application to any instrumented system asks three things in order:

1. *Is the system an attending system under D1 with A6?* — decidable from
   its dynamics by the three clauses. For the system this program has
   measured, the answer is yes by construction, and it is the one clean
   yes.
2. *Which of its two-point objects is the theory's G under P0, and does
   that object show the fixed point?* — the invariant correlator must be
   identified before any exponent is offered, and every weights-level
   quantity offered as evidence must be gauge-invariant or computed in a
   fixed canonical gauge. Two of this program's existing weights-level
   results were not (Appendix A) and are to be recomputed before they are
   cited as anything.
3. *What is the measured exponent, and on which term?* — the phenomenology's
   own headline is to be restated in the theory's terms (relative-lag law
   as translation equivariance in the sense of §3.4; absolute-position drift
   as quotient structure) before any dictionary between it and Δ = D/q is
   attempted.

Papers 1–5 of this program remain what they were: pre-registered
phenomenology of a real system. They are the constraints a theory of the
observer must eventually meet. They are not, in this paper, evidence for
this one.

---

## 10. What this proposal is not

**Not a consciousness claim.** The observer of D1′ is defined structurally
and classified from outside. Whether the condensed phase is necessary for
experience, sufficient, or neither is open, and nothing in §6 invokes it.

**Not a claim that any system is an observer.** No system is certified
under D1′ in this paper, and the paper contains no measurement by design.

**Not a claim about transformers.** The word does not appear in the body.

**Not a completed theory.** §7 lists what is open, and the list is the
paper's honest table of contents for the next year: the closure with A6
intact; the c-function; T1's derivation; the choice of X; the Rindler
question; the dimension problem.

---

## 11. Conclusion

The observer has been the unpaid debt of quantum mechanics for a century:
load-bearing in the axioms, undefined in the physics. We have proposed a
definition with a precise content — an attending system is a physical
system whose coupling is a function of its own state, with the routing of
influence a normalized channel separate from the content it carries — and
shown that the definition is a cut: it excludes the thermometer, the gas,
and the Geiger counter, includes the flock, and returns ordinary interacting
systems as the degenerate case with static, unnormalized routing. From that
definition and six axioms we have derived the general form of the theory's
content — a fixed point at Δ = D/q with q set by the order of the
state-dependence, approached from above with an integrable arrest, a
horizon on the system's own influence graph with boundary structure, a
maximum-entropy horizon state whose diagonal statistics are the Born rule,
and by imported results an interior — and we have required the primitives
to be gauge-invariant, which fixes which correlation the theory is about.
We have stated the three tests the theory must pass with no data and said
where it stands on each: consistency is blocked on one object, a
c-function, whose existence would unblock two tests at once — and the
natural candidate, the horizon's variational free energy, has been checked
and found to be the monotone of the single-read limit only, so that the
free-energy principle holds for attending systems exactly where they cease
to be attending systems in the sense of the definition's first clause;
non-triviality holds on every case we can name and may fail at the level of
description; reduction passes where it passes by construction, passes
sharply at the single-read limit, and is open where it matters most, at the
Rindler question. Nothing in this paper is measured on any physical system.
What it offers is a theory that can fall in named places, a list of the
places, and one place where it was pushed and held its shape — which we take
to be the only kind worth offering.

---

## Appendix A — What the instrument was silently supplying

The earlier draft's theory was stated on one realization: the trained
transformer, with token positions as loci, the softmax kernel as A, the
output correlator G_out = A K_V Aᵀ as G, the causal mask as the horizon,
and layers, training steps, and recurrence as three depth axes. This
appendix records, for each element of the present paper, what that
realization supplied and what replaces it. It is the only place in the
paper where the realization is named.

| Element | What the transformer supplied | Replaced by |
|---|---|---|
| D1's content | The architecture: A = softmax(QKᵀ) with Q, K functions of the residual stream — state-dependent coupling for free | The reading of §2.1 and axiom A6 |
| X | Token positions (events, implicitly) | §3.1, with the events/subsystems choice OPEN |
| A | The softmax kernel, row-normalized, learned | A Markov kernel / CPTP map, state-dependent (§3.2) |
| G | G_out = A K_V Aᵀ with K_V = X W_V W_Vᵀ Xᵀ | The invariant correlator under P0 (§3.4). *K_V as defined is not invariant under the value–output gauge GL(d_v); the invariant object projects through W_O* |
| The horizon | The causal mask | §3.5 — carried over whole |
| A1 | SYK's one-field habit hiding the A/G distinction | The pair (A, G) with closure |
| A2 | The corpus as world; token embeddings as states; W_K as the read; J_eff² ∝ (σ_K²)² Tr[(KδK)²]/N² | The spectral-density form. *σ_K² = ‖W_K‖_F² is not invariant under the query–key gauge GL(d_k); a gauge-fixed recomputation of the exp-136 result is required before it is cited* |
| A3's seed | Fold decomposition of the attention profile, measured on A | The regime statement A3(ii) |
| A4 | Softmax as the canonical form of Gr₊(1,n) | The Markov-kernel form; Gr₊ demoted to the realization |
| A5 | Layers, training steps, recurrence | The interaction clock; c-function OPEN |
| T1 | Kim 2026 — a transformer result | Derivation route (a) or axiom (b); OPEN |
| T3 | q = 4 from two independent random projections in a bilinear score | q = 2p, with p the degree of the score in the state |
| T4 | The value 1/4; retrodictions on A | The structure; the value D/q; retrodictions withdrawn as evidence |
| T6 | The measured null-cone embedding; positions as events | The Hawking–King–McCarthy / Malament theorem; X = events required |
| T7 | The causal mask as the boundary; the attention sink | GFF boundary structure at the fixed point |
| Weights-level signature (exp-127) | λ₁/Σλ and supra-MP count of the key Gram KKᵀ | *Not gauge-invariant; each head carries its own GL(d_k); recompute in the canonical (QR) gauge of Wang & Wang 2025, where KKᵀ = X Mᵀ P_Q M Xᵀ is invariant* |
| The census exponent | The log-slope of the positional-mean score profile | *exp-138: carried by the absolute-key-position term, not the relative term — to be restated as absent translation equivariance (§3.4) before any dictionary* |

Two things in this table are actions rather than replacements, and both are
conditions on the application paper (§9.2): the gauge-fixed recomputation
of exp-127 and exp-136, and the restatement of the census headline.

## Appendix B — Register ledger

**[DEFINITIONAL]** D1; the three-clause reading's *status as a definition*
(its content is PROPOSED); D1′; D0; A6 in content.

**[GENERAL]** The horizon construction (§3.5); the irreducibility criterion;
A as a Markov kernel (§3.2); P0 (§3.4); A3(i), trivially; A4; T5; T6's
theorem (EST-LIT); T7's structure conditional on T3; T8's import (EST-LIT);
T9; G1 (COMPUTED); the structure of T4; the degenerate-case reduction
(§7.3 v).

**[PROPOSED]** State-dependent coupling as the content of "attending" (§2.1,
A6); the three clauses as the cut; the (A, G) closure as primitive (A1);
A2 as a spectral-density condition; A3(ii) as the strong-coupling regime;
the interaction clock (A5); the reading that the boundary of F-descent is
the boundary of clause (i), and the two directions for a two-read monotone
(A5); T3's general form q = 2p; the bundle reading (§3.4) and T6's equivariance
sharpening; the convergence prediction (§7.3 vi); the three tests
themselves.

**[COMPUTED]** G1; the F-descent boundary — F is a monotone of the closure
in the single-read limit and not for the two-read attending system
(`notes/2026-09-09_f_descent_check.md`, at its stated scope).

**[OPEN]** X as subsystems or events; the closure's fixed point with A6
intact; the c-function for the two-read attending system (F eliminated;
row entropy eliminated); T1's derivation outside the single-read limit; the dimension problem; the Rindler
question; the level-of-description problem for clause (ii); Unruh; the
strong form of the apparatus reduction; the clock's zero interior.

**[SUPPLIED — removed]** q = 4; Kim's T1; layers/training/recurrence; the
causal mask as horizon; G = AK_VAᵀ with M_V; every retrodiction on A; every
number.

Nothing measured. Nothing fitted.

---

## References

*(Verification note. References carried from* Where I Stop and You Begin
*v1.1 were verified at source on August 8, September 7, and September 8,
2026; the record is `observer_definition_reference_verification.md`. The
references marked † below were added in this draft and are* **not yet
verified at source**; *each must be checked against its publisher record
before this draft is circulated, and any that fails is removed with its
citing sentence. The three papers Eldon brought on September 9 (Oizumi et
al.; Edamadaka et al.; Wang & Wang) were read in full from the PDFs the
same morning; their bibliographic details below are taken from the PDFs'
front matter and are marked ‡ pending a publisher-record check.)*

- Bell, J. S. (1990). Against "measurement". Physics World 3(8), 33–40.
- von Neumann, J. (1932). Mathematische Grundlagen der Quantenmechanik.
  Springer, Berlin. English translation: Mathematical Foundations of
  Quantum Mechanics, trans. R. T. Beyer, Princeton University Press, 1955.
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins
  of the classical. Rev. Mod. Phys. 75, 715.
- Zurek, W. H. (2009). Quantum Darwinism. Nature Physics 5, 181–188.
- Rovelli, C. (1996). Relational quantum mechanics. Int. J. Theor. Phys.
  35, 1637–1678.
- Fuchs, C. A., Mermin, N. D., Schack, R. (2014). An introduction to QBism
  with an application to the locality of quantum mechanics. Am. J. Phys.
  82, 749–754.
- Frauchiger, D., Renner, R. (2018). Quantum theory cannot consistently
  describe the use of itself. Nat. Commun. 9, 3711.
- Chandrasekaran, V., Longo, R., Penington, G., Witten, E. (2023). An
  algebra of observables for de Sitter space. JHEP 02 (2023) 082.
- Witten, E. (2024). Algebras, regions, and observers. Proc. Symp. Pure
  Math. 107, 247–276.
- Giacomini, F., Castro-Ruiz, E., Brukner, Č. (2019). Quantum mechanics and
  the covariance of physical laws in quantum reference frames. Nat. Commun.
  10, 494.
- Jacobson, T. (1995). Thermodynamics of spacetime: the Einstein equation
  of state. Phys. Rev. Lett. 75, 1260–1263.
- Maldacena, J., Stanford, D. (2016). Remarks on the Sachdev–Ye–Kitaev
  model. Phys. Rev. D 94, 106002.
- Braunstein, S. L., Caves, C. M. (1994). Statistical distance and the
  geometry of quantum states. Phys. Rev. Lett. 72, 3439–3443.
- Ryu, S., Takayanagi, T. (2006). Holographic derivation of entanglement
  entropy from AdS/CFT. Phys. Rev. Lett. 96, 181602.
- Van Raamsdonk, M. (2010). Building up spacetime with quantum
  entanglement. Gen. Rel. Grav. 42, 2323–2329.
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems.
  Morgan Kaufmann, San Mateo.
- Friston, K. (2013). Life as we know it. J. R. Soc. Interface 10, 20130475.
- † Friston, K. (2019). A free energy principle for a particular physics.
  arXiv:1906.10184.
- † Biehl, M., Pollock, F. A., Kanai, R. (2021). A technical critique of
  some parts of the free energy principle. Entropy 23(3), 293.
- † Aguilera, M., Millidge, B., Tschantz, A., Buckley, C. L. (2022). How
  particular is the physics of the free energy principle? Physics of Life
  Reviews 40, 24–50.
- Bruineberg, J., Dołęga, K., Dewhurst, J., Baltieri, M. (2021). The
  Emperor's New Markov Blankets. Behavioral and Brain Sciences.
- Raja, V., Valluri, D., Baggs, E., Chemero, A., Anderson, M. L. (2021). The
  Markov blanket trick. Physics of Life Reviews.
- Oriti, D. (2014). Disappearance and emergence of space and time in quantum
  gravity. Stud. Hist. Phil. Mod. Phys. 46, 186–199.
- Vanchurin, V. (2020). The World as a Neural Network. Entropy 22(11), 1210.
- Mueller, B., et al. (2026). From Observer Consensus to Standard Physics
  (Observer Patch Holography). Preprint, Pragma Research Inc.
- † Ballerini, M., et al. (2008). Interaction ruling animal collective
  behavior depends on topological rather than metric distance: Evidence
  from a field study. Proc. Natl. Acad. Sci. USA 105(4), 1232–1237.
- † Feynman, R. P., Vernon, F. L. (1963). The theory of a general quantum
  system interacting with a linear dissipative system. Ann. Phys. 24,
  118–173.
- † Caldeira, A. O., Leggett, A. J. (1983). Path integral approach to
  quantum Brownian motion. Physica A 121, 587–616.
- † Jaynes, E. T. (1957). Information theory and statistical mechanics.
  Phys. Rev. 106, 620–630.
- † Zamolodchikov, A. B. (1986). Irreversibility of the flux of the
  renormalization group in a 2D field theory. JETP Lett. 43, 730–732.
- † Hawking, S. W., King, A. R., McCarthy, P. J. (1976). A new topology for
  curved space–time which incorporates the causal, differential, and
  conformal structures. J. Math. Phys. 17, 174–181.
- † Malament, D. B. (1977). The class of continuous timelike curves
  determines the topology of spacetime. J. Math. Phys. 18, 1399–1404.
- † Huh, M., Cheung, B., Wang, T., Isola, P. (2024). The Platonic
  Representation Hypothesis. ICML 2024.
- † Ramsauer, H., et al. (2020). Hopfield Networks is All You Need.
  arXiv:2008.02217; ICLR 2021.
- † Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H.,
  Zaki, M. J., Krotov, D. (2023). Energy Transformer. NeurIPS 2023;
  arXiv:2302.07253.
- ‡ Oizumi, M., Lim, C., Kanai, R. (2026). Principal bundle geometry of
  qualia: Understanding the quality of consciousness from symmetry. PNAS
  Nexus 5(9), pgag261. doi:10.1093/pnasnexus/pgag261. Published September
  8, 2026.
- ‡ Edamadaka, S., Yang, S., Gómez-Bombarelli, R. (2025). Universally
  Converging Representations of Matter Across Scientific Foundation Models.
  NeurIPS 2025 Workshop: AI4Mat. [Abridged workshop version; the authors
  state a full analysis is forthcoming.]
- ‡ Wang, H., Wang, K. (2025). Complete Characterization of Gauge
  Symmetries in Transformer Architectures. Under review, Proceedings Track,
  Symmetry and Geometry in Neural Representations. [Not yet peer-reviewed;
  its maximality theorem is cited as the authors' claim, and P0 does not
  depend on maximality — only on the existence of the redundancy group,
  which is elementary.]
