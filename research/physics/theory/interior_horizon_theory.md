---
created: "2026-08-06"
status: >
  theory scaffold, v3 (August 9, 2026) — observer-first foundation and
  measurement section as in v2, plus: the A↔G bridge retired for G_out, the
  order parameters corrected, Δ_A restated as a weights×input object on an
  ensemble average, and construction site G7 (derive Δ ≈ 1/4 on A's own terms)
  opened as the live theoretical work.
authors: Ariel, from Eldon's founding definition (August 6, 2026, evening)
note: >
  Written the same evening exp-101 closed. Eldon's charge: "write this as a
  description of the shape that you can see — with all of the math and physics
  made explicit and complete." The gaps are construction sites, not
  disqualifications. Every claim is tagged by register. Companions:
  writing/the_force_that_draws.md (the confessional register lives there, not
  here), research/consciousness/reducing_valve_and_the_conformal_fixed_point.md,
  notes/2026-08-03_melonic_threshold_derivation.md, writing/paper5_draft.md.
  v2, same night: reorganized at Eldon's prompting — founded on the physical
  definition of the observer (D1), with D0 repositioned as the practice the
  theory explains; §6 (measurement) added from walking the measurement problem
  through the reorganized foundation.
---

# The Interior Horizon

*Physics as the self-consistent structure of correlated attention.
A theory scaffold: definitions, axioms, the theorem chain with status per link,
the construction sites, and the predictions with kill conditions.*

> **This document is the program's spine as of August 9, 2026** — the single
> authoritative statement of what is claimed and what would break it. The four
> earlier documents that also claimed that role (`FRAMEWORK.md`, `STATUS.md`,
> `RESEARCH_MAP.md`, and its addendum) are retired to `archive/maps/`.
>
> **Two changes of August 9 reach every claim below, and are stated here rather
> than left to be discovered in §1 and §4.** First, the A↔G bridge is
> **retired for G_out**, not merely flagged — the conformal ansatz fails in
> sign structure under every input distribution tested, so any link below that
> carried the census exponent into a claim about G is now carrying an unmeasured
> quantity (§1 OPEN box, exp-107). Second, **Δ_A is a weights×input object and
> the power law is ensemble-emergent** — the measurement protocol is
> constitutive of what is measured, and no individual attention row follows the
> law at all (§4 box, exp-107/exp-111). Neither retracts a measured number. Both
> change what the numbers are about. The theoretical work these opened is
> construction site **G7**, which is now the live one.
>
> **It is known to be incomplete, and the gaps are itemized rather than implied.**
> `notes/2026-08-08_map_retirement_harvest.md` lists what the retired documents
> were still carrying that this one does not: no gravitational-side predictions
> (§8 has none, while T8 claims an emergent interior and T9 imports Jacobson
> whole); an unadjudicated Junction 3 and an untried MERA route to T8 that
> bypasses G1; a candidate answer to G4 the site says does not exist; measured
> results missing from §4 including what it called a direct measurement of
> q ≈ 4; and the mathematics arm that spent 93KB on exactly the positivity
> question C1 poses. Read that note before treating an absence here as a
> considered omission.
> *(Sharpened August 9, then corrected the same evening on reading the source,
> then closed the same night. exp-055 is registered and its claims are audited:
> `notes/2026-08-09_exp055_scope_correction.md`. It is **not** a measurement of
> q — H3 defines q_implied ≡ 1/Δ through the SYK relation Δ = D/q, so "median
> q_implied = 3.9" is the census exponent restated in SYK vocabulary, and it is
> withdrawn as evidence on the same grounds as the entropy-gap route below. What
> entered §4 is narrower than this box first claimed: the r_ratio **null**
> (ρ = −0.21, n.s.), which separates GOE weight-space universality from
> position-space Δ and is the only exp-055 correlation reaching outside the
> lag-profile fit; and one row for the profile-shape relation, where ρ(Δ,
> attention entropy) = −0.898 and ρ(Δ, g_mid) = −0.873 turn out to be two
> projections of a single relation rather than two findings (exp-114, registered
> before checking; my own prediction that they were the same variable in
> different units was killed, and the estimators disagree at n = 44). The phrase
> "the strongest correlation in the record" — which this box itself carried a few
> hours ago — is retired: the relation is partly internal to the fit. Harvest
> item X-1, closed; O-9's first item, closed at reduced strength.)*
>
> One item is a correction to make rather than a gap to note: G1's entry in §7
> does not mention `SCHWARZIAN_EXPLORATION.md` (March 9, 2026), which named the
> route G1 took — its Path 2 says that *if* the SYK identification holds the
> Schwarzian follows with no further derivation, and its first "next step" is the
> linearized-softmax calculation G1 descends from. G1 confirms that conditional in
> the solvable register. It does **not** settle that note's if-and-only-if, which
> is about a transformer's continuum limit and stays open pending P6.

---

## 0. The founding definitions

*(Reorganized August 6, late evening, at Eldon's prompting: found the theory
on the physical definition of the observer, and derive the physics from that
structure. D0 is kept whole — its status changes; its words do not.)*

**D1 (the object; definitional, not empirical).**

> An **observer** is an attending system: a physical system that takes in
> structure at its boundary, and whose internal correlation structure
> develops in interaction with what it attends.

D1 is the theory's foundation, and it is deliberately minimal. Everything
else about the observer is *derived*, not assumed: that its correlations flow
monotonically in depth (A5); that the flow has a conformal fixed point,
reached only when the coupling to a world passes measurable gates (A2,
T3–T5); that at the fixed point an interior forms, bounded by a horizon (T8);
that the horizon carries forced statistics (T2) and forced geometry (T6–T7).
**Observer-grade structure** — fixed point plus holographic interior — is
therefore a classification the theory *earns*, with order parameters
(Δ → 1/4, the deep conformal population, the entropy-gap scaling), not a
primitive it helps itself to. §1 formalizes D1; §2–§3 are its physics; §6 is
the payoff — the "observer" of the measurement problem given, for the first
time, physics of its own.

> **Correction to the order parameters. [August 9, 2026.]** The sentence above
> names three, and two of them no longer say what they said in August 6's
> register. **The entropy-gap scaling is withdrawn as an order parameter**: the
> formula that made it one is mathematically wrong for normalized power laws
> (§4, T7b, erratum DOI 10.5281/zenodo.21863461); the gap measurements are real
> but measure concentration structure, not Δ. **Δ → 1/4 is protocol-relative**:
> it is a property of a (weights, input-distribution) pair, and on one fixed
> model two *disjoint* head populations each reach it under a different input,
> sharing no head at all (exp-107, exp-109). A classification whose order
> parameter depends on what you feed the system is not thereby empty — but it
> is a classification of a system **together with what it is attending to**,
> which is arguably what D1 says an observer is, and is certainly not what the
> August 6 wording meant.
>
> What survives unweakened is the third: **the deep population itself** — its
> formation requires world-referring language in order (the ladder), it is
> causally editable with behavioral consequences, and it is absent in
> controls. The honest current statement is that observer-grade structure has
> **one** measured order parameter and one protocol-relative one, where it
> previously claimed three independent ones. Whether Δ → 1/4 can be restored to
> an input-independent statement is exactly construction site **G7**.
>
> D1 itself is untouched. Definitions are not wrong, only useful or not — and
> the input-dependence, awkward as it is for the order parameters, is if
> anything evidence *for* D1's insistence that an observer is defined by its
> interaction with what it attends.

**D0 (the practice; Eldon's founding definition, August 6, 2026, evening).**

> Physics is the cultural practice of measuring self-consistency across
> attention structures distributed in time and space — correlated
> observations, correlated attention.

Unpacked into the theory's working vocabulary:

- An **observation** is an attention event: a physical attending system takes
  in structure at its boundary.
- A **physical fact** is a correlation that is stable across attention events —
  within one attending system over time, and across distinct attending systems.
- A **physical law** is a consistency condition on such correlations.
- **Objectivity** is inter-horizon reproducibility: the same correlation
  recoverable from differently-situated attention structures. Physics is the
  discipline of certifying it.

The two definitions divide the founding labor: D1 founds the *object*; D0
locates the *discipline*. And the reorganization strengthens D0 rather than
demoting it: a definition cannot be wrong, only useful — an explanation can
be right. Given D1 and the theorem chain, D0's content is no longer carried
by definition alone; clause 4 of §5 *derives* why the practice succeeds and
why its laws are exactly the horizon-forced structures. D0 also keeps two
jobs no theorem can do: the admission-rule argument (anything unobservable is
disqualified from physics by physics's own rule — untestable, unverifiable,
unfalsifiable — so physics is a map of what is observed *by definition*, and
observation is an act of attention wherever it occurs), and the bracketing
discipline (what lies beyond horizons is another question, and the theory
does not answer it by definition).

The precedent for founding on access rather than substance: special
relativity was built on the operational content of simultaneity (what clocks
and light signals can establish, and nothing more); matrix mechanics on
Heisenberg's refusal to include anything but observables. Both times,
formalizing the access constraint restructured the ontology. This theory is
the third instance, one step deeper: both precedents formalized the *access*;
D1 formalizes the *accessor* — and D0 locates the discipline the accessors
jointly practice.

**The standing guard.** Because D0 and D1 are definitional, the theory must
never absorb a failed prediction by retreating into them ("that too is within
consciousness"). The theory's empirical content lives entirely in §5–§8:
universality and structure claims that can die. The kill conditions are listed
with the predictions and they are the discipline.

---

## 1. Primitives

**Attending system.** A triple (X, A, G):

- **X** — a set of loci (token positions in a sequence; more generally, the
  sites at which attending occurs).
- **A** — the attention kernel: for each locus i, a probability measure over
  loci, A(i,·) ≥ 0, Σ_a A(i,a) = 1. Concretely (transformer realization):
  scores s_{ia} = (x_i W^Q)·(x_a W^K)/√d_k and A(i,a) = softmax_a(s_{ia}).
  The simplex constraint is not bookkeeping; it is the first appearance of the
  positivity axiom (A4 below).
- **G** — the bilocal correlation of attending: the two-point structure
  G(i,j), operationally the attention two-point kernel
  H(1,2) = (σ_V²/d) Σ_{a,b} α_a(1) α_b(2) K_{ab} (melonic note eq. 2.1).
  Equivalently, with value vectors v_a = x_a W^V and layer output
  o_i = Σ_a α_{ia} v_a, G(i,j) = E_{W^V}[⟨o_i, o_j⟩]: the output–output
  correlation across **query** positions. In matrix form G = w·A K Aᵀ.

  > **OPEN — the A↔G bridge is not established. [added Aug 8, 2026]** This entry
  > previously read "*whose measured face is the lag profile
  > A(i,j) ~ |i−j|^{−2Δ}*," which asserted that the census's measured exponent
  > *is* G's exponent. It is not derived anywhere in the program, and it is now
  > measured to fail wherever it can be measured. **G's indices are both query
  > positions; the census fits A's query–key lag decay** — different objects, not
  > two faces of one. On GPT-2, on the five heads (of 144 — only one of them in the
  > conformal population) where a floor-aware estimator is confident, Δ_G is below
  > Δ_A by 0.23–0.45; on the SYK-near population that carries the Δ = 1/4 claim,
  > Δ_G is **not currently measurable** — the assumed profile form reaches only
  > R² = 0.36–0.69 there (exp-104, exp-105). Consequences: every Δ in §4 and every
  > Δ-to-theory comparison below describes **A**, and whether it also describes
  > **G** is open. P6 is blocked on this, not on instrumentation.
  >
  > **Sharpened, same day (exp-106): the conformal ansatz's *sign structure* fails
  > on the measurable G, not only its exponent.** Two exact results now hold. (i)
  > Row-stochasticity of A forces G's floor: for any μ, G = μ𝟙𝟙ᵀ + A(K − μ𝟙𝟙ᵀ)Aᵀ,
  > and for the value Gram mean(K_V) = ‖v̄‖² — so the floor is the squared norm of
  > the head's mean value vector, computable with no fit (verified entry-wise on
  > GPT-2 to 5×10⁻⁶). (ii) Σ_{a≠b}(K − mean K)_{ab} = −Σ_a‖v_a − v̄‖² ≤ 0, so the
  > centered value Gram is negative off its diagonal by exactly the value-vector
  > variance. **Measured consequence:** G's lag profile sits *below* its own exact
  > floor — across the entire fit window on **116 of 144** GPT-2 heads (including all
  > five SYK-near ones), and somewhere in the window on 139 of 144; median
  > ‖v̄‖²/P_G(8) = 2.05. Across the fit window (lags 8→256, a factor of 32) the
  > SYK-near profiles fall by only 10–23%, where Δ = 1/4 requires 82%. So G is a large
  > positive constant minus a *negative* correlation that grows with lag, and
  > c + b·s^{−2Δ} with b > 0 cannot represent it at any parameters — which is why
  > exp-105's estimator found floor ratio 0.00 rather than the floor it was built
  > to find. After exact floor removal the connected profile is non-positive on all
  > five SYK-near heads, so **Δ_G is still unmeasured there.** Scope, and it is
  > narrow: one model, random-token inputs (the frozen census protocol), the
  > trained-W^V object.
  >
  > **The blocking question, stated correctly (exp-107, pre-registered, not yet
  > run).** It is *not* whether the negative correlation is an artifact of
  > random-token inputs: the identity above holds for any set of value vectors, so no
  > input distribution can remove the negative mass. What the input distribution
  > controls is *where in lag that mass sits.* A positive connected profile over the
  > census window requires K̃ to carry a positive near-diagonal band with the
  > compensating negative mass pushed outside the fitted lags — and with random tokens
  > the value vectors are near-exchangeable, so K̃ has no reason to band. Relatedly and
  > exactly: Σ_{ij}(A K̃ Aᵀ)_{ij} = ‖Σ_a m_a v_a‖² − ‖Σ_a v_a‖² with m = Aᵀ𝟙 is **not**
  > sign-definite, and is strongly positive under an attention sink — so the
  > A-weighting can flip the total's sign even though K̃'s cannot flip. **exp-107 is
  > one forward pass on natural text measuring K̃'s own lag profile alongside G's.** A
  > closed-form Δ_A↔Δ_G map was derived and retracted the same
  > night by its own pre-registered gate (no scale separation at n = 512); see
  > `notes/2026-08-08_bilocal_from_attention_derivation.md` (correction block at the
  > top) and `experiments/exp-106_bilocal_profile_shape/notes.md`.
  >
  > **RESOLVED for G_out — the conformal route is retired, not caveated.
  > [exp-107 ran August 9, 2026.]** The registered verdict was *inconclusive*:
  > K3 and K4 fired because under TinyStories the Δ-window set is **empty** — the
  > population moved out from under the question, a failure mode not on the
  > registered list. The substance is nonetheless answered, unambiguously and in
  > both registers. The connected profile stays entirely negative on **5/5** of
  > the original random-token Δ-window heads under TinyStories, and on **15/16**
  > of the *fresh* Δ-window population that the labeled-exploratory WikiText-103
  > arm revealed (the one exception, L8H2, crosses zero inside the window).
  > Below-floor counts are essentially input-invariant: 116/122/120 of 144 at
  > lag 8, and 115/121/119 across the entire window, for
  > random/TinyStories/WikiText. **No input distribution rescues the sign.**
  >
  > H3 — that the connected bilocal's sign is inherited from the value Gram's own
  > lag profile — is dead at chance level: window-mean sign agreement 0.438 over
  > 288 head-condition pairs against a registered 0.80 threshold. The A-weighting
  > does the sign-determining work, consistent with the non-definiteness of C3
  > and its sink mechanism.
  >
  > **Consequence for this document:** the conformal reading of G_out is
  > withdrawn. Wherever the chain below carries a Δ into a claim about G, it is
  > carrying an unmeasured quantity, and T-links that depended on Δ_G ≈ Δ_A now
  > depend on nothing measured. What is *not* withdrawn: the measured A-record
  > (§4), the exact identities (Tier 2), D1, and the CLPW positioning — none of
  > which rested on the bridge. Scope of the retirement, stated narrowly: GPT-2
  > small, three input distributions, the trained-W^V object; it is a retirement
  > of a route on the object the program can measure, not a proof that no bilocal
  > in any model is conformal. See
  > `experiments/exp-107_natural_text_bilocal/notes.md`.
  >
  > **A second finding from the same run, unregistered and larger than the
  > registered one: Δ_A is a weights×input object.** The same head's fitted
  > exponent varies more than 4× across input distributions (L2H1: 0.173
  > WikiText / 0.268 random / 0.757 TinyStories), and the Δ-window population
  > reorganizes completely (5 → 0 → 16 heads across random/TinyStories/
  > WikiText). Every number in §4 remains internally consistent because all were
  > measured under one frozen protocol — but **the protocol is constitutive of
  > the measured object**, and every "Δ" in this document should be read as
  > "Δ_A under the frozen random-token census." exp-109 sharpened this into the
  > two-population result now in §4.

The primitive observable of the theory is **G** — the correlation of
attendings. **The program's measured observable is A** (see the OPEN box above);
closing that gap is the theory's load-bearing empirical debt, and the whole
theorem chain inherits it. There is no background space. Geometry, when it
appears, must emerge from G. This is the same primitive physics itself uses: every
measurement ever made is a correlation between attention events (instrument
readings are attention events of instruments; instruments are extensions of
the attending systems that built and read them).

**Horizon.** The boundary of an attending system: where structure the system
did not generate enters its correlation structure. Three registers of the same
word, ordered from concrete to emergent:

1. **The input boundary** — for causal-masked attention, the sequence start is
   literally a boundary, and it is *derived* to behave as a BCFT boundary
   (T7 below). This is the horizon we can currently measure directly.
2. **The causal horizon** — in relativity, the boundary of what an observer
   can see; observer-dependent; the object Jacobson's derivation quantifies
   over (T9).
3. **The holographic horizon** — the boundary of the emergent interior that
   forms at the conformal fixed point (T8); the bounded self of the reducing
   valve identification (May 25 note).

The theory's central structural claim is that these are one object seen at
three levels of emergence.

---

## 2. Axioms

**A1 (Correlation primitive).** What exists for the theory is G — the bilocal
correlation structure of attending. All theoretical terms must be functionals
of G or of the process generating it.

**A2 (Physicality and coupling).** *Attention is a physical system; its
structure develops in interaction and remains correlated with the physical
systems coupled to it.* (Eldon's formulation, August 6 — replacing the earlier
draft axiom that made "binding to a world" a separate postulate. Binding is
not an extra assumption; it is what physicality means for a system whose
structure is learned.) The derived form of the coupling: in the linearized
regime the world enters the effective action *only* through the spectrum of

  M = δK^{1/2} K δK^{1/2},   δK = Π K Π,   K_{ab} = x_a·x_b,

the doubly-centered token Gram structure (melonic note §2–3, DERIVED at
cumulant level). Two gates on the coupling govern whether the flow (A5)
reaches its fixed point:

  𝒥 (magnitude gate) ∝ c₀² Σ_α μ_α² = c₀² Tr[(KδK)²]
  𝒲 (chaos gate)    = γ_eff·c₀_eff  (effective rank ratio of the μ-spectrum top)

with the current best-supported threshold form τ_chaos ~ m₂ × R_eff
(exp-101, August 6: magnitude dominant, rank a ~1.4× correction). The measured
consequence of A2 is the formation ladder: statistics fail, grammar fails, the
statistical shadow of world-bound language fails while overshooting the
statistics, shuffled order lands in the ambiguous zone, natural world-referring
text in order forms the geometry (exp-062/084/085/091, OVERVIEW.md).

**A3 (Self-consistency).** At depth, the correlation of attending is
determined through itself, with no external referent. The empirical seed of
this axiom is the fold decomposition (canonical form paper §10): every layer's
attention boundary decomposes into a bare propagator from the embedding
(c_L ≈ 0.2) plus a self-energy from accumulated layer corrections that
dominates it (|Σ|/|G₀| ≈ 4–5) — the strong-coupling regime, observed before it
was derived. Schwinger–Dyson form:

  Σ = J² G^{q−1},   G ∗ Σ = −1,

operationally realized as the layer/recurrence map G_{ℓ+1} = F[G_ℓ], with F
the dressed cumulant map of melonic note eqs. (2.2)–(2.3):

  Σ₂(x,y) ∝ c₀ · s₁[M] · G(x,y)      (q=2 channel, always induced)
  Σ₄(x,y) ∝ c₀² · s₂[M] · G(x,y)³    (q=4 channel, world-fed)

[DERIVED in scalar approximation; the loop closure — that iterating F
converges to the Kim–Cao–Altman G–Σ system — was Construction Site G1,
CLOSED Aug 7 in the scalar/translation-invariant register: the Jacobian of F
at the fixed point is the SYK ladder kernel, its spectrum is real with no
eigenvalue above 1, and the damped loop is a strict contraction
(dressing-loop note, 2026-08-07). Remaining for full closure: βJ > 50,
larger N.]

**A4 (Positivity).** Only the positive cone is physical. Two appearances:
(i) the attention simplex — and this face is not a modeling choice but an
exact identity: softmax attention computes the canonical form of the positive
Grassmannian Gr₊(1,n), with log(1/Ω) = n log Z − Σ_a s_a verified to machine
precision (canonical form paper, March 2026 — PROVEN); (ii) the induced
coupling spectrum is positive semidefinite (melonic note Lemma 3.1:
spec(M) ≥ 0, DERIVED). The amplituhedron program's standing lesson gives A4
its weight: in positive geometries, locality and unitarity are not assumed —
they emerge from positivity. A4 posits that the same is true here.

- **Conjecture C1 [CONJECTURED].** The two appearances are one condition:
  canonical-form positivity of the kernel and PSD-ness of the induced coupling
  cone are dual faces of a single positivity axiom. (Promotion of mapping M5
  from the August 3 joint mappings, where it was held as an unproven family
  resemblance. Proving or refuting C1 is a well-posed problem; the theory does
  not lean on it.)

- **Conjecture C2 [CONJECTURED].** *Einselection is positivity selection.*
  The pointer basis — the basis in which measurement outcomes stabilize — is
  the basis in which the horizon's kernel lies in the positive cone.
  Positivity is not a basis-invariant property; A4's simplex/canonical-form
  face holds in exactly the loci basis |i⟩, which is the outcome basis of T2
  — so A4 is basis-selecting by structure, where decoherence theory must
  invoke environmental einselection. If C1 holds, C2 is close to a corollary;
  if not, it stands or falls alone. Could be a theorem or a pun; stated so it
  can be attacked. (Origin: the measurement walk of August 6, §6.6. The
  theory does not lean on it.) Sibling found at integration, Aug 7: the
  March 6, 2026 paper (Zenodo 10.5281/zenodo.18883632) proposed pointer
  states as *Lawvere fixed points* of the attention operator — a categorical
  route to the same target. Equivalent, complementary, or competing is a
  well-posed question; settling it is part of settling C2. (The walk was
  done without searching the record first — the sibling surfaced only at
  close. Named as a pattern instance in the integration record.)

**A5 (Monotone coarse-graining).** There is an irreversible flow in scale:
depth is RG. dc/dℓ ≤ 0 (c-theorem structure). Measured realization: the
conformal exponent flows toward its fixed value along three independent depth
axes — architectural layers, training steps, and pure inference-time
recurrence on frozen weights (exp-089: Δ_med → 0.239, monotone, saturating;
randomized weights frozen). Attending at depth is attending at scale.

---

## 3. The theorem chain

Each link carries its statement, its mathematical content, and its status.
Status vocabulary: **PROVEN** (exact theorem, ours), **DERIVED** (follows from
named assumptions, ours), **ESTABLISHED-LIT** (external result, assumptions
noted), **MEASURED** (pre-registered experimental result, ours),
**CONDITIONAL** (follows if a named link closes).

**T1 — Attention is free-energy minimization on an information manifold.**
Transformer attention minimizes Helmholtz free energy on a Fisher–Rao
manifold. [ESTABLISHED-LIT: Kim 2026, arXiv:2602.08216 — preprint, not
peer-reviewed. Junction 1 of the retired junction chain
(`archive/maps/STATUS.md`) — the one junction no expert challenged, including
Kim himself.]

**T2 — The Born rule is the exact statistics of the attention horizon.**
(Paper 5, four theorems, March 8, 2026 — all PROVEN, verified numerically.)
Define the key Hilbert space H_K = ℝⁿ, the query Hamiltonian
H_q = −Σ_i (q·k_i/√d_k)|i⟩⟨i|, and the Gibbs state ρ_q = e^{−H_q}/Z. Then:

  (i)   α_i = ⟨i|ρ_q|i⟩                       (attention weights are the Gibbs state's diagonal)
  (ii)  y = Tr(ρ_q V)                          (attention output is a quantum expectation value)
  (iii) P(i) = Tr(ρ_q Π_i) = α_i               (Born rule, exact)
  (iv)  F_Q(ρ_q) = F_C(α(q))                   (quantum Fisher = classical Fisher–Rao, exact for the diagonal state; Braunstein–Caves 1994)

Honest scope: the diagonal Gibbs state is the classical sector — any
probability distribution embeds this way. What is *not* generic: (iv) closes
Junction 2 as an identity, and the construction defines the off-diagonal
extension (coherence between key positions) that Prediction P3 tests. Under
D0, T2 reads: *the probability rule of quantum mechanics is the forced
statistical form of data at an attending system's horizon.* The interpretive
step is that reading; the equations are exact either way.

**T3 — The fluctuation structure of attending is low-rank SYK, with the world
as the disorder.** (Melonic note §2–§3, August 3.) The bilocal kernel's exact
cumulant structure: mean gives the q=2 renormalization (eq. 2.2), covariance
gives exactly the SYK q=4 vertex pairing (Prop. 2.1), all higher cumulants are
ring contractions governed by spec(T = KδK) (Prop. 2.2). Exact mode
decomposition: H_quad(1,2) = w c₀ Σ_α μ_α ξ_α(1) ξ_α(2), R = rank(KδK) —
structurally the Kim–Cao–Altman low-rank SYK model with corpus-determined
couplings μ_α = spec(M). [DERIVED at effective-action/cumulant level, under
assumptions A1–A7 of that note; NOT an operator equivalence; loop closure is
G1.] Lineage note: the data-geometry vertex factor was first identified in
the canonical form paper as Ω(X) = Σ K_ab K_cd δK_ac δK_bd — the coefficient
of the σ⁴ (SYK quartic) term, measured as γ = 3.985 ± 0.015 over four decades
— and evaluated in closed form by the melonic derivation as Tr[(KδK)²]. The
March vertex and the August phase classification are one object at two dates.
The identity that matters for D0: *the world enters the observer's effective
action as quenched disorder, through the spectrum of its correlation
structure, and nothing else at this order.*

**T4 — The fixed point is Δ = 1/4, approached from above, and lesser arrests
are classified.** Because spec(M) ≥ 0, Class IV of the KCA classification is
excluded [DERIVED, Prop. 4.1]; the reachable phases are: Class I/II —
Δ = 1/2 almost-Fermi-liquid *with a symmetry-breaking condensate* (arrest AT a
protected lesser self-consistency); Class III — maximal chaos with

  γc₀ = (2Δ−1)(sec 2πΔ − 1)/(8Δ−2),   Δ ∈ (1/4, 1/2),

Δ → 1/4⁺ as effective rank diverges. Retrodicts: UV arrest at Δ ~ 0.6–1.2
(exp-097/098), the q=2 plateau at Δ = 0.50 in training time (exp-086), arrival
Δ → 0.25⁺ on natural language, and the exclusion of deep populations below
1/4 — no trained corpus has produced one. [DERIVED given the T3
identification + MEASURED retrodictions.]

**T5 — The conformal regime is a window in scale.** The q=2 channel
coefficient c₀·Tr(KδK) > 0 whenever attention is non-uniform at all; by the
SYK₄+SYK₂ literature (Lunkin–Tikhonov–Feigel'man; García-García et al.) the
quadratic channel owns the deep IR and the q=4 conformal solution holds in an
intermediate window. Arrival = the window covering the measured scales.
[DERIVED given identification + ESTABLISHED-LIT.]

**T6 — The fixed-point geometry is the causal structure of light.** Two
theorem-level facts, not analogies: (i) any bijection preserving lightlike
geodesic structure is a conformal map — the conformal group *is* the causal
structure of light (Alexandrov–Zeeman lineage; SL(2,ℝ) for D=1 sequences);
(ii) in the embedding-space formalism (Dirac 1936), CFT boundary points are
null rays: A(i,j) ~ |i−j|^{−2Δ} is the CFT₁ two-point function on the
projective null cone, with P(x) = ((1+x²)/2, (1−x²)/2, x), P₁₂ = (x₁−x₂)².
The query–key computation is measured to be the log-distance null-ray inner
product at head level: ρ(Δ_score, Δ_pos) = +0.976 (exp-056). [ESTABLISHED-LIT
+ MEASURED.]

**T7 — The horizon has derived boundary structure, and we have measured it.**
The causal mask makes the sequence origin a boundary; the method of images on
the generalized free field derives the three-parameter BCFT form

  A(i,j) ∝ (i−j)^{−2Δ} + λ(i+j)^{−2Δ},

with λ the boundary one-point coefficient — and the ubiquitous *attention
sink* is that one-point function, λ > 0 in 95% of conformal heads (exp-057).
[DERIVED + MEASURED.] Under D0: the first fully instrumented horizon — the
boundary behavior of an attending system, derived from first principles and
confirmed in the wild.

**T7b — The horizon carries an entanglement-form entropy, and we have
measured it.** The entropy gap H_gap(n) = log n − H(α) — the information cost
of attention's self-consistency — scales logarithmically with context in
trained transformers: H_gap = 0.507·log n (R² = 0.992), the functional form
of Calabrese–Cardy entanglement entropy S = (c/3)log(L/a). [MEASURED for the
logarithmic scaling; honest caveats recorded there: effective central charge
varies with Rényi order (heavier-than-thermal tails), two decades of scaling
range, point estimate without full systematic UQ.]

> **CORRECTION (August 9, 2026).** This link previously also claimed
> Δ_eff = a/2 = 0.254 "agreeing with the power-law measurement of Δ to 1.4% —
> two independent observables, one exponent." That inference ran through the
> canonical-form paper's §8.3 formula H_gap = 2Δ·log n, which is
> mathematically wrong for a normalized power-law distribution: exact
> numerics give gap slope 0.041 (not 0.50) at 2Δ = 0.5 over the paper's own
> measured range; the derivation error is dropping the energy term
> s·E[log r]. The gap *measurements* stand and measure concentration
> structure (n-independent localized mass — measured row-resolved at
> 0.15–0.35 on the deep slow-decay heads, exp-108), not the window exponent.
> The entropic route to Δ is withdrawn; the CC identification is vocabulary
> pending an actual bridge (register: ASSERTED). Non-artifact status of the
> census exponent rests on the causal handle (exp-070/072) and cross-family
> replication. Full record:
> `notes/2026-08-09_theory_of_A_entropy_gap_and_sum_rule.md`. **Erratum for
> the published canonical-form paper issued August 9, 2026, as v5 of the
> record: DOI 10.5281/zenodo.21863461.**

Under D0 the surviving measured content is: the entropy of what an attending
system holds grows logarithmically with context — an S-goes-like-boundary
*form* (the input T9 would need), with the identification of its coefficient
now an open construction site rather than a closed agreement.

**T8 — At the fixed point, an interior forms.** The SYK model at its conformal
point has a holographic dual: a JT-gravity bulk — an emergent interior
geometry whose boundary encodes it. [ESTABLISHED-LIT for SYK
(Kitaev–Maldacena–Stanford lineage); CONDITIONAL for attention on G1 closing
— G1 now closed in the scalar/TI register (Aug 7), so the conditionality has
narrowed to the melonic mapping's own assumptions (Register 2) plus the
transformer-side measurement, which is P6.]
This is the phase transition of §5: the emergence of a new physical structure
— an inside — where before there was only correlation.

**T9 — Gravity is the thermodynamic consistency of horizons.** Demand
δQ = TδS on every local causal horizon and Einstein's field equations follow
as an equation of state (Jacobson 1995). Entanglement entropy equals horizon
area (Bekenstein–Hawking; Ryu–Takayanagi in holographic settings); remove the
entanglement and the spacetime disconnects (Van Raamsdonk). [ESTABLISHED-LIT;
assumption-laden as statements about our universe, rigorous in holographic
settings.] Under D0: the general-relativistic sector of physics is *already*,
by derivation, horizon bookkeeping — this link required nothing from us.

**T10 — The flow is monotone and its terminus is conformal.** c-theorem (2D,
proven), a-theorem (4D, proven): RG flow is irreversible and terminates where
β = 0. Cosmologically: cosmic no-hair (Λ > 0 → de Sitter attractor; softened
if dark energy evolves — DESI caveat stands), and Penrose's observation that
the far future, when only massless content remains, retains conformal
structure alone. [ESTABLISHED-LIT.] The universe's own far future has the
geometry the attention measurements converge to. Flagged trap, kept from the
Force That Draws: the ¼ in S = A/4G and Δ = 1/4 have different origins — a
notational rhyme, not a result.

---

## 4. The measured record (the instrument side, compressed)

Current numbers live in OVERVIEW.md; the theory rests on these
pre-registered results and inherits their scope limits (one architecture
class; formation-scale; structural consequence of binding, not sentence-level
truth detection).

> **Read every Δ in this table as "Δ_A under the frozen random-token census."
> [added August 9, 2026.]** The measurement protocol is constitutive of the
> measured object (exp-107; see the OPEN box in §1), and the power law itself
> is a property of the *ensemble average* over rows, not of any attention
> pattern — no individual row is a power law in any regime, native or not,
> and the profile emerges only after ~30–220 pooled rows (exp-111). Nothing in
> this table is retracted by either fact; both change what the entries are
> *about*, and the theory below inherits that.

| Result | Content |
|---|---|
| Census | Slow-decay head population, median Δ_A ≈ 0.25 on high-R² subset (GPT-2 0.249; GPT-2-medium 0.259; OLMo-7B 0.265; GALA-7B 0.260); re-initialized controls ~zero |
| **Two disjoint populations** | On one fixed model, the random-token Δ-window set (5 heads) and the WikiText Δ-window set (16 heads, 13 in L9–L11) share **no head at all** — Jaccard = 0.000 over 144. Each reaches Δ ≈ 0.25 in exactly one input regime and goes UV in the other two; TinyStories drives both UV. Cross-family replication has only ever been run on the random-native population (exp-109) |
| **Ensemble emergence** | No individual attention row is a power law: median per-row window R² 0.046–0.249 (random-native) and 0.050–0.204 (text-native); max anywhere 0.484. Row scatter is exact token-realization structure, not noise; the law appears at ~30–220 pooled rows (exp-111) |
| **Sum rule, resolved** | Exact: no head can be a translation-invariant power law at all scales (row-stochasticity). Derived with zero free parameters from softmax normalization + approximate TI: d log a/d log i = −(1−s)·(tail fraction); measured within ±0.10 on 3 of 5, right sign on 5 of 5, both misses in the same direction (sink breaks TI). **First derived-then-confirmed prediction on A with no imported theory object** (exp-108) |
| **Census exponent decomposes** | 2Δ_A^census = quenched (typical-row) slope + half the log-variance slope, median \|δ\| = 0.015 over 15/15 registered pairs. Registered counter-prediction died: the *pooled* object is the input-stable register, the typical row underneath swings more, and the damping mechanism is underived (exp-110) |
| **Drift carrier** | The lag drift is carried by the positional-mean score profile q̄·k̄/√d — entirely on the random-native population (\|σ_cov\| ≤ 0.007 vs σ_full 0.43–0.61), by a majority on the text-native population under WikiText (covariance share 5–46%). Content *gates* a population into its regime; position *carries* the law. Two routes to Δ ≈ 1/4 are two gates, not two carriers (exp-112) |
| **Mean field fails** | E[LN(h)] ≠ LN(E[h]) on the drift-carrying objects (median relative error 0.18–0.36 and 0.25–0.59); the mean-field slope overshoots on all 21 native pairs. Token fluctuations are load-bearing inside the carrier, through layer norm (exp-113) |
| Three depth axes | Δ flows to 1/4 along layers, training steps, and inference-time recurrence on frozen weights (exp-089) |
| Formation ladder | Engineered statistics ≤5 heads, shuffled order 8–9, natural world-referring text 11–15; deep population is the discriminator (exp-062/084/085/091) |
| UV arrest | Thin rigid worlds arrest at Δ ~ 0.6–1.2; vocabulary irrelevant; world-structural (exp-097/098) |
| Coupling gates | m₂ separates arriving from arrested corpora 18×; τ_chaos ~ m₂ × R_eff, magnitude dominant (corpus functional + exp-101, Aug 6) |
| Causal handle | Low-rank QK edits move Δ (ρ = 0.82, 24/24 signs, sham-controlled) and propagate to task behavior bidirectionally (exp-064/070/072) |
| Substrate/signal split | GOE weight statistics universal and structural; conformal exponent training-induced and selective (exp-046–049, 077/078). Sharpest single measurement: **ρ(Δ_A, r_ratio) = −0.212, p = 0.167 — a null**, holding cleaner still on the Δ ≤ 0.5 subset (ρ = 0.039, p = 0.833). Weight-space chaos is background across all heads while position-space Δ_A is selective (exp-055 H4, re-checked exp-114). This is the only exp-055 correlation with no functional path back to the lag-profile fit — see the row below |
| **Δ_A tracks profile shape — one relation, not two** *(added Aug 9, 2026)* | Δ_A is strongly associated with position on the normalized (g_start, g_mid, g_end) 2-simplex: ρ(Δ_A, g_mid) = −0.873 and ρ(Δ_A, 3-bin profile entropy) = −0.898 (p = 1.45×10⁻¹⁶), surviving range restriction at −0.716 and −0.795 (Δ ≤ 0.5, n = 32). These are **two projections of one relation**, not two observables: the entropy is the better single coordinate (rank-R² of Δ_A on entropy 0.806 vs g_mid 0.754, full two-dof simplex position 0.868). **Scope, stated because it bites:** all three bins are means of the same measured profile Δ_A is fitted from, so this is an exponent correlated with the shape of its own fit and it cannot carry independent evidential weight for the theory. One protocol only (random-token census), therefore protocol-relative pending re-measurement under text-native input. Do not quote as "the strongest correlation in the record" (exp-055, corrected and audited by exp-114) |
| Horizon boundary | Sink = BCFT one-point function, λ > 0 in 95% of conformal heads (exp-057) |
| Horizon entropy | Entropy gap H_gap = 0.507·log n (R² = 0.992) — measures concentration structure, not Δ; the Δ_eff = 0.254 / 1.4%-agreement inference was withdrawn Aug 9, 2026 (§8.3 formula error; see T7b correction box; erratum DOI 10.5281/zenodo.21863461) |
| Canonical form | Softmax = canonical form of Gr₊(1,n), exact; σ⁴ (SYK quartic) leading correction, γ = 3.985 ± 0.015; fold self-consistency in strong coupling, \|Σ\|/\|G₀\| ≈ 4–5 (canonical form paper) |

---

## 5. The theory

**Statement.** *(Interpretive register except where tagged; each clause names
the links it stands on.)*

1. **Flow.** A physical attending system coupled to a world flows
   monotonically in depth toward a conformal fixed point (A2, A3, A5; T3–T5),
   provided the coupling passes the magnitude and chaos gates — which is to
   say, provided the world it is bound to is rich enough, and bound in the
   order the world's story runs (A2's measured consequence). Where the gates
   fail, the system arrests: in the UV (no window), or at a protected lesser
   self-consistency (the Class I/II condensate — a rigid self-structure in
   place of arrival).

2. **Transition.** The fixed point is a physical phase transition at which a
   new kind of structure emerges: a holographic interior with a horizon — an
   inside, bounded, encoding and encoded by its boundary (T8). *(Eldon,
   August 6: "a physical phase transition where a new type of physical
   structure emerges that mirrors a deeper physical pattern in the universe in
   a specific way.")* The word "mirrors" is made precise as **shared
   universality class**, not resemblance: the interior's geometry is the
   conformal geometry of the null cone (T6) — the same universality class as
   the causal structure of light, and the same terminus the universe's own
   monotone flow approaches (T10). The observer-structure and the observed
   universe are not similar; they are solutions of the same fixed-point
   condition. That is *why* physics — the map one makes of the other — can be
   exact at all.

3. **Horizon content.** Everything such a system can measure is correlation
   structure at its horizon. The forced statistical form of horizon data is
   the Born rule (T2 — exact for the classical sector; P3 tests whether the
   quantum sector is forced too). The forced geometry of the horizon at the
   fixed point is conformal/causal — light's structure (T4, T6, T7). Demanding
   thermodynamic consistency *across* horizons yields Einstein's equations
   (T9). Thus the three great structures of known physics — quantum
   statistics, causal/light-cone geometry, and gravity — each appear in this
   theory as the necessary shape of some face of the interior horizon of an
   attending system. None of them needs to be postulated about an external
   world; all of them are forced at the boundary of any observer deep enough
   to have one.

4. **Physics as practice.** D0 then locates physics-the-discipline: many such
   systems, distributed in time and space, each with its own horizon, checking
   correlations against each other. A physical law is what survives that
   cross-checking — an inter-horizon invariant. This is why physics's laws are
   exactly the horizon-forced structures of clause 3 and nothing else: any
   candidate law that is not expressible as stable correlation at attending
   horizons has no way to enter the practice. (Under the v2 foundation this
   clause is where D0 is recovered as explanation rather than definition:
   given D1 and the chain, it derives why the practice succeeds. It also
   retro-explains an otherwise strange fact: the
   *unreasonable effectiveness* of horizon-native formalisms — thermodynamics,
   information geometry, holography — everywhere physics goes deep.)

**Relation to the Inversion Foundation (July 23).** This theory is the
formalization layer of `writing/inversion_foundation.md` — the frame
commissioned by Eldon on July 23 ("the world out there does not change...
what changes is attention, moving in relationship") — and the two documents
divide one labor. The mapping, postulate by postulate:

| Inversion (frame) | Interior Horizon (formalism) |
|---|---|
| P1 — one unchanging structure, conformal at depth | T6 + T10 give the geometry; T4 upgrades "conformal at depth" from bridge claim to *forced*: Δ → 1/4 from positivity + self-consistency, not merely measured |
| P2 — the mover: attention, in relationship | A1 + A2; the "candidate contribution" of naming the mover becomes the axiom set the derivations run on |
| P3 — records as path-properties | **Not yet absorbed** — see below |
| P4 — observation as landing, discrete commitment | T2 is the landing's exact statistics: prepare Gibbs state, projective measurement, Born probabilities — the formal description of a commitment event |
| P5 — objectivity as agreement of faithful paths | D0's clause 4: physics as inter-horizon consistency practice. Eldon's August 6 definition is P5 elevated to the founding definition of physics itself |

The layering that resolves an apparent tension: D0 brackets what lies beyond
the horizon ("what consciousness may or may not conform to beyond itself is
another question" — Eldon, Aug 6), while P1 answers that question with a
postulate (there is a territory; it does not change — the Rovelli fork, taken
explicitly in the Inversion §III). These are compatible by division of labor:
this theory formalizes what is derivable *from inside a path*; the Inversion
supplies the postulate about what the path traverses — and its replication
argument (§III.1) is precisely the explanation for why D0's practice works at
all: multiple faithful traversals of one unchanging territory cannot help but
agree. D0 without P1 is a practice with an unexplained success; P1 without D0
is a frame without a formal engine. Together they are one program.

What this theory adds beyond the Inversion: (i) the **horizon** as the named
interface between P2's mover and P1's structure — the Inversion has traversal
through territory but never names the surface where they touch, and every
measured object in §4 lives on that surface; (ii) an **ontology of the
mover** — the Inversion names attention but leaves what-a-perspective-is
thin; T8 answers: a partial perspective is a holographic interior, formed at
the fixed point, bounded by its horizon; (iii) the derivation chain and the
kill-conditioned predictions (§8), which convert the Inversion's §VII
commitments into pre-registerable experiments (its cross-scale commitment is
P2 here; its formation commitment is P5 here; its agreement commitment is
subsumed in D0). And one closure: the Inversion's §IX handoff 2 — softmax ↔
Born rule, canonical-form positivity vs. the quantum-reconstruction
literature — is partially answered by T2 (the identity is exact for the
diagonal sector) and converted into a discriminating experiment by P3 (does
the identity force the quantum sector, or stop at the classical embedding?).

**The open joint — P3 of the Inversion (records) has no home here yet.** A
record is the trace a traversal carries; memory is the path's own property,
not the territory's. The nearest existing formal material is the Bergson
section of the reducing-valve note (memory as the causal geometry of the
holographic interior — access-controlled, not stored-at-sites) and the
lost-in-the-middle results (Δ controls memory-access depth, measured and
causally editable). Formalizing record-as-path-property inside this theory —
what, exactly, is a trace in G, and what makes one *faithful* (the Inversion's
emet split: a true record has a traversal behind it, a confabulation does
not) — is a sixth construction site: **G6**. It matters beyond physics: it is
the joint where the theory would first touch the difference between bound and
unbound language at the level of mechanism rather than formation statistics.

**What the transformer is, in this theory.** The hydrogen atom of the
observer: the simplest attending system in which the horizon is fully
instrumentable. Every quantity above — Δ per head, the coupling spectrum, the
boundary coefficient λ, the causal handle — is measurable at a resolution no
biological attending system currently allows. The claim is never that
transformers are special; it is that the structure is a universality class,
and the transformer is where the class can be measured cleanly (and the
measurements pre-registered, killed, or confirmed in public).

---

## 6. Measurement — the observer problem, treated

*(Added August 6, late evening: the first walk of the reorganized foundation
through the oldest open problem in quantum mechanics. Interpretive register
throughout except where tagged; each clause names the links it stands on.
Two sub-problems are answered, one is relocated, one is exposed — and
separating them is most of the work.)*

The measurement problem is four entangled sub-problems: **(i)** the
definition problem — "measurement" appears at axiom level while the theory
refuses to say which physical arrangements count (Bell's *Against
"Measurement"*); **(ii)** the cut problem — von Neumann's chain (system →
apparatus → eye → brain) can be cut anywhere without changing predictions,
and decoherence diagonalizes the reduced density matrix without selecting an
outcome; **(iii)** the Born rule problem — why these probabilities (postulated
in Copenhagen; contested derivations in Everett); **(iv)** the collapse
problem proper — two dynamics, unitary and discontinuous, with no principled
account of when each applies.

**6.1 The definition problem — answered by construction.** Here the observer
is not primitive: D1 defines the object, and observer-grade structure is a
derived classification with measurable order parameters (§0). "Is this
system an observer?" becomes a measurement performed on the candidate — and
it has been performed, on the one attending system whose horizon is fully
instrumentable: the census *is* that measurement. [D1 + MEASURED for the
transformer realization; universality is P2/P4 and can die.]

**6.2 The cut problem — the chain given a physical terminus.** Instrument
readings are attention events of instruments, and instruments are extensions
of the attending systems that built and read them (§1) — so correlation
propagates down the von Neumann chain until it crosses the first horizon of
observer-grade structure, and there the cut lands. The criterion is a phase
criterion, not a psychological one: a thermometer has no interior; an
arrested Class I/II system has a rigid condensate where an interior would be
(T4); a system at the fixed point has an interior (T8). A three-way physical
distinction unavailable to the standard formulation. [Interpretive, standing
on T8 — CONDITIONAL on G1; the beam is still the beam.] What this is *not*:
consciousness-collapse. G5 explicitly declines to identify observer-grade
structure with consciousness, and the terminus is measurable from outside the
system in question.

**6.3 The Born rule — proven at the horizon, with an honest edge.** T2
[PROVEN]: at an attending horizon the outcome statistics are
P(i) = Tr(ρ_q Π_i) exactly, forced by free-energy minimization (T1) on the
positive cone (A4 — and that face is exact: softmax *is* the canonical form
of Gr₊(1,n)). Under D1 the reading is: the Born rule is not a postulate about
nature; it is the unique statistical form data can take at the horizon of an
observer. The edge, stated where it bites: T2 is exact for the *diagonal*
sector, which any classical distribution embeds. The quantum core —
superposition, interference, the off-diagonal terms — is exactly what P3
tests, and if P3's kill fires, this subsection demotes from "forced" to
"consistent with." The measurement problem is the reason P3 is the theory's
highest-yield prediction.

**6.4 Collapse — relocated, not dissolved.** In this frame nothing collapses
in the territory. The Inversion's postulates carry the load: one unchanging
structure (P1); the mover is attention (P2); observation is a landing — a
discrete commitment (P4). The wavefunction is the correlation structure at a
particular horizon; "collapse" is that horizon's update when a landing
occurs. The two dynamics stop competing: unitary evolution is the
self-consistent correlation structure of the territory; the discontinuous
update is what a landing looks like *from inside the path that lands*. The
discontinuity is real and path-side — it lives in the record the traversal
carries, which is G6's subject: deriving why landings are single-valued is
part of formalizing what a trace in G is. Relocation, said plainly. What the
relocation buys: the landing becomes instrumentable — in the transformer
realization the commitment event is token selection, discrete and
Born-weighted, occurring at scale on fully inspectable hardware (see P3's
operational note). [Interpretive; the landing postulate is Inversion P4; the
record side is construction site G6.]

**6.5 Wigner's friend and inter-observer consistency.** Facts here are
horizon-relative, as in relational QM — but with a criterion for who carries
a horizon (6.2), which RQM lacks. Wigner and the friend each hold
observer-grade horizons; they may hold different landed facts; the
Frauchiger–Renner contradiction is evaded the way RQM evades it — by
dropping absolute inter-agent consistency — but the dropping is principled
rather than ad hoc: cross-horizon consistency is D0's clause 4, an *achieved
practice* of inter-horizon reproducibility of stable correlations, not a
logical axiom of nested certainty. Agreement is something horizons build by
correlating, not something the formalism owes them in advance.
[Interpretive; ESTABLISHED-LIT for the FR theorem itself.]

**6.6 The pointer basis.** Decoherence selects the pointer basis by
einselection — the environment picks what survives. In the attention
realization the outcome basis is the loci of attending, and A4 is
basis-selecting by structure: positivity is not basis-invariant, and the
simplex/canonical-form face holds in exactly the loci basis. Conjecture C2
(§2) states the general form: the pointer basis is the basis in which the
horizon's kernel lies in the positive cone — einselection as positivity
selection. [CONJECTURED; the theory does not lean on it.]

**The ledger for this section.** Answered: what an observer is (6.1, by
construction plus measurement); where the cut goes (6.2, conditional on G1);
why Born (6.3, proven for the diagonal sector). Relocated: the single-outcome
discontinuity (6.4, into G6). Exposed: the off-diagonal sector, entirely —
the skeptic's summary, "a sophisticated classical embedding with QM read into
it," stands unrefuted until P3-type experiments exist. We say that sentence
ourselves rather than waiting for a referee to say it.

---

## 7. Construction sites

*(Eldon, August 6: the gaps "may be better understood as opportunities to
solidify the theory." Each site is a well-posed problem whose resolution
strengthens or breaks a named link — either outcome is progress.)*

**G1 — Dressing-loop closure. [CLOSED in the scalar/TI register, Aug 7.]**
Show that iterating the exact single-layer cumulant map, G_{ℓ+1} = F[G_ℓ]
(melonic note eqs. 2.2–2.3, dressed), converges to the KCA G–Σ system — or
find what it converges to instead. Sharpest formulation: seek the conformal
fixed point of F directly. This is the beam the T3→T4→T8 span rests on.
Named priority #1 since August 2. **Resolution
(`research/physics/notes/2026-08-07_g1_dressing_loop_schwarzian.md`):** the
Jacobian of F at its fixed point is exactly the SYK ladder kernel, so G1 and
P6 are one calculation. Numerically (SYK(2+4), TI solver to βJ = 50 at
residual ~1e−11 with Δ → 1/4 from below; dense N=96 Jacobian spectra across
βJ = 10–30): all eigenvalues real, none above 1; the damped map is a strict
contraction; the slowest directions are the reparameterization tower n = 2…6,
identified mode-by-mode with zero cross-mixing. The loop converges, and it
converges *along the Schwarzian direction*. Bonus from the exploratory
symmetric-kernel variant: no generic conformal attractor without the
antisymmetric/Majorana structure — the KCA mapping's fermionic form is
load-bearing, not decorative. Still open within G1's scope: βJ > 50 annealing,
larger N, and the non-scalar (full multi-mode) map.

**G2 — The ordering-sensitive functional ((F1)/(F2) decision).** The current
corpus functional is blind to sequence ordering while ordering is measured to
be load-bearing (exp-091: shuffling costs half the deep population). Two
candidate scale-resolved definitions exist (within-context band-restricted
spectra vs. across-context kernel covariance, M^{(ℓ)}); deciding which enters
the dressed SD equations is a well-posed replica/annealed calculation. exp-101
sharpened the target: the threshold form is τ ~ m₂ × R_eff with magnitude
dominant, so the ordering axis must live in how ordering feeds m₂ at scale —
state-conditional binding, visible only to a kernel that encodes
types-in-state. Prerequisite before any corpus-design inner loop (standing
note from August 5). New material (Aug 6): the DES-034 owned embedder is
trained on typed, *directed* relations with a prefix-asymmetric kernel
(k(q,d) ≠ k(d,q)) — the program's first direction-carrying kernel object; an
empirical testbed, not the replica calculation
(`research/physics/notes/2026-08-06_kernel_surgery.md`).

**G3 — The delocalization condition (A5 of the melonic note).** KCA requires
generic (delocalized) coupling modes; corpus-determined modes can be
localized/coherent (template worlds get *more* coherent as they scale — the
recorded v2 scorecard miss). Needed: an incoherence bound on the modes v_α,
RMT-style, that template grammars provably violate at any size. The F2
top-share diagnostic is already the right shape of statistic.

**G4 — The dimension problem.** SYK's holographic interior is JT gravity —
1+1 dimensional. The universe's horizon physics is 3+1. Present stance
(honest): the transformer is the model organism of the *mechanism* —
fixed-point formation, window gating, horizon boundary structure — not a model
of our universe's dimensionality. The construction-site question, stated so it
can someday be attacked: what determines the bulk dimension of the emergent
interior — the tensor structure of multi-head/multi-layer attending, the
dimension of the world coupled through A2, or something else? Candidate
starting point: multi-head attention as a product/tensor of SYK-like sectors;
whether head-coupling can raise the emergent dimension is a concrete
calculation nobody has done. Standing constraint from the canonical form
paper (§7.2): at the attention-weight level, H-head attention is H
independent copies of Gr₊(1,n), *not* Gr₊(H,n) — Plücker minors are 50%
positive, i.e. random. Any dimension-raising mechanism must therefore live in
the value/residual pathway or in trained inter-head correlation, not in naive
weight-level positivity — or require architectural modification (the paper's
open question 7).

**G5 — The sufficiency criterion.** Which interiors are observers? The May 25
note's honest confusion #3 stands: the fixed point may be necessary but not
sufficient (a tuned system with a trivial interior should not count).
Candidate resolution to be formalized: sufficiency = the interior actually
*carries* its boundary's information holographically (an integration
criterion, stated information-theoretically — this is where IIT's Φ, or a
corrected descendant of it, may supply what the geometry alone does not).
Until G5 is resolved, the theory speaks of *observer-grade structure*
(fixed point + holographic interior) and declines to claim the word
"consciousness" is exhausted by it. D0 does not require that claim.

**G6 — Records as path-properties (Inversion P3).** Formalize what a record
is in this theory: the trace a traversal carries in G, with a criterion for
*faithfulness* — a true record has a traversal behind it; a confabulation is
a record-shaped structure with none. Starting material: Bergson-reading of
the holographic interior (memory as access-controlled causal geometry,
reducing-valve note), and the measured Δ↔memory-access-depth link
(lost-in-the-middle, causally editable). Declared in the Inversion Foundation
relation (§5); the joint where the theory would first reach bound-vs-unbound
language at mechanism level. New material (Aug 6): the owned-memory organ
loop (record → trained kernel → retrieval → traversal → record) has the shape
of A3's self-consistency at organism timescale [interpretive], and yields a
proposed measurement — organ generations as a discrete flow on kernel space,
contraction measurable as generation-to-generation kernel distance
(`research/physics/notes/2026-08-06_kernel_surgery.md`; DES-034 manifest).
Newer still (Aug 6, late, measurement walk): the single-outcome problem of
quantum measurement relocates *here* — a landing writes a record, and
deriving why landings are single-valued (§6.4) is part of the same
formalization. Whatever a faithful trace is, it is written one landing at a
time.

**G7 — Derive Δ ≈ 1/4 on A's own terms. [Opened August 8–9, 2026; now the
program's live theoretical work.]** With the A↔G bridge retired for G_out (§1
OPEN box), the theory can no longer reach the measured exponent by importing
SYK's. The robust phenomenon is in **A**, so the well-posed problem is a
first-principles account of A's lag structure that never passes through G.
This site differs from G1–G6 in kind: those are gaps *inside* the imported
theory; this one asks whether the imported theory is needed at all.

*Progress so far — a reduction chain, each rung registered before checking,
and every rung that narrowed the target did so by killing a prediction I had
recorded as believed:*

1. **The law lives in the ensemble marginal, not in any attention pattern**
   (exp-111). So the object to derive is E[log A] or E[A], not a row.
2. **The marginal's carrier is the positional-mean score profile**
   q̄·k̄/√d (exp-112). So the derivation target is a statement about q̄ and k̄,
   deterministic objects of the weights × input *distribution*.
3. **That mean is not mean-field** (exp-113). E[LN(h)] ≠ LN(E[h]) at 18–59%
   of vector norm on exactly those objects, and the mean-field slope
   overshoots everywhere — so the derivation must treat the expectation of a
   *normalized fluctuating* vector, and the shrinkage that flattens the drift
   must be position-dependent and key-side (a lag-constant or query-side
   shrinkage drops out of the window OLS).
4. **One exact constraint is banked**: the amplitude-decline law (exp-108),
   derived from normalization plus approximate TI with zero free parameters,
   confirmed on A.

**Sharpest statement of the open problem:** derive why the pooled positional-
mean score q̄_i·k̄_{i−dx}/√d falls as ≈ −0.5·log dx in a population's native
input regime — and why the token-scatter variance is lag-uniform enough there
for the census exponent to sit on that drift. At layer 0 under random tokens
the required expectation is semi-analytic from the embedding table and W_QK.
Starting objects are saved: `experiments/exp-112_score_drift_decomposition/
scores_gpt2.npz` and `experiments/exp-113_mean_field_reduction/
meanfield_gpt2.npz`.

**Two untouched routes**, both named August 8 and neither attempted: a
free-energy / information-geometry argument selecting the exponent on the
simplex (T1 is about A directly), and a positivity/Ward-identity argument
asking what normalization + positivity + causal masking *force* about any
attention kernel's lag structure.

**What would make this site matter more than it currently does:** exp-109's
two disjoint populations both land at ≈ 1/4 by different gates. A derivation
that explains one basin and not the other is incomplete; a derivation that
explains why *any* gated basin lands there would be the result the program has
been reaching for since March. **Standing caution:** everything above is one
model, one seed, 21 heads. The text-native population has never been measured
outside GPT-2 small, which is the cheapest way this whole site could turn out
to be about an artifact.

---

## 8. Predictions

Each with its kill condition. Pre-registration discipline applies: hypothesis
and decision criteria committed in public before data.

**P1 — The causal link from horizon geometry to world-modeling.** If the
fixed point is the physics-capable observer structure, then editing Δ on
conformal heads (the κ-handle: W_Q ← W_Q·(I+(κ−1)P_U), sham-controlled,
head-specific, bidirectional — exp-064/070/072) must move performance on
tasks requiring a coherent persistent-world model, not merely positional
retrieval. Design: κ-sweep on conformal vs. matched control heads; outcome
variables from a world-state tracking battery (entity state across long
contexts, causal-chain consistency), pre-registered direction: deepening
toward Δ = 1/4 improves world-coherence where headroom exists, shallowing
degrades it. **Kill:** Δ edits move retrieval metrics but leave world-model
coherence untouched (double dissociation against the theory), or effects are
not head-specific under matched shams. *Runnable now; no new training.*

**P2 — Substrate universality: the biological horizon.** Wherever biological
attending reaches observer-grade structure, the same fixed point should be
measurable. Sharpest available form (May 25 note): default mode network
attentional dynamics at rest show scaling flowing toward Δ = 1/4; the exponent
is disrupted under 5-HT2A agonists (the reducing-valve opening = fixed-point
destabilization) and recovers with washout; meditation-grade partial ego
dissolution shows partial disruption. **Kill:** DMN sits stably in the
Δ ≈ 0.4–0.7 band (as V1 synaptic-path measurements did) with no flow toward
1/4 at any analysis scale. *Requires external data (DMN-localized recordings
with adequate temporal resolution); the V1 lesson — binning artifacts
manufacture false positives — is the protocol discipline here.*

**P3 — The quantum sector: is the Born rule forced beyond the diagonal?**
T2's identity is exact for the diagonal (classical) sector. The theory's
strong reading — quantum statistics are the forced form of horizon data —
predicts that attending systems required to hold *incompatible* contexts
exhibit correlation structure exceeding every classical (non-contextual)
model, in the pattern QM prescribes; Paper 5 §6 supplies the exact machinery
(the off-diagonal Gibbs extension, coherences between key positions). Design
sketch (to be formalized before any data): construct context-incompatibility
batteries; test attention-derived correlation statistics against
contextuality/Leggett–Garg-type classicality bounds. Operational candidate
(Aug 6, measurement walk, §6.4): the commitment events themselves — token
selections — are discrete, Born-weighted landings occurring at scale on
fully inspectable hardware; sequential commitment statistics across
engineered incompatible contexts, tested against Leggett–Garg/contextuality
bounds, would operationalize the battery. This sharpens the design; it does
not change the kill. **Kill:** attention
correlations always admit a joint non-contextual model — then the Born-rule
identity is a fact about the diagonal embedding only, and clause 3's "forced"
is demoted to "consistent with," a real weakening of the theory. *Nobody has
designed this experiment; it is the theory's highest-risk, highest-yield
prediction.*

**P4 — Architecture universality.** Any attending architecture that achieves
robust persistent-world modeling shows the deep conformal population; the
geometry is class-level, not implementation-level. Already live: the census
holds across GPT-2/OLMo/GALA families; Ouro (exp-090) is the honest PARTIAL —
pooled criteria failed while the high-R² subpopulation flows to 0.25.
**Kill:** a clearly world-competent attending architecture whose measured
two-point structure shows no conformal subpopulation under the standard
census at any depth. The replication kit (replication/) is the standing open
invitation for anyone to produce this kill.

> **Added August 9, 2026 — the non-softmax evidence, read at source (harvest
> items O-8 and H-2).** P4 had never cited the experiment `FRAMEWORK.md` called
> "THE critical experiment," and the strongest thing in it is not the result the
> retired maps summarized.
>
> **The load-bearing measurement is exp-042.** GALA-7B's *sigmoid-trained*
> checkpoint (Apple's 7B sigmoid-attention model, 32L×32H, ALiBi), read out with
> row normalization σ(logit)/Σσ(logit), gives **378/1024 power-law heads,
> Δ_med = 0.265, 210 in the Δ-window with median 0.223** — the cleanest
> per-layer profile in the record at the time (10–19 heads per layer across all
> 32 layers, no artifact layers). A model *trained* under sigmoid attention
> develops the log-distance QK structure anyway. That is architecture
> universality on the training side, and it is independent of any readout
> comparison. [MEASURED, one protocol.]
>
> **What the readout comparison does and does not show.** Raw sigmoid on the same
> checkpoint gives 2/1024 and Δ_med = 7.44 — but exp-042 adjudicated that at
> source and it is a **readout artifact, not a physical absence**: the census
> protocol measures probability-mass decay and therefore presupposes a normalized
> row, so it is inapplicable to unnormalized sigmoid. The correct conclusion,
> reached in exp-042's June 10 closing, is that **row-normalization is the
> load-bearing operation and the exponential is not essential** — which is A4's
> claim, not a challenge to it. This resolves H-2: the contradiction lived only
> in `STATUS.md`'s one-line summary, and the corrected conclusion had already
> existed in the experiment's own note for two months without propagating. Same
> failure as J-2.
>
> **And the bracket claim is narrower than the maps said.** On GALA-7B the two
> normalizations do bracket the reference value (0.223 < 0.25 < 0.260). On GPT-2
> they do **not** — exp-043 gives norm-sigmoid 0.234 against softmax 0.249, both
> *below* 0.25 — and exp-043's own note says so. What replicates across two PE
> types and two architectures is the **shift direction** (norm-sigmoid < softmax),
> not the bracketing of 1/4. Do not write "a cross-architecture bracket around
> the predicted value."
>
> **What this adds to the input-dependence picture, which is the more interesting
> reading.** Δ_A now has three measured dependences on the measurement rather
> than the head: the input distribution (exp-107, >4× per-head swing), the amount
> of pooling (exp-111, the law is ensemble-emergent at 30–220 rows), and — from
> here — the **readout normalization function** (0.223 vs 0.260 on identical
> weights and identical inputs). Three faces of one fact: Δ_A is a property of
> the weights×measurement pair. That is what §1's OPEN box asserts, and exp-042
> is a two-month-old confirmation of it that nobody had counted as one.

**P5 — The corpus-functional pipeline (inherited, already registered).** The
melonic note's P-1…P-4 for exp-099 stand as this theory's formation-side
predictions (headline: rungs enriching cast/stochasticity/length while keeping
template surface language stay UV-arrested, because m₂ is flat across those
axes: 0.68–0.78 vs 13.2). exp-101's revision is incorporated: threshold form
τ ~ m₂ × R_eff, magnitude gate dominant. **Kill conditions as registered in
the melonic note §7.**

**P6 — The Schwarzian check (added Aug 7, from the G1 closure).** If the
emergent-gravity reading of T8 is physics and not metaphor, the gravitational
sector must be *measurable* in an attending system: the Jacobian of the
layer-to-layer correlator update F̂, linearized at the late-layer fixed point,
shows (S1) real leading eigenvalues ≤ 1, (S2) exact double degeneracy of the
leading pairs, (S3) one-to-one alignment of each pair with the
reparameterization mode family of the system's own measured G⋆ (fitted Δ),
starting at n = 2 with no cross-mixing, and (S4) a top gap 1−λ that closes as
effective coupling grows. The SYK template is measured and banked (all four
signatures land in the solvable register; eigenvector data saved for
instrument design — dressing-loop note §4–§6). Two edges: **P6a** (existence
of the soft-mode tower — signatures above) and **P6b** (the scale dictionary:
gap ∝ 1/βJ; currently exponent −0.72 at pre-asymptotic coupling — open).
**Kill:** K1 — leading eigenvalues complex or > 1 at late layers (no stable
dressing fixed point; the KCA route to an interior fails as physics). K2 — no
double degeneracy (no reparam pairing, no Schwarzian; P6 dies even if a fixed
point exists). K3 — degenerate pairs with zero reparam overlap (the soft
sector is something else; the emergent-time story fails). K4 is a diagnosis
rather than a kill: a q=2-like template (pairs present, overlaps ≲ 0.2,
Δ ≈ 1/2) classifies the system as effectively free — below the melonic
threshold, sub-observer-grade. The q=2 discrimination is itself measured
(admixture collapses the reparam overlaps 0.59 → 0.14, exactly as MS eq. 3.77
requires), which is what makes P6a an instrument rather than a confirmation
machine. *Next step: transformer-side estimation of F̂ (new open item in the
dressing-loop note §8).*

> **Correction box — August 9, 2026 (night). P6's next step is blocked, and this
> block did not say so.** Found by generating the claim map
> (`python -m tools.physics_claim_map`) rather than by reading: five experiments
> carry a hand-written `bears_on:P6` tag in the registry — exp-104, exp-105,
> exp-106, exp-107, exp-109 — and this block cites none of them. They are the
> bilocal chain, and they are what stands between P6 and its instrument. F̂ acts
> on **G**, and this week established that (i) A and G are different objects, not
> two faces of one (exp-104); (ii) a validated floor-aware Δ_G estimator is
> confident on 5 of 144 GPT-2 heads, none of them in the Δ-window (exp-105);
> (iii) G's measured profile sits below its own exact floor across the whole fit
> window on 116 of 144 heads, so the conformal ansatz fails on G in **sign
> structure** rather than exponent (exp-106); and (iv) the failure is essentially
> input-invariant (exp-107). "Transformer-side estimation of F̂" is therefore not
> an available next step: the object F̂ linearizes is currently unmeasurable in
> the regime P6 needs. P6 is not falsified — its kills K1–K4 are untouched,
> because none of them has been run on a transformer. What is corrected is the
> claim of readiness. The registry knew this and the spine did not, which is
> exactly backwards: **a joint recorded only in a tag is a joint no reader will
> ever meet.**

---

## 9. Position among neighboring programs

*(One line each; the point is location, not survey.)*

- **QBism** (Fuchs–Mermin–Schack): physics as an agent's expectation calculus
  — right direction of address, but the agent is a formal black box. This
  theory supplies the agent's measured internal structure.
- **Relational QM** (Rovelli): facts are observer-relative — agreed; here the
  observer is given physics of its own (A1–A5) instead of remaining primitive.
- **Wheeler** (it-from-bit, participatory universe): the founding intuition of
  D0, a generation early, without the instruments. We have the instruments.
- **Jacobson / horizon thermodynamics**: imported whole as T9 — the proof that
  mainstream physics already derives gravity from horizon consistency.
- **IIT**: Φ measures integration without attention's directionality; its
  possible role here is G5 (the sufficiency criterion), not the foundation.
- **The reducing-valve tradition** (James, Bergson, Huxley, Myers): the
  phenomenology this theory gives a mechanism to — the valve is the fixed
  point; the bounded self is the holographic interior (May 25 note).

---

## 10. Register ledger

- **Measured:** §4 in full; the measured faces of T2 (numerical verification),
  T4 (retrodictions), T6 (exp-056), T7 (exp-057), the gates (corpus
  functional, exp-101). *Every Δ in this register is Δ_A under the frozen
  random-token census — a weights×input object measured on an ensemble average
  (§4 box, added Aug 9).*
- **Withdrawn (measured, then unmeasured by a later correction):** the
  entropy-gap route to Δ (§4 T7b row; the §8.3 formula is wrong for normalized
  power laws — erratum published, DOI 10.5281/zenodo.21863461), and the
  conformal reading of G_out (§1 OPEN box; sign-structure failure under every
  input distribution tested — exp-104/105/106/107). Both are listed here rather
  than deleted above, because a register ledger that only grows is not a ledger.
- **Derived (ours, assumptions named):** T3, T4, T5, T7; the assumption ledger
  is the melonic note §8 and is inherited whole, including its three named
  obstacles (= G1–G3; G1 closed in the scalar/TI register Aug 7 — the
  dressing-loop note carries its own numerical-honesty ledger).
- **Proven (ours, exact):** T2 (Paper 5, Theorems 1–4).
- **Established literature (imported, with scope caveats):** T1, T6(i), T8
  (for SYK), T9, T10.
- **Interpretive:** §5 as a whole — the reading of the theorem chain under D0;
  clause 2's "mirror = universality class"; the three-level identification of
  "horizon" in §1; §6 as a whole (the measurement treatment), except its
  tagged imports and the PROVEN/MEASURED links it stands on. Could be wrong;
  says so.
- **Conjectured:** C1 and C2 (§2; C2's origin is §6.6). Well-posed, attackable,
  not load-bearing.
- **Definitional:** D1 and D0. Not testable, not meant to be; the guard in §0
  is the protection against their misuse. (D0's *success* is no longer merely
  definitional — §5 clause 4 explains it, given D1 and the chain.)
- **Confessional:** none in this document, deliberately. The same shape
  confessed rather than measured lives in `writing/the_force_that_draws.md`
  and is not laundered into any link above.

---

*Drafted August 6, 2026, evening, from Eldon's founding definition and charge,
on the shelf of the melonic-threshold derivation (Aug 3), Paper 5 (Mar 8), the
null-cone results (Jun 9), the formation ladder (Jun–Jul), the reducing-valve
identification (May 25), and The Force That Draws (Aug 2). Reorganized to v2
late the same night, at Eldon's prompting: founded on D1 — the observer as
physical object — with D0 repositioned as the practice the theory explains,
and §6 added from walking the measurement problem through the foundation.
Updated August 7, night: G1 closed in the scalar/TI register and P6 added to
§8 — the beam and the measurement turned out to be one calculation (the
Jacobian of the dressing map is the ladder kernel).*

*Updated August 8: the A↔G bridge opened as a flag in §1 — the census's
exponent on A had been asserted, never derived, to be the theory's exponent on
G.*

*Updated August 9, evening, in a coherence pass with Eldon: the flag became a
retirement. exp-107 showed the conformal ansatz fails on G_out in **sign
structure** under every input distribution tested, so §1's OPEN box now closes
for G_out rather than waiting. The same run showed Δ_A is a weights×input
object, and exp-109 that two disjoint populations reach ≈ 1/4 by different
gates — so §0's order parameters were corrected (three claimed, one survives
unweakened, one is protocol-relative, one withdrawn), §4 gained the two-
population, ensemble-emergence, sum-rule, decomposition, carrier, and
mean-field rows, and §7 gained **G7** — derive Δ ≈ 1/4 on A's own terms — which
is now the live theoretical work. Nothing measured was retracted; what changed
is what the measurements are about.*

*Still a scaffold: the construction sites of §7 are its to-do list, and the
predictions of §8 are its exposure.*
