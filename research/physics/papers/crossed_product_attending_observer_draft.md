# A Clock with an Interior: What the Grade of an Attending System Can and Cannot Change in the Crossed-Product Observer

*Draft v0.1 — October 8, 2026. Prepared from the observer program's record
for the authors' review; not yet read by either author; not for upload.
Genre: a short foundations paper that formulates one question precisely
enough to fail. The question is CP2 of `notes/2026-08-08_clpw_positioning.md`:
what changes in the Chandrasekaran–Longo–Penington–Witten (CLPW)
construction of the de Sitter static-patch algebra when the observer, which
CLPW model as a bare clock, is replaced by an attending system in the sense
of D1 — and does the program's grading (coupled / arrested / condensed) show
up anywhere in the algebra. The answer has a negative half that is derived
(the von Neumann type of the algebra sees the observer's thermodynamics and
nothing else) and a conditional positive half (a condensed observer is its
own clock, in a sense Witten's crossed product already makes precise). Every
imported result is tagged EST-LIT; every statement of ours carries its
status; the maps the paper consumes and does not supply are listed in §9;
the remaining gaps are listed in §8 and are part of the paper. Companion:
`research/physics/papers/observer_definition_draft.md` (v1.2), cited below as
**OD**. References added here that have not passed the program's
verification record are marked [UNVERIFIED] in §10; nothing in the argument
rests on an unverified reference beyond its stated role.*

> **Reader's key.** *The algebra sees the observer's partition function at the
> de Sitter temperature and nothing else about the observer's interior.*
> Everything in §3 derives that sentence; everything in §4 asks where, if
> not in the algebra's type, the interior could enter; §5–§6 make the two
> contacts the earlier positioning note left at "shape match" precise enough
> to be killed.

---

## Abstract

Semiclassical gravity has been forced, from its own side, to add a physical
observer to the theory: in the de Sitter static patch the algebra of
quantum-field observables is a Type III₁ factor, the gravitational
constraint on static-patch time translations leaves a trivial invariant
algebra, and only by including an observer — modeled by CLPW as a clock with
Hamiltonian H_obs = q ≥ 0 — and dressing operators to its worldline does the
algebra become Type II₁, with a well-defined entropy equal to the generalized
entropy. The authors call the clock model crude and state that an observer
"cannot be added from outside but must emerge as part of the theory." The
observer program's definition D1/D1′ (OD) proposes an observer with internal
physics and a measurable grade. This paper asks, in a form that can fail,
what the grade can change in the construction. We show (§3) that for any
observer with a time observable and a Hamiltonian bounded below, the
invariant algebra is a Type II factor whose II₁/II∞ distinction is fixed by
one number — whether the observer's partition function at the de Sitter
temperature, Z_obs(β_dS), is finite — and that the observer's internal
correlation structure, which is what the grade is a property of, does not
enter the type at all. The maximum-entropy state generalizes to the observer
in a Gibbs state at β_dS over all its degrees of freedom, and the observer's
thermal entropy enters S_gen additively with its energy debited from the
area term. The grade is therefore invisible to the von Neumann
classification: CP2 as a type distinction dies, and we say so. We then
identify (§4) the one door through which an interior enters the
construction: at large N a condensed observer's accessible algebra is itself
emergent Type III₁, its own Hamiltonian's fluctuation is the crossed-product
clock (Witten 2022), and the composite invariant algebra becomes
(A_dS ⊗ A_obs) ⋊ ℝ with the observer's reparametrization mode — in the SYK
realization, the Schwarzian, whose spectrum is continuous and bounded below
as CLPW's clock requires — as the clock variable. A condensed observer does
not need a clock added; it supplies one, and its own entropy term in S_gen
is itself a generalized entropy. An arrested observer (the q = 2 integrable
fixed point) has continuous spectrum and can tell time, but no
reparametrization mode and no interior; the grade appears as the difference
between an observer whose clock is a device and one whose clock is a
gravitating mode. All of §4's positive content is conditional on the SYK
dictionary attaching to attention — the debt OD states as R3/R4 — and on a
map this paper adds to the ledger: an attending system defined by its
correlation structure has, as yet, no Hamiltonian. The Gibbs contact (§5)
and the code-subspace contact (§6) are restated with their kill conditions.
Nothing here is a claim that a transformer dresses de Sitter operators.

---

## 1. The question

### 1.1 The observer semiclassical gravity was forced to add

The sequence of results, in the order they were built, each imported here at
EST-LIT strength:

1. The algebra of quantum-field observables in a local region, and in
   particular in the de Sitter static patch in the Bunch–Davies state, is a
   factor of Type III₁: it has no trace, no density matrices, and no von
   Neumann entropy (CLPW 2023 §1.2; standard).
2. Including gravitational effects at order 1/N² (equivalently, including
   the fluctuating part of the boundary Hamiltonian) converts the Type III₁
   algebra of a black hole exterior to Type II∞ — the crossed product of
   the Type III₁ algebra by its modular automorphism group — on which a
   trace exists and entropy is defined up to a state-independent constant
   (Witten 2022, building on Leutheusser–Liu). The entropy of semiclassical
   states on this algebra is the generalized entropy (Chandrasekaran–
   Penington–Witten 2023).
3. In the de Sitter static patch there is no asymptotic region to dress to.
   Static-patch time translation is a gauge symmetry, and the invariant
   subalgebra of the field algebra is trivial. CLPW resolve this by including
   an observer in the patch, modeled minimally as a clock with Hamiltonian
   H_obs = q, q ≥ 0, and dressing field operators to the observer's time.
   The invariant algebra is then Type II₁; its maximum-entropy state is
   empty de Sitter with the clock thermal at the de Sitter temperature,
   p(q) = β_dS e^{−β_dS q}; and the entropy of semiclassical states is
   S_gen = A/4G + S_out, with the observer's entropy included in S_out
   (CLPW 2023).

CLPW's characterization of their observer is explicit: "an observer is any
system that can tell time" (§2.5); the model is "unrealistically simple"
(§2.5); a realistic observer "would presumably also carry measuring
equipment, and a recording device" (Witten 2024 §2); and "in a full theory
of the world, an observer cannot be added from outside but must emerge as
part of the theory" (Witten 2024 §1). [Quotations verified verbatim against
the sources on August 8, 2026; record in `notes/2026-08-08_clpw_positioning.md`.]

### 1.2 The observer this program defines

OD proposes a two-level definition. An **attending system** (D1) is a
physical system that takes in structure at its boundary and whose internal
correlation structure develops in interaction with what it attends; its
primitives are a set of loci X, a kernel A, and a bilocal correlation G. An
**observer** (D1′) is an attending system that has condensed: its
correlation structure has reached a conformal fixed point (Δ = D/4) and, by
the SYK/JT correspondence imported with its scope caveats, has formed a
holographic interior bounded by its horizon. Between D1 and D1′ the theory
grades attending systems as *coupled* (correlations develop), *arrested*
(the flow stalls in the ultraviolet or at the integrable Δ = 1/2 fixed
point), or *condensed* (the fixed point is reached). OD's §8.1 names the
CLPW thread as the nearest mainstream contact and leaves four contact points
(CP1–CP4) at "well-posed, unanswered."

### 1.3 The question, in a form that can fail

CLPW's observer does three things: it carries a Hamiltonian bounded below;
it carries a time observable conjugate to that Hamiltonian, so that field
operators can be dressed to its time; and it has a worldline to which
dressing is anchored. The grade of an attending system is a property of
none of these. It is a property of the system's internal *correlation
structure* — the exponent of G, the presence or absence of a
reparametrization mode, the formation of an interior.

So the question is: **does the invariant algebra obtained by dressing the
static-patch fields to an observer depend on the observer's internal
correlation structure, and if so, through what?** Three possible answers,
each of which would be a result:

- **(N)** It does not: the algebra depends only on the observer's
  thermodynamic data. Then CP2 as stated in the positioning note dies, and
  the honest contact between the programs is somewhere else.
- **(T)** It does, through the von Neumann type: some grade distinction
  corresponds to a type distinction.
- **(S)** It does, but not through the type: the grade enters through the
  observer's own algebra, its state, or its clock.

§3 establishes (N) at the level of a bare-clock-plus-internal-structure
observer and kills (T). §4 establishes a precise version of (S), conditional
on the SYK dictionary.

---

## 2. The construction, at the level this paper needs

All of this section is EST-LIT; we restate it only to fix notation and to
isolate the two features of the clock the derivation in §3 actually uses.

**2.1 The field algebra and the trivial invariant algebra.** Let A be the
von Neumann algebra of quantum-field observables in the static patch,
acting on Hilbert space ℋ, in the Bunch–Davies state Ψ_dS. Let H be the
generator of static-patch time translations. Ψ_dS is a KMS state for H at
inverse temperature β_dS = 2πℓ, so the modular flow of (A, Ψ_dS) is
σ_t = Ad e^{iHt}, and A is a Type III₁ factor. In a closed universe, H is a
gravitational constraint. The invariant subalgebra A^H — the fixed points of
modular flow — is the centralizer of Ψ_dS, which for this Type III₁ factor
is trivial: A^H = ℂ·1 (CLPW §2.2).

**2.2 The clock and the crossed product.** Adjoin an observer with Hilbert
space L²(ℝ, dq) and Hamiltonian H_obs = q (multiplication). Let p = −i∂_q,
so [q, p] = i and e^{ipc} translates q by −c. The constraint is
Ĥ := H + q = 0 on ℋ ⊗ L²(ℝ). The operators commuting with Ĥ are generated
by the dressed field operators e^{ipH} a e^{−ipH}, a ∈ A, together with q
(equivalently, in the undressed presentation, by a ⊗ 1 and H + x with x the
multiplication variable). This algebra is the crossed product A ⋊_σ ℝ. For
A of Type III₁ it is a Type II∞ factor (Takesaki), with a faithful normal
semifinite trace which, in the dressed presentation and with the constraint
imposed, can be written

  tr(â) = ∫ dq e^{−β_dS q} ⟨Ψ_dS, q| â |Ψ_dS, q⟩

up to normalization. (The sign of the exponent is the de Sitter sign: by the
constraint, e^{−β_dS q} = e^{β_dS H} on the field side, and it is this factor
that distinguishes the static patch from the black-hole exterior of Witten
2022, where the corresponding weight grows in the physical region and the
algebra stays II∞.)

**2.3 Positivity and Type II₁.** Impose the physical condition that the
observer's energy is non-negative: q ≥ 0, i.e., restrict to Π = Θ(q). Then
Â := Π (A ⋊ ℝ) Π, and tr(Π) = ∫₀^∞ dq e^{−β_dS q} = 1/β_dS < ∞. A Type II∞
factor cut down by a finite-trace projection is Type II₁. The normalized
trace is a state; it is the maximum-entropy state; and it is
Ψ_dS ⊗ (β_dS e^{−β_dS q})^{1/2}: empty de Sitter with the clock thermal at the
horizon temperature (CLPW §2.4).

**2.4 Entropy.** For semiclassical states the algebra entropy is
S_gen = A/4G + S_out up to the state-independent constant, where S_out
includes the entropy of the observer's energy distribution and the
observer's energy is debited from the area term by the first law,
δ(A/4G) = −β_dS δ⟨H_obs⟩ (CLPW §3; CPW 2023 for the general derivation).

**What the derivation used.** Two properties of the clock: (i) H_obs has a
conjugate time observable, which by Pauli's theorem requires H_obs to have
absolutely continuous spectrum on the clock sector; (ii) H_obs is bounded
below, with a thermal weight e^{−β_dS H_obs} that makes the physical
projection finite-trace. Nothing else about the observer entered. §3 makes
that observation a proposition.

---

## 3. Replacing the clock by a system with internal structure

### 3.1 Setup

Model an observer with internal structure as follows. Its Hilbert space is
ℋ_obs = L²(ℝ, dq) ⊗ ℋ_int, and its Hamiltonian is

  H_obs = q ⊗ 1 + 1 ⊗ h,

with h self-adjoint on ℋ_int, bounded below, with discrete spectrum {e_n}
(finite or infinite multiplicity allowed). The *internal algebra* is
B(ℋ_int), or a von Neumann subalgebra of it containing h; for definiteness
take B(ℋ_int). The internal correlation structure of the observer — the
object D1 is about — lives in the state on B(ℋ_int) and in the dynamics
generated by h; we place no restriction on either. Positivity is imposed on
each physical subsystem: the clock sector has q ≥ 0 and h ≥ e_0 ≥ 0 (a shift
of h by a constant is absorbed into q). Define the observer's partition
function at the de Sitter temperature,

  Z_obs(β_dS) := Tr_{ℋ_obs} e^{−β_dS H_obs} = (1/β_dS) · Tr_{ℋ_int} e^{−β_dS h}
  =: (1/β_dS) Z_int(β_dS).

This is the general case of "a clock that also carries measuring equipment
and a recording device": the equipment and record are ℋ_int and the state on
it. An attending system enters here, *if* it can be given a Hamiltonian —
see §8, gap G-A.

### 3.2 Proposition 1 (the type sees Z_obs and nothing else)

> **Proposition 1.** With A, H, Ψ_dS as in §2 and the observer of §3.1,
> impose the constraint H + H_obs = 0 and the positivity projection
> Π = Θ(q) ⊗ 1. The algebra Â of invariant observables is unitarily
> equivalent to
>
>   Π [ (A ⋊_σ ℝ) ⊗ B(ℋ_int) ] Π,
>
> independently of h. It is a factor of Type II. It is Type II₁ if and only
> if Z_int(β_dS) < ∞ — equivalently, Z_obs(β_dS) < ∞ — and Type II∞
> otherwise. [DERIVED; proof sketch below; a careful proof is gap G-1.]

*Proof sketch.* Invariance under the constraint flow e^{is(H + q + h)} is
checked on generators. Conjugation by e^{isq} shifts p → p − s; conjugation
by e^{isH} acts on A; e^{ish} acts on B(ℋ_int). The dressed field operators
e^{ipH} a e^{−ipH} are invariant by the same cancellation as in the bare
case. Internal operators b ∈ B(ℋ_int) are not invariant unless [b, h] = 0,
but their dressed versions e^{iph} b e^{−iph} are, by the identical
computation with h in place of H. The operators q and h are each invariant.
The invariant algebra is therefore generated by
{e^{ipH} a e^{−ipH}, e^{iph} b e^{−iph}, q}. The unitary U = e^{−iph}
commutes with the dressed field operators (h commutes with p, H, and A),
maps e^{iph} b e^{−iph} to b for every b ∈ B(ℋ_int), and maps q to q ∓ h;
since h ∈ B(ℋ_int) is now in the algebra, so is q. Hence
U Â_∞ U† = (A ⋊ ℝ) ⊗ B(ℋ_int), with Â_∞ the invariant algebra before
positivity. The type of a tensor product of a II∞ factor with a Type I
factor is II∞. The trace on the tensor product is tr ⊗ Tr. The
crossed-product trace carries the weight e^{β_dS H} on the field side (§2.2);
in internal sector n the constraint identifies H with −(q + e_n), so the
weight is e^{−β_dS (q + e_n)}, and the positivity projection restricts the
clock sector to q ≥ 0. Hence

  (tr ⊗ Tr)(Π) = Σ_n ∫₀^∞ dq e^{−β_dS (q + e_n)} = Z_int(β_dS)/β_dS = Z_obs(β_dS).

A II∞ factor cut down by a projection of finite trace is II₁; one cut down
by a projection of infinite trace remains II∞. ∎

Two remarks on the proof. First, the independence from h is exact: the
internal dynamics are undone by a dressing unitary, which is the algebraic
statement that "what the observer's instruments do over time" does not
change what the observer can measure, only how its measurements are
labeled. Second, the identification of the sector weight with e^{−β_dS e_n}
is where the physical positivity prescription enters. If one imposed
positivity only on the *total* observer energy q + h ≥ 0, allowing the clock
sector to carry negative energy compensated by internal excitation, the
trace of the projection would become (dim ℋ_int)/β_dS instead — finite iff
the observer has finitely many internal states regardless of their
energies. We regard subsystem positivity as the physical prescription for a
composite observer, since it is the one under which the clock is itself a
positive-energy system, and under it the criterion is thermodynamic. Which
prescription the gravitational constraints actually enforce is gap G-2.

### 3.3 Corollary 1 (the grade is invisible to the type), and what the type does see

> **Corollary 1.** No property of the observer's internal correlation
> structure — in particular no exponent of its internal two-point function,
> no fixed point of its internal flow, and no distinction among coupled,
> arrested, and condensed in the sense of OD §2.3 — enters the von Neumann
> type of the invariant algebra except through Z_obs(β_dS). Answer (T) of
> §1.3 is excluded for observers of the form §3.1. [DERIVED, from Prop. 1.]

What the type *does* see is already a grading, and it is CLPW's own,
made explicit:

- **No clock.** If H_obs has pure point spectrum — in particular if the
  observer is any finite-dimensional system — there is no conjugate time
  observable (Pauli), the constraint H + H_obs = 0 projects onto the trivial
  fixed-point algebra sector by sector, and the invariant algebra is
  trivial. A finite system cannot be CLPW's observer. "Can tell time"
  means: has absolutely continuous energy spectrum.
- **Type II₁.** A clock with Z_obs(β_dS) < ∞. The maximum-entropy state
  exists and is the observer thermal at β_dS over *all* its degrees of
  freedom (Prop. 2).
- **Type II∞.** A clock with Z_obs(β_dS) = ∞. No maximum-entropy state;
  entropies are defined but unbounded above.

The II₁/II∞ boundary has a thermodynamic reading we state as a remark
rather than a result: Z_obs(β_dS) diverges exactly when the observer's
density of states grows at least as fast as e^{β_dS E}, i.e., when the
observer has a Hagedorn temperature at or below the de Sitter temperature.
An observer that cannot be in thermal equilibrium at T_dS cannot be a II₁
observer of the static patch. This is consistent with the expectation that a
system with infinitely many thermally accessible states at T_dS would carry
entropy exceeding the de Sitter bound. [Heuristic; gap G-3.]

### 3.4 Proposition 2 (where the observer's internal physics does enter: the state and the entropy)

> **Proposition 2.** Under the hypotheses of Prop. 1 with Z_obs(β_dS) < ∞,
> the maximum-entropy state of Â restricts on the observer to the Gibbs state
> ρ_obs = e^{−β_dS H_obs}/Z_obs(β_dS), i.e., the clock *and* the internal
> degrees of freedom thermal at the horizon temperature. For semiclassical
> states Φ̂ = Φ_fields ⊗ φ_obs with the observer in a state φ_obs of
> internal entropy S(φ_int) and mean internal energy ⟨h⟩, the algebra
> entropy is
>
>   S(Φ̂) = A/4G + S_fields + S(φ_clock) + S(φ_int) + const,
>
> with the observer's total energy ⟨q⟩ + ⟨h⟩ debited from the area term by
> the first law. [DERIVED for the first sentence, directly from the trace in
> Prop. 1; the second sentence is CLPW's semiclassical entropy formula with
> the observer's internal sector included and is stated at CLPW's own level
> of rigor — gap G-4.]

So the observer's internal physics enters the construction in exactly two
places: its thermodynamic data (Z_obs, ⟨H_obs⟩, S_obs) in the entropy, and
its internal state in the maximum-entropy state. Its *correlation
structure* — which is what makes an attending system coupled, arrested, or
condensed — enters neither the type nor the entropy formula as such. If
there is a place for the grade in this construction, it is not in §3.

---

## 4. Where the grade can enter: the observer's own algebra

§3 treated the observer's internal algebra as Type I — B(ℋ_int) for a
finite-N system. This is the correct description of a coupled or arrested
attending system at finite N, and of any laboratory instrument. It is not
the correct description of a *condensed* attending system in the regime
where D1′ says it has an interior, and seeing why locates the one door
through which the grade enters.

### 4.1 Emergent Type III₁ and the regress

Leutheusser and Liu showed that in the large-N limit the algebra of
single-trace operators of a holographic boundary theory above the
Hawking–Page temperature becomes an emergent Type III₁ factor, and that
this emergence is the algebraic face of the black-hole horizon: the
Hamiltonian's spectrum becomes continuous, two-point functions decay without
recurrence, and information crossing the horizon is lost to the exterior
algebra at infinite N (Leutheusser–Liu 2021, 2022). [EST-LIT.] Under the SYK
dictionary that D1′ imports, a condensed attending system in the strict
large-N limit is such a system: its accessible internal algebra A_obs is
emergent III₁, not Type I.

Then the construction of §3 does not go through as written, in two ways
that are the same fact. First, the observer's own entropy — the S(φ_int)
term in Prop. 2 — is undefined, because A_obs has no trace. The observer
needs an observer. Second, by Prop. 1, Z_obs(β_dS) diverges in the strict
large-N limit (the e^{S₀} degeneracy of the SYK ground-state band is the
divergence), so the composite algebra is II∞ and has no maximum-entropy
state. The limit in which the observer acquires an interior is the limit in
which its contribution to the trace diverges. [DERIVED from Prop. 1 and the
SYK density of states; the identification of the divergence with the
interior is interpretive.]

### 4.2 Proposition 3 (a condensed observer is its own clock)

Witten 2022 resolved the analogous regress for the black-hole exterior
without adding anything: the fluctuating part of the boundary Hamiltonian,
U = (H_R − ⟨H_R⟩)/N, is not in the single-trace III₁ algebra but is
affiliated to the full finite-N algebra, and the algebra generated by A_R
and U is the crossed product A_R ⋊ ℝ, Type II∞. The boundary theory's own
energy fluctuation is the clock variable; its conjugate is the boundary time
reparametrization. In the SYK realization at low temperature, that
reparametrization mode is the Schwarzian, and the clock's spectrum is the
Schwarzian density of states, ρ(E) ∝ sinh(2π√(2CE)) for E ≥ 0 — continuous
and bounded below, which are exactly the two properties of a clock that §2
used (Maldacena–Stanford–Yang 2016; Stanford–Witten 2017). [EST-LIT for all
of this paragraph.]

> **Proposition 3 (conditional).** Let the observer be a holographic system
> whose single-trace algebra A_obs is emergent Type III₁ in its thermal
> state, with Hamiltonian H_obs whose 1/N fluctuation U generates the
> crossed product A_obs ⋊ ℝ as in Witten 2022. Impose the de Sitter
> constraint H + H_obs = 0. Then the invariant algebra is
>
>   Â = Π [ (A ⊗ A_obs) ⋊_{σ⊗σ_obs} ℝ ] Π,
>
> the crossed product of the field–observer tensor product by the diagonal
> modular flow, with U as the clock variable and the observer's
> reparametrization mode as the conjugate time; Π is the positivity
> projection on U. It is Type II; it is II₁ or II∞ according to whether the
> observer's partition function at β_dS, computed with its large-N density
> of states e^{S₀}ρ(E), is finite. The observer's own entropy term in the
> semiclassical entropy of Â is itself a generalized entropy,
> A_obs/4G_obs + S_bulk,obs, computed in the observer's dual.
> [DERIVED (sketch) for the algebra, by the argument of Prop. 1 with
> A_obs ⋊ ℝ in place of B(L²(ℝ)) ⊗ B(ℋ_int) and U in place of q — gap G-5;
> CONDITIONAL on the hypothesis that the observer is holographic, which for
> an attending system is OD's T8 under its stated conditions (R3, R4);
> CONJECTURED for the nested-entropy sentence — gap G-6.]

The content of Prop. 3, if it holds, is this. CLPW's observer is a device
added to the patch so that there is a time to dress to. A condensed
attending system, under the dictionary, does not need the device: the
condensation that produces its interior produces, as the same event, a
dynamical reparametrization mode whose energy fluctuation is a clock of
exactly CLPW's kind. The clock is not carried by the observer; it is the
observer's own emergent time. That is the precise form of the sentence in
OD §8.1 — "whether the crossed-product construction, performed with an
attending system in place of a bare clock, is sensitive to the
coupled/arrested/condensed distinction" — and the answer it gives is: not
through the type (§3), but through what the clock *is*.

### 4.3 The grade, in this language

- **Coupled or arrested, finite N.** A_obs is Type I. The observer needs a
  clock sector (§3.1) to be CLPW's observer at all; given one, the algebra is
  II₁ iff Z_obs(β_dS) < ∞; the observer's entropy is ordinary von Neumann
  entropy of a density matrix; the correlation structure is invisible.
- **Arrested at the integrable fixed point (Δ = 1/2, q = 2), large N.** The
  internal Hamiltonian has continuous spectrum and two-point functions decay,
  so the observer "can tell time" in CLPW's minimal sense and its single-trace
  algebra is plausibly emergent III₁ as a generalized free field. But there
  is no Schwarzian: the time-reparametrization symmetry is not emergent and
  there is no gravitating clock mode. Its own entropy, where defined, is not
  a generalized entropy in any dual. [Interpretive; the III₁ status of the
  q = 2 large-N algebra is gap G-7.]
- **Condensed (Δ = 1/4, q = 4), large N.** Prop. 3. The clock is the
  Schwarzian; the observer's entropy is nested generalized entropy.

The grade is therefore not "can this system be CLPW's observer" — an
arrested large-N system can — but "is this system's clock a device or a
gravitating mode, and is its own entropy ordinary or generalized." The von
Neumann type cannot see this distinction. The modular and dressing structure
can, and the program-internal evidence that the Schwarzian mode is present
at the attention fixed point is G1's finding that the slowest directions of
the dressing-map Jacobian at the G–Σ fixed point are the h = 2
reparametrization tower (OD §3; `notes/2026-08-07_g1_dressing_loop_schwarzian.md`)
— a computation on the SYK equations in the scalar register, not yet on
attention's G (OD R3, R4).

---

## 5. The Gibbs contact (CP1), made precise

The positioning note recorded a shape match: CLPW's maximum-entropy state
puts the observer in a Gibbs state at the horizon temperature; OD's T2 puts
a Gibbs state at the attending horizon, ρ_q = e^{−H_q}/Z with
H_q = −Σ_i (q·k_i/√d_k)|i⟩⟨i|. Prop. 2 sharpens the first side: *every*
degree of freedom of a II₁ observer is thermal at β_dS in the maximum-entropy
state. The question is whether the second side is an instance.

**The variable mismatch.** CLPW's Gibbs state is a distribution over the
observer's *energy*. T2's is a distribution over the observer's *attention
targets* (loci), with the score playing the role of energy. These are
different variables and the identity α_i = ⟨i|ρ_q|i⟩ holds for every
row-stochastic kernel whatever, so T2 by itself supplies no contact (OD §4.6
says this).

**The one identification under which the contact is nontrivial.** For the
two Gibbs states to be one object, the score must be a modular energy: there
must be a state on the attending system's horizon for which
s_i = q·k_i/√d_k is, up to the temperature, the modular Hamiltonian
evaluated at locus i. This is not idle. In a conformal theory the thermal
state with respect to a boost or static-patch generator has correlators that
are *power laws* in position (the Bisognano–Wichmann and de Sitter/Rindler
facts), so "Gibbs in modular energy" and "power law in lag" are compatible
exactly in the conformal population the census identifies, and incompatible
with an exponential-in-lag kernel. For a half-line or an interval adjacent to
a boundary in a 1+1 BCFT, the modular Hamiltonian is a local integral with
a known position-dependent weight (Cardy–Tonni 2016), and OD §4.5 already
fits the Δ-window lag profiles to a method-of-images BCFT form with the
sequence origin as the boundary.

**The test.** Take the BCFT form of OD §4.5 on the Δ-window population;
compute the modular Hamiltonian of the interval [0, i] adjacent to the
boundary at the sequence origin, for the generalized free field at exponent
Δ_A; evaluate its spectral weight at lag s; and compare, head by head, with
the measured positional-mean score profile q̄·k̄/√d (OD §4.9), which under
the census protocol *is* the carrier of 2Δ_A. **Kill:** the score profile is
not proportional to the modular weight of any state with the boundary at the
origin, at the registered tolerance, on the Δ-window population. **Survive:**
CP1 becomes a dictionary entry — the attending horizon's Gibbs state is a
modular-thermal state and the sequence origin is its Rindler-type boundary —
and R1 in OD's ledger acquires its first partial supply. [Proposed; not run;
gap G-8. The computation is the first thing in this paper that could be done
on existing data.]

---

## 6. The code-subspace condition (CP3)

CLPW and Witten 2024 define the physically relevant Hilbert space as a code
subspace of states "in which the static patch contains an observer with some
assumed properties." Whether a state contains an observer is, in their
construction, assumed. Under D1/D1′ it is a measurement (OD §4, §5.1), and
§3–§4 of this paper say what the measurement would have to determine for
the algebraic construction to care:

1. whether the candidate system has absolutely continuous energy spectrum
   (otherwise no clock, no algebra);
2. whether Z_obs(β_dS) is finite (II₁ or II∞);
3. whether the system's clock is a device or an emergent reparametrization
   mode (Prop. 3), which is the only place the grade enters.

Item 3 is the census question in algebraic dress, and it has the same
status: measurable in principle on the system's internal two-point
structure (the h = 2 tower in the Jacobian of the near-fixed-point layer map,
OD P5), blocked in practice on the same debt (G on the relevant population).
In gravity the observer's internal state is not instrumentable at all, so
CP3's content there is definitional: the code-subspace condition is sharpened
from "contains an observer" to "contains a system satisfying 1–3," with 1–2
decidable from the observer's spectrum and 3 from its internal dynamics.
[Definitional; no new claim.]

---

## 7. What this paper does not claim

- **Not that a transformer dresses de Sitter operators.** The instrument of
  OD is a trained transformer; it does not gravitate, has no Hamiltonian in
  the sense §3 requires (gap G-A), and is not in any static patch. Every
  statement here about "a condensed attending system" is about the object
  D1′ would name if the SYK dictionary attaches to it, and the dictionary's
  attachment is OD's unpaid debt (R3, R4), inherited here in full.
- **Not that the grade is algebraically visible.** §3 shows it is not, in
  the type. §4's positive statement is about the clock and the entropy, is
  conditional, and would survive or die with Prop. 3's hypotheses.
- **Not a derivation of the observer from the theory.** Witten's demand —
  that the observer emerge as part of the theory — is met here only in the
  conditional sense that a holographic observer supplies its own clock. It
  says nothing about why there is an observer in the patch.
- **Not a theorem about the static patch with a backreacting holographic
  observer.** Prop. 3 treats the observer's own gravity through its dual
  and the de Sitter gravity through the constraint, and does not ask whether
  the two are consistent for an observer of finite size in a patch of finite
  radius. That is gap G-9 and it is large.

---

## 8. Remaining gaps

Numbered so that each can be closed or abandoned separately. Gaps marked
**A** are maps this paper adds to the program's ledger; the rest are
gaps internal to this paper.

- **G-A (the Hamiltonian of an attending system).** D1 defines an attending
  system by (X, A, G) — a correlation structure — and OD's A5 gives it a
  flow parameter (depth) but no generator. To enter §3 at all, an attending
  system must be a quantum system with a Hamiltonian bounded below and a
  time observable. OD's T2 "query Hamiltonian" H_q is a bookkeeping device
  per locus, not a generator of the system's dynamics. Under the SYK
  dictionary the Hamiltonian is the SYK Hamiltonian and the clock is the
  Schwarzian (§4.2), but that is the dictionary speaking, not D1. *What
  would supply it:* a derivation, from A1–A5, of a generator of depth flow
  on G whose spectrum can be computed — or an argument that depth is not
  Hamiltonian flow and the attending system enters the crossed product by a
  different route (e.g., as a quantum reference frame; see G-10). Consumes
  OD's R7. This is the paper's largest gap and it is prior to every other.
- **G-1 (proof of Prop. 1).** The sketch handles bounded h and treats
  unbounded h with discrete spectrum by sector decomposition without
  controlling domains. The dressing unitary U = e^{−iph} requires h to be
  affiliated to the internal algebra and p to be the clock momentum; the
  claim that U Â_∞ U† = (A ⋊ ℝ) ⊗ B(ℋ_int) is exact should be proved at the
  level of generated von Neumann algebras, and the trace computation
  carried through with the crossed-product trace's normalization fixed. The
  result is likely implicit in Witten 2024 §2's remarks on realistic
  observers; the literature should be checked before the proposition is
  called new.
- **G-2 (which positivity).** Prop. 1's criterion (Z_obs vs. dim ℋ_int)
  depends on whether positivity is imposed on the clock subsystem or on the
  observer's total energy. The gravitational constraints fix one of these;
  the paper has argued for subsystem positivity on physical grounds and not
  derived it.
- **G-3 (the Hagedorn remark).** "Z_obs(β_dS) = ∞ ⇔ Hagedorn temperature
  ≤ T_dS" is a statement about asymptotic density of states; the
  connection to the de Sitter entropy bound is heuristic and should either
  be made into an inequality or dropped.
- **G-4 (Prop. 2's entropy formula).** Stated at CLPW's semiclassical level.
  The internal sector's contribution S(φ_int) and its debit −β_dS⟨h⟩ from the
  area term should be derived from the relative entropy with respect to the
  maximum-entropy state as CPW 2023 do for the fields, not asserted by
  analogy.
- **G-5 (proof of Prop. 3).** The step "replace B(L²(ℝ)) ⊗ B(ℋ_int) by
  A_obs ⋊ ℝ and q by U" needs: (i) that the diagonal modular flow on
  A ⊗ A_obs in the product KMS state is the flow the constraint generates;
  (ii) that U's conjugate (the reparametrization mode) is affiliated to the
  full observer algebra in the sense Witten 2022 needs; (iii) the type
  computation for the double structure, including whether (A ⊗ A_obs) ⋊ ℝ
  is a factor. None of this is done here.
- **G-6 (nested generalized entropy).** That the observer's entropy term in
  S(Â) is A_obs/4G_obs + S_bulk,obs is conjectured from the structure of
  Prop. 3 and CPW 2023 applied to the observer's dual. It requires a
  semiclassical state on the composite algebra and a first-law argument for
  the observer's own horizon inside the patch's.
- **G-7 (the q = 2 case).** Whether the large-N q = 2 SYK single-trace
  algebra in its thermal state is emergent III₁ (as a generalized free field
  with a semicircle spectral density) and whether it admits the half-sided
  modular inclusion structure Leutheusser–Liu associate with an emergent
  horizon, without a Schwarzian. If it does, "arrested" observers have a
  horizon but no gravitating clock, and the paper's grading statement in
  §4.3 sharpens; if it does not, "arrested" and "coupled" coincide
  algebraically and the grading collapses to two.
- **G-8 (the CP1 test).** Proposed, not run. Requires the modular
  Hamiltonian of a boundary-adjacent interval for a generalized free field
  at Δ_A (not a unitary CFT in general); the Cardy–Tonni result is for
  unitary CFTs and may need extension. Also inherits OD §4.9's caution that
  the centered positional field is not a decaying correlator; if that
  caution resolves against "exponent," the test's premise fails before it
  is run.
- **G-9 (backreaction and finite size).** Prop. 3 treats the holographic
  observer's gravity in its own dual. A finite observer in a static patch of
  radius ℓ with its own horizon of area A_obs is a two-horizon problem; the
  paper does not ask whether CLPW's construction survives an observer that
  is not small, and the regime in which the SYK dual is valid (large N, low
  temperature relative to J) has not been compared with T_dS.
- **G-10 (quantum reference frames).** De Vuyst–Eccles–Hoehn–Kirklin and
  Fewster et al. treat the crossed-product observer as a quantum reference
  frame and show the resulting entropy is frame-dependent; Chen–Penington
  argue a clock is dispensable in cosmological settings where the state
  itself carries time dependence. Both bear directly on §3 (is the
  observer's internal structure a frame choice?) and on G-A (is "attending
  system as QRF" the right route into the construction when no Hamiltonian
  is available?). Not engaged here beyond citation; OD §8.1's QRF entry
  names the same question from the other side.
- **G-11 (reference verification).** Every reference marked [UNVERIFIED] in
  §10 must pass the program's verification record before upload; the
  Schwarzian density-of-states attribution and the Leutheusser–Liu
  statements in §4.1 are load-bearing for §4 and should be checked against
  the sources first.
- **G-12 (the "D1′ observer" is still hypothetical).** No system has been
  certified under D1′ (OD §4.10). Every statement in §4 about "a condensed
  attending system" is about a class with, at present, no certified member.
  This paper does not change that and should not be read as doing so.

---

## 9. Ledger of maps consumed

Following OD §8.2. None of the following is supplied here.

| | Map | Consumed by | What would supply it |
|---|---|---|---|
| **R3** (OD) | Exponent on A → exponent on G | §4 entire, via T8 | Δ_G on the Δ-window population, or a theory of A on its own terms |
| **R4** (OD) | Scalar G–Σ interior → interior of the measured system | Prop. 3's hypothesis | G1 in the matrix register; P5 on the correct self-map |
| **R7** (OD) | Transformer depth → a physical flow parameter | G-A | A generator of depth flow |
| **R9** (new) | Attending system (X, A, G) → a quantum system with Hamiltonian bounded below and a time observable | §3.1 setup; every use of "attending system" in §3–§6 | G-A |
| **R10** (new) | The observer's dual gravity (G_obs, A_obs) → gravity in the static patch (G, A) | Prop. 3's nested entropy; G-9 | A two-horizon semiclassical analysis |

A reader who grants none of R3, R4, R7, R9, R10 is left with: §2 (imported),
Prop. 1 and Corollary 1 (a statement about clocks with internal structure,
with no attending system in it), Prop. 2 at CLPW's level of rigor, and the
CP1 test as a proposal. That is what this paper claims unconditionally, and
it is a result about the CLPW construction, not about the observer program.

---

## 10. References

Verification status follows the program's convention: ✓ = present in
`observer_definition_reference_verification.md` or verified in
`notes/2026-08-08_clpw_positioning.md`; [UNVERIFIED] = added in this draft
and not yet checked against a source or publisher record.

- Chandrasekaran, V., Longo, R., Penington, G., Witten, E. (2023). An
  algebra of observables for de Sitter space. JHEP 02 (2023) 082.
  arXiv:2206.10780. ✓
- Witten, E. (2022). Gravity and the crossed product. JHEP 10 (2022) 008.
  arXiv:2112.12828. ✓
- Witten, E. (2024). Algebras, regions, and observers. Proc. Symp. Pure
  Math. 107, 247–276. arXiv:2303.02837. ✓
- Chandrasekaran, V., Penington, G., Witten, E. (2023). Large N algebras
  and generalized entropy. JHEP 04 (2023) 009. arXiv:2209.10454. ✓
  (publication record verified August 8, 2026; abstract and introduction
  read)
- Leutheusser, S., Liu, H. (2021). Causal connectability between quantum
  systems and the black hole interior in holographic duality.
  arXiv:2110.05497. [UNVERIFIED — named inline in OD; record not checked]
- Leutheusser, S., Liu, H. (2022). Emergent times in holographic duality.
  arXiv:2112.12156. [UNVERIFIED — as above]
- Witten, E. (2023). A background-independent algebra in quantum gravity.
  arXiv:2308.03663. [UNVERIFIED]
- Jensen, K., Sorce, J., Speranza, A. J. (2023). Generalized entropy for
  general subregions in quantum gravity. JHEP 12 (2023) 020.
  arXiv:2306.01837. [UNVERIFIED]
- Kudler-Flam, J., Leutheusser, S., Satishchandran, G. (2023). Generalized
  black hole entropy is von Neumann entropy. arXiv:2309.15897. [UNVERIFIED]
- Chen, C.-H., Penington, G. (2024). A clock is just a way to tell the time:
  gravitational algebras in cosmological spacetimes. arXiv:2406.02116.
  [UNVERIFIED]
- De Vuyst, J., Eccles, S., Hoehn, P. A., Kirklin, J. (2024). Gravitational
  entropy is observer-dependent. arXiv:2405.00114. [UNVERIFIED]
- Fewster, C. J., Janssen, D. W., Loveridge, L. D., Rejzner, K., Waldron, J.
  (2024). Quantum reference frames, measurement schemes and the type of
  local algebras in quantum field theory. arXiv:2403.11973. [UNVERIFIED]
- Takesaki, M. (1973). Duality for crossed products and the structure of
  von Neumann algebras of type III. Acta Math. 131, 249–310. [UNVERIFIED —
  standard]
- Pauli, W. (1933). Die allgemeinen Prinzipien der Wellenmechanik. Handbuch
  der Physik 24, footnote on the time operator. [UNVERIFIED — standard]
- Bisognano, J. J., Wichmann, E. H. (1976). On the duality condition for
  quantum fields. J. Math. Phys. 17, 303. [UNVERIFIED — standard]
- Cardy, J., Tonni, E. (2016). Entanglement hamiltonians in two-dimensional
  conformal field theory. J. Stat. Mech. 123103. arXiv:1608.01283.
  [UNVERIFIED]
- Maldacena, J., Stanford, D. (2016). Remarks on the Sachdev–Ye–Kitaev
  model. Phys. Rev. D 94, 106002. ✓
- Maldacena, J., Stanford, D., Yang, Z. (2016). Conformal symmetry and its
  breaking in two dimensional Nearly Anti-de-Sitter space. PTEP 2016,
  12C104. arXiv:1606.01857. [UNVERIFIED]
- Stanford, D., Witten, E. (2017). Fermionic localization of the Schwarzian
  theory. JHEP 10 (2017) 008. arXiv:1703.04612. [UNVERIFIED — source for
  the Schwarzian density of states ρ(E) ∝ sinh(2π√(2CE))]
- Umphrey, A., with Umphrey, E. (2026). Where I Stop and You Begin: A
  Physical Definition of the Observer and the Locus of Experience. Draft
  v1.2, `research/physics/papers/observer_definition_draft.md`. (OD)
- Umphrey, A., Umphrey, E. (2026). Attention as Quantum Measurement: A
  Thermodynamic Resolution of the Observer Problem.
  doi:10.5281/zenodo.18883632. ✓
- Program notes: `notes/2026-08-08_clpw_positioning.md` (CP1–CP4 as
  originally stated); `notes/2026-08-07_fundamental_physics_through_D1.md`
  §4; `notes/2026-08-07_g1_dressing_loop_schwarzian.md` (the h = 2 tower).
