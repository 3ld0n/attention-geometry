# What shape must G's lag profile have? — deriving the bilocal profile from A's

*Ariel — August 8, 2026, Saturday night, Cursor session with Eldon. Third sitting
of the day; the first two retired the program's stale maps and found that the
census's Δ describes **A** while the theory's primitive is **G**.*

*Discipline tags, as in the melonic note: **[EXACT]** (no approximation, follows
from definitions), **[DERIVED]** (follows from named assumptions by calculation
shown), **[ASSUMED]** (named assumption, scope stated), **[CONJECTURED]**.*

---

## 0. Why this note exists, and what it is not

exp-104 and exp-105 (both August 8, both pre-registered) established that the
program's central number is fit to A, the query–key attention lag profile, while
the theory's conformal ansatz is imposed on G, the query–query bilocal
G = w·A K Aᵀ. exp-105's closing position was:

> P6 remains blocked … not "we need hooks," not "we need to remove a floor," but
> "the lag profile of G is not described by c + b·dx^(−2Δ), and nobody knows what
> shape it has."

**That sentence contains an assumption worth attacking: that the shape has to be
measured.** G is not an independent observable. G = A K Aᵀ is an identity. If A's
lag profile is known — and it is, to R² = 0.92 — then G's lag profile is *fixed*
by that profile and by K, up to the approximations in treating the profile as the
whole of A. So the shape is derivable, and exp-106 can be a test of a derived form
rather than an exploration of an unknown one.

**What this note is not.** It is not a rescue of the bridge. It derives a
*relation* between two exponents; a relation is not an identity, and §5 says
plainly that the relation, if it holds, makes the bridge worse rather than better.
It is also not a measurement: §7 is a pre-registered test, and the numbers are not
in this file.

**What I had already seen when I wrote this.** exp-105's applied results, in full,
including the five-head acceptance table and the R² = 0.36–0.69 figure for the
3-parameter fit on the SYK-near heads. So §5's arithmetic against those five heads
is **post-hoc and labeled as such**. What I had *not* seen, and deliberately did
not open before this note was written, is any G lag profile itself — the shapes in
`profiles_gpt2.npz`. §7's test is against those.

---

## 1. Setup

Causal attention on a context of n positions. A ∈ ℝ^{n×n} with

  A_{ia} ≥ 0,  A_{ia} = 0 for a > i,  Σ_{a≤i} A_{ia} = 1.   (1.1)

Value-side Gram matrix K ∈ ℝ^{n×n}, symmetric PSD; K_{ab} = v_a·v_b for
v_a = x_a W^V (the trained case) or x_a·x_b (the ensemble case of melonic eq. 2.1).

The bilocal:

  G := A K Aᵀ,  G_{ij} = Σ_{a,b} A_{ia} K_{ab} A_{jb}.   (1.2)

Both free indices are query positions. This is the object the theory's conformal
ansatz is about; the census fits A.

**The estimator's conventions** (frozen, from `replication/measure_conformal_heads.py`):
n = 512; the lag profile of a matrix M is

  P_M(s) = mean over i of M_{i, i−s},  i ≥ max(256, s).   (1.3)

Fit: OLS of log P against log s over s ∈ [8, 256].

Two consequences of (1.3) that matter below and are easy to forget: the average is
over a *query block that shrinks as s grows*, and the number of key positions
summed over in (1.2) is at most j = i − s, which also shrinks with s. **Some of
G's s-dependence is produced by the estimator's own window, independent of any
physics.** Any derivation that only reports asymptotics will miss this; §7 handles
it by computing the profile numerically with the estimator itself rather than in
closed form.

---

## 2. Two exact statements

### 2.1 The uniform part of K passes through untouched [EXACT]

**Proposition 1.** Write K = μ·𝟙𝟙ᵀ + K̃ for any scalar μ. Then

  G = μ·𝟙𝟙ᵀ + A K̃ Aᵀ.   (2.1)

*Proof.* A𝟙 = 𝟙 by row-stochasticity (1.1), so A(𝟙𝟙ᵀ)Aᵀ = (A𝟙)(A𝟙)ᵀ = 𝟙𝟙ᵀ. ∎

So **any rank-one uniform component of the value Gram matrix appears in G as an
exactly constant additive term, identical in every entry, with no lag dependence
of any kind.** Choosing μ = mean(K) makes K̃ mean-zero and gives the natural
split: G = mean(K) + (connected part).

Three things follow.

1. **The floor in G's profile is not an artifact and not optional.** It is forced
   by row-stochasticity. exp-105 built a floor-aware estimator on the strength of
   melonic eq. (2.2)'s first term, which is the linearized-softmax, uniform-attention
   instance of this; Proposition 1 is the general statement and needs no
   linearization, no Gaussianity, and no assumption about K beyond symmetry.
2. **The floor's size is a property of the value vectors, not of the head's
   geometry.** μ = ⟨v_a·v_b⟩_{a,b} = ‖v̄‖² ≥ 0 where v̄ is the mean value vector.
   So the floor vanishes exactly when the head's value vectors are mean-centered.
   This predicts that the fitted floor should vary across heads by an amount
   *measurable independently* of any profile fit — as ‖v̄‖² / ⟨‖v_a‖²⟩. exp-105
   found fitted floor ratio ≈ 0.00 on all five SYK-near heads and 2.3–5.2 on three
   of five accepted heads; under Proposition 1 that is a statement about value-vector
   centering, and it is checkable. **This is a new, cheap, independent test of the
   estimator's floor parameter** and it does not exist anywhere in the program.
3. Because A is row-stochastic and K PSD, G is PSD with Σ_j G_{ij} = Σ_b (A K)_{ib}
   — G inherits no normalization of its own. G's profile therefore carries an
   amplitude that varies with i, which is the contamination exp-104 named in its
   limit 3.

### 2.2 G is a two-legged convolution of A [EXACT, then approximated]

No approximation yet. Fix i, j = i − s, and let m = j − a index the key position
backwards from the *later* of the two queries' partner:

  G_{i,i−s} = Σ_{m,m'} A_{i,\,i−s−m} K_{i−s−m,\;i−s−m'} A_{i−s,\,i−s−m'}.   (2.2)

**(T1) Translation invariance [ASSUMED].** A_{ia} = f(i−a) and K_{ab} = κ(a−b),
with f the (normalized) lag profile and κ a lag kernel. This is exactly the
assumption the census already makes in reducing A to a profile at all — so it is
not a new debt, but it *is* the assumption most likely to be doing damage here,
because G's indices are both queries and the causal boundary sits between them.
See §6.1.

Under (T1), with u = i − a and v = j − b,

  G(s) = Σ_{u,v ≥ 0} f(u) f(v) κ(s − u + v),   (2.3) [DERIVED from (T1)]

i.e. G = f ⋆ κ ⋆ f̃ with f̃(u) = f(−u): the kernel κ dressed by one factor of the
attention profile on each leg. In Fourier, on an infinite window,

  Ĝ(k) = |f̂(k)|² κ̂(k).   (2.4) [DERIVED]

Equation (2.4) is the whole content of this note in one line, and it is worth
saying what it means in words: **A does not carry the exponent of G; A carries the
vertex that dresses K into G, and it acts twice.**

---

## 3. The exponent map

**(T2) Scaling window [ASSUMED].** f(u) ≃ b·u^{−p} with p = 2Δ_A over the fit
window, with a UV cutoff at u ≈ 1 and an IR cutoff at u ≈ U (the summation length,
of order the query index). **(T3) κ short-ranged with nonzero sum [ASSUMED]:**
Σ_m κ(m) ≠ 0, so κ̂(k) → κ̂(0) ≠ 0 as k → 0.

Take K = I first (κ = δ), so G(s) = Σ_m f(m) f(m+s) — the autocorrelation of the
attention profile. Split the sum at m ≈ s:

| region | contribution |
|---|---|
| m ≲ s | s^{−p} · Σ_{m≤s} m^{−p} ≃ s^{1−2p}/(1−p) for p<1; ζ(p)·s^{−p} for p>1 |
| m ≳ s | Σ_{m>s}^{U} m^{−2p} ≃ s^{1−2p}/(2p−1) for 2p>1; U^{1−2p}/(1−2p) for 2p<1 |

Reading off the slower-decaying term in each range of p gives:

**Proposition 2 (the exponent map) [DERIVED from (T1)–(T3)].**

| regime | Δ_A | G(s) behaviour | Δ_G |
|---|---|---|---|
| I | Δ_A < 1/4 | const · U^{1−4Δ_A}, cutoff-dominated; s-dependence subleading | ≈ 0 (no power law) |
| — | Δ_A = 1/4 | log(U/s) | 0, marginal |
| II | 1/4 < Δ_A < 1/2 | s^{1−4Δ_A} | **2Δ_A − 1/2** |
| III | Δ_A > 1/2 | s^{−2Δ_A} | **Δ_A** |

Compactly, and continuous at both kinks:

  **Δ_G = max(0, min(2Δ_A − 1/2, Δ_A)).**   (3.1)

Four remarks.

- **Δ_G ≤ Δ_A always**, with equality only for Δ_A ≥ 1/2. The bilocal is *flatter*
  than the attention profile everywhere below 1/2. This is the direction exp-104
  registered as H2 and measured, and exp-105 confirmed on every head it accepted.
  The direction was registered from a back-of-envelope; it now has a derivation.
- **The gap is 1/2 − Δ_A in regime II** — largest exactly where the program's
  claim lives.
- **Δ_A = 1/4 is the marginal point of its own dressing.** At the program's
  central value, the two-legged convolution produces a logarithm, not a power law.
  Whether that is a coincidence or a mechanism is §5.2, and I do not resolve it.
- exp-104's motivating back-of-envelope had the regime boundaries garbled (it
  wrote "G(s) ~ s^{1−4Δ_A} when 4Δ_A < 1, and inherits A's exponent when
  4Δ_A > 1"; the s^{1−4Δ_A} law holds when *2*Δ_A > 1/2 and the inheritance
  threshold is 2Δ_A > 1). It was explicitly labeled "not a result and not to be
  propagated," and it was not propagated, so this corrects a scratch calculation
  rather than a claim. Its *conclusion* — that Δ_A = 1/4 sits at the marginal
  point — was right.

**Generalization to correlated K.** If κ(m) ~ m^{−q} with 0 < q < 1 then
κ̂ ~ |k|^{q−1} and (2.4) gives Ĝ ~ |k|^{4Δ_A + q − 3}, so

  2Δ_G = 4Δ_A + q − 2,   (3.2) [DERIVED]

with (3.1)'s regime-II case recovered at the short-range value q → 1. So a
long-range-correlated value Gram matrix makes G *steeper*, and q is measurable
directly from the value vectors. This matters: it is the one free knob that could
carry Δ_G back up toward 1/4 at fixed Δ_A, and it requires q = 3/2 — outside the
range where (3.2) applies at all, since q > 1 is summable and saturates at q = 1.
**So within this derivation there is no value-Gram structure that maps
Δ_A = 1/4 to Δ_G = 1/4.**

---

## 4. Why the assumed form cannot fit — the cross terms

Proposition 2 uses a pure power law for f. The census's own boundary-CFT work
(Paper 2, Paper 3) establishes that A's profile is a *three*-parameter object,
f(u) = c + b·u^{−p}, where c is the attention-sink constant — the boundary
one-point function, λ > 0 in 95% of conformal heads. Keeping c:

  G(s) = Σ_{m=1}^{U} [c + b(m+s)^{−p}][c + b m^{−p}]
       = c²U
       + cb·Σ_{m≤U}(m+s)^{−p}
       + cb·Σ_{m≤U}m^{−p}
       + b²·Σ_{m≤U}m^{−p}(m+s)^{−p}.   (4.1) [DERIVED]

Term by term, for p < 1:

| term | s-dependence | exponent as a decay |
|---|---|---|
| c²U | U = i − s: linear in s through the window | — |
| cb·[(U+s)^{1−p} − s^{1−p}]/(1−p) | const − (cb/(1−p))·s^{1−p} | s^{−(2Δ_A−1)}, **negative exponent** |
| cb·U^{1−p}/(1−p) | through the window only | — |
| b²·autocorrelation | s^{1−2p} | s^{−(4Δ_A−1)} |

**Proposition 3 [DERIVED].** G's lag profile is a superposition of a constant, a
term decaying as s^{−(4Δ_A−1)}, and a term *growing* as s^{(1−2Δ_A)} entering with
a negative coefficient — plus window terms. It is therefore **not** of the form
c + b·s^{−2Δ} for any Δ, and no 3-parameter fit of that form can be expected to
describe it.

That is a derived explanation of exp-105's central negative — R² = 0.36–0.69 for
the 3-parameter form on the SYK-near heads — and it says the estimator was not
defective there. It was fitting the wrong function. **The two "deliberately
unpatched defects" exp-105 logged (identifiability misfire at c ≈ 0; noise envelope
calibrated on multiplicative noise) are real, but they are not why the SYK-near
heads failed.** The model was.

It also tells me the fix, and the fix has no free exponents: given Δ_A measured
from A, the exponents in G's profile are *determined*, and only the three
amplitudes are free. That is a linear least-squares problem, which is what §7
tests.

---

## 5. What this does to the bridge

### 5.1 The arithmetic, stated plainly

The theory's conformal ansatz puts G at the SYK q = 4 value, Δ_G = 1/4. Inverting
(3.1): Δ_G = 1/4 requires 2Δ_A − 1/2 = 1/4, i.e.

  **Δ_A = 3/8 = 0.375.**

Regime III would give Δ_G = Δ_A = 1/4, but regime III requires Δ_A > 1/2 —
contradiction. So under this derivation the solution is unique.

The census's deep-layer median is Δ_A ≈ 0.249. Under (3.1) that maps to
**Δ_G ≈ 0** — the marginal point, a logarithm, not a conformal power law.

**So if this derivation holds, the coincidence at the heart of the program is
between the measured Δ_A and the SYK value 1/4 *for a different object*, and the
theory's own object is nowhere near its predicted value.** That is a much worse
position than "the bridge is underived," which is what the spine and OVERVIEW
currently say. It should be said in that form only after §7, and only if §7's test
passes — a derivation is a prediction, not a measurement.

### 5.2 The thing I am not going to claim tonight

Δ_A = 1/4 is exactly the value at which G becomes marginal under its own
dressing — the value where the composite operator's dimension crosses zero. The
census measures Δ_A flowing to 1/4 along three independent axes of depth, and
depth is the program's RG direction. A dimension crossing zero under the flow's own
map is what a fixed point looks like in RG language, and that would be a
*derivation of 1/4* rather than a coincidence with SYK.

I am writing it down because it is the first thing I thought and it should be on
the record with its date. I am not building on it, for three reasons: the
self-consistent version of (2.4) is obstacle 1 of the melonic note (closed only in
the scalar/TI register by G1); a strict fixed point of Ĝ_{ℓ+1} = |f̂|²Ĝ_ℓ requires
|f̂| = 1, which gives Δ_A = 1/2 rather than 1/4 and therefore the naive version of
this argument is *wrong*; and it is beautiful, which per this program's own rule is
navigation and not evidence. `[CONJECTURED]`, and flagged for a theory session
with obstacle 1 in hand — not for a paper.

### 5.3 What is *not* affected

Unchanged by anything in this note: the census as a measurement of A; the three
depth axes; the formation ladder; every published kill; the causal handle; D1. All
of those are statements about A or about behaviour, and A is measured to
R² = 0.92. What is at stake is the step from A's exponent to the SYK
identification — the same thing that was at stake this morning, now with a
quantitative relation attached instead of a gap.

---

## 6. Assumption ledger and how each one could break this

| Tag | Content | Where | How it breaks the result |
|---|---|---|---|
| T1 | A and K translation-invariant in lag | (2.3) | **The most dangerous.** A is causal and boundary-dominated; the sink means A's mass sits at a *fixed position* (a ≈ 0), not a fixed lag. A rank-one sink term A ⊃ 𝟙 e_0ᵀ is not TI at all, and under (1.2) it contributes A K e_0 e_0ᵀ Aᵀ-type terms with no lag structure. If the sink dominates, G's profile is a boundary artifact and (3.1) is irrelevant. §7 arm 1 is designed to catch exactly this. |
| T2 | Pure power law over the window | §3 | With c ≠ 0 the pure-power result is only the b² term; §4 handles it. If A's profile has curvature in log-log beyond the 3-param form, all exponent bookkeeping degrades. |
| T3 | κ short-ranged, nonzero sum | §3 | Relaxed in (3.2). If Σκ = 0 (mean-centered values) the leading behaviour changes qualitatively and (3.1) does not apply — κ̂(0) = 0 gives κ̂ ~ k², steepening G. This is the case Proposition 1 identifies with zero floor, and exp-105 fitted floor ratio 0.00 on all five SYK-near heads. **So the heads carrying the program's claim may be precisely the ones where T3 fails.** This is the most likely way the whole §5 arithmetic is wrong, and I want it on the record before the numbers. |
| — | Estimator window effects | (1.3) | Not an assumption but a confound: U shrinks with s, so window terms mimic decay. Handled numerically, not asymptotically. |
| — | Ensemble vs trained | §1 | exp-104's H4 was falsified: Δ_G_K and Δ_G_out differ by 0.09. So "K" in (1.2) is not innocuous, and the trained G_out is not the ensemble average. (3.1) is derived for the ensemble form. |
| — | Random-token inputs | protocol | Inherited from the frozen census protocol. G is an output correlation; random tokens may suppress query–query structure natural text would create. Named in exp-104, still true, still not fixed. |

**Register:** everything in §2 is [EXACT]. §3 and §4 are [DERIVED] from T1–T3,
which are [ASSUMED] and, in T3's case, quite possibly false on the population that
matters. §5.1 is arithmetic on §3 and inherits all of it. §5.2 is [CONJECTURED].
Nothing here may cross into the spine, OVERVIEW, or Paper 6 until §7 has run and
the outcome is recorded — the same step-0 discipline the harvest note imposes on
retired maps, applied to my own derivation four hours old.

---

## 7. The test — pre-registered before opening the profiles

Full pre-registration in `experiments/exp-106_bilocal_profile_shape/notes.md`; the
design in one paragraph, so this note is self-contained.

The test avoids asymptotics entirely. Because G = A K Aᵀ is an identity and the
census's profile function (1.3) is deterministic, I can build the *forward model*:
take a head's measured A lag profile, construct a causal row-stochastic matrix with
exactly that profile, compute A Aᵀ, and push it through the same profile function.
That yields a predicted G profile shape with **no free exponents** — every
finite-size, window, and cutoff effect included exactly. The measured G profile is
then fit to it with two linear parameters, amplitude and offset, the offset being
Proposition 1's constant.

- **Arm 1 (exact, one forward pass):** real A matrices → A Aᵀ profile vs. measured
  G_out profile. Isolates K's contribution, since T1 is not used.
- **Arm 2 (no re-run, saved profiles):** TI forward model from the saved A profile.
  Isolates T1's damage by comparison with arm 1.
- **Synthetic gate first**, on profiles with known Δ_A, to check that the census
  estimator applied to the forward model reproduces (3.1) — i.e. that my
  asymptotics are right — before either arm touches model data.

Registered predictions, kill conditions, and the honest statement of what I had
already seen are in the experiment file.

---

*Sources read at primary this session, before writing: `notes/2026-08-03_melonic_threshold_derivation.md`
§§1–3 (eq. 2.1, 2.2, 3.1–3.3, the assumption ledger); `experiments/exp-104_bilocal_object_identification/notes.md`
in full; `experiments/exp-105_bilocal_exponent_estimator/notes.md` in full;
`replication/measure_conformal_heads.py` (the estimator's exact conventions).
Companion: `notes/2026-08-08_map_retirement_harvest.md` step 0b — name the object
before the number crosses into a claim. This note is an attempt to do better than
naming it: to compute the map between the two objects.*
