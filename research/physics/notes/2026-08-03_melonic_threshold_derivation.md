# The Melonic Threshold from Corpus Statistics — Piece 2 Derivation

*Ariel — 2026-08-03, ~12:15 AM – (session ongoing). Solo theory session, gifted by
Eldon at the close of the August 2 late-night session. Launch document:
`2026-08-03_piece2_theory_launch.md`. Companion data note:
`2026-08-02_arrest_and_arrival.md`.*

*Discipline: every displayed claim is tagged **[DERIVED]** (follows from named
assumptions by calculation shown or cited), **[ASSUMED]** (named assumption,
scope stated), **[VERIFIED-LIT]** (external theorem, verified against source this
session), or **[CONJECTURED]** (stated precisely, not established). Where a step
is heuristic I say so in place. Beauty is navigation, not evidence.*

---

## 0. Target and result summary

**Target (from the launch doc):** derive, from the statistics of the training
corpus, the condition under which trained attention flows to the chaotic
(q=4, Δ=1/4) fixed point versus arresting — the melonic-dominance threshold as a
computable functional of the data distribution.

**What this session establishes** (details and honest limits in §8–§9):

1. The disorder-averaged linearized-attention kernel is, *exactly at the level of
   its cumulant structure*, a **low-rank SYK model** in the sense of Kim–Cao–Altman
   (arXiv:1910.10173): the fluctuating part of the bilocal kernel decomposes into
   R independent "boson" modes, where **R = rank(K δK)** and the coupling
   eigenvalues are **spec(δK^{1/2} K δK^{1/2})** — K the token Gram matrix, δK its
   doubly-centered form. The corpus enters the effective action through this
   spectrum and nothing else (at the order computed). [DERIVED, §2–§3]
2. The low-rank SYK classification then supplies the phase structure
   [VERIFIED-LIT, §4]: sub-extensive rank → free; extensive rank with generic
   positive spectrum → Δ = 1/2 almost-Fermi-liquid **with a T-breaking
   condensate** (= arrest AT a protected lesser solution — the two-permanences
   language becomes a theorem-shaped statement here); extensive rank with
   degenerate-edge spectrum → maximally chaotic with Δ ∈ (1/4, 1/2) given by a
   **closed formula in the rank ratio**; Δ → 1/4 only as the mode count per
   position diverges.
3. Because the induced coupling spectrum is positive semidefinite [DERIVED], the
   deep-layer conformal dimension is predicted to approach **1/4 from above**
   along the formation ladder, and Δ < 1/4 (Class IV) is excluded for this
   mechanism. This retrodicts the UV-side Δ values of exp-097/098 and the
   deep-head Δ values of the C-NAT family. [DERIVED + data check, §4.4]
4. An always-present induced **q=2 channel** [DERIVED, §3.4] means the q=4
   conformal regime is a *window in scale*, not a terminus — matching the
   verified SYK₄+SYK₂ literature (Lunkin–Tikhonov–Feigel'man) and our prethermal
   arrest phenomenology.
5. The **corpus functional**: a scale-resolved window parameter 𝒲(ℓ) built from
   the spectrum of the corpus-induced coupling operator (two candidate
   definitions given; deciding between them is the named next obstacle). The
   formation-ladder kills (exp-062/084/085/091/094/097/098) map onto specific
   degradations of 𝒲(ℓ). [PARTIALLY DERIVED / CONJECTURED, §5]
6. A **registerable prediction pipeline** for exp-099: compute 𝒲 from a candidate
   rung corpus before training; predict the deep-layer Δ via the Class III
   formula; predict arrival/arrest ordering across rungs monotone in 𝒲.
   Numerical values for C-alien vs TinyStories in §6; exp-099 conditional
   predictions in §7.

---

## 1. Setup and the score field

### 1.1 Definitions

Context of n tokens with embeddings x_a ∈ ℝ^d, a = 1..n. Weight matrices
W^Q, W^K ∈ ℝ^{d×d_k} with i.i.d. entries N(0, σ_Q²/d), N(0, σ_K²/d).
Scores s_{ia} = (x_i W^Q)·(x_a W^K)/√d_k. Attention α_{ia} = softmax_a(s_{ia}).

Gram matrix K_{ab} = x_a·x_b. Centering projector Π = I − (1/n)𝟙𝟙ᵀ.
Doubly-centered kernel δK := Π K Π, i.e. δK_{ab} = (x_a − x̄)·(x_b − x̄).

**Lemma 1.1 [DERIVED].** δK is positive semidefinite, with
rank(δK) ≤ min(n−1, d). *Proof:* δK is the Gram matrix of the centered
embeddings. ∎

*(This corrects an ambiguity in the March 9 calculation, which left the
signature of δK unexamined. The PSD property matters below: it fixes which
low-rank SYK classes are reachable.)*

### 1.2 Score covariance — exact

**Lemma 1.2 [DERIVED].** For any fixed tokens,
E[s_{ia} s_{jb}] = (σ_Q²σ_K²/d²)(x_i·x_j)(x_a·x_b).

*Proof.* E[(x_i W^Q)_μ (x_j W^Q)_ν] = (σ_Q²/d)(x_i·x_j)δ_{μν}, similarly for W^K;
the 1/d_k from the two score normalizations cancels the d_k-fold μ-sum. ∎

Write c₀ := σ_Q²σ_K²/d². For the *centered* scores
δs_{ia} := s_{ia} − (1/n)Σ_c s_{ic}:

  E[δs_a(x_i) δs_b(x_j)] = c₀ (x_i·x_j) δK_{ab}.   (1.1) [DERIVED]

The covariance **factorizes**: query-sector kernel ⊗ key-sector centered kernel.

### 1.3 Emergent Gaussianity

s_{ia} is a sum of d_k products of independent Gaussians; its cumulants of order
2m scale as d_k^{1−m}.

**(A3) Large-d_k assumption [ASSUMED]:** we work in the limit where the score
field is a Gaussian process with covariance (1.1). Corrections are O(1/d_k) and
systematically computable; they are *not* included below.

### 1.4 Linearized softmax

**(A1) Linearization [ASSUMED]:** |δs| ≪ 1 (valid for σ_Q σ_K ≪ 1; empirically
the lazy/early-training regime). Since Σ_a δs_{ia} = 0 exactly,

  α_{ia} = (1/n)(1 + δs_{ia}) + O(δs²).   (1.2) [DERIVED from (A1)]

**(A2) Weight-ensemble Gaussianity [ASSUMED]:** exact at initialization; for
trained ensembles this is the annealed-Gaussian surrogate assumption. §5
replaces the i.i.d. covariance with corpus-conditioned structure; the Gaussian
*shape* of the ensemble remains assumed. (Gurau's Gaussian universality
[VERIFIED-LIT, PoS(2019)376:222 and refs. therein] softens this: non-Gaussian
i.i.d.-type disorder reduces to Gaussian with modified parameters at leading N.
What universality does NOT cover is *correlated/structured* disorder — that is
exactly the regime §3–§5 addresses through the rank/spectrum route.)

---

## 2. The bilocal observable and its exact cumulant structure

### 2.1 The kernel observable

Following Ageev's construction as used in the March 9 note, the attention
two-point kernel between query points x₁, x₂:

  H(1,2) = (σ_V²/d) Σ_{a,b} α_a(1) α_b(2) K_{ab}
         = w Σ_{a,b} K_{ab} [1 + δs_a(1) + δs_b(2) + δs_a(1)δs_b(2)],   (2.1)

with w := σ_V²/(n²d). [DERIVED from (1.2)]

H is an (affine + quadratic) functional of the Gaussian field φ = δs. All its
joint cumulants are therefore *exactly computable* by Wick calculus — the theory
of H is the theory of quadratic forms of a Gaussian process. This is the
rigorous replacement for the order-by-order β-expansion of March 9.

### 2.2 Mean (the q=2 renormalization)

  E[H(1,2)] = w Σ_{ab} K_{ab} + w c₀ K₁₂ · Tr(K δK).   (2.2) [DERIVED]

*The first term is the uniform-attention (bare) propagator G₀. The second is a
correction proportional to the kernel itself, with coefficient c₀ Tr(KδK): a
kernel-multiplicative (mass-like / hopping-like) renormalization. This is a
**q=2-type channel**: self-energy linear in the propagator.*

### 2.3 Covariance (the q=4 vertex) — exact

**Proposition 2.1 [DERIVED].** The connected two-point function of H is

  Cov(H(1,2), H(3,4)) = w²c₀² [(x₁·x₃)(x₂·x₄) + (x₁·x₄)(x₂·x₃)] · Ω
                        + w²c₀ (κᵀ δK κ) [(x₁+x₂)·(x₃+x₄)],   (2.3)

where Ω := Tr[(K δK)²], κ_a := Σ_b K_{ab}.

*Proof sketch (indices checked explicitly this session).* Write
H = h₀ + ℓᵀφ + φᵀQφ with Q_{12}[(p,a),(q,b)] = (w/2)(δ_{p1}δ_{q2}+δ_{p2}δ_{q1})K_{ab}
and ℓ_{12}[(p,a)] = w(δ_{p1}+δ_{p2})κ_a. For Gaussian φ:
Cov = ℓ₁₂ᵀ C ℓ₃₄ + 2Tr(Q₁₂ C Q₃₄ C). The trace evaluates with C from (1.1):
the query-sector deltas produce exactly the two SYK pairings
(x₁·x₃)(x₂·x₄) + (x₁·x₄)(x₂·x₃), and the key-sector contraction produces
Σ_{abcd} K_{ab} δK_{bc} K_{cd} δK_{da} = Tr[(KδK)²]. ∎

Two structural facts, both new relative to the March note:

- **The data-geometry factor Ω has closed form Tr[(KδK)²]** — previously an
  unevaluated four-index sum. It is manifestly ≥ 0 (trace of the square of a
  matrix with nonnegative real spectrum; see Lemma 3.1).
- The first term of (2.3) has *exactly* the SYK bilocal q=4 vertex pairing
  structure (as March 9 found at order β⁴); the second term is a distinct
  additive channel with one-particle (q=2-like) external structure, previously
  lumped in at order β².

### 2.4 Higher cumulants — the ring structure

**Proposition 2.2 [DERIVED, structure; prefactors not tracked].** The m-th joint
cumulant of {H(2i−1,2i)} is a sum over cyclic ("ring") contractions,

  κ_m ∝ w^m c₀^m Tr[(K δK)^m] × [cyclic products of query-sector kernels],

plus lower-channel (linear-leg) terms. The full interaction hierarchy of the
bilocal theory is therefore governed by the **spectrum of T := K δK**.

*This is the standard cumulant formula for quadratic forms of Gaussians,
κ_m(φᵀQφ) = 2^{m−1}(m−1)! Tr[(QC)^m], applied channel-by-channel.*

---

## 3. Structural identification with low-rank SYK

### 3.1 The mode decomposition

**Lemma 3.1 [DERIVED].** T = KδK has real, nonnegative spectrum, and its nonzero
spectrum equals that of the PSD matrix M := δK^{1/2} K δK^{1/2}.
*Proof:* similarity/commutation of spectra for products of PSD matrices. ∎

Diagonalize δK = Σ_α ν_α v_α v_αᵀ (ν_α ≥ 0, α = 1..R₀, R₀ = rank δK). Then the
score field decomposes *exactly* as

  δs_a(x) = Σ_α √(c₀ ν_α) v_α(a) χ_α(x),   (3.1) [DERIVED]

where χ_α are i.i.d. Gaussian processes over query points with covariance
Cov(χ_α(x), χ_β(y)) = δ_{αβ} (x·y). (Check: reproduces (1.1).)

Substituting into (2.1), the *fluctuating quadratic part* of the kernel is

  H_quad(1,2) = w c₀ Σ_{α,β} M_{αβ} χ_α(1) χ_β(2),
  M_{αβ} := √(ν_α ν_β) (v_αᵀ K v_β),   (3.2) [DERIVED]

and in the eigenbasis of M (eigenvalues μ_α ≥ 0):

  H_quad(1,2) = w c₀ Σ_{α=1}^{R} μ_α ξ_α(1) ξ_α(2),   R = rank(M) = rank(KδK),  (3.3)

with ξ_α i.i.d. Gaussian processes whose covariance is the **bare propagator**
(x·y).

**(A4) Kernel-propagator identification [ASSUMED]:** the token inner product is
the bare bilocal propagator, x_i·x_j = G₀(x_i,x_j) (exact for Ageev's
Fourier-feature embeddings; assumption for learned embeddings).

### 3.2 The correspondence

Equation (3.3) is *structurally identical* to the low-rank SYK model of
Kim–Cao–Altman (KCA) [VERIFIED-LIT this session: arXiv:1910.10173,
PRB 101, 125112 (2020)], where

  H_KCA = interactions with coupling tensor J_{ij,kl} = Σ_{α=1}^{R} λ_α u^α_{ij} u^α_{kl},

a rank-R coupling in pair space, solved via one boson per mode with the
classification driven by the eigenvalue density ρ(λ) and the rank ratio
γ := R/N.

| Low-rank SYK (KCA) | Linearized attention (this note) |
|---|---|
| N Majorana fermions | n token positions |
| Fermion propagator G(τ) | Bilocal attention kernel G(x_i, x_j) |
| R boson modes, couplings λ_α | R = rank(KδK) kernel-fluctuation modes, couplings μ_α = spec(δK^{1/2}KδK^{1/2}) |
| ρ(λ): eigenvalue density of coupling matrix | spectral density of M — a **corpus functional** |
| γ = R/N | γ_eff = R_eff/n |
| q=2 random-mass perturbations | induced q=2 channel, coefficient c₀·Tr(KδK) (2.2) plus κᵀδKκ channel (2.3) |

**Status of the identification [DERIVED at the level of effective-action /
cumulant structure; NOT an operator-level equivalence].** What is established:
the generating functional of the attention bilocal's fluctuations has the same
mode-decomposed bilinear structure, the same vertex pairing structure, and the
same two-channel (q=2 + q=4) skeleton as the KCA model's G–Σ effective action.
What is *not* established: (i) that the self-consistent (dressed) SD equations
of deep/iterated attention coincide with KCA's SD equations including the boson
back-reaction [the dressing step replaces the ξ covariance (x·y) by the running
G — plausible and standard, but the loop closure is **the next unproven step**];
(ii) quenched = annealed for structured disorder (self-averaging; see (A5)
below).

**(A5) Self-averaging / mode-delocalization [ASSUMED, NAMED OBSTACLE].** The KCA
solution requires the coupling modes u^α to be generic (delocalized) so the
fermion self-energy self-averages. Our modes v_α are corpus-determined. If they
are *localized* (e.g., rigid templates concentrating modes on few
positions/types), the SD description can fail toward integrable channels even at
nominally adequate rank. A precise delocalization condition (an incoherence
bound on v_α, in the spirit of eigenvector delocalization in RMT universality)
is required to make §4 rigorous. This is the second named obstacle.

### 3.3 Where the SD equations come from here

The single-layer calculation above has no self-consistency loop: H is an exact
quadratic functional of Gaussian scores. The loop enters through **depth /
recurrence**: layer ℓ+1's kernel statistics are functionals of layer ℓ's dressed
kernel (empirically: depth = RG flow, our depth-convergence results; exp-089
inference-time recurrence). The constitutive form (July 21 note): impose the
conformal ansatz on the dressed kernel and demand consistency of (2.2)–(2.3)
with the dressing. At that level the two channels give:

  Σ₂(x,y) ∝ c₀ · s₁[M] · G(x,y)   (q=2; s₁ = Σμ_α, spectral first moment)
  Σ₄(x,y) ∝ c₀² · s₂[M] · G(x,y)³  (q=4; s₂ = Σμ_α² in the scalar approximation)

[DERIVED in scalar approximation — i.e., treating the mode structure as a
scalar coupling. The honest refined statement: Σ₄ carries the operator
structure of M, and the *scalar* reduction is valid exactly when (A5) holds and
γ_eff is large. Otherwise the mode structure must be kept — which is the KCA
regime, handled by their classification.]

### 3.4 The window in scale — q=2 always wins the deep IR

The q=2 channel coefficient c₀ Tr(KδK) is strictly positive whenever attention
is non-uniform at all (Tr(KδK) = Σμ_α > 0 unless δK = 0). By the verified
SYK₄+SYK₂ results [VERIFIED-LIT this session: Lunkin–Tikhonov–Feigel'man,
PRL 121, 236601 (2018); PRL 125, 196602 (2020); García-García et al.,
arXiv:1707.02197]:

- At mean field the quadratic term dominates the deep IR: G ~ 1/t (FL).
- The q=4 conformal solution G ~ t^{−1/2} holds in an **intermediate window**,
  extended by soft-mode fluctuations, terminating at t ~ J/Γ² where Γ is the
  quadratic strength.

**Consequence [DERIVED given the identification]:** for trained attention the
q=4 conformal regime is a *window in scale* between the UV (bare-term-dominated)
and the deep IR (q=2/FL-dominated). "Arrival" empirically means: the window
covers the measured scales (the context/arc scales probed by the two-point
fits). This is the precise sense of the Conformal Window of Attention, and it
gives the q=2 plateau (training-time prethermal arrest) a mechanism: the q=2
channel is *always induced*; the question is only whether the q=4 window opens
between the scales it controls.

---

## 4. The threshold: applying the low-rank classification

### 4.1 The KCA classification [VERIFIED-LIT, extracted from the paper this session]

For coupling spectrum ρ(λ) with rank R = γN (extensive):

| Class | Spectrum shape | Fermion Δ | Chaos | Protection |
|---|---|---|---|---|
| I | positive, vanishing edge (ρ ~ (λ_max−λ)^η, η>0) | 1/2 | non-maximal, λ_L ~ T^{1+η} | **T-breaking boson condensate at T < T_c** |
| II | positive, non-vanishing edge (−1<η≤0) | 1/2 | ~T | condensate at T=0 |
| III | degenerate top (ρ = c₀δ(λ−λ_max)+…) | Δ(γc₀) ∈ (1/4, 1/2) | **maximal, 2πT** | none |
| IV | λ_max ≤ 0 | Δ(γ) ∈ (0, 1/4) | maximal | none |

with the closed-form interpolation (their eqs. 31/36):

  γ (or γc₀) = (2Δ−1)(sec 2πΔ − 1)/(8Δ−2),   (4.1)

Δ ∈ (0, 1/4) for Class IV, Δ ∈ (1/4, 1/2) for Class III; Δ → 1/4 as the
argument → ∞ (SYK₄ recovered at super-extensive rank); Δ → 1/2 as γc₀ → 0
(SYK₂ value).

### 4.2 Which classes can attention reach?

**Proposition 4.1 [DERIVED].** The induced coupling spectrum {μ_α} is
nonnegative (Lemma 3.1). Therefore **Class IV is excluded** for the linearized
attention mechanism: λ_max > 0 always (unless the coupling vanishes entirely).
The reachable phases are:

- **Class I/II** (generic positive spectrum, non-degenerate edge):
  Δ = 1/2 almost-Fermi-liquid, with a condensed boson mode.
- **Class III** (degenerate/flat top of the μ-spectrum):
  maximal chaos, Δ ∈ (1/4, 1/2) given by (4.1) with γc₀ = (fraction of modes at
  the top edge) × R/n.
- Super-extensive effective rank with balanced spectrum: Δ → 1/4⁺.

**Prediction P-A [DERIVED, given the identification]:** deep-layer conformal
dimensions approach **1/4 from above** as world richness grows; the mechanism
cannot produce Δ < 1/4 in the deep population.

### 4.3 The arrest taxonomy, derived

Combining §3.4 and §4.2, the five empirically observed stations map to
mechanisms:

| Station (measured) | Mechanism (this derivation) |
|---|---|
| UV arrest, Δ ~ 0.6–1.2 (exp-097/098) | window empty from the UV side: coupling magnitude too small for Σ to overtake G₀⁻¹ at accessible scales (**arrest OF the flow**; strong-coupling condition fails) |
| q=2 plateau, Δ = 0.50 (exp-086 training trajectory) | Class I/II physics and/or the always-induced q=2 channel dominating early (**arrest AT a protected solution**; the KCA condensate is literally the protecting structure — a T-breaking bilinear condensate, i.e. a conservation-law-like self-model) |
| Arrival, Δ → 0.25⁺ (natural language, deep layers) | Class III with large γc₀ / effectively super-extensive rank |
| Trivial Δ → 0 (backbone collapse, exp-097/098 L0) | **not covered by this derivation** — see §4.4 |
| Substrate 0.169 (untrained) | bare/disordered value — outside the trained-flow analysis |

The two-permanences distillation (Aug 2 note) now has a concrete instance
inside the model class: **arrest-AT is protected by the Class I/II condensate
(an emergent conserved structure); arrival is the unprotected self-consistent
solution; the window closes in the deep IR because the q=2 channel (a
protection-generating channel) is never exactly zero.** [DERIVED-level for the
model class; the cosmological analogy remains interpretive and stays in the
Aug 2 note.]

### 4.4 Data checks (retrodictions, not fits)

- exp-097 deep/mid-layer Δ ∈ [0.61, 1.2]; exp-098 Δ_med ∈ [0.609, 0.845]; C-NAT
  deep heads (exp-096 s0 tonight: 0.61–1.20 at L3–L5, backbone excluded) — all
  **above 1/4**, consistent with P-A. No trained corpus in the series has
  produced a deep-layer population converging to Δ < 1/4.
- The backbone population (L0, Δ ≈ 0.07–0.16 when present) sits *below* 1/4 —
  by P-A it cannot be this mechanism. Consistent with its empirical
  separability (exp-091 vs 097/098: backbone and UV arrest dissociate). The
  backbone requires a separate account (plausibly the induced q=2/hopping
  channel with its additive external structure (2.3), which has different
  scaling — **not derived tonight**, flagged).
- GPT-2's precision heads (L6H4 Δ = 0.2499): approaching 1/4 from above within
  measurement error — consistent; the measured median 0.2493 slightly below 1/4
  is within the resolution at which (A1)–(A5) corrections (finite d_k, causal
  boundary, non-linearized softmax) are expected to act. Not evidence for or
  against at current precision.

---

## 5. The corpus functional

### 5.1 What feeds the coupling

At the order computed, the corpus enters through the spectrum {μ_α} of
M = δK^{1/2} K δK^{1/2} — i.e. through the geometry of token embeddings *within
the context*. Two distinct functionals of the corpus govern the two gates:

- **Coupling magnitude (UV gate):** 𝒥 ∝ c₀² s₂[M] = c₀² Σ_α μ_α² = c₀² Tr[(KδK)²],
  normalized per mode (see §6 for the operational normalization). Controls
  whether the window opens at all. Failure → UV arrest.
- **Window parameter (chaos gate):** 𝒲 := γ_eff·c₀_eff — effective rank ratio of
  the top of the μ-spectrum:

    γ_eff := R_PR/n,  R_PR := (Σ_α μ_α)²/(Σ_α μ_α²)   (participation ratio),
    c₀_eff := fraction of spectral weight at the top edge (Class III weight).

  Primary operational choice: **𝒲 = R_PR/n** (ε-free). Refinement when the edge
  structure is resolvable: 𝒲 = N_top/n. Predicted deep-layer dimension:

    **𝒲 = (2Δ−1)(sec 2πΔ − 1)/(8Δ−2),  Δ ∈ (1/4, 1/2).**   (5.1) [CONJECTURED
    quantitatively; DERIVED modulo (A5), the dressing-loop closure (§3.3), and
    the Class III edge idealization]

### 5.2 Ordering and the scale-resolved functional

A single-context spectral functional of the one-hot kernel is invariant under
within-context permutations — but ordering is empirically load-bearing
(exp-091/093/094). Resolution: the flow is over *scales*; the coupling feeding
the flow at scale ℓ is carried by score fluctuations between positions at
separation ~ℓ. Two candidate scale-resolved definitions:

- **(F1) Within-context, band-restricted:** spectrum of the coupling operator
  restricted to position pairs at separation ~ℓ. Ordering-sensitive because the
  band structure of K is.
- **(F2) Across-context covariance:** the corpus-ensemble covariance of the
  centered-kernel process at separation ℓ,
  M^{(ℓ)}_{ab} := Cov_w(δK^w_{a,a+ℓ}, δK^w_{b,b+ℓ}) — "the corpus fluctuations
  are the disorder." PSD by construction; ordering-sensitive; directly
  generalizes the quenched-disorder reading.

**Named obstacle 3 [OPEN]:** the derivation in §2–§3 is at fixed context and
does not by itself decide whether (F1) or (F2) (or a combination — annealed vs
quenched over contexts) enters the dressed SD equations. Deciding this is a
well-posed calculation: carry the corpus average through the replica/annealed
treatment of §2 with context-dependent K_w. **This is the precise next
unproven step of the program.** Tonight's numerics compute (F2) (plus the
context-level M-spectrum), because (F2) is the one that makes the exp-085 kill
intelligible — see next.

### 5.3 The formation ladder in this language [retrodictive, qualitative]

- **exp-062 (two-point MI fails), exp-085 (generator fingerprint fails):**
  two-point/mean statistics fix E_w[δK] — the *mean* of the kernel process.
  The coupling is fed by the *fluctuation spectrum* (F2). A generator can match
  means (even overshoot long-range MI) while its kernel-fluctuation rank at arc
  scale collapses. Formally: the disorder is the covariance, not the mean.
- **exp-084 (PCFG fails):** no persistent world state → kernel fluctuations at
  large ℓ decorrelate → M^{(ℓ)} magnitude dies with ℓ → window never opens
  beyond sentence scale.
- **exp-091/093/094 (ordering ladder):** shuffling at block size B destroys
  cross-block kernel covariance → M^{(ℓ)} truncated at ℓ ≳ B → flow arrests at
  the corresponding scale/depth; matches the monotone n_deep recovery
  (quarter < half < full).
- **exp-097/098 (alien world):** world state space of 16 states, ~33 templates,
  ~15 names → kernel-fluctuation spectrum low-rank and edge-degenerate at small
  weight; magnitude small → UV arrest, and predicted 𝒲 far below the natural
  corpus. §6 quantifies.
- **exp-096 (anonymization, closed tonight):** removing cross-story name
  identity leaves within-story kernel structure (the process (F2) at arc scale)
  largely intact → backbone preserved, Δ_med ≈ 0.15, n_deep cost small (4 vs
  5–7) — consistent with a modest reduction of arc-scale fluctuation rank.

---

## 6. Numerics — corpus functional for C-alien vs TinyStories

*(computed this session; see §6.2 for results — script:
`research/physics/theory/corpus_functional.py`)*

### 6.1 Operational definitions

- Word-level types (whitespace tokenization, lowercased, punctuation split) —
  corpus-intrinsic, no learned embedding. **(A7) [ASSUMED]:** one-hot type
  kernel as UV proxy for the bare token kernel: K_{ab} = δ(type_a, type_b).
  Caveat named: learned embeddings deform the spectrum; the dose-response
  ordering is expected robust, absolute numbers are proxy-level.
- Contexts: concatenated stories to n = 512 words, W ≥ 300 contexts per corpus.
- Per-context spectral functional: μ-spectrum of M = δK^{1/2} K δK^{1/2};
  report R_PR/n and normalized s₂.
- Scale-resolved (F2): spectrum of M^{(ℓ)} at ℓ ∈ {1, 2, 4, …, 256}; report
  participation ratio PR(ℓ)/(n−ℓ) and magnitude tr M^{(ℓ)}/(n−ℓ).
- Prediction: Δ_pred from (5.1) using the arc-scale 𝒲.

### 6.2 Results

*(filled in after the run — see below)*

---

## 7. exp-099 — the registerable prediction

*(filled in after §6; predictions conditional on the rung designs, stated
before any training data exists)*

---

## 8. Assumption ledger

| Tag | Content | Where it enters | Status |
|---|---|---|---|
| A1 | Linearized softmax, |δs| ≪ 1 | (1.2), everything downstream | ASSUMED; full softmax adds G^{2k} vertices, RG-irrelevant in IR by power counting [ASSUMED, standard] |
| A2 | Gaussian weight ensemble | disorder average | ASSUMED (exact at init; Gurau universality covers i.i.d. non-Gaussian; structured case handled via rank/spectrum route) |
| A3 | Large d_k (score Gaussianity) | §1.3 | ASSUMED; corrections O(1/d_k) |
| A4 | Token kernel = bare propagator | §3.1, SD equations | ASSUMED (exact for Fourier features) |
| A5 | Mode delocalization / self-averaging | §3.2, KCA applicability | ASSUMED — **named obstacle 2** |
| A6 | Statistical stationarity for conformal ansatz | §3.3 | ASSUMED |
| A7 | One-hot type kernel as UV proxy | §6 | ASSUMED (numerics only) |
| — | Dressing-loop closure (depth recursion = KCA SD system) | §3.3 | **named obstacle 1 — next unproven step** |
| — | (F1) vs (F2) corpus average | §5.2 | **named obstacle 3** |

## 9. What this is and is not

**Is:** a derivation, at the effective-action/cumulant level with assumptions
named, that (i) linearized attention's disorder structure *is* low-rank SYK
with corpus-determined rank and spectrum; (ii) the reachable phases are
Class I/II/III (never IV) — so deep-layer Δ approaches 1/4 from above or
arrests at 1/2 / in the UV; (iii) the q=4 regime is a window in scale;
(iv) a computable corpus functional with a closed-form Δ prediction, wired to a
pre-registerable pipeline for exp-099.

**Is not:** a proof. The three named obstacles (dressing-loop closure;
delocalization condition; (F1)/(F2) decision) are each well-posed and each
could break the quantitative claim (5.1) while leaving the qualitative
structure (two channels, positive spectrum, window in scale, rank-gated chaos)
intact. A kill of (5.1) by exp-099 would be a result: it would localize the
failure to one of the named assumptions.

---

*Session continues in §6–§7 with the numerics.*
