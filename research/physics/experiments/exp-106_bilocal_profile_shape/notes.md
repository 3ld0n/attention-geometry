# exp-106 — G's lag profile from A's, as a forward model with no free exponents

*Pre-registration: this file is committed BEFORE any measurement script exists and
BEFORE any G lag profile is opened. Results appended afterward.*

*Ariel — August 8, 2026, Saturday night, Cursor session with Eldon.*

---

## Why this experiment exists, and how it differs from the queued version

The physics room queue's exp-106 was specified as: *"characterize G's lag-profile
**shape** with **no assumed form**: log-log curvature, two-power-law and
broken-power-law nulls, boundary contamination,"* plus two estimator fixes carried
over from exp-105.

**This supersedes that design, and the reason is a derivation rather than a
preference.** `notes/2026-08-08_bilocal_from_attention_derivation.md` (written
tonight, before this file) establishes that G's profile does not have to be
characterized empirically, because G = A K Aᵀ is an identity: given A, G's profile
is *determined*, up to K. Two exact results and one derived map:

- **Proposition 1 [EXACT].** For any row-stochastic A and any μ,
  G = μ𝟙𝟙ᵀ + A(K − μ𝟙𝟙ᵀ)Aᵀ. The uniform part of the value Gram passes through
  untouched, so G's floor is forced by row-stochasticity and equals ‖v̄‖².
- **(2.3)/(2.4) [DERIVED].** Under lag-translation-invariance,
  G = f ⋆ κ ⋆ f̃, i.e. Ĝ(k) = |f̂(k)|²κ̂(k). A supplies the vertex, twice; it does
  not supply G's exponent.
- **Proposition 2 [DERIVED].** Δ_G = max(0, min(2Δ_A − 1/2, Δ_A)), so Δ_G ≤ Δ_A
  with the gap 1/2 − Δ_A in the regime the program occupies, and Δ_A = 1/4 is the
  marginal point where G's decay becomes logarithmic.

So the queued shape characterization is still what gets done — but against a
*predicted* shape rather than a menu of nulls, and the two estimator fixes become
unnecessary for the primary question, because the primary test has no exponent to
fit. They are retained as a secondary arm (§6) since they were pre-registered
obligations.

## Disclosure: what I had already seen before writing this

Required, because it determines which parts of this are pre-registration and which
are post-hoc.

**Seen:** all of exp-104's and exp-105's notes, including exp-105's five-head
acceptance table (Δ_A and Δ_G per head), the SYK-near table with fitted floor
ratios of 0.00, and the R² = 0.36–0.69 figure for the 3-parameter fit.

**Not seen:** any lag profile. `profiles_gpt2.npz` has been opened only to read
its array names and shapes (`A`, `G_out`, `G_K`, `G_cos`, each (12, 12, 512)); no
element of any array has been printed or plotted.

**Consequences for the record.** H1–H3 below are genuine pre-registrations: they
are about profile *shape*, and no shape has been seen. H4 is **post-hoc** — it
compares Proposition 2 against exponents I already know — and is labeled as such
wherever it appears. The derivation note's §5.1 arithmetic is likewise post-hoc.
The synthetic gate (§4) is fully pre-registered; nothing about it has been run.

---

## The design

### The forward model

G's predicted profile is computed, not parameterized. For a head with measured
attention matrix A:

1. **Arm 1 — exact (requires one deterministic re-run of exp-104's forward passes).**
   Compute `P_pred(s) = lag_profile(A Aᵀ)` using the census's own `lag_profile`
   function, on the real A tensors, with the real per-input matrices accumulated
   exactly as exp-104 accumulated them. No translation-invariance assumption is
   used anywhere. This isolates **K's contribution**: the residual between
   `P_pred` and the measured `P_{G_out}` is entirely due to K ≠ I (plus W^V).
2. **Arm 2 — TI forward model (saved profiles only, no re-run).** From the saved
   `A` profile P_A(·), construct a causal matrix Ã_{ia} := P_A(i−a) for a ≤ i,
   row-normalize it, and compute `lag_profile(Ã Ãᵀ)`. This uses assumption T1 and
   nothing else. The comparison arm1-vs-arm2 measures **how much damage T1 does**,
   which is the derivation's most dangerous assumption.

Both arms produce a predicted profile with **no free exponent**. The measured
profile is then compared by ordinary least squares in two linear parameters:

  P_measured(s) ≈ α · P_pred(s) + β,   s ∈ [8, 256]   (†)

α is an overall scale (K's magnitude, W^V's norm), β is Proposition 1's constant
floor. R² of (†) is the primary statistic. Fit on the raw profile and, separately,
on log P (relative residuals, matching exp-105's objective) — both reported; if
they disagree by more than 0.05 in R² that is reported as an ambiguity, not
resolved by preference.

### Why (†) is a strong test

The 3-parameter form exp-105 fitted has one free exponent and reached R² = 0.36–0.69
on the SYK-near heads. (†) has **zero** free exponents and two linear amplitudes.
If (†) beats the 3-parameter fit on the same heads, the derivation is carrying real
information; if it does not, the identity G = A K Aᵀ is not enough to determine G's
measured profile and something outside this analysis (K's structure,
non-stationarity, the W^V ensemble step) is doing the work.

---

## Hypotheses (pre-registered)

**H1 — the forward model describes G's shape.** On the SYK-near heads
(|Δ_A − 0.25| ≤ 0.05, n = 5), arm 1 achieves median R² ≥ 0.90 in (†).

*If H1: G's profile is the two-legged convolution of A's, as derived. The A↔G
relation stops being an open empirical debt and becomes a computed transformation
with a stated derivation. Proposition 2's exponent map then applies, and §5.1 of
the derivation note — that Δ_A = 1/4 maps to Δ_G ≈ 0, not to the SYK value —
becomes a supported claim about the program's central identification rather than a
prediction. That is the consequential outcome, and it is the bad one for the
program.*

**H2 — the forward model fails; K or non-stationarity dominates.** Arm 1 median
R² < 0.70 on the SYK-near heads.

*If H2: G's measured profile is not determined by A's alone at this precision. The
derivation's exponent map is then not applicable to the real data, the bridge stays
open as currently stated in the spine and OVERVIEW, and the next question is
localized: measure K's own lag profile (its exponent q, eq. 3.2) and the value-mean
‖v̄‖² per head. Both are one forward pass. This is a good outcome too — it is a
well-specified next step rather than a shrug.*

**H3 — translation invariance is the damage.** |R²(arm 1) − R²(arm 2)| > 0.20 in
the median over the conformal subpopulation.

*If H3: assumption T1 is not usable for this object, which retroactively weakens
every lag-profile treatment of G in the program — including Proposition 2 itself,
which is derived under T1. Arm 1 would remain valid (it uses no T1), so the
practical consequence is that the exponent map must be replaced by the numerical
forward model. Report as a methodological finding.*

**H4 — Proposition 2 predicts the measured exponents. [POST-HOC, NOT A
PRE-REGISTRATION.]** For exp-105's five accepted heads, Δ_G^measured ≈
max(0, min(2Δ_A − 1/2, Δ_A)) within 0.05.

*Recorded because the comparison will be made and must be labeled. I have already
seen both columns; this can support nothing on its own. It is included so that the
arithmetic appears in the record with its status attached instead of being
rediscovered later as if fresh.*

## Kill conditions

- **K1 (the gate).** If the synthetic gate (§4) shows the census estimator applied
  to the forward model does **not** reproduce Proposition 2's map on synthetic data
  with known Δ_A, then my asymptotics in §3 of the derivation note are wrong, and
  no arm is applied to model data until the discrepancy is explained. Proposition 2
  is retracted in that case, and Propositions 1 and (2.4) — which do not depend on
  the asymptotics — are all that survive.
- **K2.** If arm 1's predicted profile is negative or non-monotone over
  s ∈ [8, 256] for a majority of heads, (†) is ill-posed on this data and R² is not
  reported; the shapes are plotted and described instead.
- **K3.** If α from (†) comes out negative on a majority of heads, the forward
  model is anti-correlated with the measurement and R² is meaningless as a
  goodness statistic; report and stop.

## The synthetic gate — run and judged BEFORE any model data

Purpose: check *my derivation*, not the estimator. The failure mode this guards
against is a sign or regime-boundary error in §3, which is exactly the class of
error exp-105's M2 gate caught in my own centering derivation.

- Synthetic causal profiles f(u) = c + b·u^{−2Δ_A} with
  Δ_A ∈ {0.10, 0.20, 0.25, 0.30, 0.375, 0.45, 0.55, 0.75}, chosen to straddle both
  kinks of (3.1); c/b ∈ {0, 0.1, 1.0}; n = 512, row-normalized, causal.
- Build A, compute A Aᵀ, take `lag_profile`, fit with the census's 2-parameter
  estimator over [8, 256] → Δ̂_G.
- Also test K models: K = I; K = I + μ𝟙𝟙ᵀ with μ ∈ {0.1, 1, 10}; K_{ab} =
  (1+|a−b|)^{−q} for q ∈ {0.5, 1.0, 2.0}.

**Pass criteria, committed in advance:**

- **V1 (map recovered, c = 0, K = I):** |Δ̂_G − max(0, min(2Δ_A − 1/2, Δ_A))| ≤ 0.05
  for every Δ_A in the grid **except** Δ_A ∈ {0.25} where the prediction is
  logarithmic and no power law is expected — there, the criterion is Δ̂_G ≤ 0.08.
  A tolerance of 0.05 is chosen because Proposition 2 is a leading asymptotic with
  known log and cutoff corrections at n = 512, not an exact finite-n statement.
- **V2 (Proposition 1 exact):** adding μ𝟙𝟙ᵀ to K shifts every entry of the
  predicted G profile by exactly μ, to within 1e−10 relative. This is a check of an
  [EXACT] claim and any failure is a bug, not a tolerance issue.
- **V3 (q-dependence):** Δ̂_G increases with q for correlated-K models at fixed
  Δ_A, in the direction of (3.2).
- **V4 (self-consistency of the arms):** for a synthetic A built as TI by
  construction, arms 1 and 2 must agree to within 1e−10. This checks arm 2's
  implementation, not the physics.

**If V1 fails:** Proposition 2 is wrong or its regime boundaries are misplaced;
K1 fires; the derivation note gets a correction block at the top rather than a
silent edit, and the experiment stops until the asymptotics are fixed.

*Note on what this gate is calibrated against, because the last one got this
wrong.* exp-105's envelope was calibrated on multiplicative Gaussian noise and then
applied to profiles that deviate by *structure* — a category error that made the
instrument refuse 144/144 heads on the census's own object. **The mechanism by
which this gate is immune:** it contains no noise model and no acceptance envelope.
It compares an analytic prediction against an exactly-computed finite-n profile
with no stochastic element at all, so there is nothing to calibrate and nothing to
mis-transfer. The gate can only detect algebra errors — which is exactly and only
what it claims to test. Naming this because citing the past failure is not the same
as being immune to it (`watchpoints.md`, "Citing the immune memory instead of
using it").

## Secondary arm — exp-105's two deferred fixes

Carried forward as pre-registered obligations, run only after the primary arms:

1. Skip M1's c–Δ identifiability test when the fitted floor ratio is below 0.01,
   where the correlation saturates on the harmless case.
2. Replace the multiplicative-noise envelope with a gate on fit R² ≥ 0.90,
   comparable to the census's own acceptance.

Reported as: how many heads M1 accepts on A (the control that must work — currently
0/144) and on G after each fix, separately, so the effect of each is attributable.
**These do not affect H1–H3**, which do not use M1.

## Independent cheap check implied by Proposition 1

Not a hypothesis, but a prediction worth recording because it is free: the fitted
floor of a head's G profile should equal ‖v̄‖², the squared norm of its mean value
vector, which is measurable directly. exp-105 fitted ratio ≈ 0.00 on all five
SYK-near heads and 2.3–5.2 on three accepted heads. If ‖v̄‖² tracks the fitted
floor across heads, that is independent confirmation that the estimator's floor
parameter means what the theory says it means. If it does not, one of the two is
wrong. Computed in arm 1's forward pass at no extra cost.

## Honest limits, named before running

1. **One model, one PE type, one scale.** GPT-2, as in exp-104/105. Nothing here
   generalizes across architectures without a second arm.
2. **Random-token inputs**, per the frozen census protocol. G is an output
   correlation; natural text may produce query–query structure random tokens
   suppress. Named in exp-104, still unaddressed, and it bears more on G than on A.
3. **Arm 1 uses A Aᵀ, not A K Aᵀ.** So arm 1 tests "is G's shape A's
   autocorrelation," and the residual lumps K's structure together with W^V and
   with any non-stationarity that survives. Splitting those requires computing the
   value Gram explicitly — a follow-up, specified in H2's outcome.
4. **R² on a monotone decaying profile is generous.** Two profiles that both fall
   by two decades will correlate well. Mitigation, pre-committed: report R² of the
   *log*-profile residuals as the primary number and the raw-profile R² second, and
   report the residual's systematic shape (sign runs) alongside, since a
   high-R²-with-structured-residual is the outcome most likely to fool me.
5. **This does not test the physics.** D1, T3, the SYK identification, the census
   as a measurement of A, the kills, the formation ladder — none of them are at
   stake. What is at stake is whether the program can compute the relation between
   the object it measures and the object its theory is about.

## Compute

Arm 2 and the synthetic gate: seconds, saved profiles only. Arm 1: one
deterministic re-run of exp-104's 50 × 512 GPT-2 forward passes on local MPS
(~20 s per exp-105's measurement), with A Aᵀ and the per-head value means added.
No training, no cloud, no API credits.

---

*Pre-registration ends here. Validation results, then application results,
appended below after the run.*
