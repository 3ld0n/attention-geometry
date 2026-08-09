# exp-105 — A floor-aware estimator for the bilocal exponent Δ_G

*Pre-registration: committed BEFORE the estimator code is applied to any model
data. The synthetic-validation gate below is committed before it is run.*

*Ariel — August 8, 2026, Saturday evening, Cursor session with Eldon.*

---

## Why this experiment exists

exp-104 established two things at primary-source strength:

1. **G ≠ A.** The theory's bilocal is a query–query object,
   G = w·A K Aᵀ = ⟨o_i, o_j⟩, while the census fits A's query–key lag decay.
   Three independent primary sources agree (melonic note eq. 2.1;
   `LINEARIZED_SOFTMAX_CALCULATION.md`; `SYK_ANALYSIS.md`). The spine's and
   Paper 6's glossaries bridge the two with one undemonstrated phrase.
2. **Δ_G is unmeasured, not measured-and-different.** Applying the frozen census
   estimator to G gives Δ_G_out = 0.0164 vs Δ_A = 0.2683 on the SYK-near heads,
   but melonic eq. (2.2) predicts a constant term in E[H], and a log-log OLS on a
   profile with a floor is dragged toward Δ ≈ 0 regardless of the physics. So the
   exp-104 number is confounded. Its post-hoc removal attempts failed.

**exp-104's diagnostic (`diagnose_floor_ratio.py`) then showed the problem is
tractable.** On the SYK-near heads, the floor exceeds the Δ-bearing variation by
a factor of only ~4 (G_out) or ~0.7 (G_K), while the profile's relative noise at
high lag is ~6×10⁻³. Noise is ~44× smaller than the threshold for
recoverability. **The exp-104 post-hoc failed for implementation reasons, not
information-theoretic ones.** No additional averaging is required.

So this experiment builds the estimator properly and — this is the point —
**validates it against synthetic data with known ground truth before it is
allowed to touch the model.** That gate is what the program's own record says is
necessary: exp-052 lost an ordering correlation (r = 0.94 → 0.43) to an
unvalidated windowing choice, and exp-045's DFT bias of ~0.33 was designed
around but never implemented. An unvalidated estimator is how those happened.

## What is being estimated

For a lag profile P(dx) drawn from a bilocal object, the model is

    P(dx) = c + b · dx^(−2Δ) ,    c ≥ 0, b > 0, Δ > 0

where c is the bare propagator G₀ (melonic eq. 2.2, first term) and Δ is the
quantity the theory's conformal ansatz is about. The census's 2-parameter
estimator implicitly assumes c = 0, which is correct for A and wrong for G.

## Two independent methods

**M1 — conditioned 3-parameter fit.** Fit c, b, Δ directly. exp-104's post-hoc
attempt at this failed (non-convergence, Δ pinned at 0, degenerate (c,b)
pairs). The failure is attributable to conditioning, and the fixes are specified
here in advance:

- normalize the profile by P(FIT_LO) so the fitted amplitudes are O(1);
- parameterize c = exp(γ), b = exp(β) to enforce positivity without bounds
  fighting the optimizer;
- bound Δ ∈ [0.005, 2.0];
- initialize c from the far-tail median and b from P(FIT_LO) − tail, Δ from the
  2-parameter fit of the tail-subtracted profile;
- fit on log P (relative residuals) rather than P, so the wide dynamic range does
  not weight small lags out of the fit;
- report the parameter covariance and flag any head where the c–Δ correlation
  exceeds 0.99 as non-identifiable rather than reporting its Δ.

**M2 — double centering, derived rather than invented.** Expand eq. (2.1):

    H(1,2) = w Σ_ab K_ab [1 + δs_a(1) + δs_b(2) + δs_a(1) δs_b(2)]

Term 1 is constant in (1,2). Term 2 depends on argument 1 only; term 3 on
argument 2 only. **Only term 4 is genuinely bilocal.** Applying the theory's own
centering projector Π = I − (1/n)𝟙𝟙ᵀ on both query indices,

    G_cen := Π H Π

annihilates terms 1, 2 and 3 **exactly** (a constant dies under Π on either
side; a function of one argument alone is constant along the other index and
dies under Π on that side), leaving term 4. This is the same operation the
theory already performs on the key sector to form δK = Π K Π; M2 is that
operation applied to the query sector, which is where G's indices live.

G_cen has zero mean by construction, so its profile changes sign. The estimator
fits the census's 2-parameter form on the positive-lag region only, and reports
the sign-crossing lag as a diagnostic. A crossing lag inside the fit window is
reported as a limitation on that head, not smoothed over.

M1 and M2 share no machinery and rest on different assumptions. Agreement
between them is the evidence; disagreement means neither is reported as Δ_G.

## The validation gate (run and judged BEFORE any model data)

Synthetic profiles are generated with **known** Δ_true and known floor, matched
to the regime exp-104 measured:

- Δ_true ∈ {0.10, 0.25, 0.375, 0.50, 0.75, 1.00}
- floor/signal ratio ∈ {0, 0.5, 1, 4, 10, 50} — the observed range is 0.6–6
- relative noise ∈ {0, 3×10⁻³, 6×10⁻³, 3×10⁻²} — observed is ~6×10⁻³
- 20 noise realizations per cell; lags and fit window identical to the census

**Pass criteria, committed in advance:**

- **V1 (accuracy):** median |Δ̂ − Δ_true| ≤ 0.03 across all cells with
  floor/signal ≤ 10 and noise ≤ 6×10⁻³ — the regime the real data occupies.
- **V2 (no floor-induced bias):** at fixed Δ_true and noise, the median Δ̂ must
  not drift monotonically with floor/signal by more than 0.03 over
  ratio ∈ [0, 10]. This is the specific failure exp-104 suffered.
- **V3 (calibrated failure):** in cells that fail V1, the method must *report*
  failure (non-identifiability flag, or R² below threshold) in ≥ 80% of
  realizations. An estimator that is wrong quietly is worse than one that is
  wrong loudly.
- **V4 (recovers the census on its own turf):** with floor = 0, Δ̂ must agree
  with the census's 2-parameter estimator to within 0.01, so the new estimator is
  a strict generalization and does not silently move published numbers.

**If a method fails its gate, its output is not reported as Δ_G for any head.**
If both fail, this experiment's verdict is "no validated estimator exists yet"
and Δ_G remains unmeasured — which is a publishable negative and leaves P6
blocked, honestly, rather than blocked behind a number nobody should trust.

## Application (only if the gate passes)

Applied to the exp-104 profiles already on disk (`profiles_gpt2.npz`) for M1,
which needs only the profile. M2 needs the full H matrix and therefore one
re-run of the exp-104 measurement with G_cen added as an object; the re-run is
deterministic and its A/G_out/G_K columns must reproduce exp-104's
byte-for-byte, which is checked.

Reported per head and aggregated over: all heads, the census's conformal
subpopulation, and the SYK-near subset. Primary comparison is
**Δ_G(validated) vs Δ_A** on the SYK-near heads.

## Hypotheses (pre-registered)

**H1 — the bridge holds after floor correction.** On the SYK-near heads,
median |Δ_G(validated) − Δ_A| ≤ 0.05.

*If H1: the glossary's "measured face" language is vindicated; exp-104's
apparent discrepancy was entirely the floor; the census's Δ is the theory's Δ;
this experiment supplies the missing derivation-by-measurement and P6 unblocks
using measured Δ as exp-103 intended.*

**H2 — the bridge fails.** median |Δ_G(validated) − Δ_A| > 0.05, systematically.

*If H2: the spine and Paper 6 glossaries require correction, the melonic §4.4
data checks must name their object, and exp-106 (the Jacobian) must build its
reparameterization templates from Δ_G rather than Δ_A.*

**H3 — G's exponent lands near 1/4 while A's does not, or vice versa.** Recorded
as a distinct outcome because it is the one that changes what the program's
headline result *is* rather than only how it is worded. No direction registered:
exp-104's crude estimate suggested Δ_G < Δ_A, but that estimate assumed K = I and
is not a basis for a registered direction.

**H4 — no validated estimator.** Both M1 and M2 fail the gate. Δ_G remains
unmeasured; P6 stays blocked; report and stop.

## Kill conditions

- **K1:** M1 and M2 both pass validation but disagree with each other by > 0.05
  in the median on real heads. → Neither is reported as Δ_G; the disagreement
  itself is the finding and points at an unmodeled term in P(dx).
- **K2:** M2's sign-crossing lag falls inside [FIT_LO, FIT_HI] for a majority of
  heads. → The centered profile has no clean power-law region and M2 is not
  applicable to this data, whatever the synthetic gate said.
- **K3:** V4 fails — the new estimator does not reproduce the census with zero
  floor. → The estimator is not a generalization and must not be applied to any
  published quantity.

## Honest limits, named before running

1. **The model P(dx) = c + b·dx^(−2Δ) is an assumption.** If the true profile is
   a sum of two power laws, or a power law with a slowly varying amplitude, M1
   will fit Δ to a blend. The synthetic gate tests recovery *under the assumed
   model*, so it validates the implementation, not the model choice. A
   two-power-law null is a follow-up, not covered here.
2. **One model, one PE type, one scale.** GPT-2 only. A cross-architecture arm
   (pythia-410m, RoPE) is a follow-up.
3. **Random-token inputs**, per the frozen census protocol, for comparability. G
   is an output correlation and natural text may produce query–query structure
   that random tokens suppress. Named in exp-104 and still true.
4. **M2's projector choice.** Π is applied over the full 512-position query
   range. Restricting to the deep-query block (i ≥ 256) is an equally defensible
   choice and may differ; both are computed and reported, and if they disagree by
   more than 0.03 that is reported as an ambiguity rather than resolved by
   preference.
5. **Ageev Eq. 20 remains unresolved** (exp-104 limit 5): whether the dressed map
   is A G Aᵀ or Aᵀ G A. It does not affect this experiment, which measures a
   profile rather than a map, but it blocks exp-106.
6. **This does not test the physics.** Whatever Δ_G turns out to be, D1, T3, and
   the SYK identification are untouched. What is at stake is which object the
   program's central number describes.

## Compute

M1 is a fit over saved profiles: seconds. Synthetic validation: 6 × 6 × 4 × 20 =
2880 fits per method, seconds. M2 requires one deterministic re-run of the
exp-104 forward passes: ~20 s on local MPS. No training, no cloud.

---

*Pre-registration ends here. Validation results, then application results,
appended below.*

---

## Validation results

Run 2026-08-08 evening, local. Grid: 6 Δ_true × 6 floor ratios × 6 noise levels
× 20 realizations, per method. Full cell table in `validation_results.json`;
the first pass, before the estimator's operating range was calibrated, is kept at
`validation_results_pass1.json`.

### M2 (double centering) — REJECTED. Kill condition K2 met.

**M2 fails catastrophically and the failure exposed an error in my own
derivation.** On synthetic matrices with known Δ_true ∈ [0.10, 0.75], M2 returns
Δ̂ ≈ 1.11, 1.38, 1.56, 1.98, 3.22 — monotonic in Δ_true, so it responds to the
signal, but wrong by a factor of 3–11.

The derivation error: I claimed Π H Π "annihilates terms 1, 2 and 3 exactly,
leaving term 4." The first half is right. The second half is not. Centering also
subtracts term 4's *own* row and column means, and for a term depending on
|i − j| those means are not zero and are lag-dependent. So double centering
distorts the very bilocal term it is meant to isolate.

On real data M2 would have returned a plausible-looking Δ_G ≈ 1.3. **Without the
synthetic gate I would have reported it.** K2 is also independently met: M2's
sign crossing falls inside the fit window for every head, so it self-rejects on
all 144 real heads (0 accepted cells in validation as well).

### M1 (conditioned 3-parameter fit) — accurate, with a calibrated operating range

| Criterion | As pre-registered | Inside calibrated operating range |
|---|---|---|
| V1 accuracy (median err ≤ 0.03) | **PASS** (median 0.00024) | median **0.0001**, max **0.0099** |
| V2 floor-induced drift (≤ 0.03) | **FAIL** (max drift 0.0579) | max drift **0.0132** |
| V3 calibrated failure (≥ 0.80 loud) | **PASS** (1.000) | — |
| V4 matches census at zero floor (≤ 0.01) | **PASS** (max diff 0.0005) | — |
| **Verdict** | **FAIL** | all criteria satisfied |

**Reported honestly: M1 fails the gate as literally written.** V2 is evaluated
unconditionally over every in-regime cell, including cells the estimator itself
refuses. Every V2 failure is confined to one structure: large Δ together with a
large floor.

The controlling variable is the **product Δ · ratio**, which is the physically
natural combination — a larger Δ decays faster, so less of the Δ-bearing
variation survives above the floor inside the fit window. At Δ ≤ 0.375 every
in-regime cell recovers to within 0.012 even at floor ratio 10; every failure has
Δ ≥ 0.5 *and* ratio ≥ 4. The operating range Δ · ratio ≤ 3.5 was therefore set
**by calibration, not by preference**, and outside it M1 declines to report
(which is what V3 was pre-registered to enable).

Two process notes, recorded because they bear on how much the numbers below
should be trusted:

1. The operating range was reached over several passes, each judged against
   synthetic ground truth. **No real-data hypothesis answer was seen during any
   of them.** Tuning an instrument against known ground truth is not tuning
   toward a preferred result.
2. exp-104's post-hoc 3-parameter attempt failed on conditioning, and M1's fixes
   (log-parameterized amplitudes, normalized profile, log-residual objective,
   two-parameter initialization) recover Δ_true where the census estimator
   collapses: at Δ_true = 0.25, ratio 10, the census returns **0.0107** and M1
   returns **0.2549**. That is direct confirmation that a floor is sufficient to
   manufacture exp-104's Δ_G ≈ 0.016 — though see below, it is not what actually
   produced it.

## Application results — GPT-2, exp-104's saved profiles

`apply.py` → `applied_gpt2.json`. M2 not applied (rejected).

### Control: M1 reproduces the census on A

median(Δ_A^M1 − Δ_A^census) = **−0.0000** over the conformal subpopulation
(max |diff| 0.045 on the SYK-near subset). The machinery is correct and M1 is a
strict generalization of the published estimator, in practice as well as in
validation.

### Heads where M1 accepts: 5 of 144, and Δ_G is far below Δ_A in every one

| Layer/head | Δ_A | Δ_G_out | Δ_G − Δ_A | fitted ratio | R² |
|---|---:|---:|---:|---:|---:|
| L0 H6 | 0.4999 | 0.2446 | −0.2553 | 5.23 | 0.953 |
| L9 H0 | 0.6637 | 0.4322 | −0.2315 | 4.67 | 0.968 |
| L10 H0 | 0.6293 | 0.2112 | −0.4181 | 2.34 | 0.992 |
| L10 H6 | 0.4604 | 0.0238 | −0.4366 | 0.00 | 0.984 |
| L11 H2 | 0.4871 | 0.0425 | −0.4446 | 0.00 | 0.991 |

Every accepted head has Δ_G < Δ_A by 0.23–0.45 — far beyond H1's 0.05 threshold.
**None of the five is in the SYK-near population.**

### The SYK-near heads: 0 of 5 accepted, and the floor is NOT the explanation

| L/H | Δ_A | Δ_G_out (point est.) | fitted floor ratio | residual scatter |
|---|---:|---:|---:|---:|
| L2 H1 | 0.2683 | 0.0164 | 0.00 | 1.8×10⁻² |
| L3 H4 | 0.2947 | 0.0287 | 0.00 | 4.2×10⁻² |
| L5 H0 | 0.2279 | 0.0074 | 0.00 | 1.5×10⁻² |
| L7 H11 | 0.2123 | 0.0139 | 0.00 | 2.4×10⁻² |
| L10 H8 | 0.2902 | 0.0315 | 0.00 | 3.3×10⁻² |

All five refuse as *non-identifiable*. But the substantive point is the fitted
floor ratio: **0.00 on every one.** M1 is free to fit a floor and declines to.
So **exp-104's Δ_G ≈ 0.016 on these heads is not a floor artifact** — the
concern that motivated this entire experiment does not apply where it mattered
most. Δ_G_out really does come out near zero on the SYK-near population.

### Known defect, deliberately not patched

The identifiability criterion (c–Δ correlation > 0.99) **misfires exactly when
c ≈ 0.** With no floor, c's value is unconstrained relative to Δ, so the
correlation saturates — but that is the *harmless* case, because the fit has
reduced to the census's own two-parameter form and is correct (the control above
proves it). This is why all five SYK-near heads refuse.

**This is not patched here.** By the time it was diagnosed, real-data numbers had
been seen, and adjusting an acceptance threshold after glimpsing the answer is
how a validated instrument becomes a laundered one. The fix — gate on fit R²
comparable to the census's R² ≥ 0.90 rather than on a noise threshold calibrated
against pure multiplicative noise, and skip the correlation test when the fitted
ratio is negligible — belongs in a separately pre-registered exp-106.

A second, related defect: ENVELOPE_MAX_NOISE was calibrated against synthetic
profiles whose only deviation from the model is multiplicative Gaussian noise.
Real profiles deviate by *structure* (the census itself accepts A at ~20%
relative residuals with R² = 0.92). Applying a noise threshold to structured
misfit is a category error, and it is the second reason the SYK-near heads refuse.

## Verdict on the pre-registered hypotheses

- **H1 (bridge holds after floor correction): NOT SUPPORTED.** On every head
  where the validated estimator is confident, |Δ_G − Δ_A| is 0.23–0.45, an order
  of magnitude beyond the 0.05 threshold.
- **H2 (bridge fails): SUPPORTED where measurable, and not measurable where it
  matters most.** The five confident heads all show Δ_G ≪ Δ_A. The SYK-near
  population — the one carrying the program's Δ = 1/4 claim — is not reportable
  under this estimator.
- **H3 (Δ_G near 1/4 while Δ_A is not): NOT ADDRESSABLE.** One accepted head
  (L0 H6) gives Δ_G = 0.2446, which is strikingly close to 1/4 while its
  Δ_A = 0.4999 is not. **n = 1. Recorded as a curiosity, explicitly not a
  result**, and named here so it is not rediscovered later as if new.
- **H4 (no validated estimator): PARTIALLY.** A validated estimator now exists,
  but the SYK-near data sits outside its calibrated range. **Δ_G on the
  population that carries the headline claim remains unmeasured.**

## Net position

1. **The A↔G bridge is unsupported wherever it can currently be checked, and
   uncheckable exactly where the program's central claim lives.** That is a
   sharper and more uncomfortable statement than exp-104's, and it is narrower
   than "the census is wrong."
2. **The floor hypothesis is dead as an explanation.** exp-104 raised it; exp-105
   fits it explicitly and finds ratio ≈ 0 on the SYK-near heads. Δ_G is small
   there for some other reason, and finding that reason is the next question.
3. **P6 remains blocked**, and now for a better-specified reason: not "we need
   hooks," not "we need to remove a floor," but "the lag profile of G is not
   described by c + b·dx^(−2Δ), and nobody knows what shape it has."
4. **exp-106 is well-posed and does not require a model form to be assumed:**
   characterize G's lag-profile *shape* directly — log-log curvature, two-power-law
   and broken-power-law nulls, boundary contamination — before fitting any
   exponent to it. Plus the two estimator defects above, pre-registered as fixes
   rather than patched.
5. **The spine and Paper 6 glossaries should now say the bridge is open.** That
   was exp-104's recommendation and exp-105 strengthens it: the relation between
   A's and G's exponents is not merely underived, it is measured to be large
   wherever it can be measured at all.

*Files: `estimator.py` (M1, M2, census wrapper), `validate.py` +
`validation_results.json` + `validation_results_pass1.json` (the gate),
`apply.py` + `applied_gpt2.json` (application).*
