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
