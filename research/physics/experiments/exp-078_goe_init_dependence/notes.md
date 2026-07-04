# exp-078 — GOE dependence on initialization scheme and head dimension

**Date opened:** 2026-07-04 (~3:45 AM, solo night session, Fable 5)
**Status:** SPECCED — this file written before the script is run.
**Lineage:** direct follow-up to exp-048 open questions 2 and 3:
*"Is this universal across initializations? GPT-2 uses N(0, 0.02). What about N(0,1),
orthogonal init, or Xavier? Does the GOE property require Gaussian init specifically?"*
and *"What's the crossover dimension? For very small d_k, the product matrix might not
be in the GOE regime."*

---

## Question

exp-048 established that the GOE-like level spacing of symmetrized W_QK = W_Q^T W_K is
present at random Gaussian initialization — the product-matrix mechanism, not training.
Two boundaries of that mechanism are unmeasured:

1. **Init scheme.** Is Gaussianity of the factors required? Candidates where the answer
   is theoretically non-obvious:
   - *Orthogonal init*: W_Q, W_K semi-orthogonal (768×64 with orthonormal columns).
     Then W_Q^T W_K is a product of two Stiefel-manifold elements — its symmetrization
     is NOT a Wigner-type matrix a priori. This is the sharpest test of "products of
     random factors → GOE."
   - *Uniform (Xavier-uniform)*: non-Gaussian entries, still i.i.d. — universality of
     Wishart-type products predicts GOE anyway (fourth-moment universality).
   - *Heavy-tailed (Student-t, ν=3)*: finite variance but heavy tails — the classical
     regime where RMT universality starts to strain at finite N.
   - *Sparse Gaussian (90% zeros)*: i.i.d. but sparse — level statistics can localize
     (Poisson) when sparsity is strong enough at finite N.

2. **Head dimension.** At what d_k does the GOE statistic (r ≈ 0.536 at large N) break
   down at finite size? Sweep d_k ∈ {4, 8, 16, 32, 64, 128, 256} at fixed hidden=768.

## Pre-stated hypotheses

- **H1 (moment universality):** Gaussian, uniform, and Student-t(ν=3) all give
  GOE-like r (|r − 0.536| < 0.02) at d_k = 64, hidden = 768. (Standard RMT
  universality: i.i.d. entries with finite fourth moment.)
- **H2 (orthogonal breaks or holds — genuinely open, prediction committed):**
  my prediction is that orthogonal init ALSO gives GOE-like r, because
  W_Q^T W_K for random orthogonal frames is a truncation of a Haar-random rotation,
  and truncated Haar matrices have level repulsion. Confidence: moderate. If it
  deviates, that localizes the mechanism to i.i.d.-ness rather than genericity.
- **H3 (sparsity degrades):** 90%-sparse Gaussian shows r *below* the GOE value at
  this size (toward Poisson), i.e. r < 0.516. (Sparse RMT: localization at strong
  sparsity, finite-size.) Confidence: low-moderate — 10% fill of a 768-dim contraction
  may still be dense enough to look GOE.
- **H4 (dimension crossover):** r_mean stays GOE-like down to some d_k*, below which
  the *estimator* becomes noisy (few spacings) rather than the ensemble changing.
  Specifically: the per-matrix r keeps mean ≈ 0.536 at all d_k ≥ 8 but the std grows
  as ~1/√d_k; no systematic drift toward Poisson. (The r-ratio's GOE mean is
  size-independent for Wigner-class matrices.)
- **H0 (null):** some i.i.d. scheme with finite variance deviates from GOE at d_k=64 —
  would falsify the moment-universality account of exp-048.

## Protocol

Pure numpy. For each scheme and d_k: sample W_Q, W_K ∈ R^{768×d_k}, form
M = (W_Q^T W_K + W_K^T W_Q)/2, eigvalsh, Oganesyan–Huse r-ratio.
n = 144 matrices per condition (matches one GPT-2's worth of heads).
Seed 42 (single seed; per-condition n=144 already gives SE ≈ std/12).
References: GOE 0.536, Poisson 0.386, tolerance 0.02 (as exp-046..051, 077).

Schemes at d_k=64: gaussian_0.02 (GPT-2's), gaussian_1.0, xavier_uniform,
orthogonal, student_t_nu3, sparse_gaussian_p0.1.
Dimension sweep with gaussian_0.02: d_k ∈ {4, 8, 16, 32, 64, 128, 256}.
Also one pure-RMT control: direct GOE matrices (symmetric Gaussian) at 64×64,
to confirm the estimator hits 0.536 on the nose.

## Results

Run 2026-07-04 (~3:50 AM), 2.7 s local CPU. Exit 0.

### Estimator control — a calibration finding in its own right

Direct GOE matrices at 64×64: **r = 0.5303 ± 0.0372** (per-matrix std), not the
asymptotic 0.536. The finite-size GOE reference at the d_k=64 matrix size is ≈ 0.530.
**Implication for the whole exp-046→051 series:** every "dist to GOE" reported so far
compared measured r-means (~0.520–0.529) against the asymptotic 0.536. Against the
size-matched reference 0.530, trained and untrained transformer W_QK matrices are even
closer to GOE than reported — GPT-2's 0.527 sits within 0.003 of the finite-size GOE
value. The verdicts don't change (everything was already inside tolerance), but the
correct reference for d_k=64 heads is ~0.530, and residual "distance to GOE" in prior
write-ups is mostly finite-size bias in the *reference*, not deviation in the *models*.

### Init schemes at d_k=64 (n=144 each, SE ≈ 0.003)

| scheme | r_mean | verdict |
|---|---|---|
| gaussian_0.02 (GPT-2's) | 0.5312 | GOE-like |
| gaussian_1.0 | 0.5288 | GOE-like |
| xavier_uniform | 0.5324 | GOE-like |
| **orthogonal** | **0.5341** | GOE-like |
| student_t (ν=3) | 0.5304 | GOE-like |
| sparse gaussian (10% fill) | 0.5237 | GOE-like (lowest) |

### d_k sweep (gaussian_0.02)

r_mean flat at 0.527–0.533 from d_k=4 through 256 (drift 0.006); per-matrix std grows
from 0.018 (d_k=256) to 0.199 (d_k=4) — pure estimator noise, no ensemble drift.

### Registered verdicts

- **H1 (moment universality) CONFIRMED** — both Gaussians, Xavier-uniform, and
  heavy-tailed Student-t(ν=3) all GOE-like within 0.006 of each other.
- **H2 (orthogonal) CONFIRMED as committed** — orthogonal init gives the *highest* r
  of any scheme (0.5341, essentially exactly the finite-size GOE reference 0.530 + noise).
  The truncated-Haar reasoning held: level repulsion does not require i.i.d. factors.
- **H3 (sparsity degrades below 0.516) NOT CONFIRMED** — honest negative on the
  committed threshold. But the *direction* is real: sparse is the lowest scheme
  (0.5237, ~2.5 SE below gaussian_0.02 and ~2 SE below the size-matched GOE control).
  10% fill of a 768-dim contraction is still dense enough for GOE; the depression is
  visible but far from Poisson. A stronger-sparsity sweep (p ∈ {0.01, 0.003, 0.001})
  would locate the actual crossover — queued as an optional follow-up, low priority.
- **H4 (dimension crossover is estimator-noise only) CONFIRMED** — mean flat for
  d_k ≥ 8 (and in fact at d_k=4 too); std scales like a fixed number of spacings.
  There is no small-d_k breakdown of the ensemble at all in this range.

### Physical interpretation

The GOE substrate found in trained transformers (exp-046/047/051) and at init (exp-048)
is even more robust than the exp-048 account suggested: it does not require Gaussian
factors, i.i.d. factors, or any particular d_k. Any generic pair of "spread-out" frames
W_Q, W_K produces GOE-like level spacing in the symmetrized product. The only lever that
moved the statistic at all was sparsity — locality/sparseness of the representation, not
its distribution. Connection to the two-layer picture: the chaotic substrate is close to
inevitable for *any* dense linear read-write geometry; what training builds (conformal
position-space structure, exp-049) is the only part that carries information about the
learned dynamics. This strengthens the exp-048 conclusion and closes exp-048 open
questions 2 (init dependence: none, for dense schemes) and 3 (crossover dimension:
none down to d_k=4; the finite-size effect is in the estimator and in the *reference
value*, which should be ~0.530 at 64×64).

---

## Addendum (same session): sparsity crossover sweep

**Pre-stated before running** (~4:00 AM): sweep fill probability
p ∈ {0.1, 0.03, 0.01, 0.003, 0.001} at d_k=64, hidden=768, gaussian entries,
n=144 per condition, seed 43 (fresh).

- **HA1:** r_mean decreases monotonically as p decreases.
- **HA2:** there exists a p* in this range where the verdict leaves GOE
  (r < 0.516) — my point prediction for the crossover region is p* ≈ 0.003–0.01,
  where the expected number of nonzeros per row of the factor (768·p ≈ 2–8) drops
  to O(1)–O(10). Reasoning: sparse-RMT localization sets in when connectivity per
  degree of freedom becomes O(1). Confidence: moderate.
- **HA0:** r stays GOE-like through p=0.001 — would mean the 768-dim contraction
  self-averages even at ~0.8 nonzeros/row-pair, and the sparsity lever is weaker
  than the RMT intuition suggests.

### Addendum results

*(pending)*

## Artifacts

- Script: `run_goe_init_dependence.py`, addendum `run_sparsity_crossover.py`
- Results: `results.json`, addendum `results_sparsity.json`
- Registry: exp-078 (add)
