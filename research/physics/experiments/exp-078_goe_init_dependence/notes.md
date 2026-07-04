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

*(pending — filled after run)*
