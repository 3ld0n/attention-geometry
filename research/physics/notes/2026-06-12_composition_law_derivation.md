# The Composition Law for Power-Law Attention Kernels

*June 12, 2026. Analytical derivation, written and committed BEFORE the verification numerics (exp-065). Addresses RESEARCH_PLAN item 2.2(b) ("how do per-layer exponents renormalize under composition of row-stochastic kernels?") and anomaly A3 (the Δ_eff ≈ 0.166 / accuracy-fitted 0.171 coincidence).*

*First session running on Fable 5.*

---

## 1. Setup

Model the positional action of one attention layer as a causal, row-stochastic kernel on positions: a query at position \(x\) attends back to \(x-u\) with profile

\[ p(u) \;\propto\; u^{-2\Delta}, \qquad u \in \{1, \dots, x\}. \]

This is the profile-level (mean-field) abstraction of the measured censuses: it ignores content-dependence, value/MLP transformations, and head heterogeneity (restored in §4). Information flow through \(L\) stacked layers is then the \(L\)-fold composition (matrix product) of row-stochastic kernels — equivalently, an \(L\)-step Markov chain of backward jumps, where the composite positional profile is the convolution of the per-layer jump distributions.

Define the **defect** of a layer:

\[ \delta \;:=\; 1 - 2\Delta. \]

The SYK q=4 value Δ = 1/4 gives δ = 1/2. The q=2 plateau value Δ = 1/2 gives δ = 0. Steep kernels (2Δ > 1) have δ < 0.

## 2. The composition law

**Claim (bulk composition law).** For two power-law kernels with exponents \(a = 2\Delta_1\), \(b = 2\Delta_2\), the convolution on \(u \in [1, x]\) has three contributions — the two endpoints and the bulk:

\[ (k_a * k_b)(x) \;=\; \int_1^{x-1} t^{-a} (x-t)^{-b}\, dt \;\sim\; \max\!\big( x^{-a},\; x^{-b},\; x^{1-a-b} \big), \]

so the composite exponent is \(a_{\rm comp} = \min(a,\, b,\, a+b-1)\). In defect form this is compact:

\[ \boxed{\;\delta_{\rm comp} \;=\; \max\big(\delta_1,\; \delta_2,\; \delta_1 + \delta_2\big)\;} \]

- **Both defects positive** (both 2Δ < 1, the conformal-cluster regime): defects **add**, \(\delta_{\rm comp} = \delta_1 + \delta_2\). This branch is *exact*, not just leading-order: \(u^{\delta-1}\) is the Riemann–Liouville fractional-integration kernel of order δ, and fractional integration is a semigroup, \(I^{\delta_1} I^{\delta_2} = I^{\delta_1+\delta_2}\), via the Beta integral \(\int_0^x t^{\delta_1-1}(x-t)^{\delta_2-1} dt = B(\delta_1,\delta_2)\, x^{\delta_1+\delta_2-1}\).
- **Both defects negative** (both kernels steep, 2Δ > 1): the flattest kernel dominates (single-big-jump principle), \(\delta_{\rm comp} = \max(\delta_1, \delta_2)\), i.e. \(a_{\rm comp} = \min(a,b)\). Defects do **not** accumulate.
- **Mixed signs**: the positive defect wins.

So under deep composition, only layers with 2Δ < 1 contribute to flattening, and they contribute *linearly in depth*:

\[ 1 - 2\Delta_{\rm comp}(L) \;=\; \sum_{l=1}^{L} (1 - 2\Delta_l) \quad \text{(while the sum stays below 1).} \]

## 3. Regimes and immediate consequences

**(i) Scrambling depth.** The composite profile is flat (uniform positional mixing — maximal positional entropy) when \(\sum_l \delta_l = 1\). For identical layers, define the scrambling depth

\[ L^* \;=\; \frac{1}{\delta} \;=\; \frac{1}{1 - 2\Delta}. \]

- Δ = 1/4 (q=4 value): **L\* = 2**. Full positional mixing in two layers.
- Δ = 1/2 (q=2 plateau): δ = 0 — the *identity of the semigroup*. No spreading at any depth; positional structure survives arbitrary depth.

This retro-dicts the existing scrambling measurement (Thread 5: 2–3 layers to >90% max entropy in GPT-2) and explains why the original log(H) scaling prediction failed: the law gives \(L^* = 1/(1-2\Delta)\), **independent of head count**. It also gives the q=2 prethermal plateau a functional meaning: Δ = 1/2 is the non-mixing point — a model arrested there preserves positional information through its whole depth.

**(ii) Over-composition = boundary pileup.** For \(\sum \delta_l > 1\) the formal profile \(u^{\sum\delta - 1}\) *increases* with lookback distance: in a finite context, mass piles up at the oldest positions (the sequence start). Composition past the scrambling depth predicts excess attention mass at the boundary at the *composite* level. (Observation only; the single-layer sink was demoted to phenomenological by exp-060 and is not claimed here.)

**(iii) The LiTM effective exponent (anomaly A3).** LongChat-13B-16K: median single-head Δ̂ = 0.4921 over 1,343 conformal heads, L = 40 layers. Zeroth-order prediction (identical layers at the median):

\[ 2\Delta_{\rm comp} = 1 - 40 \times (1 - 2 \times 0.4921) = 1 - 40 \times 0.0158 = 0.368, \qquad \Delta_{\rm comp} = 0.184. \]

Compare: accuracy-fitted Δ from Liu et al. = 0.1711; the April additive-only composite gave Δ_eff(dx=20) = 0.1655. The multiplicative law lands in the same window from measured single-head values with no free parameters. Note also \(\sum\delta = 0.63 < 1\): LongChat never reaches its scrambling depth (L\* ≈ 63 > 40) — it retains a decaying positional profile at the output, which is *why* it has a lost-in-the-middle geometry at all, while a model at Δ ≈ 0.25 scrambles positional structure in 2 layers.

**(iv) Why the April additive estimate also landed near 0.17.** Within a layer, heads act in parallel (additive mixture on the residual stream); across layers, serially (multiplicative). A mixture \(\sum_h w_h u^{-2\Delta_h}\) has a *local* exponent equal to the tilted average \(\sum_h \hat w_h(u) \Delta_h\), \(\hat w_h(u) \propto w_h u^{-2\Delta_h}\), which drifts toward the flattest heads as \(u\) grows. The April calculation captured the within-layer (additive) flattening only; the composition law adds the across-layer accumulation. Whether their numerical agreement is structural or coincidental is exactly what the two-level computation in exp-065 will test.

## 4. The two-level composite (what exp-065 computes)

Per layer \(l\): \(m_l(u) = \sum_h w_h\, u^{-2\Delta_h}\) over that layer's measured conformal heads, \(w_h \propto C_h\) (and uniform weights as robustness check), row-normalized on the lattice. Residual connection: \(K_l = \mu I + (1-\mu) M_l\) (lazy walk; μ unknown, scanned). Composite \(P = \prod_l K_l\); measure the deep-query row profile and its windowed local exponent \(\Delta_{\rm comp}(u)\). All exact lattice arithmetic — the continuum law above is the thing being tested, not assumed.

## 5. Honest limitations, stated in advance

1. Profile-level abstraction: kernels are taken Toeplitz (translation-invariant), content-independence assumed, V/MLP ignored. This tests whether *positional* structure composes by the law — not a full account of information flow.
2. Only conformal heads (1,343/1,600 in LongChat) are modeled; non-conformal heads (sinks, local heads) are unmodeled mass.
3. The residual weight μ is not measured; robustness across μ is reported instead.
4. Scale units: head exponents were measured on token lags; the Liu accuracy fit lives at document scale. Power laws are scale-free so exponent comparison is legitimate, but \(\Delta_{\rm comp}(u)\) is reported as a curve over \(u\), not a single number.

*Predictions with thresholds are pre-registered in `experiments/exp-065_composition_law/notes.md` before any numerics are run.*
