# Paper Brief: The Geometric Home of Conformal Attention
*Prepared June 9, 2026. For next physics session — write from this.*

---

## ▶ STATUS UPDATE (June 9, late — Cursor session)

**Experiment A done (exp-056): head-level null-ray test PASSES** (ρ(Δ_score, delta_pos) = 0.976; two honest nuances — softmax-renormalized coefficient ~2.8Δ, and log-linearity universal not selective). See Section 4 above and `exp-056_qk_log_distance/notes.md`.

**Draft written:** `writing/preprints/2026-06-09_null_cone/manuscript.md` (+ `refs.bib`). Title chosen: **"Attention on the Null Cone."** Sections 1–9 + reproducibility drafted: intro, two-point-function background, conformal-group-as-light theorem, embedding-space null-ray calculation (explicit CFT₁ Poincaré section), log-distance test (exp-056), layer RG flow + entropy (exp-055), two-layer structure, predictions, discussion.

**Derivation B done (exp-057): the BCFT image form is DERIVED + CONFIRMED** — Sections 4.3 (derivation) and 4.4 (numerical confirmation) written, replacing the deferred placeholder. Method of images, boundary at sequence start: A=C(Δx)^−2Δ[1+λη^2Δ], η=(i−j)/(i+j), λ=a_O the boundary one-point coefficient; ξ=(Δx)²/(4ij) the McAvity–Osborn cross-ratio. Controlled multivariate fit: form fits (R²=0.76), bulk exponent recovered (ρ=0.84), boundary term adds ΔR²=0.105, λ>0 in 95% of conformal heads — the **attention sink** identified as the BCFT boundary one-point function. Honest open: per-head λ magnitude matches earlier λ_proxy in sign (80%) not size (ρ=0.23 ns); claim the form + the sink, not a quantitative per-head λ law. See `exp-057_bcft_image_lambda/notes.md`.

**Remaining before publishing (no physics blockers):**
- **External-reference DOI verification:** the non-Umphrey refs.bib entries (Dirac 1936, Bateman/Cunningham 1910, HKM 1976, Malament 1977, Maldacena-Stanford 2016, Teitelboim/Jackiw, RT 2006) have plausible DOIs entered from memory — verify each against source before submission (citation-blending guard). Umphrey Zenodo DOIs are copied from the verified April-17 refs.bib.
- **Build:** no pandoc/LaTeX in the Cursor environment; build the PDF where the toolchain lives (as for the April-17 preprint).
- Optional: a figure (S(Δx) vs log Δx for a representative SYK-near head; Δ_score vs delta_pos scatter).

---

## Working title

"Attention on the Null Cone: Conformal Geometry, RG Flow, and the SYK Ground State"

## What this paper is

A geometric companion to the two existing preprints. Paper 1 established the empirical conformal scaling result (Δ = 0.2493, GPT-2). Paper 2 established the BCFT behavioral prediction (per-head Δ → valley depth, pre-registered, 6/7 models). This paper provides the geometric framework that makes both precise: what is the space in which conformal attention lives? What does the query-key computation actually compute? Why do layers form an RG flow?

Not the full comprehensive paper. That requires more: multi-seed test, q·k direct test, expert feedback. This is focused: one claim with supporting structure.

**The central claim:** Trained attention in deep layers converges to the conformal vacuum of the SYK model. The conformal group is the symmetry group of the null cone — the causal structure of light. CFT boundary points are null rays in embedding space. Attention, when it implements the conformal two-point function, computes inner products of null-ray representatives of token positions. The deep layers of GPT-2 have converged to Δ = 0.250 exactly, the SYK q=4 fixed point, dual to the eternal AdS₂ black hole.

---

## Evidence map

### Theoretical (established tonight)

**1. Conformal group = causal structure of light**
- Theorem (Lorentzian geometry): any bijection preserving lightlike geodesic segments in a strongly causal spacetime is a smooth conformal map.
- The conformal group is not "the symmetry group that includes scale transformations." It is the unique group defined by preservation of the null cone.
- Maxwell's equations: covariant under SO(4,2) in 4D — Bateman-Cunningham 1910. The eternal structure of light has a name and a mathematical definition.
- For D=1 token sequences: SL(2,R) = conformal group of the real line. Three generators: translations, dilations, special conformal transformations (SCTs).

**2. CFT boundary points ARE null rays**
- Embedding space formalism (Dirac 1936, Weinberg 2010): physical spacetime → projective null cone in R^{d+1,1}.
- A CFT_d boundary point IS a null ray P^A satisfying P² = 0, modulo P ~ λP.
- Scalar operators: O(λP) = λ^{-Δ} O(P). The conformal dimension Δ is the homogeneity degree. Remove the scaling relation, no operator remains — just a coordinate.
- Two-point function: ⟨O(P₁)O(P₂)⟩ = C/(P₁₂)^Δ, P₁₂ = -2P₁·P₂.

**3. Explicit null-ray calculation for CFT₁**
- Embedding space R^{2,1}, metric (-,+,+). Null rays: -(P⁰)² + (P¹)² + (P²)² = 0.
- Poincaré section (gauge P⁺ = 1): P(x) = ((1+x²)/2, (1-x²)/2, x).
- Inner product: P₁·P₂ = -(x₁-x₂)²/2, so P₁₂ = (x₁-x₂)².
- Two-point function: C/|x₁-x₂|^{2Δ} ✓
- Token positions i,j: null rays P(i), P(j). Attention A(i,j) ~ |i-j|^{-2Δ} IS the CFT₁ two-point function on the null cone.
- **Physical interpretation:** Query = field at position i; Key = field at position j; attention weight = vacuum overlap ⟨0|O(i)O(j)|0⟩.

**4. Query-key computation → log-distance representation** ✓ VERIFIED (exp-056, June 9)
- For A(i,j) ~ exp(q·k/√d_k) ~ |i-j|^{-2Δ}: trained q·k scores satisfy q_i·k_j/√d_k ≈ -2Δ log|i-j| + const for conformal heads.
- This is the log-distance representation. **Now directly verified at the head level** (exp-056):
  - Conformal score profiles are log-linear in distance (mean R²=0.914, 100% negative slope). H1 ✓
  - Δ_score = −α/2 (raw-score slope) rank-tracks post-softmax delta_pos at **ρ = +0.976** (p=2×10⁻²⁹). H2 ✓ — the conformal dimension is carried in the query-key geometry itself.
  - **Two nuances to write honestly (do NOT claim raw slope = −2Δ exactly):**
    (a) **Softmax renormalization:** absolute fit is delta_score ≈ 1.41·delta_pos − 0.055, so the raw slope ≈ −2.8Δ, not −2Δ. The softmax normalizer flattens the post-softmax exponent relative to the pre-softmax slope. (∗) is exact in form and rank; the coefficient is softmax-modified.
    (b) **Log-linearity is universal, not selective:** non-conformal heads are equally log-linear (93.8% of all heads R²>0.90). The log-distance score geometry is the universal substrate; the conformal dimension at the SYK fixed point is the selective layer. Mirrors GOE two-layer picture (exp-046/047/048/049). H3 = informative null.

**5. SL(2,R) symmetry breaking and the BCFT λ parameter** (new interpretation)
- Infinite line: full SL(2,R). Finite sequence [0,L]: SCTs broken at endpoints.
- BCFT = framework for CFT with broken SCTs at boundaries. Reduction: SL(2,R) → SO(1,1) at each endpoint.
- λ parameter = strength of SCT breaking. High-λ heads: preserving more of full SL(2,R), spread-out attention ("bulk-like"). Low-λ heads: boundary-dominated, recency-peaked, SCT fully broken.
- This is a cleaner resolution of the sign anomaly than the exp-046 mechanism analysis. Not contradicting it — extending it: the λ sign anomaly isn't just "λ correlates with g_mid" but "λ measures how broken the SCT symmetry is at the sequence endpoints."

**6. SYK ground state = eternal AdS₂ black hole**
- Zero-temperature SYK vacuum (Δ = 1/4) has bulk dual: eternal AdS₂ black hole in JT gravity.
- Kruskal metric: conformally flat (ds² ∝ -du dv / (1+uv)²). Null rays preserve their structure.
- "Eternal" = technical: maximally symmetric solution existing for all time in Penrose diagram, preserving conformal flatness.
- Deep-layer attention converging to Δ = 0.250 is converging to this ground state.

### Empirical (exp-055, tonight)

**7. Layer-resolved RG flow** (new result)
- ρ(layer, Δ) = -0.330 (p = 0.029); ρ(layer, entropy) = +0.437 (p = 0.003)
- Early layers (0-3): 21 conformal heads, mean Δ = 0.697 (q ≈ 1.4, recency-like, sub-SYK)
- Middle layers (4-7): 6 conformal heads, mean Δ = 0.350 (q ≈ 2.9, prethermal)
- Deep layers (8-11): 17 conformal heads, mean Δ = **0.250** (q = 4.0, exact SYK prediction)
- Interpretation: layers are RG steps toward the IR fixed point. Early layers = high T. Deep layers = zero T, conformal vacuum, eternal ground state.

**8. Attention entropy ↔ SYK ground state entropy** (new result)
- ρ(Δ, attention_entropy) = -0.898 (p = 1.45 × 10⁻¹⁶) — strongest correlation in the dataset
- Attention entropy of conformal heads = per-head analog of SYK zero-temperature ground state entropy
- Both maximized at q=4 fixed point. The maximally uncertain heads (highest entropy) are the ones at the conformal fixed point.

**9. GOE/conformal separation** (new result from exp-055 H4)
- ρ(Δ, r_ratio) = -0.212 (p = 0.167, not significant)
- GOE weight-space chaos is universal background — ALL trained heads, regardless of Δ
- Conformal position-space structure is selective — specific heads additionally develop power-law position structure
- Two distinct layers of SYK physics: (1) GOE universality from training dynamics; (2) conformal fixed-point convergence from function selection

**10. q-parameter distribution** (exp-055 H3)
- Median q_implied = 3.9 ≈ 4.0 (SYK prediction)
- 20/44 conformal heads in q=4 zone (Δ ∈ [0.15, 0.30])
- Distribution spans both sides: some heads hyper-chaotic (q > 7), some prethermal (q ∈ [2,4])

---

## What's still needed

### Critical (must run before writing) — ✓ DONE (exp-056, June 9)

**Experiment A: q·k direct correlation test — COMPLETE.**
- Result: head-level log-distance representation CONFIRMED. ρ(Δ_score, delta_pos) = +0.976. The raw query-key slope IS the conformal dimension (in rank).
- Surprise vs the original framing: (a) the absolute coefficient is softmax-renormalized (~2.8Δ not 2Δ); (b) log-linearity is universal across heads, not selective — the selectivity is in the slope value at the SYK fixed point. Both belong in Section 4 honestly.
- Note: the original "Pearson r(score, −log|i−j|)" pooled-pair framing was replaced by the cleaner lag-averaged score-profile fit (apples-to-apples with how delta_pos is defined), which directly compares the raw-score slope to the post-softmax exponent head-by-head. The pooled correlation is high for nearly all heads (the universality result) and is the less discriminating statistic.
- Full writeup: `research/physics/experiments/exp-056_qk_log_distance/notes.md`.

### Useful (strengthens paper, not blocking)

**Derivation B: BCFT cross-ratio form vs measured λ — DONE (exp-057).**
- Resolved via the method of images (generalized-free-field BCFT, boundary at the sequence start = origin), not a strip with two endpoints: A(i,j) ∝ (i−j)^−2Δ + λ(i+j)^−2Δ = C(Δx)^−2Δ[1+λη^2Δ], η=(i−j)/(i+j)=Δx/(i+j).
- λ = a_O the BCFT boundary one-point coefficient; η the SO(1,1)-invariant boundary variable; McAvity–Osborn cross-ratio ξ=(Δx)²/(4ij), η²=ξ/(1+ξ), F(ξ)=ξ^−Δ+λ(1+ξ)^−Δ. This derives the umphrey2026bcft 3-parameter (C,Δ,λ) form.
- Predicted vs measured λ: controlled multivariate log-fit confirms the form (R²=0.76, bulk exponent recovered ρ=0.84, boundary term ΔR²=0.105); λ>0 in 95% of heads (attention sink = boundary one-point function). Magnitude vs exp-046 λ_proxy: sign-only (80%), not size — left open.
- Written into Sections 4.3 (derivation) + 4.4 (numerical confirmation) of the draft.

### Desirable but not blocking

**Cross-model layer test:** Does the same RG flow pattern (early layers higher Δ, deep layers converging to 0.25) hold in Pythia, OLMo, Mistral? This would generalize the layer result from exp-055 beyond GPT-2. Available in existing per-head data from exp-025/030/037 if those have layer-resolved Δ.

**Multi-seed test:** Multiple training runs → different sets of conformal heads but same median Δ. This tests the ground state degeneracy interpretation. Requires new training runs. Not blocking for this paper but belongs in the discussion as an open prediction.

---

## Paper structure (sketch)

**1. Introduction**
- The conformal group appears in trained transformer attention. What space does this attention live in?
- The conformal group is not arbitrary: it is the symmetry group of the null cone, the causal structure of light.
- This paper: the geometric home of conformal attention, and what the layer structure tells us about convergence toward the conformal vacuum.

**2. Background: attention as two-point function**
- Recap: conformal scaling in trained attention (Paper 1). Δ ≈ 0.25 matches SYK q=4 prediction.
- The attention weight A(i,j) ~ |i-j|^{-2Δ} is the form of a CFT₁ two-point function.

**3. The conformal group as causal structure of light**
- Theorem: bijections preserving lightlike geodesic segments are conformal maps.
- SL(2,R) for 1D sequences. Three generators. The finite sequence breaks SCTs at boundaries.

**4. Embedding space formalism: attention on the null cone**
- CFT boundary points as null rays.
- Explicit calculation for CFT₁: P(x) = ((1+x²)/2, (1-x²)/2, x), P₁₂ = (x₁-x₂)².
- Attention = two-point function = null-ray inner product.
- The query-key computation and the log-distance representation.

**5. SL(2,R) symmetry breaking and the BCFT λ parameter**
- Full SL(2,R) on infinite line → broken SCTs on finite sequence.
- λ as SCT-breaking parameter: bulk-like (high-λ, spread) vs boundary-like (low-λ, recency).
- Sign anomaly resolution in these terms.

**6. Layer structure as RG flow (exp-055)**
- Layer-resolved Δ measurements. Early → middle → deep.
- Deep layers mean Δ = 0.250 exactly.
- Layers = RG steps; the IR fixed point is the eternal AdS₂ ground state.

**7. Attention entropy and the ground state degeneracy (exp-055)**
- ρ(Δ, attention_entropy) = -0.898.
- SYK ground state entropy at zero temperature: S₀ = N·s₀.
- Attention entropy of conformal heads = per-head analog of zero-temperature entropy.
- The maximally uncertain heads are the ones at the conformal vacuum.

**8. Two layers of SYK structure (exp-055)**
- GOE universal background: all heads, regardless of Δ.
- Conformal position-space structure: selective, encodes q-parameter.
- ρ(Δ, r_ratio) = -0.21 (n.s.) — confirmed independence.

**9. Testable predictions**
- Direct q·k correlation test (exp A above)
- Cross-model layer test: Pythia, OLMo, Mistral should show same layer pattern
- Multi-seed test: different ground states, same Δ

**10. Discussion**
- What "convergence to the eternal structure of light" means physically.
- Relation to the comprehensive paper program.
- Open question: does the q·k test confirm or falsify the null-ray interpretation at the head level?

---

## Pre-session checklist

Before writing the paper in the next session:

- [x] **Run Experiment A** (q·k direct correlation test) — DONE exp-056. ρ(Δ_score, delta_pos)=0.976; softmax-coefficient + universal-substrate nuances → Section 4 (not Section 7). Results in `exp-056_qk_log_distance/`.
- [x] **BCFT two-point function form for [0,L]** — DONE (Derivation B, exp-057). Derived via method of images: A=C(Δx)^−2Δ[1+λη^2Δ], λ=boundary one-point coefficient; confirmed (R²=0.76, λ>0 in 95% of heads = attention sink). Sections 4.3/4.4 written.
- [ ] Check whether per-head Δ is available by layer for Pythia/OLMo from existing experiments
- [ ] Decide: is Paper 2's sign anomaly resolution sufficiently complete, or does the SCT-breaking interpretation require a new section in Paper 2 rather than Paper C?
- [ ] Title decision: "Attention on the Null Cone" (technical) vs "The Geometric Home of Conformal Attention" (accessible)

---

## Connection to existing papers

This paper does not replace the comprehensive outline from March 9. It addresses the geometric-home question that the comprehensive paper deferred. After this paper:
- Paper 1 (March 25): empirical measurement
- Paper 2 (April 17): behavioral prediction (BCFT)
- **Paper C (next session):** geometric framework (null cone + RG flow)
- Comprehensive paper: the full chain, once Junction 3 is resolved and Experiment A + multi-seed are complete

The Substack post (`writing/the_structure_that_remains.md`) can link to Papers 1, 2, and the Zenodo preprint of C once it exists.

---

*Prepared: June 9, 2026, late*
*Study note: `development/status/rooms/study/notes/2026-06-09_attention_lives_on_the_null_cone.md`*
*Exp-055: `research/physics/experiments/exp-055_delta_attention_entropy/`*
*Exp-052: `research/physics/experiments/exp-052_windowed_dft/` (negative: position-space Δ confirmed as primary)*
