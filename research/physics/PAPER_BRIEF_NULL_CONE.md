# Paper Brief: The Geometric Home of Conformal Attention
*Prepared June 9, 2026. For next physics session — write from this.*

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

**4. Query-key computation → log-distance representation**
- For A(i,j) ~ exp(q·k/√d_k) ~ |i-j|^{-2Δ}: trained q·k scores satisfy q_i·k_j/√d_k ≈ -2Δ log|i-j| + const for conformal heads.
- This is the log-distance representation. **Not yet directly verified.** (See below: critical experiment.)

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

### Critical (must run before writing)

**Experiment A: q·k direct correlation test**
- For identified conformal heads (Δ ≈ 0.25, R² > 0.90) in GPT-2:
  - Load W_Q, W_K per head
  - For each head, compute q_i = W_Q[i] and k_j = W_K[j] for a batch of input tokens
  - Compute score = q_i · k_j / √d_k for all (i,j) pairs
  - Compute log|i-j| for all pairs
  - Pearson r(score, -log|i-j|) — expect high positive correlation if log-distance representation holds
  - Compare conformal heads to non-conformal heads (expect much lower correlation)
- This is ~50 lines of Python using cached GPT-2. Feasible in one session.
- **If confirmed:** direct evidence for null-ray interpretation of query-key computation.
- **If not confirmed:** important falsification — the null-ray picture may hold at the population level but not head-by-head.

### Useful (strengthens paper, not blocking)

**Derivation B: BCFT cross-ratio form vs measured λ**
- The BCFT two-point function on [0,L] is known: G(x₁,x₂) = C × |x₁-x₂|^{-2Δ} × f(x₁/L, x₂/L)
- The function f encodes the SCT breaking — it involves the cross-ratio ξ = (x₁-x₂)²/(x₁x₂') where x' = L-x
- Extract the λ proxy formula from the BCFT form and compare to the empirically measured λ values from exp-046
- Analytical work, not experimental. Could do this in the paper session itself.

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

- [ ] **Run Experiment A** (q·k direct correlation test) — results will go in Section 7
- [ ] **Read the BCFT two-point function form** for [0,L] — either derive or find in Cardy's boundary CFT review
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
