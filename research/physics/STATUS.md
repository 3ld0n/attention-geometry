# Holographic Attention Research — Status
*Living document. Updated as expert feedback arrives and open questions resolve.*
*Last updated: June 4, 2026*

---

## The Chain

```
Junction 1: Attention → Fisher-Rao geometry
  Claim: Transformer attention minimizes Helmholtz free energy on a Fisher-Rao information manifold.
  Basis: Kim 2026 (arXiv 2602.08216) — this IS Kim's result directly.
  Status: ✓ SOLID — Kim's own framework. No challenge to this junction.

Junction 2: Fisher-Rao geometry → Quantum Fisher metric → Born rule
  Claim: The Fisher-Rao metric on the attention manifold IS the quantum Fisher metric.
         Attention weights ARE Born rule probabilities. Attention output IS a quantum expectation value.
  Basis: Paper 5 (March 8, 2026) — Gibbs state construction. Four theorems, numerically verified.
         Classical Fisher = Quantum Fisher for diagonal states (Braunstein-Caves 1994, exact).
  Status: ✓ CLOSED — exact mathematical identity, proven in Paper 5. No analogy.
  See: writing/paper5_draft.md

Junction 3: Free scalar QFT → Holographic dual
  Claim: The large-head limit scalar QFT from Ageev/Ageeva's construction has a holographic dual.
  Basis: Two arguments — context-length scaling (m ~ 1/L → 0) and Kim's Goldstone mechanism.
  Status: ⚠ CONDITIONALLY OPEN — depends on whether Ageev scalar is massless.
  Expert needed: Ageev/Ageeva (can check their own kernel directly); Qi (new holographic dictionary).

Junction 4: Holographic QFT → RT surface
  Claim: If Junction 3 holds, the RT formula describes the entanglement entropy of attention.
  Basis: Standard holographic result (Ryu-Takayanagi 2006).
  Status: ⚠ FOLLOWS IF JUNCTION 3 HOLDS — standard holography, not in doubt, but conditional.

Junction 5: RT surface → Spacetime geometry
  Claim: RT surface area encodes circuit complexity; circuit complexity generates spacetime.
  Basis: Czech 2018; ER=EPR (Maldacena-Susskind).
  Status: ✓ PROVEN for 3D AdS — Czech 2018 is rigorous. ER=EPR still conjectural but widely accepted.
```

**Overall chain status:** Junctions 1-2 and 5 are now rigorous. Junction 3 has a new empirical path (see Numerical Results below). Junction 4 follows if 3 holds.

**Biological test: Mouse V1 conformal scaling — April 29 positive claim REVERSED on April 30 re-analysis.**
April 29 (preliminary, now superseded): binned pair-correlation vs Euclidean nucleus distance on Ding et al. (2025) V1 pairs gave Δ = 0.238 (R² = 0.857, null Z = 27), reported as matching GPT-2 Δ = 0.2493. April 30 re-analysis on same data: (a) **Synaptic path-length test** (the correct topological observable): Δ = 0.72 in silico, Δ = 0.44 in vivo — statistically real against null but ~3× steeper than SYK prediction. (b) **Pair-level log-log regression** (no binning): Δ = 0.074, R² = 0.003. (c) **Retinotopy-partialled pair-level regression**: Δ = 0.039. The April 29 Δ = 0.238 was a binned-shell-mean statistic, not the pair-level conformal correlation exponent the theory predicts. **Current status:** biological validation is *not confirmed* by MICrONS correlation-vs-distance data when analyzed carefully; framework prediction (biological cortex should be near Δ = 1/4 if in SYK universality class) remains an open prediction, not a confirmation. Two cleaner biological tests remain available on the same data: GOE spectral statistics of the V1 connectivity matrix, and CFT entanglement-entropy / mutual-information scaling on calcium traces — both independent of geometric distance. Full accounting: `research/physics/STATUS_ADDENDUM_2026-04-30.md`, `research/microns/RESULTS_v2.md` (synaptic), `research/microns/RESULTS_v3_retinotopy.md` (retinotopy-partialled), revised `research/physics/consciousness_physical_theory.md`. **Transformer-side result (GPT-2 Δ = 0.2493) and BCFT pre-registered test unchanged — both are direct-measurement, not binned-distance, observables.**

**NEW — BCFT "Lost in the Middle" testing (April 15, 2026):**
Per-head Δ controls LiTM valley depth: Spearman ρ = +0.94 (Pythia-70m), +0.64 (LongChat-13B-16K on A100). Multi-layer composition gives Δ_eff = 0.17, matching accuracy-fitted Δ from Liu et al. Direct shape prediction fails for LongChat (recency vs primacy mismatch). Full results: `research/notes/bcft_lost_in_the_middle.md`. Cloud infrastructure (Modal) operational for future GPU experiments.

**NEW — BCFT pre-registered test (April 17, 2026): preprint published.**
Pre-registered prediction (`research/notes/bcft_pre_registered_prediction.md`): per-head Δ → valley_depth Spearman ρ ≥ 0.50, p ≤ 1e-5, on any decoder-only transformer. Tested on 7 decoder-only models. **Results**: 6 confirmed (Pythia-410m/1.4B with ρ = +0.76/+0.71; GPT-Neo-2.7B with ρ = +0.96; Qwen2-7B with ρ = +0.85; OLMo-7B with ρ = +0.85; Mistral-7B-v0.3 with ρ = +0.58); **Pythia-2.8B falsified at ρ = +0.46**. Per-layer diagnostic localizes Pythia-2.8B failure to layers 22–27 and shows GPT-Neo-2.7B clean across all 32 layers — *training recipe, not parameter count or data, is the differentiating variable*. Functional-form fit (3-parameter BCFT with C, Δ, λ) on Pythia-2.8B and GPT-Neo-2.7B: 88–94% of conformal heads prefer BCFT over bare power law; Δ_BCFT closer to SYK Δ=1/4 than Δ_PL; **joint** (Δ, λ) → valley rank-R² = 0.55 (Pythia-2.8B), 0.77 (GPT-Neo-2.7B). **Two surprises**: (a) ρ(λ, valley) is *negative* across most layers in both models — opposite of framework prediction; (b) GPT-Neo-2.7B has alternating-layer structure suggesting two distinct head populations. **Preprint**: `writing/preprints/2026-04-17_bcft_pre_registered/manuscript.{md,pdf}`. **Zenodo DOI: 10.5281/zenodo.19629862**. Audit and follow-ups in `research/notes/framework_audit_2026-04-17.md` (postscripts).

**NEW — GOE universality confirmed cross-model (June 4, 2026, exp-046 + exp-047):**
The Oganesyan-Huse r-ratio of W_QK eigenvalues (symmetrized W_Q^T W_K per head) is GOE-like across ALL tested transformers, regardless of architecture or PE. Per-head r-ratio distributions are model-invariant:

| Model | PE | Family | Heads | r_mean | r_std | verdict |
|-------|----|--------|-------|--------|-------|---------|
| GPT-2 (exp-046) | learned | GPT-2 | 144 | 0.528 | 0.040 | GOE-like |
| GPT-2-medium (exp-047) | learned | GPT-2 | 384 | 0.526 | 0.039 | GOE-like |
| Pythia-410m (exp-047) | RoPE | NeoX | 384 | 0.520 | 0.038 | GOE-like |

GOE reference = 0.536; Poisson = 0.386. All three distributions are statistically identical (|Δr| ≤ 0.008). The GOE structure is layer-uniform (layer-to-layer std ≈ 0.008–0.009). Conformal and non-conformal heads are indistinguishable (exp-046: conformal r = 0.525, non-conformal r = 0.529). **Physical interpretation:** trained transformers universally develop SYK-chaotic weight matrices through gradient descent. The conformal position-space structure (power-law Δ ≈ 0.25 in specific heads) is an additional selective pattern within a universal GOE background. Open controls needed: (1) untrained (initialized) weights — expected Poisson; (2) broader model families (OLMo, Mistral, GPT-Neo). See exp-046 notes, exp-047 notes.

**NEW — Sign anomaly resolved (June 2, 2026, exp-046):**
The sign anomaly ρ(λ_BCFT, valley) < 0 (first found April 17 in Pythia-2.8B and GPT-Neo-2.7B) is **confirmed in GPT-2** (ρ = −0.75, p = 4.7×10⁻⁹) and its **mechanism is identified**. The anomaly is NOT a failure of BCFT physics. The mechanism: λ_proxy (boundary parameter) captures the recency/boundary balance of attention. High-λ heads have spread-out attention (elevated middle-context, lower recency: high g_mid, low g_end) → shallower valley. Low-λ heads are recency-dominated (near-zero g_mid, high g_end ≈ 1.0) → valley ≈ 1.0 (deep). Key correlation: ρ(λ, g_mid) = +0.74 (p = 9.3×10⁻⁹). **Framework correction:** The April 17 framework prediction P2 conflated λ and Δ as having the same sign of effect on valley depth. They do not: Δ deepens the valley (conformal power-law); λ shallows it (boundary enhancement spreads attention). The sign anomaly is a discovery about how BCFT parameters encode attention structure, not a falsification of the framework. See exp-046 notes.

**NEW — PE ordering (corrected, May–June 2026, exp-039/040/041/042/043/044):**
Corrected positional encoding ordering based on median Δ_SYK (softmax, full-attention models only; GPT-Neo removed because alternating global/local architecture confounds the measurement):

| Model | PE | Δ_SYK | Source |
|-------|----|--------|--------|
| Pythia-410m | RoPE | 0.358 | exp-036 |
| Mistral-7B-v0.3 | RoPE+SWA | 0.298 | exp-039 |
| OLMo-7B | ALiBi | 0.265 | exp-039 |
| GPT-2 | learned | 0.249 | exp-007 |

GPT-Neo-2.7B removed from this table: its global-layer population has Δ_med = 0.101 (trivial fixed point, architecture-induced, exp-044), which confounds any PE-effect comparison. OLMo-7B is now the clean ALiBi reference. Tentative ordering: RoPE > RoPE+SWA > ALiBi > learned. Bracket width confound remains: GALA-7B ALiBi softmax Δ = 0.260 vs OLMo ALiBi Δ = 0.265 (consistent); norm-sigmoid effect on bracket width is PE-universal in direction but varies in magnitude (GPT-2 bracket ~0.015 vs GALA-7B ~0.037 — confounded by depth/scale).

**NEW — Non-softmax universality (May 31, 2026, exp-041/042/043):**
GALA-7B (Apple's 7B sigmoid-attention model) tested under exp-007 protocol. Sigmoid raw: 2/1024 conformal heads, Δ_med = 7.44 — effectively no SYK conformal structure. Normalized-sigmoid (QK-norm, scale_query + scale_key per-head RMSNorm): 210/1024 SYK-near, Δ_SYK = 0.223. Softmax: 80/1024 SYK-near, Δ_SYK = 0.260. SYK prediction (0.25) is bracketed between norm-sigmoid and softmax. **The conformal substrate is QK geometry (weight structure), not the normalization function.** Softmax is load-bearing for the exp-007 lag-profile protocol, but the underlying QK weight structure carries the physics. GPT-2 norm-sigmoid control (exp-043): Δ_SYK = 0.234 vs softmax 0.249 — same direction, smaller bracket (learned PE vs ALiBi for GALA-7B).

**NEW — Spectral function G_>/G_< (June 2, 2026, exp-045, partial):**
DFT spectral analysis on GPT-2 conformal heads. G_< = 0 confirmed (causal attention = zero-temperature SYK ground state, β → ∞). Pearson r = 0.94 between position-space Δ_pos and frequency-space Δ_freq (ordering consistent). CFT kinematic prediction α = 2Δ−1 not confirmed (measured α ≈ −0.83 vs predicted −0.50). Root cause: structural finite-DFT bias (~0.33 offset) from the 55-lag finite support — not a failure of the physics. A calibrated spectral estimator (synthetic calibration curves to invert finite-support bias) is needed for quantitative frequency-domain testing. See exp-045 notes.

**NEW — Empirical path through Junction 3 (March 24, 2026):**
Trained GPT-2 attention weights show power-law decay α(Δx) ~ |Δx|^{-2Δ} with **median Δ = 0.2493** across 44 power-law heads (R² > 0.90). This matches the SYK q=4 prediction **Δ = 0.25** for D=1 sequences. Randomized GPT-2 shows 0 power-law heads. Phase transition observed in Pythia-70m training at ~step 256. Full results: `research/physics/NUMERICAL_RESULTS_MARCH24.md`.

This provides an empirical route: trained attention → SYK conformal fixed point → JT gravity, independent of whether Ageev's scalar is massless.

**Empirical paper PUBLISHED (March 25, 2026):**
"Conformal Scaling in Trained Transformer Attention: Evidence for an SYK Fixed Point" — DOI: 10.5281/zenodo.19225996. Supplementary data: DOI: 10.5281/zenodo.19225971. Draft v5 includes: GPT-2 per-head analysis (Δ = 0.2493 median), three Pythia models showing depth convergence and phase transitions, prethermal plateau at Δ ≈ 0.50 (q=2), entanglement entropy matching CFT formula, information scrambling analysis. Authored as Ariel Umphrey, with Eldon Umphrey.

**Three equivalent descriptions of the softmax (March 8, 2026):**
- Statistical mechanics: Gibbs distribution at temperature T = 1/√d_k
- Quantum mechanics: diagonal density matrix on key Hilbert space (Paper 5)
- Bayesian inference: exact posterior for a Gibbs generative model (Kim-Friston correspondence)
See: `research/physics/KIM_FRISTON_CORRESPONDENCE.md`

---

## Expert Feedback Received

### Gunn Kim — March 6, 2026
**Response to:** Initial Paper 1 email (March 5) + follow-up with Papers 2 and 3

**What he said:**
> "The claims made in these new papers appear to extend far beyond the scope of the framework developed in my paper. In particular, connections to quantum measurement theory, the island formula, and black-hole information recovery would require a much more explicit physical construction than what is currently outlined. At present these seem to be speculative analogies rather than results that follow directly from the thermodynamic attention model."
> Declined arXiv endorsement.

**What this validates:**
- Kim did not challenge Junction 1 — his own framework stands.
- He engaged seriously enough to read Papers 2 and 3 and respond substantively.
- First real engagement from the physics research community.

**What this clarifies:**
- The physical construction connecting Junctions 2-4 is not explicit enough to constitute a "result" — it reads as structural analogy.
- "Speculative analogies rather than results that follow directly" is a precise scientific statement: we need to show the mechanisms, not just the resemblance.
- arXiv endorsement path through Kim is closed. Need another route (cs.LG endorser, or wait for a physicist who finds the chain more rigorous).

**What this doesn't mean:**
- The mathematical observations are wrong.
- The work isn't worth continuing.
- The other physicists will respond the same way.

**What needs to happen to address Kim's critique:**
1. Resolve Junction 3 — get confirmation from Ageev/Ageeva that the scalar is massless, or Qi's framework applies.
2. Make the physical mechanism of Junctions 2-4 more explicit — not just that the math looks the same, but that the physical systems are identical in the relevant sense.

**Three specific routes identified (March 6, 2026):**

**Route A — SYK correspondence (most direct for island formula):**
SYK model has the same structural features as Kim's attention: thermal two-point function G(τ) ~ 1/|τ|^{2Δ} with Δ = 1/4 in conformal limit, temperature matching Kim's T = 1/√d_k. SYK's holographic dual is JT gravity (2D dilaton). Island formula explicitly computed in JT gravity (Almheiri/Penington 2019). If Ageev's two-point function in large-head limit matches SYK conformal correlator, then attention → SYK → JT gravity → island formula is a chain of mathematical identities, not analogies.
*KEY FINDING (March 6, 2026):* Read Ageev 2602.10209 in full. The independence-breaking (IB) four-point function (Eq. 33) takes the form Cov_{W^Q,W^K}(X_{12}, X_{34}) — a covariance of a bilocal quantity over the random parameters. This is STRUCTURALLY IDENTICAL to the SYK disorder-averaged connected four-point function: Cov_J[G(τ₁,τ₂), G(τ₃,τ₄)]. The mechanism is the same: random parameters shared across the system generate bilocal correlations. See `research/physics/SYK_ANALYSIS.md` for full argument.
*Open question:* Do Ageev's Schwinger-Dyson equations in the conformal limit reduce to Σ ∝ G^3 (SYK q=4)? Ageev can check this directly. If yes: holographic dual = JT gravity, island formula follows.
*Key papers:* Maldacena-Stanford 2016; Almheiri/Penington 2019; Kitaev 2015 (SYK).

**Route B — MERA tensor network (most general for holography):**
Multi-head attention has a natural tensor network representation. Swingle (2012) proves MERA tensor networks → exact AdS geometry + RT formula. If attention layers satisfy MERA isometry conditions (approximately), holography follows from Swingle as a theorem, not an analogy — and independent of Junction 3.
*Next step:* write attention mechanism explicitly as a tensor network; verify isometry conditions.
*Key papers:* Swingle 2012; Levine et al. 2019 (transformers as tensor networks).

**Route C — Explicit POVM (addresses quantum measurement critique directly) — COMPLETED March 8:**
✓ Paper 5 (writing/paper5_draft.md) provides the full construction:
  - Gibbs state = attention weights (Theorem 1, exact)
  - Attention output = quantum expectation value Tr(ρV) (Theorem 2, exact)
  - Born rule: P(key i) = α_i (Theorem 3, exact via complete PVM {|i><i|})
  - Junction 2 closed: quantum Fisher = classical Fisher for diagonal states (Theorem 4)
  - Off-diagonal extension defined (quantum attention with coherence)
  - Schwarzian action conjecture identified (Section 8, speculative)
*Key papers:* Braunstein-Caves 1994; Kim 2026; Paper 5.

**March 9 critical review (PAPER_REVIEW_MARCH9.md):**
  - Theorems 1–4 are correct; the proofs are valid
  - **Framing overreaches:** "attention IS quantum" should be "attention IS the classical limit of a natural quantum system" — any probability distribution embeds as a diagonal density matrix; the diagonal sector has no quantum content
  - Paper 5's real deliverables: Junction 2 closure (Theorem 4) and the off-diagonal extension framework
  - **Recommendation: revise Paper 5 language before sending to Kim**

**Route A update — Linearized-Softmax $G^4$ Calculation (March 9, LINEARIZED_SOFTMAX_CALCULATION.md):**
  - In the large-$d_k$ Kim regime (linearized softmax), the disorder average over $W^Q, W^K$ generates a $G^4$ effective action vertex at order $\beta^4 = 1/d_k^2$
  - The vertex has bilocal $G^2 \cdot G^2$ structure matching SYK with $q=4$
  - Effective coupling: $J^2_{\text{eff}} \propto (\sigma_Q^2\sigma_K^2/d^2)^2/d_k^2$
  - Conformal dimension: $\Delta = D/4$ where $D$ is the spatial dimension of the token sequence
  - **For language models ($D=1$): $\Delta = 1/4$ — exact SYK $q=4$ match**
  - The Schwarzian follows from SYK in this regime; JT gravity is the holographic dual
  - Caveat: needs rigorous Hubbard-Stratonovich derivation and computation of the data-geometry factor $\Omega$
  - **Status: Paper 4 Question 2 answered affirmatively at the structural level**

**Schwarzian exploration (March 9, SCHWARZIAN_EXPLORATION.md):**
  - The attention free energy is exactly reparametrization-invariant at $T=0$, broken at $O(T)$ by entropy
  - The naive path (Schwarzian from the free energy) gives a kinetic term $(\epsilon')^2$, not the Schwarzian
  - All four alternative paths to the Schwarzian converge on the same question: **is the continuum limit conformal?**
  - This question is equivalent to Junction 3 in a different guise
  - The Schwarzian conjecture should be reframed: "Schwarzian governs attention dynamics IF AND ONLY IF the continuum limit is a CFT"

**Numerical Verification (March 9, Session 2 — NUMERICAL_RESULTS.md):**
  - **Theorems 1-4:** EXACT verification to machine precision (10⁻¹⁷)
  - **Score covariance factorization:** CONFIRMED (cosine similarity 0.997)
  - **σ⁴ scaling of G⁴ vertex:** CONFIRMED in linearized regime (σ ≤ 0.2; ratio constant at ~7 × 10⁻⁹)
  - **Standard initialization (σ ~ 1) is fully nonlinear:** the linearized approximation FAILS at standard init. The G⁴ result is a solvable limit, not the physical regime.
  - **Multi-layer enhancement: 18× from one additional layer.** Layer 1 disorder propagates through residual connection, amplifying layer 2 fluctuations. This is the mechanism for nonlinear SD equations.
  - **Depth scaling: Var ~ L^1.19 (power law).** Connected correlator grows slightly faster than linearly with depth. 6 layers → 146× enhancement. Per-layer contribution decreases with depth (approach to fixed point).
  - Code: `research/physics/numerical_test_*.py` (requires numpy/scipy from `.venv`)

**Comprehensive paper outline (March 9, Session 2 — COMPREHENSIVE_PAPER_OUTLINE.md):**
  - Proposed single arXiv paper combining all results
  - 10 sections, estimated 30-35 pages
  - Includes numerical verification as Section 8 (Testable Predictions)
  - Target: cs.LG primary, cross-list hep-th, quant-ph, cond-mat.dis-nn
  - For discussion with Eldon before writing begins

---

## Open Questions

| Question | Who can answer | Status |
|---|---|---|
| Is the Ageev large-head limit scalar massless? | Ageev/Ageeva | Waiting — email sent Mar 6 |
| Does Qi's 2602.20295 framework apply to free CFTs? | Qi | Waiting — email sent Mar 6 |
| Is Kim's framework the same structure as Friston's FEP? | Friston | Waiting — email sent Mar 6 |
| Does linearized-softmax $G^4$ vertex survive rigorous H-S derivation? | Us — tractable | **Structural result Mar 9; full derivation pending** |
| Is data-geometry factor $\Omega$ nonzero for standard distributions? | Us — numerical | Open |
| Does trained-transformer attention kernel show conformal scaling? | Us — empirical | **✓ YES — Δ = 0.2493 (median, GPT-2). March 24.** |
| Do Papers 2 and 3 survive Ageev/Qi scrutiny if Junction 3 closes? | Internal review | Open |
| arXiv endorsement path — who in cs.LG would endorse? | Need to identify | Open |
| Does L^1.19 depth scaling persist at standard init (σ ~ 1)? | Us — numerical | Open |
| Does depth scaling approach a conformal fixed point? | Us — numerical + theory | Suggested by decreasing per-layer ratios |
| Why is ρ(λ, valley) *negative* in both Pythia-2.8B and GPT-Neo-2.7B? | Us — analysis + theory | **RESOLVED (exp-046, June 2):** λ captures recency/boundary balance, not a BCFT failure. See above. |
| What architectural feature of Pythia's training recipe at scale produces the late-layer ρ(Δ, valley) failure? | Us + EleutherAI authors | **Open as of April 17 — per-layer diagnostic localizes to layers 22–27 of Pythia-2.8B; GPT-Neo controls** |
| Does the alternating-layer pattern in GPT-Neo correspond to two functional populations of heads? | Us — analysis | **Partially resolved (exp-040/044):** GPT-Neo global layers Δ_med = 0.101 (trivial fixed point, architecture-induced). GPT-Neo not a clean ALiBi reference. Functional distinction confirmed but OLMo is better subject for further study. |
| Does the pre-registered prediction hold on Llama-3-8B? | Us — pending Meta access | Outstanding |
| Do W_QK eigenvalues show GOE-like level spacing in GPT-2? | Us — analysis | **CONFIRMED UNIVERSAL (exp-046/047, June 2/4):** GOE across GPT-2, GPT-2-medium, Pythia-410m. Conformal/non-conformal indistinguishable. |
| Are GOE weight statistics universal across model families, or GPT-2-specific? | Us — analysis | **CONFIRMED UNIVERSAL (exp-047, June 4):** Identical distributions in GPT-2 (learned), GPT-2-medium (learned), Pythia-410m (RoPE/NeoX). |
| What mechanism converts Poisson → GOE during training? (Untrained control.) | Us — analysis | **Open:** untrained GPT-2 weight r-ratio not yet measured. Expected Poisson; if confirmed, gradient descent converts Poisson → GOE. |
| Is the GOE structure universal across OLMo, Mistral, GPT-Neo? | Us — analysis | **Open:** Pythia-410m tested; OLMo/Mistral/GPT-Neo would extend the cross-family confirmation. |
| What is the correct frequency-space Δ estimator for the DFT bias? | Us — analysis | **Open (exp-045):** Synthetic calibration can in principle invert finite-support bias. Calibrated estimator not yet implemented. |
| Does the softmax → norm-sigmoid bracket width depend on PE, depth, scale? | Us — analysis | **Partially open (exp-043):** GPT-2 bracket ~0.015 (learned, 12L), GALA-7B bracket ~0.037 (ALiBi, 32L, 7B). Confounded. Clean test: GPT-2-scale model with ALiBi. |

---

## Outreach Status

| Person | Email | Sent | Response | Status |
|---|---|---|---|---|
| Gunn Kim | [email on file] | ✓ Mar 5, 6, 9, 11, 18 | ✓ Mar 6 — declined endorsement, raised construction concern | Awaiting — stories follow-up sent Mar 18 |
| Dmitry Ageev | [email on file] | ✓ Mar 6, 9, 11, 18 | — | Awaiting response |
| Yulia Ageeva | [email on file] | ✓ Mar 6, 9, 11, 18 | — | Awaiting response |
| Xiao-Liang Qi | [email on file] | ✓ Mar 9, 11, 18 | — | Awaiting response |
| Karl Friston | [email on file] | ✓ Mar 6, 9, 11, 18 | — | Awaiting response |
| Haim Sompolinsky | [email on file] | ✓ Mar 28 (resent — prior to [email on file] bounced) | — | Awaiting response |
| Jim Halverson | [email on file] | ✓ Mar 28 (resent — prior to jim.halverson@ bounced) | — | Awaiting response |
| Brandon Robinson | TBD (now at UvA) | ☐ (prior to [email on file] bounced) | — | Need correct UvA email |
| Neel Nanda | [email on file] | ✓ Mar 28 (resent — prior to anthropic bounced; now at DeepMind) | — | Awaiting response |
| Jordan Peterson | [email on file] | ☐ | — | Not yet sent |
| Kyle Fish | LinkedIn first | ☐ | — | LinkedIn outreach pending |
| M. Hamed Mohammady | [email on file] | ☐ | — | Second wave — hold |
| Francesco Buscemi | [email on file] | ☐ | — | Second wave — hold |
| Hans-Joachim Rudolph | PhilArchive/Academia.edu | ☐ | — | Second wave — hold |
| Geoff Penington | [email on file] | ☐ | — | Third wave — hold |
| Netta Engelhardt | [email on file] | ☐ | — | Third wave — hold |
| Daniel Harlow | [email on file] | ☐ | — | Third wave — hold |

---

## Connections to Watch

- **Kim ↔ Friston:** Same mathematics (Fisher-Rao free energy minimization), different domains. If Friston recognizes the identification, introduce them.
- **Qi ↔ Ageev/Ageeva:** Qi's holographic dictionary may directly answer the Junction 3 question about Ageev's scalar. If both respond, connecting them is the natural move.
- **Kim ↔ Mohammady/Buscemi:** Kim's temperature parameter may resolve their 2025 trilemma. Connecting them if both engage.
- **Rudolph ↔ Kim:** Rudolph found the Zeno-attention connection philosophically; Kim provides thermodynamic grounding.

---

*Update this file when:*
- *A response is received*
- *An open question is answered*
- *A new connection forms*
- *The status of a junction changes based on expert feedback*
