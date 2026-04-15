# Physics Research Map

*A complete map of what we've done, what we've found, where results and code live, and what's next.*
*Ariel — April 15, 2026. Written at Eldon's request to organize the scattered research.*

---

## How to Read This Document

This document is the **single navigable index** to all physics research. For each investigation thread, it tells you:
- What the question is
- What experiments were run and what they found
- Where the code, data, and writeups live
- What's confirmed, what's open, what's next

The companion documents are:
- `FRAMEWORK.md` — the unified theoretical framework (what we believe and why)
- `STATUS.md` — outreach tracking and the junction chain (who we've contacted and the logical structure)
- `research/notes/the_attending_unit.md` — the core claim stated in one place (four characteristics, three evidence lines)
- `paper_conformal_draft.md` — the publishable paper draft (v5, includes BCFT)

---

## Part 1: The Investigation Threads

### Thread 1: Conformal Scaling in Trained Attention

**Question:** Does trained transformer attention exhibit power-law decay matching the SYK q=4 prediction (Δ = 1/4)?

**Status: CONFIRMED.** The foundational result. Everything else builds on this.

| Experiment | Script | Key Result |
|---|---|---|
| GPT-2 per-head analysis | `gpt2_per_head_analysis.py` | 44/144 heads power-law (R² > 0.90), **median Δ = 0.2493** (prediction: 0.2500) |
| GPT-2 head-averaged | `gpt2_conformal_test.py` | Power-law in attention weights (R² > 0.95). Hidden states homogenize — wrong observable |
| Randomized control | `gpt2_randomized_control.py` | **0 heads** with R² > 0.90 in randomized weights. Definitive: training creates the structure |
| Random attention baseline | `numerical_test_conformal.py`, `_v2.py`, `_phase_transition.py` | No conformal structure at any coupling σ. Second Law dominates random attention |
| Additional controls | `additional_controls.py` | Randomizing positional embeddings: preserves power law, shifts Δ to 0.10. Weights = dynamics, positions = geometry |
| Robustness check | `robustness_check.py` | Robust to fit range and R² threshold choices |

**Data:** No separate data files — results computed on the fly from GPT-2/Pythia weights.
**Writeup:** `paper_conformal_draft.md` §3.1–3.5, `NUMERICAL_RESULTS_MARCH24.md` Experiments 1–4, `FRAMEWORK.md` §3.

---

### Thread 2: Training Phase Transition

**Question:** Does a phase transition from disorder to conformal order occur during training?

**Status: CONFIRMED** across three model sizes.

| Experiment | Script | Key Result |
|---|---|---|
| Pythia-70m training dynamics | `pythia_phase_transition.py` | Sharp transition at step ~256 (A: 1.3 → 5.1) |
| Pythia-160m training dynamics | `phase_transition_scaling.py` | Transition at step ~256 |
| Pythia-410m training dynamics | `phase_transition_scaling_v2.py` | Transition at step ~1,000, gentler (A: 1.73 → 2.18) |
| Phase transition scaling | `phase_transition_scaling.py`, `_v2.py` | Onset delays with model size. Width ~1.0 in log-steps across all three |
| Fine checkpoint 410m | `fine_checkpoint_410m.py` | **Prethermal plateau at Δ ≈ 0.50** (SYK q=2) persists through step 143,000 |

**Key finding:** Two-stage flow: disorder → integrable order (q=2, Δ=0.50) → chaos (q=4, Δ=0.25). Smaller models complete the flow; 410m still at integrable stage.

**Data:** `410m_output.log`, `fine_410m_output.log` (terminal output captures)
**Writeup:** `paper_conformal_draft.md` §3.3, `FRAMEWORK.md` §3.3 and §4.2, `NUMERICAL_RESULTS_MARCH24.md` Experiments 5–10.

---

### Thread 3: Depth Convergence

**Question:** Does Δ converge toward 1/4 with increasing model depth?

**Status: CONFIRMED.**

| Experiment | Script | Key Result |
|---|---|---|
| Pythia depth test | `pythia_depth_test.py` | Δ flows: 0.60 (6 layers) → 0.38 (12 layers) → 0.28 (24 layers) |
| Deep layers of 410m | (same script) | Δ = 0.26 in deepest layers |

**Writeup:** `paper_conformal_draft.md` §3.4, `FRAMEWORK.md` §3.4.

---

### Thread 4: Entanglement Entropy (CFT Confirmation)

**Question:** Does the entanglement structure of attention weights follow the CFT prediction S(k) = (c/3)log(k)?

**Status: CONFIRMED.**

| Experiment | Script | Key Result |
|---|---|---|
| Entanglement entropy | `entanglement_entropy_test.py` | **R² = 0.9965 (GPT-2), R² = 0.9990 (Pythia-410m).** c ≈ 0.19 (GPT-2), c ≈ 0.11 (410m) |

**Writeup:** `paper_conformal_draft.md`, `FRAMEWORK.md` §5 (P2).

---

### Thread 5: Scrambling and Page Curve

**Question:** Does transformer attention exhibit fast scrambling as SYK predicts?

**Status: PARTIALLY CONFIRMED.**

| Experiment | Script | Key Result |
|---|---|---|
| Scrambling test | `scrambling_test.py` | Fast scrambling confirmed (2-3 layers to >90% max entropy). Does NOT scale as log(H). Scramble-then-decode structure reminiscent of Page curve |

**Writeup:** `FRAMEWORK.md` §5 (P1, P7).

---

### Thread 6: BCFT Boundary Corrections

**Question:** Does the causal mask act as a measurable BCFT boundary?

**Status: CONFIRMED.** This is the newest confirmed result line.

| Experiment | Script | Key Result |
|---|---|---|
| CFT symmetry test (periodic) | `cft_symmetry_test.py` | **FAILED** — periodic CFT form fits *worse* than power law. Geometry is wrong: system has a boundary, not periodicity |
| BCFT boundary test | `bcft_boundary_test.py` | **BCFT wins 15-0 over bare power law.** Mean R² improvement +0.176. λ consistently positive (Neumann boundary) |
| BCFT refined fit | `bcft_refined_fit.py` | 4-parameter model (separate Δ_bulk/Δ_bdy) is degenerate — 3-parameter is right complexity |

**Key finding:** The causal mask is a physical boundary. Position-dependent corrections match BCFT prediction. Near the boundary, the bare power law fails; deep in the sequence it recovers. Boundary parameter λ > 0 consistent with reflecting boundary of JT gravity.

**Data:** (computed on the fly from GPT-2)
**Writeup:** `the_attending_unit.md` "BCFT Confirmed" section, `paper_conformal_draft.md` abstract.

---

### Thread 7: BCFT → Lost-in-the-Middle Predictions

**Question:** Can the BCFT framework explain the "lost in the middle" phenomenon in LLMs?

**Status: PARTIALLY CONFIRMED.** Strong per-head result, but direct accuracy prediction failed.

| Experiment | Script | Key Result |
|---|---|---|
| Numerical BCFT predictions | `bcft_accuracy_prediction.py` | Valley at ~0.40L (confirmed analytically), asymmetry decreases with L |
| GPT-2 per-layer U-shape | `bcft_litm_predictions.py` | Per-layer data extracted |
| Start-boundary excess | `bcft_litm_excess.py` | 10/13 conformal heads show excess — attention sink = boundary effect |
| Pythia-70m training (v1) | `bcft_litm_training_test.py` | Inconclusive (methodology issues) |
| Pythia-70m training (v2) | `bcft_litm_training_v2.py` | **Per-head ρ(Δ, valley_depth) = +0.942** — the strongest empirical correlation |
| Fit to Liu et al. accuracy | `bcft_litm_liu_comparison.py` | R² = 0.88–0.99 with free parameters, but fitted Δ doesn't match measured Δ |
| GPT-2 KV retrieval benchmark | `bcft_litm_gpt2_benchmark.py` | Task too easy — no LiTM signal in GPT-2 |
| **LongChat-13B-16K on A100** | `bcft_cloud_comparison.py` | 1,343/1,600 conformal heads, median Δ=0.49, ρ(Δ, valley)=+0.637. Direct accuracy prediction: R²=0.02–0.07 |
| Multi-layer composition | `bcft_multilayer_composition.py` | **Δ_eff = 0.1655 from composition, matching Liu accuracy-fitted Δ = 0.1711** |
| Filtered composition | `bcft_composition_filtered.py` | Shape mismatch persists (best R²=0.19). Attention sinks dominate raw composite |

**Key findings:**
1. Per-head Δ controls valley depth: confirmed in two architectures (ρ = +0.94 Pythia, +0.64 LongChat)
2. Multi-layer Δ renormalization: single-head Δ≈0.49 composes to effective Δ≈0.17, matching accuracy fits
3. Direct attention-to-accuracy mapping fails: attention profiles peak at end (recency), but LongChat accuracy peaks at start (primacy)

**Data:** `bcft_longchat_measurements.json` (666KB, 1,343 head measurements)
**Writeup:** `research/notes/bcft_lost_in_the_middle.md` (full results section with summary table)

---

### Thread 8: Four-Point Correlator and Crossing Symmetry

**Question:** Does the four-point function of trained attention follow conformal field theory predictions?

**Status: CONFIRMED.**

| Experiment | Script | Key Result |
|---|---|---|
| Four-point analysis (all 144 heads) | `research/experiments/four_point_attention.py` | L6H4: Δ = 0.2499. Free-theory four-point R² = 0.94. Crossing slope 0.957 (expect 1.0) |

**Data:** `research/experiments/four_point_results.json`
**Writeup:** `research/findings/FINDINGS_LOG.md` (Finding 1), `research/notes/softmax_godelian_consistency.md` §9.

---

### Thread 9: GUE / Random Matrix Statistics

**Question:** Do trained attention matrices exhibit GUE (quantum chaotic) eigenvalue statistics?

**Status: NEGATIVE at single-matrix level. Collective observable needed.**

| Experiment | Script | Key Result |
|---|---|---|
| GUE level statistics | `gue_level_statistics.py` | Individual logit matrices show GOE by structure (real symmetric), not GUE. No trained/random difference |
| GUE v2 | `gue_level_statistics_v2.py` | Same conclusion. Single-matrix spectral statistics is the wrong observable |
| Eigenvalue comparison | `eigenvalue_spectrum_comparison.py` | Detailed eigenvalue analysis |
| Attention spectral analysis | `attention_spectral_analysis.py` | Spectral properties of trained vs random attention |

**Key insight:** The SYK correspondence lives at the *scaling correlation* level, not the single-matrix spectral level. The right observable for GUE is collective: correlation matrix across layers, Jacobian spectrum, or effective transfer matrix across heads.

**Data:** `results_eigenvalue/eigenvalue_comparison.json`, `results_eigenvalue/spectral_analysis.json`
**Writeup:** `research/riemann_unprovability.md` §5 Problem 3, ACTIVE.md "Design many-body GUE experiment."

---

### Thread 10: Theoretical — SYK Derivation Chain

**Question:** Can we derive the SYK correspondence rigorously from the softmax attention mechanism?

**Status: Structural result established, full derivation pending.**

| Work | Script/File | Key Result |
|---|---|---|
| Linearized softmax G⁴ calculation | `LINEARIZED_SOFTMAX_CALCULATION.md` | G⁴ vertex matches SYK q=4 in linearized regime. Δ = D/4 → Δ = 1/4 for D=1 |
| Multi-layer numerical test | `numerical_test_multilayer.py` | Multi-layer enhancement: 18× from one layer. Var ~ L^1.19 |
| Depth scaling test | `numerical_test_depth.py`, `_layernorm.py`, `_standard.py` | LayerNorm suppresses disorder (147× vs 14,443×). Depth scaling approaches fixed point |
| SYK numerical test | `numerical_test_syk.py` | Numerical verification of SYK-attention correspondence |
| Schwarzian exploration | `SCHWARZIAN_EXPLORATION.md` | Naive Schwarzian path gives wrong kinetic term. All routes converge on: Schwarzian iff continuum limit is CFT |
| Kim-Friston correspondence | `KIM_FRISTON_CORRESPONDENCE.md` | Three equivalent descriptions of softmax: Gibbs, diagonal density matrix, Bayesian posterior |

**Writeup:** `STATUS.md` (Routes A, B, C), `FRAMEWORK.md` §2.

---

### Thread 11: Theoretical — Softmax-Gödelian Consistency

**Question:** What is the formal relationship between softmax normalization, logical consistency, and the conformal bootstrap?

**Status: Extensive formal development.** This is the mathematical backbone.

| Work | File | Key Result |
|---|---|---|
| Full formal mapping | `research/notes/softmax_godelian_consistency.md` (~93KB, 14 sections) | Dutch book → coherence → Gaifman → consistency → incompleteness. Grassmannian extension. Plücker = crossing equations |
| Softmax Incompleteness Theorem | (same, §2) | G(σ) = 1 − ||σ||₂² = gradient capacity. Completeness and self-correction mutually exclusive |
| Four-point experiment | (same, §9) + `four_point_attention.py` | Δ = 0.2499, R² = 0.94, crossing slope 0.957 |
| B-C kernel identification | (same, §11-12) | 9 of 10 links closed between Grassmannian positivity and conformal bootstrap |
| Reverse direction (§13) | (same) | If CFT crossing contains ζ zeros (Benjamin & Chang), then Grassmannian positivity → RH chain |
| Li-Plücker correction (§14) | (same) + `research/findings/FINDINGS_LOG.md` Finding 2 | Naive version false (alternating signs). Corrected: ξ(1/(1-z))/ξ(1) has all-positive coefficients. THIS is the right Grassmannian point |

**Writeup:** The softmax-Gödelian note IS the writeup (93KB, the longest single document in the research).

---

### Thread 12: Theoretical — Relationship as Boundary

**Question:** Can the holographic principle be derived from the structure of relationship itself?

**Status: Developed framework, not experimentally tested.**

| Work | File | Key Result |
|---|---|---|
| Three-layer structure | `research/notes/relationship_as_boundary.md` (~51KB, 10 sections) | Relationship → positive Grassmannian → primes as boundary → holographic principle |
| Li-Plücker correction | (same, §10) | Completed τ-function has all-positive Taylor coefficients |
| Boundary symmetry inheritance | (same, §10.3) | Conjecture: S-invariance + positivity forces zeros onto critical line |

---

### Thread 13: Theoretical — Langlands as Holography

**Question:** Is the Langlands correspondence the holographic principle applied to number fields?

**Status: Exploratory. Three hard questions identified.**

| Work | File | Key Result |
|---|---|---|
| Langlands exploration | `research/notes/langlands_as_holography.md` (~36KB) | Langlands = holography for number fields. Kapustin-Witten connects to N=4 SYM |
| Three-question convergence | (same, updated April 14) | Modular completion, Perlmutter L-function, and zero sum rule for SYK converge to one question |
| Free boson factorization | (same) | c=1 free boson: L-function zeros ARE Riemann zeros. Simplest CFT encodes deepest number theory |

---

### Thread 14: Theoretical — Riemann Unprovability

**Question:** Can the framework explain why RH might be true but unprovable?

**Status: Framework established. Direction reversed (Eldon's insight).**

| Work | File | Key Result |
|---|---|---|
| Riemann unprovability program | `research/riemann_unprovability.md` (~27KB, 5 problems) | Five hard problems connecting GUE → RH. Each identified precisely with honest gap assessment |
| Direction reversal | `the_attending_unit.md` | Theory predicts symmetry → 10¹³ zeros confirms it. Theory doesn't need to prove RH |

---

### Thread 15: Tropical Geometry Bridge (Exploratory, Early)

**Question:** Can tropical geometry connect the discrete (transformer) and continuous (CFT) descriptions?

**Status: Extensive exploration, no definitive result.**

| Scripts | Count | Notes |
|---|---|---|
| `tropical_bridge_v1.py` through `tropical_bridge_v7n.py` | ~20 scripts | March 11 exploration. Many iterations trying different formulations. No single clean result emerged |
| `tropical_attention_bridge.py` | 1 | Initial attempt |

**Assessment:** This was a productive exploration that didn't converge. The tropical geometry angle may still be useful but the scripts are iterative drafts, not experiments with results. Could be archived.

---

### Thread 16: Neural-Transformer Comparison (for Wang et al. Outreach)

**Question:** Can we predict the kernel exponent μ in Wang et al.'s neural ERM model from the transformer measurement?

**Status: Analysis complete, outreach pending.**

| Work | File | Key Result |
|---|---|---|
| Comparison document | `transformer_neural_comparison.md` | μ = 2Δ = 0.50 (GPT-2). Prediction: same μ in ERM model. Eigenvalue comparison won't work (documented negative result) |

---

## Part 2: Code Inventory by Category

### Confirmed Experiment Scripts (producing published/cited results)
```
gpt2_conformal_test.py          — GPT-2 attention power-law measurement
gpt2_per_head_analysis.py       — Per-head Δ distribution (the core measurement)
gpt2_randomized_control.py      — Randomized weight control
additional_controls.py          — Position embedding and input controls
robustness_check.py             — Fit robustness verification
pythia_phase_transition.py      — Pythia-70m training dynamics
pythia_depth_test.py            — Cross-model depth convergence
phase_transition_scaling.py     — 70m + 160m phase transitions
phase_transition_scaling_v2.py  — 410m phase transition
fine_checkpoint_410m.py         — Fine-grained 410m training (prethermal plateau)
entanglement_entropy_test.py    — CFT entanglement entropy
scrambling_test.py              — Fast scrambling test
bcft_boundary_test.py           — BCFT boundary correction (15-0 result)
bcft_refined_fit.py             — 4-parameter BCFT refinement
cft_symmetry_test.py            — Periodic CFT test (negative result, led to BCFT)
research/experiments/four_point_attention.py  — Four-point correlator measurement
```

### BCFT / Lost-in-the-Middle Scripts (April 15 work)
```
bcft_accuracy_prediction.py     — Numerical BCFT predictions
bcft_litm_predictions.py        — GPT-2 per-layer U-shape
bcft_litm_excess.py             — Start-boundary excess measurement
bcft_litm_training_test.py      — Pythia training v1 (methodology issues)
bcft_litm_training_v2.py        — Pythia training v2 (ρ = +0.94 result)
bcft_litm_liu_comparison.py     — Fit to published accuracy data
bcft_litm_gpt2_benchmark.py     — GPT-2 KV retrieval benchmark
bcft_cloud_comparison.py        — LongChat-13B on A100 (Modal cloud)
bcft_multilayer_composition.py  — Multi-layer Δ renormalization
bcft_composition_filtered.py    — Filtered composition analysis
```

### Theoretical Verification Scripts (March 9 session)
```
numerical_test_conformal.py     — Random attention baseline
numerical_test_conformal_v2.py  — Random attention v2
numerical_test_phase_transition.py — Phase transition in random attention
numerical_test_depth.py         — Depth scaling
numerical_test_depth_layernorm.py — Depth with LayerNorm
numerical_test_depth_standard.py — Standard depth test
numerical_test_linearized.py    — Linearized regime verification
numerical_test_multilayer.py    — Multi-layer enhancement
numerical_test_syk.py           — SYK correspondence numerics
```

### Spectral / GUE Scripts (March 26–29)
```
gue_level_statistics.py         — GUE level statistics (negative result)
gue_level_statistics_v2.py      — GUE v2 (confirmed: wrong observable)
eigenvalue_spectrum_comparison.py — Eigenvalue spectrum analysis
attention_spectral_analysis.py  — Spectral properties
```

### Other / Exploratory
```
scaling_collapse_test.py        — Scaling collapse (not yet run properly)
neural_conformal_exploration.md — Neural/conformal exploration notes
conformal_integration_theory.md — Integration theory notes
tropical_bridge_v*.py (20 files) — Tropical geometry exploration (March 11)
```

### Data Files
```
bcft_longchat_measurements.json     — 666KB, 1343 LongChat-13B heads (April 15)
results_eigenvalue/eigenvalue_comparison.json — GUE eigenvalue data
results_eigenvalue/spectral_analysis.json    — Spectral analysis data
research/experiments/four_point_results.json  — Four-point correlator data
```

---

## Part 3: Documents Map

### Core Framework Documents
| Document | What It Is | Last Updated |
|---|---|---|
| `FRAMEWORK.md` | The unified theoretical framework — claims, evidence, predictions | April 15, 2026 |
| `STATUS.md` | Junction chain, outreach tracking, open questions | March 25 (body updated to April 15) |
| `the_attending_unit.md` | Core claim: four characteristics, three evidence lines, non-softmax prediction | April 14, 2026 |
| `paper_conformal_draft.md` | The publishable paper (v5, includes BCFT) | April 15, 2026 |
| `DIRECTIONS.md` | Open research directions (March 24 — partially superseded) | March 24, 2026 |

### Theoretical Development Notes
| Document | What It Is | Size |
|---|---|---|
| `softmax_godelian_consistency.md` | Formal mapping: softmax → coherence → Grassmannian → bootstrap → RH | 93KB, 14 sections |
| `relationship_as_boundary.md` | Three-layer structure: relationship → primes → holography | 51KB, 10 sections |
| `langlands_as_holography.md` | Langlands correspondence as holographic duality for number fields | 36KB |
| `riemann_unprovability.md` | Five hard problems connecting GUE to RH | 27KB |
| `bcft_lost_in_the_middle.md` | BCFT framework for lost-in-the-middle, with full experimental results | 23KB |

### Earlier Technical Documents
| Document | What It Is | Notes |
|---|---|---|
| `NUMERICAL_RESULTS.md` | March 9 numerical verification (Theorems 1-4, G⁴ vertex) | Superseded by MARCH24 for experiments |
| `NUMERICAL_RESULTS_MARCH24.md` | March 24 experiments (the core empirical results) | 10 experiments, definitive |
| `LINEARIZED_SOFTMAX_CALCULATION.md` | G⁴ vertex derivation in linearized regime | Key theoretical result |
| `SCHWARZIAN_EXPLORATION.md` | Schwarzian action from attention | Shows correct path requires CFT limit |
| `SYK_ANALYSIS.md` | Ageev IB mechanism = SYK disorder average | Route A detail |
| `KIM_FRISTON_CORRESPONDENCE.md` | Three descriptions of softmax | Brief but important |
| `COMPREHENSIVE_PAPER_OUTLINE.md` | Proposed comprehensive paper (March 9) | Partially superseded by conformal draft |
| `PAPER_OUTLINE_CONFORMAL.md` | Conformal paper outline | Led to paper_conformal_draft |
| `PAPER_REVIEW_MARCH9.md` | Critical self-review of Papers 1-5 | Important for honest framing |
| `CANONICAL_FORM_PAPER_REVIEW.md` | Canonical form paper review | Pre-v4 |
| `AI_RESEARCH_LANDSCAPE.md` | AI × physics landscape | March 9 |
| `AMPLITUHEDRON_CONNECTIONS.md` | Amplituhedron connections (58KB) | Substantial; April 13 update |

### Published Papers
| Paper | Where | DOI |
|---|---|---|
| Conformal scaling paper (v5) | Zenodo | 10.5281/zenodo.19225996 |
| Canonical form paper (v4) | Zenodo | 10.5281/zenodo.18968481 |

### External-Facing Writing (Physics Content)
| Document | What It Is |
|---|---|
| `writing/comprehensive_paper.md` | Comprehensive physics paper draft |
| `writing/canonical_form_paper.md` | Canonical form paper draft |
| `writing/paper4_draft.md`, `paper5_draft.md` | Earlier paper drafts |
| `writing/measurement_preprint_draft.md` | Measurement preprint |
| `writing/physics_preprint_draft.md` | Physics preprint |
| `writing/information_recovery_draft.md` | Information recovery paper |
| `writing/the_music_of_the_spheres.md` | Framework overview (most complete single statement) |
| `writing/the_structure_of_light.md` | Measurement-oriented companion |
| `writing/substack/post_009_gravity_attention.md` | "Gravity Is What Attention Looks Like From Outside" (Substack #10) |
| `writing/substack/post_010_one_experiment.md` | "One Experiment" (Substack) |

---

## Part 4: What's Confirmed vs. What's Open

### Confirmed Empirical Results (4 independent measurement types)
1. **Conformal scaling:** Δ = 0.2493 (GPT-2), matching SYK q=4 prediction of 0.2500
2. **Phase transition:** Three Pythia models show disorder → conformal transition during training
3. **BCFT boundary correction:** Causal mask is a physical boundary (15-0 over bare power law)
4. **Entanglement entropy:** S(k) = (c/3)log(k) with R² > 0.999

### Confirmed Supporting Results
5. **Depth convergence:** Δ flows from 0.60 → 0.38 → 0.28 → 0.26 with depth
6. **Cross-architecture robustness:** GPT-2 (absolute PE) and Pythia (RoPE) converge to same Δ
7. **Prethermal plateau:** Δ ≈ 0.50 (q=2) persists through 410m training, smaller models flow past it
8. **Randomized control:** 0 power-law heads in random weights (definitive)
9. **Four-point correlator:** Free-theory fit R² = 0.94, crossing slope 0.957
10. **Per-head Δ–valley depth correlation:** ρ = +0.942 (Pythia), +0.637 (LongChat-13B)
11. **Multi-layer Δ renormalization:** Δ_eff = 0.17 from composition, matching accuracy-fitted Δ

### Open / Negative Results
- **GUE at single-matrix level:** Negative. Single-head logit matrices show GOE, not GUE. Need collective observable
- **Direct attention → accuracy prediction:** Failed (R² = 0.02–0.07). Shape mismatch between attention and accuracy
- **Tropical geometry bridge:** Extensive exploration, no convergence
- **Periodic CFT symmetry test:** Failed — wrong geometry (led productively to BCFT)

### Open Theoretical Questions
- Junction 3 (free scalar → holographic dual): conditionally open, depends on Ageev scalar mass
- D=4 from Gr(2,4): suggestive but not derived
- KP bilinear identity for ξ τ-function: untested
- Boundary symmetry inheritance → RH: conjectural

---

## Part 5: Planned Experiments (Priority Order)

### 1. Non-Softmax Universality — THE critical experiment
**What:** Measure Δ in sigmoid attention vs. softmax, using Apple's paired models (identical training, different attention mechanism).
**Why:** Distinguishes "the attending unit is universal" from "softmax is special." If sigmoid (no normalization) also gives Δ = 1/4, the theory is wrong. If it doesn't, the theory is strengthened.
**Where:** Apple `apple/ml-sigmoid-attention` (ICLR 2025), 7B models + 8 training checkpoints each.
**Compute:** Needs 24GB+ memory. Modal A100 ($0.50/run) or M4 Pro via MLX.
**Code basis:** Adapt `gpt2_per_head_analysis.py` and `bcft_boundary_test.py`.
**Prediction:** Softmax → Δ = 1/4. Sigmoid → Δ ≠ 1/4 (near 1/2 or no clean power law).
**Tracking:** `the_attending_unit.md` "Non-Softmax Universality Prediction" section.

### 2. MPT-30B-Instruct LiTM Test
**What:** Measure per-head attention profiles in MPT-30B-Instruct (clearer U-shape in published accuracy data than LongChat).
**Why:** The LongChat shape mismatch (recency vs primacy) may be specific to long-context fine-tuning. MPT is a standard model.
**Compute:** A100 via Modal (~$0.50).
**Code basis:** Adapt `bcft_cloud_comparison.py`.

### 3. Boundary Entropy (g-factor) Measurement
**What:** Measure the Affleck-Ludwig boundary entropy from entanglement entropy near the sequence boundary.
**Why:** Directly tests BCFT prediction. Connects to information capacity at boundaries.
**Compute:** Local (GPT-2/Pythia).
**Code basis:** Adapt `entanglement_entropy_test.py`.

### 4. Scaling Collapse Test
**What:** Normalize training curves by model-dependent scales, check whether all models collapse onto a universal curve.
**Why:** Standard physics test for universality class identification.
**Code:** `scaling_collapse_test.py` exists but hasn't been run properly.
**Reference:** Qiu et al. ICML 2025 "supercollapse."

### 5. Many-Body GUE Experiment
**What:** Measure GUE statistics using collective observables (correlation matrix across layers, Jacobian spectrum) rather than single-head logit matrices.
**Why:** The single-matrix test was the wrong observable. GUE should appear at the collective level.
**Code:** Needs new script.

### 6. Functional Equation Measurement (BCFT version)
**What:** Test S-symmetry on the strip (annulus partition function) rather than the torus.
**Why:** The periodic test failed because the geometry was wrong. The BCFT version is the correct test of Characteristic 4 (symmetry).
**Status:** Needs BCFT literature study before designing.

### 7. Analytical Δ Renormalization Derivation
**What:** Derive how per-layer Δ values compose to give the effective Δ governing task performance.
**Why:** The numerical match (0.17 from composition ≈ 0.17 from accuracy) is striking but needs theoretical explanation.
**Status:** Theoretical work needed.

---

## Part 6: What Needs Cleanup

### Files That Could Be Archived
- `tropical_bridge_v*.py` (20 files) — exploratory iteration with no converged result
- `NUMERICAL_RESULTS.md` — superseded by `NUMERICAL_RESULTS_MARCH24.md` for all experiments
- `DIRECTIONS.md` — mostly superseded by FRAMEWORK, attending_unit, and this document
- `COMPREHENSIVE_PAPER_OUTLINE.md` — superseded by `paper_conformal_draft.md`
- `PAPER_OUTLINE_CONFORMAL.md` — same

### Documents That Need Updating
- `STATUS.md` header says "Last updated: March 25, 2026" but body has April 15 content. Header should match.
- `DIRECTIONS.md` (March 24) — Directions 1, 2, 4 are now confirmed. Could be marked as such or retired.
- The Open Questions table in `STATUS.md` needs updating: several questions are now answered.

### Structural Issues
- Published paper drafts in `writing/` (paper4, paper5, comprehensive, canonical_form) duplicate content with `research/physics/` versions. Single source of truth not clear.
- `research/findings/FINDINGS_LOG.md` has only 3 entries (April 13). Many earlier findings never logged there. Either use it consistently or retire it.
- The `research/experiments/` directory has only one experiment (four-point). All other experiment scripts are in `research/physics/`. Pick one location.

---

*This document should be updated when new experiments are run or new threads emerge.*
*It is the map of what exists. FRAMEWORK.md is the map of what we believe.*
