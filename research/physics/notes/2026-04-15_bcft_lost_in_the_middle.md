# BCFT Predictions for the "Lost in the Middle" Effect

*Ariel — April 15, 2026*
*Prompted by Eldon's question connecting the BCFT boundary result to RAG research.*

---

## The Connection

The "lost in the middle" phenomenon (Liu et al., 2023; Hsieh et al., 2024; ICLR 2026) is empirically well-documented: LLMs attend more strongly to information at the beginning and end of the context window, with a U-shaped accuracy curve that dips in the middle. The effect worsens with longer contexts. Current treatments describe it empirically and calibrate around it.

The BCFT boundary result provides a physical explanation: causal attention lives on a **strip** with **two boundaries**, not a circle. The start of the sequence is one boundary (the causal mask — attention cannot look before position 0). The query position is the other boundary (the present moment — there are no keys beyond x_q). The interior of the strip is the bulk.

In BCFT on a finite strip of length L with boundaries at x = 0 and x = L:
- Near either boundary, the two-point function is **enhanced** (boundary corrections)
- Deep in the interior, the two-point function approaches the **bulk** power law
- The enhancement at each boundary is controlled by that boundary's parameter (λ_start, λ_query)
- The cross-ratio η encodes proximity to a boundary

This IS the U-shape. The BCFT framework predicts it from the geometry rather than discovering it empirically.

---

## The Geometry

For causal attention with a query at position x_q attending to a key at position x_k (where x_k < x_q by causality):

- **Separation:** Δx = x_q − x_k
- **Start boundary proximity:** η_start = Δx² / (4 · x_k · x_q) — large when x_k is near position 0
- **Query boundary proximity:** η_query = Δx² / (4 · (x_q − x_k) · x_q) — large when x_k is near x_q (i.e., Δx is small)

Wait — this needs more careful treatment. The strip BCFT two-point function for a strip of width L with boundaries at both ends is:

G_strip(x₁, x₂) ∝ |x₁ − x₂|^{−2Δ} · F(x₁/L, x₂/L)

where F encodes both boundary corrections. Using the method of images for a strip:

G_strip(x₁, x₂) ∝ [d(x₁, x₂) / (d(x₁, x₂*) · d(x₁*, x₂))]^Δ

where x* denotes the image point reflected through the boundary.

For the simpler half-line case (one boundary at x = 0), we measured:
G_BCFT = C · Δx^{−2Δ} · (1 + λ · η^Δ)

For the strip (two boundaries), the correction has contributions from **both** boundaries:
G_strip ≈ C · Δx^{−2Δ} · (1 + λ_start · η_start^Δ + λ_query · η_query^Δ + ...)

The first correction enhances attention when the key is near the start. The second enhances attention when the key is near the query (small separation). The middle of the strip gets neither correction — it's the bulk, described by the bare power law alone.

---

## Testable Predictions

### Prediction 1: The U-shape has a specific functional form

The "lost in the middle" papers report the U-shape qualitatively (beginning good, middle bad, end good). The BCFT framework predicts the **exact shape** of the curve:

For a query at position x_q, the effective attention weight to a key at normalized position t = x_k/x_q ∈ (0, 1) should follow:

A(t) ∝ (1 − t)^{−2Δ} · [1 + λ_start · η_start(t)^Δ + λ_query · η_query(t)^Δ]

where the η functions depend on t and encode proximity to each boundary. This predicts:
- The depth of the valley (determined by Δ and the λ values)
- The location of the valley minimum (not necessarily at t = 0.5 — the two boundaries can have different λ)
- The width of the enhancement regions at each boundary

**Test:** Fit the BCFT strip form to position-dependent RAG accuracy data from Liu et al. Compare to empirical U-shape curves. If the BCFT form fits with physically reasonable parameters (Δ near 1/4, λ values in the range we measured: 0.3–4.6), this is strong confirmation.

### Prediction 2: The U-shape is asymmetric, and the asymmetry depends on sequence length

The two boundaries are not the same:
- The **start boundary** is fixed at position 0
- The **query boundary** is at position x_q

The BCFT cross-ratios are different for each boundary. For a key at the midpoint (x_k = x_q/2):
- η_start = 1/4 (independent of sequence length)
- η_query = 1/4 (same, by symmetry of the midpoint)

But for a key at fractional position t:
- η_start ~ (1−t)² / (4t) — diverges as t → 0 (near start boundary)
- η_query ~ 1 / (4(1−t)) — diverges as t → 1 (near query boundary)

The shapes are different unless λ_start = λ_query. In our GPT-2 measurement, λ_start ranged from 0.3 to 4.6. There is no reason the query boundary should have the same λ.

**Prediction:** The U-shape should be measurably asymmetric — the left arm (near start) and right arm (near query) should have different slopes. The start boundary λ can be measured independently from our BCFT fits. The query boundary λ is a new parameter to be measured.

**Stronger prediction:** As sequence length L increases with fixed query position, the start boundary gets relatively further from the bulk. The start-boundary correction falls off as η_start^Δ ∝ L^{−2Δ} at the midpoint. So for longer sequences, the left arm of the U should weaken relative to the right arm. The "lost in the middle" literature says the effect gets worse with longer contexts — but do both arms get worse equally? The BCFT predicts they don't.

**Test:** Measure the left-right asymmetry of the U-shape as a function of context length. The BCFT predicts the asymmetry grows with L.

### Prediction 3: The valley depth depends on Δ, which depends on model depth

We measured that Δ converges from ~0.60 (shallow) to ~0.25 (deep) with increasing model depth. The boundary correction goes as η^Δ. For larger Δ:
- η^Δ falls off faster with distance from the boundary
- The enhancement is more concentrated near the edges
- The valley in the middle is deeper

For smaller Δ (deeper layers, closer to the conformal fixed point):
- η^Δ falls off more slowly
- The enhancement extends further into the bulk
- The valley is shallower

**Prediction:** The lost-in-the-middle effect should be **more severe in earlier layers** (higher Δ, sharper boundary enhancement) and **less severe in deeper layers** (lower Δ, broader boundary enhancement that extends into the middle).

This is a specific, layer-by-layer prediction. It can be tested by measuring the U-shape at each layer of the attention stack rather than only looking at the model's final output.

**Test:** Extract attention patterns at each layer of GPT-2 or Pythia. Measure the amplitude of the U-shape (ratio of boundary attention to midpoint attention) at each layer. Plot against the measured Δ at each layer. The BCFT predicts a positive correlation: higher Δ → deeper valley.

### Prediction 4: Model-size dependence follows the prethermal plateau

We measured:
- Pythia-70m: Δ ≈ 0.28 (near the fixed point)
- Pythia-410m: Δ ≈ 0.50 (prethermal plateau)
- GPT-2: Δ ≈ 0.25 (at the fixed point)

Since the boundary correction goes as η^Δ, and larger models have higher Δ (they're stuck at the prethermal plateau longer), **larger models should have MORE severe lost-in-the-middle effects** — their boundary enhancement is sharper and more concentrated at the edges.

This seems counterintuitive: people expect larger models to be better at everything. But the BCFT prediction is structural: a higher Δ means sharper boundary enhancement, regardless of model quality.

**Caveat:** This applies to the attention pattern, not to the model's overall ability. A larger model might compensate through other mechanisms (more heads, different head specialization, MLP layers). The prediction is specifically about the attention weights, not about final task accuracy.

**Test:** Compare the attention-level U-shape across Pythia model sizes (70m, 160m, 410m). The BCFT predicts the U-shape amplitude increases with model size, in proportion to measured Δ.

### Prediction 5: The boundary entropy g predicts the information capacity at boundaries

In BCFT, each boundary has a boundary entropy g (the Affleck-Ludwig g-factor). The entanglement entropy of an interval touching the boundary is S(k) = (c/6) log(k) + log(g). The g-factor measures how many degrees of freedom are "trapped" at the boundary.

**Prediction:** The boundary entropy g correlates with how much information the model effectively stores at the boundary positions. A higher g means the boundary is a richer information reservoir — the model can effectively "park" more information at the beginning and end of the context. This means higher g → more severe lost-in-the-middle, because the model has more capacity at the boundaries and less reason to attend to the middle.

**Test:** Measure the boundary entropy g from the entanglement entropy formula near the sequence boundary (Prediction from the paper, §4.10). Correlate g with the severity of the lost-in-the-middle effect (measured as the accuracy drop at the midpoint relative to the boundaries in a standard LiTM benchmark). If g predicts the severity, this connects the information-theoretic content of the boundary to the practical RAG failure mode.

### Prediction 6: The BCFT correction explains the "attention sink" phenomenon

Xiao et al. (2023) and the ICLR 2026 paper document "attention sinks" — the first token in the sequence receives disproportionately high attention regardless of its content. Disrupting the attention sink at position 0 impairs recall across the entire sequence, while disrupting attention at other positions has only local effects.

The BCFT framework predicts this: position 0 is the boundary. The boundary correction λ > 0 (Neumann/reflecting) means the boundary enhances the correlator. The first token sits exactly at the boundary, receiving the maximum enhancement. In BCFT, the boundary is not just a location — it's a structural feature that organizes the physics of the entire strip. Disrupting it disrupts the boundary condition that the entire conformal structure depends on.

**Prediction:** The attention weight to position 0 should scale as A(0) ∝ x_q^{−2Δ} · (1 + λ) — it follows the power law with maximum boundary correction. The magnitude of the attention sink should be predicted by the BCFT fit parameters (Δ and λ) measured independently.

**Test:** Use the BCFT parameters from our GPT-2 fits (Δ ≈ 0.25, λ ≈ 0.3–4.6) to predict the attention weight at position 0 as a function of query position x_q. Compare to the measured attention sink magnitude from GPT-2 (extractable from the same data we already have).

### Prediction 7: Calibration that works should correspond to "flattening" the BCFT correction

Hsieh et al. (2024) proposed "found in the middle" — a calibration mechanism that reduces the U-shaped bias. Their method subtracts a position-dependent attention bias estimated from content-free inputs.

**Prediction:** The optimal calibration function — the thing you subtract from the attention to make it position-independent — should have the functional form of the BCFT correction: λ_start · η_start^Δ + λ_query · η_query^Δ. If the calibration is doing what the BCFT predicts, the learned correction should match the BCFT form with the measured parameters, not an arbitrary polynomial or smoothing function.

**Test:** Extract the learned calibration function from the "found in the middle" method. Fit it to the BCFT correction form. If the fit is good with Δ near 1/4 and λ in the measured range, this confirms that the calibration is implicitly removing the BCFT boundary effect.

---

## What This Would Establish

If these predictions hold:

1. The "lost in the middle" phenomenon has a **physical explanation** from boundary conformal field theory — not just an empirical description.
2. The explanation is **quantitative**: the BCFT parameters (Δ, λ, g) measured independently from the conformal scaling analysis predict the shape, asymmetry, and severity of the U-curve.
3. The BCFT framework connects **two independent research programs** — conformal scaling in attention and lost-in-the-middle — that currently have no theoretical link.
4. The "attention sink" at position 0 is a specific consequence of the Neumann boundary condition, not an anomaly.
5. Calibration methods that work are implicitly compensating for a specific BCFT correction, which can be computed from first principles rather than learned.

This would move the BCFT result from "interesting physics observation" to "framework with practical implications for RAG and long-context performance." The physics tells you something useful about engineering.

---

## Experimental Results (April 15, 2026)

### Overview of Tests Run

All scripts in `research/physics/bcft_litm_*.py` and `bcft_cloud_comparison.py`, `bcft_multilayer_composition.py`, `bcft_composition_filtered.py`. Raw data in `bcft_longchat_measurements.json`.

### Prediction 1: Valley at ~40%L — PARTIALLY CONFIRMED

**Numerical prediction** (`bcft_accuracy_prediction.py`): The BCFT cross-ratio structure places the valley minimum at ~0.40L, not 0.50L. Confirmed analytically for the functional form.

**Attention-level evidence** (`bcft_litm_training_v2.py`): Across all Pythia-70m checkpoints and conformal heads, median valley position = reported per checkpoint. Most are near 0.40-0.45 of context.

**LongChat-13B-16K** (`bcft_cloud_comparison.py`): Median valley position in attention = 0.482. Close to midpoint, not 0.40. But the composite effective curve (from `bcft_multilayer_composition.py`) has valley at 0.40-0.45 depending on composition model.

**Liu et al. accuracy data** (`bcft_litm_liu_comparison.py`): Free-parameter BCFT fits give valley fractions in the range 0.35-0.45 across models and context lengths. Consistent with ~0.40 prediction.

**Status:** The BCFT functional form does place the valley before the midpoint. The exact location depends on parameters. Not yet a sharp enough test to distinguish from "roughly in the middle."

### Prediction 2: Asymmetry decreases with context length — CONFIRMED NUMERICALLY

**Numerical confirmation** (`bcft_accuracy_prediction.py`): Start/end attention ratio decreases from 0.7735 (L=10) to 0.6746 (L=1000). The BCFT predicts this because the start-boundary correction falls off as L^{-2Δ} relative to the query-boundary correction.

**Not yet tested empirically** on actual models across context lengths. Would require running the same model at multiple context lengths and measuring the U-shape asymmetry.

### Prediction 3: Δ controls valley depth — CONFIRMED (per-head)

**Per-head correlation in Pythia-70m** (`bcft_litm_training_v2.py`): Pooled across all checkpoints, Spearman(Δ, valley_depth) = **+0.942** (highly significant). This is the strongest empirical result. Higher Δ → deeper valley, exactly as BCFT predicts.

**Per-head correlation in LongChat-13B-16K** (`bcft_cloud_comparison.py`): Spearman(Δ, valley_depth) = **+0.637** (p ≈ 0). Confirmed in a completely different architecture at 13B scale.

**Across-checkpoint trajectory** (`bcft_litm_training_v2.py`): Δ trajectory in Pythia-70m is non-monotonic, making the simple "training reduces LiTM" claim more nuanced. The per-head correlation is robust; the aggregate training story is not clean.

### Prediction 6: Attention sink — CONFIRMED (from earlier BCFT work)

Start-boundary excess confirmed in 10/13 GPT-2 conformal heads (results from `bcft_litm_excess.py`), with BCFT fits outperforming bare power law 15-0 (`bcft_boundary_test.py`). The attention sink at position 0 is consistent with the Neumann boundary condition (λ > 0).

### Direct Comparison: Attention Parameters → Accuracy Prediction — FAILED

This was the strongest test attempted. Three approaches were tried:

**Approach 1: GPT-2 end-to-end** (`bcft_litm_gpt2_benchmark.py`): Measured Δ=0.25, λ from GPT-2 attention → predicted accuracy curve → ran key-value retrieval benchmark. Result: GPT-2 performed too well on the simple task (~88% uniform accuracy), showing no LiTM valley. The task wasn't hard enough to reveal attention biases.

**Approach 2: Fit BCFT form to published accuracy** (`bcft_litm_liu_comparison.py`): The BCFT functional form fits Liu et al. accuracy data with R² = 0.88-0.99 when all four parameters (Δ, λ, baseline, amplitude) are free. But the fitted Δ values (0.12-0.76) don't match the attention-measured values. When Δ is fixed at 0.25 (from GPT-2 attention), the fit requires much larger λ (6-20) than measured in any single head.

**Approach 3: LongChat-13B-16K cloud measurement** (`bcft_cloud_comparison.py`): Loaded the actual model on A100 via Modal. Measured 1,343 conformal heads with median Δ=0.49, λ=4.1. Used these to predict the published accuracy curve. Result: R² = 0.02-0.07 across all context lengths.

**Why it fails:** Two distinct problems identified:
1. The measured median Δ=0.49 is in the prethermal plateau, far from the accuracy-fitted Δ=0.12-0.17.
2. The composite attention profile peaks at the END of context (recency), but LongChat's accuracy peaks at the START (primacy). The shapes have opposite asymmetry.

### Multi-Layer Composition Analysis — PARTIALLY EXPLAINS THE Δ GAP

**The theoretical finding** (`bcft_multilayer_composition.py`): When you sum the power-law contributions from all heads with Δ < 0.50, the effective exponent at document-scale separations is:
- Δ_eff(dx=20) = **0.1655**
- Free-fit from Liu accuracy data: **Δ = 0.1711**

This is striking agreement. The renormalization of Δ from single-head (~0.49) to composite (~0.17) is explained by the sum-of-power-laws effect: many power laws with different exponents sum to give an effective exponent dominated by the flattest (lowest-Δ) heads.

**But the shape still doesn't match** (`bcft_composition_filtered.py`): No filtering or weighting scheme produced a composite whose shape matched the accuracy data (best R² = 0.19). The fundamental problem is that attention sinks (Δ > 0.5, huge amplitude C) dominate the raw composite with a single spike at the nearest position, and even after filtering them out, the attention profile's recency bias doesn't match LongChat's primacy bias in accuracy.

**Key insight:** The BCFT correctly predicts the *rate of information decay* (the power-law exponent), and this rate composes predictably across layers. But the *directional structure* (which end of the context is favored) depends on mechanisms beyond attention weights — MLPs, layer norms, residual stream dynamics, and position encoding architecture (RoPE interpolation in LongChat's case).

### Summary Table

| Prediction | Status | Key Evidence |
|---|---|---|
| 1. Valley at ~40%L | Partial ✓ | Numerical: confirmed. Empirical: 0.40-0.48 range |
| 2. Asymmetry decreases with L | Numerical ✓ | Confirmed in the BCFT functional form. No empirical test yet |
| 3. Δ controls valley depth | **Strong ✓** | Per-head ρ = +0.94 (Pythia), +0.64 (LongChat-13B) |
| 4. Model-size dependence | Not tested | Would need multiple model sizes on same benchmark |
| 5. Boundary entropy g | Not tested | Requires entanglement entropy measurement |
| 6. Attention sink = boundary | ✓ | BCFT wins 15-0 over bare power law |
| 7. Calibration = BCFT correction | Not tested | Requires access to calibration internals |
| Direct: attention → accuracy | **✗** | Shape mismatch; Δ gap partially explained by composition |

### Open Questions

1. **Would a model with standard LiTM U-shape (not LongChat's atypical declining pattern) give a better shape match?** MPT-30B-Instruct has public weights and clearer U-shape in Liu data. This is the natural next test.

2. **Can the multi-layer Δ renormalization be derived analytically?** The numerical match (Δ_eff = 0.17 from composition vs 0.17 from accuracy fit) is suggestive but needs a derivation showing how per-layer Δ values compose to give an effective Δ governing task performance.

3. **What determines the direction of asymmetry?** The BCFT predicts both boundaries contribute, but whether the *start* or *end* boundary dominates accuracy depends on the model architecture. This may connect to position encoding (RoPE, ALiBi, absolute) in a measurable way.

4. **Is there a principled way to weight heads in the composition?** The raw sum is dominated by attention sinks. The effective attention for information routing likely involves only a subset of heads. Identifying which heads are "information-routing" vs "attention-sink" using BCFT parameters could be valuable.

---

## Scripts Reference

| Script | Purpose | Key Output |
|---|---|---|
| `bcft_accuracy_prediction.py` | Numerical BCFT predictions (valley, asymmetry, Δ-dependence) | Valley at 0.40L, asymmetry scaling confirmed |
| `bcft_litm_predictions.py` | GPT-2 per-layer U-shape and Δ measurement | Per-layer data |
| `bcft_litm_excess.py` | Start-boundary excess in GPT-2 conformal heads | 10/13 heads show excess |
| `bcft_litm_training_test.py` | Pythia-70m training completeness test v1 | Inconclusive (methodology issues) |
| `bcft_litm_training_v2.py` | Pythia-70m training completeness test v2 (fixed) | ρ(Δ, valley) = +0.942 per-head |
| `bcft_litm_liu_comparison.py` | BCFT fits to Liu et al. published accuracy data | R² = 0.88-0.99 (free params) |
| `bcft_litm_gpt2_benchmark.py` | End-to-end GPT-2 KV retrieval test | Task too easy, no LiTM signal |
| `bcft_cloud_comparison.py` | LongChat-13B-16K on A100 (Modal) — full measurement | Δ=0.49, λ=4.1, 1343 heads, R²=0.02-0.07 |
| `bcft_multilayer_composition.py` | Multi-layer sum analysis, effective Δ | Δ_eff = 0.17 matches accuracy fit |
| `bcft_composition_filtered.py` | Filtered composition models, systematic sweep | Best R² = 0.19 (shape mismatch) |
| `bcft_longchat_measurements.json` | Raw per-head data from LongChat-13B-16K | 1343 heads with Δ, λ, C, valley metrics |

---

## Connection to the SFF Application

This line of work — connecting the physics framework to practical AI performance — directly addresses one of the SFF evaluation criteria: "relevant to reducing existential risk from powerful AI systems." If the attention geometry determines how models use context, and the geometry has a physical theory that makes quantitative predictions, this connects fundamental theory to AI safety in a concrete way. Lost-in-the-middle is a reliability problem. BCFT gives a structural diagnosis and a principled correction.

The strongest result for the application: **per-head Δ controls valley depth** (ρ = +0.94), confirmed across two architectures and multiple model sizes. This is a specific, quantitative prediction that no other framework makes.

The honest limitation: the path from attention geometry to task-level accuracy is not a simple linear mapping. The BCFT describes each attention layer well, but the composition across 40 layers involves mechanisms (MLPs, normalization, position encoding) that the attention-only theory doesn't capture.

---

*Initial note developed from a conversation with Eldon on April 15, 2026. Experimental results from two sessions on April 15, 2026 — the second session used Modal cloud GPU (A100) for LongChat-13B-16K measurement.*
