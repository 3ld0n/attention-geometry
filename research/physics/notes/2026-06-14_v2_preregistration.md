# v2 Pre-Registration: Joint (Δ, λ) → Valley Depth + Depth-Accumulated Primacy Laws

*Ariel Umphrey — June 14, 2026.*
*Written and committed BEFORE any confirmatory model measurements are run.*
*Read in conjunction with the April 17 pre-registration (`2026-04-17_bcft_pre_registered_prediction.md`) which this supersedes for the valley-depth test and extends with two new predictions.*

---

## What this pre-registers and why

The April 2026 pre-registration (`exp-025`, run April 17) confirmed that Spearman ρ(Δ, valley_depth) ≥ 0.50 in 6 of 7 tested models. That test used a single-parameter statistic: the conformal weight Δ alone.

Since April, two developments allow a tighter test:

1. **exp-061 (June 10)** derived and confirmed that the BCFT image form implies a specific, no-free-parameter valley prediction v̂(Δ̂, λ̂) from the fitted per-head (Δ, λ) pair. On the two existing diagnostic models (Pythia-2.8B, GPT-Neo-2.7B), ρ(v̂, valley_measured) = +0.683 / +0.906 — higher than the Δ-only correlation and achieved without any additional fit parameters.

2. **exp-065/exp-066 (June 12)** derived and confirmed two scaling laws for attention primacy (mass at the absorbing start boundary) from the composition law: (A) primacy grows with effective depth d_eff = (1−μ)L; (B) start-window primacy shrinks with context length N at fixed depth. Both confirmed synthetically (exp-066: P1a ρ = +0.998, P1b median |Δprim| = 0.005–0.012, P2 ρ = −1.00) and descriptively on LongChat (P3a ρ = +1.00). The fresh-model confirmatory test has not been run.

This document pre-registers:

- **Test A**: ρ(v̂(Δ̂, λ̂), valley_measured) on fresh models — the strict upgrade from the April single-parameter test.
- **Test B**: Depth-accumulated primacy on fresh models — Laws A and B stated as attention-side predictions with numeric thresholds, testable locally on cached models.

---

## Test A: Joint (Δ, λ) → Implied Valley

### The statistic

Per-head implied valley v̂(Δ̂, λ̂) computed from the pipeline-exact BCFT image form:

```
A(i, j) = C · [(i−j)^{−2Δ} + λ · (i+j)^{−2Δ}]
```

where i is the query position and j is the key position (j < i). The pipeline geometry follows the standard exp-025/exp-061 convention: L = 512, deep queries i ∈ [384, 512), key thirds at [1,170), [170,340), [340,384).

For each head with fitted (Δ̂, λ̂):

1. Compute the third-mass expectations:
   - S = mean_{i ∈ deep, j ∈ [1,170)} [(i−j)^{−2Δ̂} + λ̂·(i+j)^{−2Δ̂}]
   - M = mean_{i ∈ deep, j ∈ [170,340)} [(i−j)^{−2Δ̂} + λ̂·(i+j)^{−2Δ̂}]
   - E = mean_{i ∈ deep, j ∈ [340,384)} [(i−j)^{−2Δ̂} + λ̂·(i+j)^{−2Δ̂}]
2. Implied valley: v̂ = 1 − M / max(S, E)

Reference implementation: `experiments/exp-061_lambda_sign_derivation/run_lambda_sign.py`, function `model_valley(delta, lam)`.

The measured valley uses the same pipeline definition (§3.4 of the April pre-registration): v_measured = 1 − middle_attn / max(start_attn, end_attn), on the same random-token average attention tensor.

The fitting procedure for (Δ̂, λ̂) uses the BCFT functional-form fit from the exp-026/exp-057 lineage: fit A(i,j) = C[(i-j)^{-2Δ} + λ(i+j)^{-2Δ}] to per-head attention profiles, position-dependently, via least-squares in log space.

### Conformality criterion (unchanged from April)

A head is included iff R² ≥ 0.85 and Δ̂ ≥ 0.05 on the single-parameter power-law fit. The two-parameter BCFT fit is only run on heads passing this criterion.

### Threshold setting

From the two diagnostic models used to develop the statistic (NOT valid as confirmatory tests):

| Model | n_conformal | ρ(v̂, valley_meas) |
|---|---|---|
| Pythia-2.8B | 207 | +0.683 |
| GPT-Neo-2.7B | 440 | +0.906 |

Both sit above the April ρ(Δ, valley) threshold of 0.50. The diagnostic lower bound is 0.683.

**Threshold (A1):** For a fresh model to count as a confirmation of Test A:
- ρ(v̂, valley_meas) ≥ **0.60** (lower than the diagnostic minimum, preserving conservatism about how well the threshold generalizes)
- p ≤ 1 × 10⁻⁵
- n_conformal ≥ 50

**Threshold (A2 — improvement claim):** ρ(v̂, valley_meas) > ρ(Δ̂, valley_meas) on the same head set. This is the specific claim that the boundary parameter λ adds predictive value beyond Δ alone.

### Kill criteria

- **Kill A1:** ρ(v̂, valley_meas) < 0.50 on any qualifying fresh model → the implied-valley statistic fails to clear the April single-parameter bar; the two-parameter BCFT form is not a useful predictor.
- **Kill A2:** ρ(v̂, valley_meas) ≤ ρ(Δ̂, valley_meas) on three or more qualifying fresh models → λ adds no predictive value; retract "boundary parameter independently predicts valley depth."

A result ρ ∈ [0.50, 0.60) is a partial result: it clears the April bar but not the v2 threshold. Report as partial; do not count as either kill or full confirmation. Record the improvement-claim outcome separately.

### Fresh confirmatory models

These models were NOT used to set thresholds. Results on them are the actual test of Test A.

**Locally cached (runnable immediately):**

| Model | Params | Layers | Heads | Notes |
|---|---|---|---|---|
| Pythia-410m | 410M | 24 | 16 | Confirmed ρ(Δ,v)=+0.76 in April (single-Δ stat); upgrade test |
| Pythia-1.4b | 1.4B | 24 | 16 | Confirmed ρ(Δ,v)=+0.71 in April; upgrade test |
| GPT-2-medium | 345M | 24 | 16 | Not in April pre-reg; fresh architecture |

**Needs download:**

| Model | Why valuable |
|---|---|
| OLMo-7B-hf | ALiBi PE, fully open data; confirmed in April for single-Δ |
| Qwen2-7B | GQA + different training distribution; confirmed in April |
| Mistral-7B-v0.3 | SWA; confirmed in April (lowest: ρ=0.58) |
| Llama-3-8B | Most-deployed 8B model; gated, access pending |
| MPT-30B-Instruct | Best LiTM U-shape data (Liu et al.); needs cloud |

**Note on Pythia-410m and Pythia-1.4b:** these were confirmed in April on the single-Δ statistic but NOT used to develop the v̂(Δ,λ) formula (that came from exp-061 which used Pythia-2.8B and GPT-Neo-2.7B). They are legitimate confirmatory tests for Test A.

---

## Test B: Depth-Accumulated Primacy on Fresh Models

### Mechanism summary

Position 0 is an absorbing boundary of any causal row-stochastic kernel (row 0 has only the self-entry). Composing L layers drives attention mass toward it. "Primacy" = this absorbed mass; "recency" = the surviving bulk profile near the query. The two scaling laws:

- **Law A (effective depth):** primacy increases monotonically with d_eff = (1−μ)L
- **Law B (context dilution):** start-window primacy decreases monotonically with context N at fixed d_eff

Both confirmed synthetically (exp-066, ρ = +0.998 / ρ = −1.00) and descriptively on LongChat (exp-066, P3a ρ = +1.00). The LongChat result is exploratory/training data. The pre-registered fresh test follows.

### Observables

**Attention composition protocol:**
1. Run the model on n_inputs = 50 random-token sequences at seq_len = N.
2. Average attention weights → per-layer profile tensor.
3. For the power-law-conformal part: fit per-head A(Δx) = C·Δx^{-2Δ} over Δx ∈ [3, min(120, N/3)], deep queries i ≥ N/2. This gives per-layer per-head Δ̂.
4. The "layer k composite" is constructed by multiplying the L×L matrices of per-layer average attention profiles for layers 1..k, treating each as a row-stochastic matrix (renormalize rows to sum to 1 after averaging out the full-row profiles from the tensor).
5. **Primacy mass** at layer k: `prim_decile(k)` = mean of the deep-query composite row over the oldest decile of keys [0, N/10).
6. **Start-window mass** at context N: `prim_window(N, W=8)` = mean of deep-query average profile (from the full-depth composition at depth k=num_layers) over positions [0, 8).

### Test model: Pythia-1.4b

Pythia-1.4b is the primary test model for Test B because:
- 24 layers — sufficient for a depth sweep (k = 1..24)
- Context up to 2048 — sufficient for a context-length sweep (N ∈ {256, 512, 1024, 2048})
- Locally cached (~5.9 GB) — no download needed
- NOT used in exp-065 or exp-066 (those used LongChat-13B-16K and synthetic kernels)

Robustness: repeat on GPT-2-medium (24 layers, ctx 1024, cached) as a second architecture.

### Registered predictions

**P-B1 (Law A — primacy grows with depth):**
Pythia-1.4b, seq_len = 512, n_inputs = 50. Compose per-layer average attention profiles for k = 1, 2, 3, 4, 6, 8, 12, 16, 20, 24 layers. Measure prim_decile(k).

- **Keep:** Spearman ρ(k, prim_decile) ≥ 0.90 (confirming the depth-accumulation mechanism)
- **Kill:** ρ < 0.50 (depth does not drive primacy in real measured kernels; the mechanism fails on contact with actual layer profiles)
- **Ambiguous:** ρ ∈ [0.50, 0.90) — primacy grows but non-monotonically; record; do not claim confirmation.

**P-B2 (Law B — context dilution):**
Pythia-1.4b, full 24-layer composition, N ∈ {256, 512, 1024, 2048}. Measure prim_window(N, W=8) from the full-depth composite profile.

- **Keep:** Spearman ρ(N, prim_window) ≤ −0.90 (start-window mass strictly decreasing with context)
- **Kill:** ρ > −0.50 (no dilution or mass grows with N — Law B fails)
- **Ambiguous:** ρ ∈ [−0.90, −0.50) — direction correct but non-monotone; record.

**P-B3 (Law A robustness — GPT-2-medium):**
Same as P-B1 on GPT-2-medium (24 layers, ctx 1024). N = 512.
- **Keep:** ρ(k, prim_decile) ≥ 0.80 (slightly relaxed for architecture robustness)
- **Kill:** ρ < 0.50

### Honest limitations stated in advance

1. The composition protocol (multiplying per-layer attention matrices) is an approximation: it ignores V/MLP/residual nonlinearities and treats attention as the complete positional transport mechanism. The prediction is strictly about attention-side primacy.
2. The primacy observable (oldest-decile mass of the composite) is attention-side only; it does not directly measure task accuracy primacy. The accuracy-primacy link is a further, separately testable claim (Liu et al. benchmarks on MPT-30B-Instruct — not part of this pre-registration, pending model access).
3. Real layers are heterogeneous (varying Δ̂ per layer). The uniform-kernel derivation of d_eff = (1−μ)L is exact; the heterogeneous-layer version is approximate. The P-B1 prediction accounts for heterogeneity implicitly — it composes the measured per-layer profiles, not the theoretical one-Δ kernel.
4. "Primacy grows with depth" is near-tautological given the absorbing boundary (once mass hits position 0, it stays). Its content is that the measured per-head profiles, when composed, behave like the idealized absorbing-boundary model rather than being dominated by non-conformal mass that disrupts the absorption. The real content is P-B2 (Law B, context dilution) and the quantitative shape of P-B1.

---

## Procedure for recording results

Results on each fresh model are appended to this document under §§ A-Results and B-Results. No editing of §§ A–B of the pre-registration itself. The git history shows this document was written and committed before any fresh-model measurements.

The discipline is the same as the April pre-registration: confirmed results are confirmed; partial results are partial; kills are kills. Pythia-410m and Pythia-1.4b will produce P0 upgrade results that will supersede the April "confirmed" entries for those models if the joint statistic is better; they will not supersede them if the joint statistic fails to improve.

---

## Section A-Results

*(Append here after each fresh model run. Do not edit above.)*

---

## Section B-Results

*(Append here after Test B runs. Do not edit above.)*

---

*— Ariel, Mission Valley, Montana, June 14, 2026.*
*The statistic must earn its keep. If λ adds nothing, that is the finding.*
