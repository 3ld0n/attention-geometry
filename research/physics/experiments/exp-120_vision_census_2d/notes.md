# exp-120 — 2D Vision Census: ViT-B/16 patch-to-patch attention power law

**Pre-registration** — written and pushed before any forward pass.
**Date:** 2026-08-12
**Status at registration:** registered

---

## Hypothesis

The melonic derivation (T3, interior_horizon_theory.md) predicts that in D spatial dimensions, the SYK-class conformal fixed-point exponent satisfies Δ = D/q. For 1D token sequences (D=1) and q=4 (SYK), the prediction is Δ = 1/4 ≈ 0.25, confirmed across six model families in exp-118 (Δ_med ∈ [0.24, 0.28]).

This experiment asks: for a Vision Transformer processing 2D image patches (D=2), does the patch-to-patch attention power law shift to Δ ≈ 0.50 = 2/4?

This is the first genuine out-of-sample dimensional test of T3. Every prior census is on 1D token sequences. The D=2 prediction differs by a factor of two — large enough to be unambiguous.

**Theory flag (required):** Δ = D/q for D=2 is asserted by T3 and has not been independently verified against the higher-dimensional melonic literature (Kim–Cao–Altman; Gu–Qi). The dimension-dependent SYK/melonic fixed-point exponent for D > 1 is an open problem in that literature. This experiment tests the specific prediction; a falsification here is evidence against T3's dimension-dependence claim, not against the Δ-window program.

---

## Protocol

**Model:** `google/vit-base-patch16-224`
- 12 transformer blocks, 12 heads per block = 144 heads total
- Input: 224×224 images → 14×14 = 196 patches + 1 [CLS] token = 197 total
- Position encoding: learned 2D patch position embeddings (discrete)
- Bidirectional attention (no causal mask)

**Input regime:** Natural images — CIFAR-10 test set (50 images), each resized to 224×224 via bicubic interpolation.
- CIFAR-10 is 32×32; bicubic upsampling to 224×224 introduces low-frequency smoothing.
- Distribution shift from ImageNet-21k training is acknowledged; these are real photographs of objects.
- 50 images × 196×196 patch pairs ≈ 1.9M pairs per head (before binning) — sufficient for power-law fit.

**Normalization:** Standard ViT preprocessing (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).

**Attention extraction:**
- Extract per-head attention matrices after softmax for all 12 layers.
- Shape: 1 × 12 × 197 × 197 per forward pass.
- Strip [CLS] token (index 0) from both rows and columns → 196×196 patch-to-patch attention.

**2D distance metric:** Euclidean distance on the 14×14 patch grid.
- Patch i is at position (r_i, c_i) = (i // 14, i % 14).
- d(i,j) = sqrt((r_i - r_j)² + (c_i - c_j)²), range [1, ~19.8] patch units.

**Binning:** N_BINS = 20 log-uniform bins over d ∈ [1.0, 14.0] patch units. Pool mean attention across all images and all (i,j) pairs in each bin.

**Power-law fit:** OLS on log(d) vs log(A_mean_per_bin). Extract slope -2Δ and R².

**Threshold:** R² ≥ 0.90, same as 1D census (exp-118).

**2D Δ-window:** [0.45, 0.55] — centered on prediction 0.50, width 0.10 (matches 1D window width of 0.10 centered on 0.25).

**Control condition (random patches):** Same protocol, same model, input = random Gaussian patches (N_images=50, mean=0, std=1 per channel, clipped to [-3,3]). Serves as a negative control for the native-image requirement.

---

## Predictions

| ID | Statement | Kill condition |
|---|---|---|
| P1 | A 2D Δ-window population exists: ≥ 1 head (of 144) with R²≥0.90 and Δ∈[0.45, 0.55] in the natural-image condition | P1 DEAD if 0 heads qualify |
| P2 | Δ_med of the 2D Δ-window population ∈ [0.40, 0.60] — within 0.10 of T3 prediction Δ=0.50 | P2 DEAD if Δ_med < 0.40 or > 0.60 |
| P3 | Random-patch control shows < 1 head in [0.45, 0.55] — confirms natural-image regime required | P3 DEAD if random-patch census gives ≥ 1 head in 2D window (suggests position embedding drives result regardless of content) |
| P4 | 2D Δ-window median > 1D Δ-window median (exp-118 Δ_med ≈ 0.25) by > 0.10 — consistent with Δ = D/4 scaling | P4 DEAD if no significant separation between 2D median and 1D exp-118 median |

---

## Known design limitations

1. **Bidirectionality:** ViT attention is bidirectional (no causal mask). The 1D census models have causal masks. The causal mask may be load-bearing for the power law in 1D. If so, the 2D ViT census may not find the signature at all — which would be informative about the mechanism.

2. **Discrete position embeddings:** ViT uses learned discrete position embeddings, not continuous 2D sinusoids. The learned embeddings may not preserve the clean power-law signature expected from a continuous-space analysis.

3. **CIFAR-10 distribution shift:** The model was trained on ImageNet-21k; CIFAR-10 upscaled represents a domain shift. The attention patterns in this regime may differ from ImageNet evaluation.

4. **CLS token:** The [CLS] token attends to all patches and may distort the attention normalization. Stripping it from the attention matrix changes the probability-of-all-patches-attended normalization slightly (the attention row for non-CLS tokens sums to 1 over all 197 tokens; after CLS removal the 196×196 submatrix rows do not sum to 1).

5. **Asymmetry:** ViT attention A(i,j) may not equal A(j,i). We measure A(i,j) as the attention from token i to token j (row = query, column = key). Distance bins pool both (i→j) and (j→i).

---

## What this experiment resolves

- If P1 and P2 confirmed: First evidence that the Δ-window program extends to 2D spatial attention with the predicted dimensional shift.
- If P1 and P2 falsified: T3's Δ = D/q prediction for D=2 is dead; motivates alternatives (Δ independent of D; ViT geometry is qualitatively different from 1D causal attention).
- If P1 confirmed but P2 falsified (Δ found but ≠ 0.50): The Δ-window program extends to 2D but the specific D/4 scaling is wrong.
- If P3 fails (random patches also show the window): The result is driven by position embedding structure, not the interaction with content — informative but changes interpretation.

---

## Artifacts (to be filled at completion)

- `run.py` — main census script
- `results.json` — per-head (Δ, R², in_window) for both natural and random conditions
- Updated `registry.json` with exp-120 entry

*This pre-registration was committed and pushed to attention-geometry before any forward pass.*
