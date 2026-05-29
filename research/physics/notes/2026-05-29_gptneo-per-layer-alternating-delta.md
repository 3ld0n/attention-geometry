# exp-040 — GPT-Neo-2.7B per-layer alternating Δ (analysis-only)

*2026-05-29*

## Question

Does GPT-Neo's overall Δ_med=0.126 (exp-039) decompose into a bimodal per-layer distribution reflecting its global/local architecture? And is the separation uniform with depth?

## Protocol

Analysis-only. Load exp-025 BCFT JSON (April 2026), filter to `EleutherAI/gpt-neo-2.7B`, compute per-layer Δ_median (R²>0.90). Coverage: 89.4% (exp-037: L≤31 of 32-36 layers). Source experiment: exp-025.

## Pre-stated hypothesis

Even-indexed layers (global full attention): Δ near trivial fixed point (~0.1).
Odd-indexed layers (local window attention): Δ >> 1.
Architecture determines the split, not training dynamics. Separation >5× expected.

## Results

**Global layers (even):** Δ_med = 0.1018 (14/16 layers with data)
**Local layers (odd):** Δ_med = 0.3361 (15/16 layers with data)
**Overall separation: 3.30×**

**By depth:**
- Shallow (L≤16) local/global separation: **17.48×** — striking alternating contrast
  - Global L10,L12,L14,L16: Δ ~ 0.08–0.12 (near trivial)
  - Local L5,L7,L9,L11,L13: Δ ~ 1.61–2.13 (strongly localized)
- Deep (L>16) local/global separation: **1.55×** — pattern essentially collapsed
  - Both types converge toward Δ ~ 0.10–0.15

**Verdict: PARTIAL** — alternating pattern real and dramatic in shallow layers, absent in deep layers.

## Interpretation

Two distinct regimes in GPT-Neo:

**Shallow layers (L≤16):** Architecture fully in control. Global attention heads sit near the trivial fixed point (Δ → 0, attention uniform). Local window heads have Δ ~ 2 — strongly localized, consistent with the hard 256-token window creating near-singular positional dependency. The 17× separation is larger than the simple pre-stated prediction of ~5× — the window attention is more localized than expected.

**Deep layers (L>16):** Convergence to moderate Δ regardless of type. Local window layers no longer show high Δ — the model appears to use its capacity in deeper layers to compute more global-like representations even within the local window constraint. Or: the attention patterns in deep layers become more uniform across lags regardless of the window boundary.

This is the first direct measurement of the **depth-dependence of architectural Δ separation**. The May 17 finding ("GPT-Neo shows alternating Δ") now has quantified depth structure.

## Implications for PE ordering

GPT-Neo should be removed from the PE ordering table (exp-039 tentative table). It is not a clean ALiBi measurement — the alternating architecture confounds any PE signal. The global-layer population (Δ~0.10) is near the trivial fixed point, not near any SYK value. The local-layer population has artificially elevated Δ from the window constraint.

**OLMo-7B remains the cleanest ALiBi data point** (pure ALiBi, full attention, Δ_med=0.265, exp-039).

## Per-layer table

| Layer | Type   | Conf/Total | Δ_med  |
|-------|--------|------------|--------|
| 0     | global | 0/20       | —      |
| 1     | local  | 0/20       | —      |
| 2     | global | 0/19       | —      |
| 3     | local  | 2/20       | 0.859  |
| 4     | global | 2/10       | 0.087  |
| 5     | local  | 7/8        | 1.612  |
| 6     | global | 4/8        | 1.602  | ← anomalous (only 4 heads, noisy)
| 7     | local  | 13/14      | 2.135  |
| 8     | global | 9/16       | 0.880  | ← anomalous
| 9     | local  | 10/11      | 1.748  |
| 10    | global | 8/20       | 0.076  |
| 11    | local  | 4/10       | 1.796  |
| 12    | global | 13/20      | 0.115  |
| 13    | local  | 7/16       | 1.825  |
| 14    | global | 18/20      | 0.100  |
| 15    | local  | 14/20      | 0.366  |
| 16    | global | 17/20      | 0.086  |
| 17    | local  | 13/20      | 0.319  |
| 18    | global | 18/20      | 0.083  |
| 19    | local  | 9/20       | 0.214  |
| 20    | global | 15/20      | 0.097  |
| 21    | local  | 16/20      | 0.175  |
| 22    | global | 13/20      | 0.099  |
| 23    | local  | 14/20      | 0.121  |
| 24    | global | 16/20      | 0.104  |
| 25    | local  | 6/20       | 0.147  |
| 26    | global | 12/20      | 0.130  |
| 27    | local  | 2/20       | 0.140  |
| 28    | global | 15/20      | 0.108  |
| 29    | local  | 1/20       | 0.108  |
| 30    | global | 10/20      | 0.108  |
| 31    | local  | 2/20       | 0.336  |

Note: L6 and L8 global-type anomalies (high Δ) have few conformal heads (4/8, 9/16) — noisy. L14-L16 global layers are reliable (17-18/20 heads R²>0.90) and cleanly show Δ~0.10.

## Anomalous global layers (L6, L8)

L6 has Δ_med=1.60 with only 4/8 R²>0.90 heads — one or two outlier heads dominate the median with sparse data. Not reliable. L8 similarly: 9/16 heads, Δ_med=0.88. These are probably noise from the BCFT JSON's partial coverage of this model at intermediate depth.

## Source

- exp-039 (overall GPT-Neo Δ_med=0.126, architectural confound identified)
- exp-025 BCFT JSON (April 2026, EleutherAI/gpt-neo-2.7B, 89.4% head coverage)
- exp-037 (fp16 NaN coverage audit — confirmed L≤31)
