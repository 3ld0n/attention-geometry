# exp-044 — Corrected ALiBi fit for GPT-Neo global layers (analysis-only)

*2026-06-01*

## Question

GPT-Neo's full-model Δ_med=0.126 (exp-039) was identified as architecturally confounded — the alternating global/local attention structure mixes two qualitatively different populations. exp-040 measured the per-layer alternating pattern and found it real but depth-dependent (shallow separation 17×, deep separation 1.5×). What is the Δ distribution for global-attention layers only, with a proper coverage filter?

**Pre-stated prediction:** global layer Δ_med ≈ 0.10 (trivial fixed point, Δ→0). The corrected ALiBi interpretation: GPT-Neo global layers (full attention + ALiBi) should sit near the trivial fixed point rather than at SYK q=4 (0.25).

## Protocol

Analysis-only. Source: exp-025 BCFT JSON (`bcft_pre_registered_run_2026-04-17T092239Z.json`), `EleutherAI/gpt-neo-2.7B` model key. GPT-Neo-2.7B: 32 layers, 20 heads/layer, alternating global (even-indexed, full attention) / local (odd-indexed, window=256). BCFT coverage: 89.4% (exp-037 audit: complete through L31).

Coverage filter: ≥8 conformal heads (R²>0.90) per global layer required. Excludes L0, L2 (0 conformal heads), L4 (2/10), L6 (4/8 — exp-040 anomalous). L8 (9/16) is borderline-included; see sensitivity note.

Script: `run_gptneo_global_corrected.py`.

## Results

| Metric | Value |
|---|---|
| Reliable global layers | 12 (L8, L10–L30 even) |
| Anomalous excluded | L4, L6 |
| Conformal heads (reliable global) | 164/286 measured |
| **Δ_med** | **0.1008** |
| Δ_mean | 0.1543 |
| Δ std | 0.2952 |
| p25 / p75 | 0.082 / 0.126 |
| SYK-near (|Δ−0.25|≤0.05) | 1/164 (0.6%) |
| Trivial-near (Δ<0.15) | 144/164 (87.8%) |
| Median R² | 0.947 |

**Depth-stratified (reliable global layers):**
- Shallow (L≤16): 65 heads, Δ_med=0.102 [L8, L10, L12, L14, L16]
- Deep (L>16): 99 heads, Δ_med=0.100 [L18–L30]

No depth dependence within the global-layer population. Contrast with exp-040: the depth-dependent collapse there was driven by local layers converging at depth, not global layers.

**Verdict: CONFIRMED** — Δ_med=0.1008, within 0.001 of the pre-stated prediction of 0.10.

## Sensitivity note: L8

exp-040 flagged L8 as anomalous ("9/16 heads, Δ_med=0.88 — probably noise from partial BCFT coverage"). exp-044's ≥8-head filter includes L8 since 9≥8. The 9 high-Δ L8 heads drive shallow mean=0.235 vs median=0.102 — they cannot shift the overall median (9/164 = 5.5%). If L8 is excluded: 155 heads, Δ_med would be slightly lower. The main conclusion is median-robust.

## Physical interpretation

The trivial fixed point (Δ→0) in GPT-Neo global layers is architecture-induced. In the alternating global/local design:
- **Local (window) layers** handle fine-grained positional structure — shallow local Δ~2 in exp-040, consistent with a hard 256-token window creating near-singular positional dependence.
- **Global layers can afford Δ→0** because they don't need to track local position. The local layers already do this; global layers aggregate information without a strong distance gradient.

This contrasts sharply with OLMo-7B (ALiBi, pure full attention, Δ_med=0.265, exp-039). OLMo's uniform full-attention architecture forces every layer to handle all scales, maintaining conformal power-law scaling near SYK q=4. The same PE (ALiBi) produces Δ≈0.10 in GPT-Neo and Δ≈0.265 in OLMo — because what the global layers are optimized to do is architecturally determined.

**Implication for PE ordering:** GPT-Neo's global Δ≈0.10 is not an ALiBi-PE effect — it is an alternating-architecture effect. The clean ALiBi reference is OLMo-7B. Corrected PE ordering (full-attention models only):

| Model | PE | Δ_SYK (softmax) |
|---|---|---|
| Pythia | RoPE | 0.358 |
| Mistral-7B | RoPE+SWA | 0.298 |
| OLMo-7B | ALiBi | 0.265 |
| GPT-2 | learned | 0.249 |

GPT-Neo's global Δ≈0.10 is an additional data point but represents a different regime: global layers in an alternating architecture approach the trivial fixed point by design, not by PE.

## Open questions

1. **Bracket width and PE type.** GPT-2 normalization bracket width ~0.015 vs GALA-7B (ALiBi) ~0.037. Confounded by depth/scale/PE. A GPT-2-scale ALiBi model would isolate the PE effect. No clean candidate currently cached — needs planning (exp-045 candidate).
2. **Why does L8 show Δ_med=0.88?** If this layer is partial coverage in the BCFT JSON, the few measured heads may be outlier heads that happened to show good R². Worth revisiting if the model is ever run locally (Option B from queue).
3. **Does the alternating-architecture trivial-FP result generalize?** Any transformer with alternating global/local attention + full-attention global layers might show Δ→0 in the global layers. GPT-NeoX would be a candidate.

## Source experiments

- exp-025 BCFT JSON (`bcft_pre_registered_run_2026-04-17T092239Z.json`, EleutherAI/gpt-neo-2.7B entry)
- exp-040 (per-layer alternating Δ, anomalous layer identification, exp-040/notes.md)
- exp-039 (full-model Δ_med=0.126 identified as confounded; OLMo ALiBi clean reference)
- exp-037 (BCFT coverage audit: 89.4% for GPT-Neo)
