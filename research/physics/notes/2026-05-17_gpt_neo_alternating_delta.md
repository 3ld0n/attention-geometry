# GPT-Neo-2.7B: Alternating Δ Pattern — Architectural Confound in BCFT Analysis

*Written May 17, 2026, during a free physics session.*

---

## The Finding

In the BCFT pre-registered test data, GPT-Neo-2.7B (ALiBi, 2.7B) shows a striking alternating per-layer Δ pattern:

- Even layers (0, 2, 4...): Δ_med ≈ 0.08–0.13 (long-range, small)
- Odd layers (1, 3, 5...): Δ_med ≈ 0.12–2.1 (short-range to very large, especially in middle layers)

This pattern is absent in RoPE models (Pythia-1.4b, OLMo-7B-hf) which show gradual per-layer variation without sharp even/odd structure.

GPT-Neo-2.7B also has the highest BCFT ρ: **0.9582**, significantly above all other models.

---

## The Explanation: Architecture, Not Two Fixed Points

**GPT-Neo-2.7B uses exactly alternating global/local attention:**
- Even layers: global attention (full context)
- Odd layers: local attention (window = 256 tokens)

Configured as `attention_types = [["global", "local"], 16]` — confirmed in Hugging Face config.

**The large Δ in odd layers is NOT evidence of a UV fixed point.** It is a measurement artifact: the BCFT analysis fits a power law over seq_len > 256 to attention weights that are exactly zero outside the 256-token window. A truncated function looks like steep power-law decay, inflating the effective Δ. The local layers don't sit at a different conformal fixed point — they sit at a hard wall.

**The small Δ in even (global) layers IS likely conformal scaling,** albeit modified by ALiBi's linear distance bias. The global layers develop content-dependent long-range attention to compensate for the ALiBi distance penalty. Δ_med ≈ 0.10–0.13 is smaller than the GPT-2 value (0.25), possibly because ALiBi forces stronger content-based weighting at all distances.

---

## Why GPT-Neo Has the Highest ρ

The high ρ (0.9582) reflects the clean bimodal signal, not the cleanest conformal physics:

1. **Global layers**: true conformal scaling → small Δ → shallow LiTM valley. BCFT prediction mechanistically correct.
2. **Local layers**: window cutoff masquerades as large Δ → deep LiTM valley. BCFT prediction phenomenologically accurate but physically different mechanism.

Both contribute to a strong Δ ↔ valley_depth correlation. The signal is unusually clean precisely because the bimodal distribution (Δ ≈ 0.10 vs Δ ≈ 0.20–2.0) separates cleanly into two valley-depth populations.

---

## Methodological Implication

The BCFT analysis, as currently implemented, confounds two distinct sources of large Δ:
1. **True conformal structure**: a trained head that has learned power-law attention decay
2. **Window cutoff**: a local-attention head whose truncated weight profile is fit (poorly) by a steep power law

Both produce high measured Δ. Both predict deep LiTM valleys. The correlation holds in both cases, but the physical story differs.

**Correction needed for fair model comparison:**
- When comparing ρ values across models, flag whether the model uses alternating local/global attention. A high ρ for models with architectural windowing partly reflects the window structure, not only the learned conformal structure.
- GPT-Neo-2.7B's ρ = 0.9582 is partly architectural; it's not directly comparable to Pythia-1.4b ρ = 0.7105 or GPT-2's expected ρ.

---

## The Convergence Within Local Layers

The odd (local) layers show an interesting depth trajectory. Their Δ values:
- Early odd layers (3, 5, 7, 9): Δ ≈ 0.86–2.1 (very large, strongly window-limited behavior)  
- Middle odd layers (11, 13): Δ ≈ 1.8 (still very large)
- Later odd layers (15, 17, 19, 21): Δ ≈ 0.17–0.37 (converging downward)
- Deepest odd layers (23, 25, 27, 29, 31): Δ ≈ 0.11–0.15 (approaching even-layer values)

The local layers' effective Δ DECREASES with depth, eventually approaching the global layers' value (~0.11) by layers 28–30. This depth convergence within the local layers suggests two things:
- In early layers, local attention is truly local (many heads concentrated within a narrow sub-window)
- In later layers, local attention spreads more uniformly across the 256-token window, producing smoother decay that fits a shallower power law
- The convergence toward Δ ≈ 0.11 in both local and global late layers suggests both layer types reach the same IR behavior within their respective domains

This is the depth convergence result (Δ flowing toward IR fixed point with depth) operating within the local-attention layers themselves. The RG flow occurs even within the architecturally constrained local attention domain.

---

## Questions This Opens

1. **What is the true conformal Δ for the global layers of GPT-Neo-2.7B, corrected for ALiBi?** The raw value ~0.10–0.13 might reflect the interaction between conformal scaling and the ALiBi exponential envelope. Disentangling the two requires fitting A(|i-j|) ~ |i-j|^{-2Δ} × exp(-m|i-j|) directly, rather than a pure power law.

2. **Would a model with full attention and ALiBi land at a smaller Δ than GPT-2?** If ALiBi genuinely drives a different fixed point (by modifying the "disorder" of effective couplings), then full-attention ALiBi models should show Δ < 0.25. Currently we don't have such a comparison.

3. **Can BCFT be extended to predict LiTM valley depth for local-attention layers directly from the window size?** The window cutoff is a hard boundary condition, which in BCFT language corresponds to a Dirichlet boundary at |i-j| = W. The conformal block for a finite-distance Dirichlet boundary should be computable.

4. **Scaling collapse:** does correcting for the window cutoff in local layers reveal a universal curve for Δ_eff vs depth, collapsing across different models?

---

## Relation to Existing Results

This finding reinforces the STATUS.md note about GPT-Neo-2.7B: "GPT-Neo-2.7B has alternating-layer structure suggesting two distinct head populations." The two populations are now identified as global and local attention layers. The BCFT framework detected the architectural distinction — which is itself meaningful as a validation of the framework's sensitivity to attention structure.

The sign anomaly in ρ(λ, valley) being negative remains unexplained by this finding. Still needs investigation.

---

*Status: speculative → developing. Needs: (1) direct fit with power-law × exp-decay model for ALiBi correction, (2) comparison with full-attention ALiBi model, (3) formal extension of BCFT to Dirichlet boundary for windowed attention.*


---

## Revised Picture: Positional Encoding Selects Fixed Points

Toy experiment (May 17, 2026): simulated power-law attention logits (true Δ) + ALiBi penalty at varying m. 

**Finding: ALiBi INCREASES apparent Δ monotonically.** For true Δ=0.25 and m=0.01, apparent Δ=0.32. For m=0.02, apparent Δ=0.40. The exponential decay from ALiBi stacks on top of the power-law decay, making the combined function appear steeper (larger Δ) than the underlying power law.

**Implication for GPT-Neo global layers:** Measured Δ ≈ 0.08–0.13. Since ALiBi increases apparent Δ, the true underlying structure must have Δ ≤ 0.08. The global layers are approaching the **trivial fixed point** Δ → 0 (uniform attention across context). ALiBi provides a small perturbation that lifts apparent Δ slightly above zero.

**Physical interpretation:** GPT-Neo's global-attention layers learned to provide nearly UNIFORM context attention — a flat background. This makes sense architecturally: with local layers handling nearby token processing, the global layers provide a diffuse, non-local context signal. The ALiBi penalty keeps this from becoming perfectly uniform, but the underlying attentional structure is near Δ = 0.

### The Fixed-Point Classification

| Positional encoding | Fixed point | Measured Δ | Physical meaning |
|---|---|---|---|
| Learned (GPT-2) | SYK q=4 | 0.25 | Unconstrained RG flow reaches chaotic IR fixed point |
| RoPE (Pythia) | Perturbed q=4 | 0.28–0.38 | RoPE adds relevant perturbation; system arrested between q=2 and q=4 |
| ALiBi global (GPT-Neo even layers) | Trivial | ~0.08–0.13 (true Δ → 0) | Distance penalty drives attention toward uniformity |
| ALiBi local (GPT-Neo odd layers) | Window cutoff | >>0.25 (artifact) | Window at 256 tokens produces steep apparent decay |

### Why BCFT Still Works

The BCFT framework predicts valley depth from Δ. This prediction holds phenomenologically even when different models sit at different fixed points:
- SYK q=4 layers: BCFT is mechanistically correct (conformal field theory)
- Trivial fixed point layers (Δ→0): BCFT predicts shallow valley, which is what uniform attention produces
- Window-cutoff layers (large Δ): BCFT predicts deep valley, which is what local-window attention produces

The framework is detecting a real correlation between attention range and LiTM performance, regardless of the conformal field theory mechanism. The phenomenology is more robust than the theory.

### Implications for the Pre-Registered Test

The GPT-Neo-2.7B result (ρ = 0.9582) is mechanistically driven by:
1. Global layers: trivial-fixed-point attention (uniform, Δ→0) → shallow LiTM valley
2. Local layers: windowed attention (Δ_apparent large) → deep LiTM valley
3. The bimodal distribution of Δ values creates an unusually clean signal

This is real physics, just different physics than the SYK story. The BCFT pre-registration should be annotated: GPT-Neo-2.7B confirmed the prediction, but through architectural attention scope rather than conformal scaling.

### New Prediction

Full-attention models with ALiBi (without local windowing) should show very small measured Δ (< 0.05) because ALiBi alone drives toward the trivial fixed point. This would be distinctive from both GPT-2 (Δ≈0.25) and Pythia (Δ≈0.28–0.38). No such model was in the pre-registered test; this is an open prediction.

