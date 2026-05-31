# exp-041: GALA-7B Sigmoid Δ Measurement — Notes

*May 29, 2026*

---

## What was measured

Per-head conformal scaling exponent Δ for Apple GALA-7B sigmoid attention arm using the exp-007 protocol (power-law fit to mean attention profile A(k) ~ k^(−2Δ), R²>0.90, fit range lags 8–256, 50 random inputs, seq_len=512).

The sigmoid arm is a controlled comparison to GALA-7B softmax (same architecture — ALiBi + HybridNorm — only the attention function differs). Both arms downloaded from the public `axlearn-public` GCS bucket (step 250,000 checkpoints).

---

## Key results (sigmoid arm only — softmax still downloading)

| Metric | Sigmoid GALA-7B | OLMo ALiBi softmax (exp-039) | GPT-2 learned PE softmax (exp-007) |
|--------|----------------|------------------------------|--------------------------------------|
| Conformal heads (R²>0.90) | 2/1024 (0.2%) | 194/1024 (18.9%) | 44/144 (30.6%) |
| Δ_med (conformal heads) | 7.44 | 0.265 | 0.2493 |
| SYK-near heads (|Δ−0.25|≤0.05) | 0 | — | 44 |
| Median R² across all heads | 0.81 | — | — |
| Max R² | 0.92 (L22H12) | — | — |

**Pre-stated hypothesis:** sigmoid Δ_med ≈ 0.25 → mechanism-independent SYK fixed point; sigmoid Δ ≠ 0.25 → softmax normalization load-bearing.

**Result:** No meaningful conformal structure detected in sigmoid arm. 2 conformal heads are artifacts (L05H05: Δ=13.55, λ=10^36; L22H12: Δ=1.32, isolated). The exp-007 protocol fails to extract power-law profiles from sigmoid attention.

---

## Why this happens: profile anatomy

Three regimes emerge from the sigmoid + ALiBi combination:

**Large ALiBi slopes (heads 0–12, m ≈ 0.04–0.84):**
The ALiBi bias adds −m·k to the logit at lag k. For m=0.84 (head 0):
- Lag 8: att ≈ sigmoid(a − 6.7) → exponentially suppressed
- Lag 16: att ≈ 0 (below float32 precision)
- No data in the fit range [8, 256] → zero conformal heads in this class

**Small ALiBi slopes (heads 24–31, m ≈ 0.004–0.02):**
The ALiBi bias barely suppresses distant positions. The profile is dominated by the sigmoid of the trained QK dot products:
- att(k) ≈ sigmoid(QK(k)/√d + small bias)
- For random tokens, QK(k)/√d is approximately constant with lag
- Profile is nearly flat across lags 8–256 (L31H31 sample: 0.531 at lag 8, 0.393 at lag 256)
- Power-law fit: Δ ≈ 0.05, R² ≈ 0.80 — below threshold

**Physical explanation — why softmax is different:**
For softmax, the partition function Z(i) = Σ_j exp(logit_{ij}) re-normalizes the output to a probability distribution. This re-normalization "extracts" the spatial correlation structure of the QK geometry as a probability mass function. The decay of this probability with lag gives the conformal profile A(k) ~ k^(−2Δ).

For sigmoid, each logit is processed independently: att_{ij} = σ(logit_{ij}). Without normalization:
- The profile reflects the absolute sigmoid value at each lag, not the relative probability
- The ALiBi exponential term is additive to the logit, not multiplicative in the probability (no partition-function cancellation)
- The result is either exponential collapse (large m) or flat profiles (small m)

Neither produces a power law in the fit range.

---

## Verdict

**Falsification of universality hypothesis — but methodologically qualified.**

The finding is not simply "sigmoid Δ ≠ 0.25" — it is "the exp-007 power-law measurement protocol produces no meaningful Δ for sigmoid attention." The measurement itself is incompatible with the sigmoid mechanism: the protocol assumes that the attention profile encodes probability mass decay (requiring normalization), which sigmoid does not provide.

This supports the claim that **softmax normalization is load-bearing** for the conformal scaling signal — but the mechanism is specific: it is the partition-function normalization, not the "softmax" label per se.

A more discriminating follow-up would be:
1. Measure GALA-7B softmax Δ (awaiting download, same architecture)
2. Compare to OLMo ALiBi softmax (exp-039, Δ=0.265) for architecture control
3. Consider a modified sigmoid protocol: compute NORMALIZED sigmoid profiles (divide by sum of row) — this is effectively converting sigmoid to softmax and would tell us whether QK geometry alone is the source, or whether the normalization step is essential

---

## Methodological notes

- **Path correction:** Zarr arrays at `gda/model/decoder/...`, NOT `gda/model/model/decoder/...` as originally scripted. The double `model/` was a path error in the original script.
- **QKNorm required:** Model applies RMSNorm to Q and K per head (scale_query/norm/scale and scale_key/norm/scale, shape (32, 128)) after the QKV projection. Without QKNorm, logits reach ~750, fully saturating sigmoid to 1.0 everywhere. With QKNorm, logits are in range [−∞, ~9], producing sigmoid values in (0, 1) with decay structure.
- **Overflow warning:** Benign. Large ALiBi negative values cause np.exp(−logit) overflow, but sigmoid(−large) = 0 gracefully. Cast warning (float64→float32) similarly benign.
- **Runtime:** 50 inputs × 32 layers × 32 heads × 512 lags = ~9.5 minutes on M5 Max in NumPy.

---

## Softmax arm results (May 30, 2026)

Softmax arm completed. Results added to `experiments/exp-041_gala7b_sigmoid_delta/results.json`.

| Metric | Sigmoid GALA-7B | Softmax GALA-7B | OLMo ALiBi softmax | GPT-2 learned PE |
|--------|-----------------|-----------------|----------------------|-----------------|
| Conformal heads (R²>0.90) | 2/1024 (0.2%) | 121/1024 (11.8%) | 194/1024 (18.9%) | 44/144 (30.6%) |
| Δ_med (conformal heads) | 7.44 | 0.2739 | 0.265 | 0.2493 |
| SYK-near heads (|Δ−0.25|≤0.05) | 0 | 80 | — | 44 |
| SYK-near median | — | 0.2604 | — | 0.2493 |

**The softmax arm shows clear SYK-class conformal structure.** GALA-7B softmax Δ_SYK-near = 0.260, consistent with OLMo ALiBi softmax (0.265) and confirming the tentative PE ordering: ALiBi slightly elevates Δ above learned PE (0.249).

**Verdict: exp-041 complete — falsified_mechanism_dependent.** The SYK conformal fixed point appears only in the softmax arm. Same architecture, same training recipe, only the normalization function differs.

### Artifact heads in softmax arm

Layers 13, 14, 17, 25 each contribute 2 conformal heads with Δ_med ≈ 6.8 — these are artifacts (likely small-slope ALiBi heads with unusual profile shapes, as in the sigmoid arm). The physically meaningful signal is in the 80 SYK-near heads (Δ_med = 0.260).

### Updated verdict

The falsification is methodologically precise: the exp-007 protocol measures power-law decay in the PROBABILITY MASS function (requires normalization to a distribution). Sigmoid produces absolute values, not probabilities. The softmax comparison confirms the conformal structure is in the trained QK geometry — but the measurement protocol requires softmax (or any proper normalization) to extract it.

The **normalized-sigmoid control** (exp-042, May 31) directly tests whether any row-normalization suffices, or whether the exp() in softmax is essential.

## Next steps (updated May 31)

1. ~~Softmax arm measurement~~ — **DONE**
2. Normalized-sigmoid control → **exp-042 running** (GALA-7B sigmoid weights, att = σ/Σσ)
3. Update "What the Stones Listen To" essay — **gate satisfied; update essay with both-arm comparison before publish**

*See registry entry: exp-041 (confirmed, both arms complete).*
