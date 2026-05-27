# exp-034 — GPT-2 temperature–BCFT sweep

*May 26, 2026. REBUS × Junction 1 local test (see `notes/2026-05-26_rebus_junction1_formalization.md`).*

## Hypothesis (pre-stated)

Increasing softmax temperature (lowering precision) drifts median Δ away from SYK 0.25 and weakens BCFT boundary parameter λ before power-law structure breaks.

## Protocol

- Model: `gpt2`, eager attention, manual block-by-block forward with logits scaled by `1/(√d_k · T)` before softmax (validated max diff 0 vs HF at T=1.0).
- Temperatures: T ∈ {0.5, 1.0, 2.0}.
- 50 random-token inputs, seq_len=256; exp-007 decay fit (R²>0.90); exp-023 BCFT λ on conformal heads (R²>0.90, 0.15<Δ<0.40).

## Results

| T | Δ_med (PL heads) | |Δ−0.25| | PL heads (R²>0.90) | conformal heads | λ_med |
|---|------------------|---------|---------------------|-----------------|-------|
| 0.5 | **0.468** | 0.218 | 36 | 12 | **0.017** |
| 1.0 | **0.243** | 0.007 | 45 | 23 | **1.635** |
| 2.0 | **0.328** | 0.078 | 59 | 18 | **1.562** |

T=1.0 reproduces exp-007 (Δ_med=0.2493) within 0.01 — protocol check passes.

**Baseline-head tracking** (23 heads conformal at T=1.0, refit at each T):

| T | Δ_med (same heads) | λ_med (same heads) |
|---|--------------------|--------------------|
| 0.5 | 0.427 | 0.230 |
| 1.0 | 0.226 | 1.635 |
| 2.0 | **0.118** | **4.256** |

## Interpretation

**Confirmed:** Conformal dimension is **not invariant under temperature** — trained T=1 sits at Δ≈1/4; perturbing T moves Δ away. Drift is real.

**Partially falsified (simple REBUS mapping):** Δ is **not monotone** in T (0.5→1.0→2.0 gives 0.47→0.24→0.33). λ does **not** weaken at high T globally — it **collapses at low T** (high precision: λ_med≈0.02) and **strengthens on the baseline conformal set at T=2** (λ_med≈4.26) while Δ on those heads drops toward trivial (0.12).

**Reading:** Precision (inverse T) and conformal scaling couple: the SYK fixed point is a property of **trained** softmax at T=1, not a generic feature of the architecture under temperature rescaling. Low-T sharpening steepens apparent decay (Δ↑ toward q=2 band); high-T flattening on already-conformal heads may compress interior while inflating boundary correction — needs finer η-resolved analysis, not just global λ.

## Status

`confirmed_drift` — drift confirmed; naive “T↑ → λ↓” falsified.

## Next

- Finer sweep around T=1 (e.g. 0.8, 1.0, 1.2) to map the basin.
- Per-head η-collapse plots at each T (exp-023 phase 3) on baseline set.
- Sigmoid falsification (exp-033) remains the mechanism test — temperature is softmax-precision, not architecture change.
