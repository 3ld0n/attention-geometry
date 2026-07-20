# exp-087 — RAND Early Burst Scaling with Model Size

**Pre-registration date:** 2026-07-20  
**Status:** Pre-registered (results not yet run)  
**Follows:** exp-086 (longitudinal Δ-spectrum), finding 3 from 2026-07-20 session

---

## Motivation

exp-086 measured the full training trajectory of Pythia-70m (6L × 8H = 48 heads). At step 512, the RAND condition showed 9/48 = 18.75% SYK-near heads — a transient early burst that collapses to 5/48 by step 4000. The layer architecture analysis (2026-07-20 note) identified this burst as arising from random initialization: at step 512, training has barely begun, and heads near the conformal attractor from initialization are still there before semantic training pulls them away.

The 16f hypothesis (queued 2026-07-20): *In larger models (more heads per layer), the RAND step-512 burst should scale with total heads — more random draws, more hits near the conformal attractor.* If the conformal attractor is a per-head property (determined by d_k and the power-law fitting window), the fraction of SYK-near heads at step 512 RAND should be approximately constant across model sizes with the same d_k.

---

## Architecture comparison

| Model | Layers | Heads/layer | d_k | Total heads |
|-------|--------|-------------|-----|-------------|
| Pythia-70m | 6 | 8 | 64 | 48 |
| Pythia-410m | 24 | 16 | 64 | 384 |

Both models use RoPE positional encoding and d_k=64. The GOE universality results (exp-047, exp-051, exp-077) confirm that the spectral substrate is architecture-independent at the same d_k.

---

## Hypotheses (pre-registered, commit to follow)

**H_scale_const:** The fraction of RAND SYK-near heads at step 512 is approximately constant across model sizes with the same d_k. Criterion: Pythia-410m fraction = SYK-near / 384 lies within ±0.05 absolute of 0.1875 (the 70m fraction), i.e., 0.1375 ≤ f ≤ 0.2375, i.e., 53 ≤ n ≤ 91 SYK-near heads.

**H_scale_layer_fixed:** The number of SYK-near heads per layer is approximately constant. Criterion: Pythia-410m SYK-near / 24 layers ≈ 9/6 = 1.5 per layer (= 36 total). Range: 1.0–2.0 per layer = 24–48 total.

**H_scale_supralinear:** SYK-near fraction increases with model size. Criterion: Pythia-410m fraction > 0.2375.

**H_scale_sublinear:** SYK-near fraction decreases with model size. Criterion: Pythia-410m fraction < 0.1375.

Notes on the hypotheses:
- H_scale_const and H_scale_layer_fixed are partially compatible (their ranges overlap at 36–48)
- If n ≈ 36–48: both H_scale_const and H_scale_layer_fixed consistent
- If n ≈ 53–91: H_scale_const only
- If n < 36: H_scale_sublinear (structural conformal heads saturate)
- If n > 91: H_scale_supralinear

The physical picture predicts H_scale_const (same d_k → same per-head probability at initialization), but structural heads might introduce a fixed-per-layer component that makes the total count less than proportional.

---

## Measurement protocol

Matches exp-086 exactly:
- `FIT_LOW = 3, FIT_HIGH = 50, R2_THRESHOLD = 0.85`
- `SYK_LOW = 0.20, SYK_HIGH = 0.30`
- `SEQ_LEN = 256, N_RAND = 20, RAND_SEED = 87`
- Checkpoint: step512 (same training step as the 70m burst)
- RAND condition only (random token sequences, same vocabulary)
- Analysis: BCFT power-law fit per head, count SYK-near

Additional output: per-layer SYK-near count (to test H_scale_layer_fixed).

---

## Existing data

Pythia-70m step 512 RAND (from exp-086 crossover_fine_results.json):
- SYK-near: 9/48 heads (18.75%)
- SYK-near heads: L1H0, L1H4, L2H5, L2H6, L3H0, L3H1, L3H6, L4H3, L4H7
- Structural heads (L4H3, L4H7) are among the burst: 2 of the 9 are architecturally stable

---

## What Pythia-410m data reveals

After running:
- Total SYK-near count at step 512 RAND
- Per-layer distribution (to identify structural attractor layers in 410m)
- Comparison against the 70m per-layer rates

---

## Verdict criteria

| Result | Verdict |
|--------|---------|
| n ≥ 53 and n ≤ 91 (fraction 0.14–0.24) | H_scale_const CONFIRMED |
| n in 24–48 (1.0–2.0 per layer) | H_scale_layer_fixed CONFIRMED |
| n > 91 (fraction > 0.24) | H_scale_supralinear CONFIRMED |
| n < 24 (fraction < 0.06) | H_scale_sublinear CONFIRMED |
| n in 36–48: both CONST and LAYER_FIXED consistent | Ambiguous — need additional size point |

---

## Connection to exp-085

If exp-085 (generational transmission) returns H_transmission_yes (~10-15 SYK-near in Pythia-70m): the generated text contains enough world-reference to recruit semantic conformal heads. Combined with exp-087: the structural head fraction should appear as a substrate in 410m regardless of the transmission verdict. This would let us ask: what fraction of the conformal heads in larger models are "background structural" vs. "semantically recruited"?

---

*This note constitutes the pre-registration. Results will be appended below after the run.*

---

## Results (appended after run)

*TBD*
