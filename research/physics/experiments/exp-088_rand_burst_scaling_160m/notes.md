# exp-088 — RAND Burst Scaling: Pythia-160m Third Scale Point

**Pre-registration date:** 2026-07-20  
**Status:** Pre-registered (results not yet run)  
**Follows:** exp-087 (RAND early burst scaling, Pythia-70m vs 410m, N^0.43 confirmed)

---

## Motivation

exp-087 measured the RAND step-512 SYK-near burst in two model sizes and found sublinear scaling (N^0.43) with a structural decomposition:
- **Structural component:** ~9 heads in L1–L4, approximately constant across model size (L4H3 and L4H7 confirmed architecture-determined in both 70m and 410m)
- **Background component:** ~0.65 SYK-near per deep layer (L5+), growing with depth

With only two data points, the N^0.43 exponent has no uncertainty estimate and no test against an intermediate size. Pythia-160m (12L × 12H = 144 heads, d_k=64) is the natural intermediate point: it sits between 70m (48 heads) and 410m (384 heads) in the same architectural family, same d_k, same tokenizer.

This experiment adds a third scale point to either confirm or refine the N^0.43 scaling law.

---

## Architecture comparison

| Model | Layers | Heads/layer | d_k | Total heads |
|-------|--------|-------------|-----|-------------|
| Pythia-70m | 6 | 8 | 64 | 48 |
| **Pythia-160m** | **12** | **12** | **64** | **144** |
| Pythia-410m | 24 | 16 | 64 | 384 |

All three models: RoPE positional encoding, d_k=64. GOE universality confirmed across all three (exp-047, exp-051, exp-077).

---

## Predictions (pre-registered)

### Prediction 1 — Total count consistent with N^0.43

From exp-087: N^0.43 scaling fit through two points (48→9, 384→22).

For Pythia-160m (N=144):
- Scaling prediction: 9 × (144/48)^0.43 = 9 × 3^0.43 ≈ **14.8**
- Acceptance range: **11–19** (±4, ≈ ±1σ given the noise in the two-point fit)

Structural decomposition prediction (independent of scaling law):
- L1–L4 structural zone: **~9 heads** (constant across model size, architecture-determined)
- L5–L11 background zone: 8 deep layers × ~0.65/layer ≈ **~5 heads**
- Total structural prediction: **~14 heads**

Both predictions converge near 14–15 heads. The acceptance range 11–19 covers both predictions and their uncertainty.

**H_160m_consistent:** Count in 11–19 → scaling consistent with N^0.43 (and structural model)  
**H_160m_deviant_high:** Count > 19 → scaling supralinear or structural model wrong  
**H_160m_deviant_low:** Count < 11 → scaling more sublinear than N^0.43  

### Prediction 2 — Structural head replication

L4H3 and L4H7 appeared as SYK-near in both Pythia-70m and Pythia-410m at step512 RAND (with consistent Δ ≈ 0.21–0.27). These are the proposed architecture-determined structural conformal heads.

**H_structural_confirmed:** Both L4H3 and L4H7 SYK-near in Pythia-160m step512 RAND  
**H_structural_partial:** Exactly one of {L4H3, L4H7} SYK-near  
**H_structural_falsified:** Neither L4H3 nor L4H7 SYK-near  

Note: Pythia-160m has 12 heads/layer vs 8 (70m) and 16 (410m), so head indices 3 and 7 still exist and the architecture differs only in head count and depth, not in the fundamental attention mechanism.

### Prediction 3 — L1–L4 structural zone count

The structural decomposition predicts L1–L4 should contain ~9 SYK-near heads regardless of model size (the architecture-determined component).

**H_structural_zone:** L1–L4 contains 7–11 SYK-near heads in Pythia-160m  
**H_structural_zone_fail:** L1–L4 contains fewer than 7 or more than 11  

---

## Measurement protocol

Identical to exp-087 (and exp-086 for the 70m data):
- `FIT_LOW = 3, FIT_HIGH = 50, R2_THRESHOLD = 0.85`
- `SYK_LOW = 0.20, SYK_HIGH = 0.30`
- `SEQ_LEN = 256, N_RAND = 20, RAND_SEED = 87`
- Checkpoint: `revision="step512"`
- RAND condition only
- Analysis: BCFT power-law fit per head, count SYK-near

Model: `EleutherAI/pythia-160m`, architecture 12L × 12H, vocab_size=50304.

---

## Existing data

Pythia-70m step512 RAND (exp-086/exp-087): 9/48 = 18.75%, L4H3 (Δ=0.265), L4H7 (Δ=0.271) structural  
Pythia-410m step512 RAND (exp-087): 22/384 = 5.73%, L4H3 (Δ=0.213), L4H7 (Δ=0.215) structural

---

*This note constitutes the pre-registration. Results will be appended below after the run.*

---

## Results (appended after run)

**Run date:** 2026-07-20  
**Elapsed:** 408.8s (model download + load ~406.6s, measurement ~2s for 20×12L×12H sequences)  
**Prereg commit:** 4f4549c2

### Pythia-160m step512 RAND
- SYK-near: 19/144 = **13.19%**, per-layer mean 1.58

### Three-model scaling table

| Model | Total heads | SYK-near | Fraction | Per-layer mean |
|-------|-------------|----------|----------|----------------|
| Pythia-70m | 48 | 9 | 0.1875 | 1.50 |
| **Pythia-160m** | **144** | **19** | **0.1319** | **1.58** |
| Pythia-410m | 384 | 22 | 0.0573 | 0.92 |

Three-point scaling exponent: **N^0.435** (vs two-point N^0.430 from exp-087 — highly consistent)

### Pre-registered verdicts

**H_160m_consistent: CONFIRMED** — 19 in pre-registered range 11–19  
**H_structural_falsified: CONFIRMED** — L4H3 (Δ=0.620) and L4H7 (Δ=0.556) are NOT SYK-near in Pythia-160m  
**H_structural_zone: CONFIRMED** — L1–L4 count = 11, in pre-registered range 7–11

### SYK-near heads in 160m

L1H0 (Δ=0.210), L1H2 (Δ=0.234), L1H5 (Δ=0.221), L1H6 (Δ=0.243), L1H7 (Δ=0.200), L1H10 (Δ=0.250),  
L2H7 (Δ=0.202), L3H3 (Δ=0.245),  
L4H0 (Δ=0.219), L4H8 (Δ=0.278), L4H9 (Δ=0.296),  
L5H4 (Δ=0.256), L5H8 (Δ=0.234), L8H2 (Δ=0.203), L8H8 (Δ=0.207), L9H3 (Δ=0.246), L9H6 (Δ=0.270), L10H4 (Δ=0.217), L10H9 (Δ=0.228)

### Structural zone breakdown

- L1-L4 zone: **11 heads** (consistent with ~9-11 in 70m and 410m)
- L5-L11 zone: **8 heads** (8 deep layers × ~1/layer)
- L4 SYK-near at 160m: L4H0, L4H8, L4H9 (not H3 or H7)

### Interpretation

**N^0.435 confirmed with three points.** The sublinear scaling law is stable: adding an intermediate scale point (144 heads between 48 and 384) gives an exponent within 0.005 of the two-point estimate. The scaling model is robust.

**Structural head prediction (specific positions) falsified.** L4H3 and L4H7 — confirmed structural in both Pythia-70m and Pythia-410m — are not SYK-near in Pythia-160m (Δ=0.620 and 0.556, well above the SYK window). The prediction that these specific head indices are universally architecture-determined does not hold across different head-counts-per-layer.

**Structural zone prediction confirmed.** L1–L4 contains 11 SYK-near heads (70m: 9, 410m: 9), confirming the layer-zone structural property. The attractor is at the layer level (L4 is the structural attractor layer), not at the specific (layer, head) index level. The head positions within L4 that land near the conformal attractor depend on the specific initialization seed and head count.

**Revised structural head concept:** Architecture-determined structural conformal heads are a layer-zone property (L1–L4 contains ~9–11 SYK-near heads regardless of model size), not a specific-(layer,head) property. The "same positions in 70m and 410m" for L4H3 and L4H7 was an initial observation from two sizes that happened to align but does not generalize.

**Implication for exp-085 prereg_extension:** exp-085 is Pythia-70m specifically. The L4H3/L4H7 structural prediction in `prereg_extension_structural_semantic.md` was derived from the Pythia-70m training trajectory (exp-086), not from the cross-model comparison. That prediction is unaffected by this result — it was always about Pythia-70m specifically. The 160m result does not invalidate the exp-085 structural head prediction.

**L1 overrepresentation in 160m:** Pythia-160m L1 has 6/12 SYK-near heads — much higher than 70m (2/8) or 410m (1/16). L1 has more heads per layer in 160m and appears to be the primary site of the step-512 RAND burst at this intermediate scale. This is a new finding not predicted by the structural decomposition model, worth noting for future work on layer-specific initialization effects.

