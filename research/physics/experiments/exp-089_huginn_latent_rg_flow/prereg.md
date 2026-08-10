# exp-089 Pre-Registration: Latent-Iteration RG Flow in Huginn

**Committed before any model download or results.**
**Hypothesis: inference-time recurrence in a trained depth-recurrent transformer produces RG flow toward the conformal fixed point (Δ → 0.25).**

---

## Motivation

exp-086 confirmed that training steps operationalize conformal depth: Spearman ρ(training_step, SYK_near_fraction) = 0.862 for RAND condition on Pythia-70m. Training time flows toward the conformal fixed point.

Huginn-0125 (arXiv:2502.05171, Geiping et al.) is a depth-recurrent transformer where the same core_block runs r times at inference (test-time compute scaling). Recurrence count r is a *continuously adjustable depth variable on a fixed, trained model* — no confound from different model sizes or training regimes.

If exp-086's training-time finding reflects a genuine RG flow (not just a correlation artifact from model scale), then inference-time recurrence on a trained model should produce the same convergence: as r increases, attention statistics in the recurrent block should flow toward the conformal fixed point.

Inbox item: [from: cursor, 2026-07-20] latent-iteration RG flow as test-time knob.

---

## Model

- **Model:** `tomg-group-umd/huginn-0125` (Huginn-0125)
- **Parameters:** 3.5B
- **Architecture:**
  - Prelude: 2 SandwichBlocks (run once per forward pass)
  - Core block: 4 SandwichBlocks (run r times, shared weights)
  - Coda: 2 SandwichBlocks (run once)
- **Attention:** CausalSelfAttention — 55 heads, head_dim=96, n_embd=5280, standard MHA (no GQA), RoPE (rope_base=50000)
- **block_idx encoding:** at recurrence step i (0-indexed), core_block layers receive block_idx = 2+4i, 3+4i, 4+4i, 5+4i
- **Decode formula:** for block_idx ≥ 2: step = (block_idx - 2) // 4, layer = (block_idx - 2) % 4

---

## Protocol

### Input
- N=20 random token sequences (uniform samples from vocab, no special tokens)
- Sequence length S=128 tokens
- Batch: single batch of 20 sequences
- Device: MPS (bfloat16 model weights)

### Attention capture
For each (step i, layer l) in the recurrent block:
1. Intercept CausalSelfAttention.forward after RoPE application
2. Compute causal attention weights manually: A = softmax(Q·K^T / sqrt(d_k) + causal_mask)
3. Compute mean distance-decay profile: mean_A(d) = mean over all (q,k) pairs with q-k=d and q≥cutoff_low

### Fitting (per head h)
Following exp-007/exp-049 protocol:
- Log-log OLS: log(mean_A(d)) = -2Δ·log(d) + C
- Distance range: d ∈ {cutoff_low, ..., max_dist} where cutoff_low=3, max_dist=32
- Minimum valid distances: 15 (requiring S≥18 for at least 1 query per distance)
- R² threshold: 0.90 → "conformal head"
- SYK-near criterion: Δ ∈ [0.20, 0.30] AND R² > 0.90

### Steps measured
Both conditions run with num_steps=64. Attention is captured at every step 0 through 63.
Report SYK_near fraction at steps: {1, 2, 4, 8, 16, 32, 64} (1-indexed: first, second, fourth, etc. recurrence application).

### Randomized condition
Reinitialize core_block weights only (all 4 SandwichBlocks) using the same init scheme as the pretrained model (following _init_weights in RavenPreTrainedModel). Prelude and coda retain trained weights. Same N=20 random sequences, num_steps=64.

---

## Registered Hypotheses

### H_rg_flow (primary)
In the TRAINED model: Spearman ρ(step, SYK_near_fraction) > 0, p < 0.05

This is the monotone convergence claim. The fraction of SYK-near heads should increase (or be non-decreasing) as recurrence step count increases.

**KEEP** verdict: ρ > 0 AND p < 0.05
**KILL** verdict: ρ ≤ 0 OR p ≥ 0.05

### H_rand_flat (secondary)
In the RANDOM model: Spearman ρ(step, SYK_near_fraction) ≤ 0 OR p ≥ 0.05

If the conformal convergence requires trained weights, randomized core_block should show no monotone trend.

**KEEP** verdict: ρ ≤ 0 OR p ≥ 0.05 in random condition
**FAIL** verdict: ρ > 0 AND p < 0.05 in random condition (would suggest architectural, not learned, RG flow)

### H_comparison (secondary)
At step 64: trained SYK_near_fraction > random SYK_near_fraction

**KEEP** verdict: trained > random at step 64
**FAIL** verdict: trained ≤ random at step 64

### H_initial (exploratory, not primary)
At step 1: trained SYK_near_fraction > random SYK_near_fraction

Tests whether any structured conformal signal exists from the first recurrence application.

---

## Decision Rules

- If H_rg_flow KEEP and H_rand_flat KEEP and H_comparison KEEP: **CONFIRMED — inference-time RG flow**
- If H_rg_flow KEEP but H_comparison FAIL: **PARTIAL — monotone but random catches up**
- If H_rg_flow KILL: **FLATLINE — no inference-time RG flow at this scale**
- If H_rand_flat FAIL: **ARCHITECTURAL — random weights also flow (structure, not learning)**

---

## Connection to Prior Work

- exp-086: Training steps operationalize conformal depth (ρ=0.862 for RAND, ρ=0.728 for NAT). This experiment tests the inference-time analogue.
- exp-088: Structural conformal heads are a layer-zone property (L1-L4 zone, N^0.435 scaling). The Huginn architecture has 4 core_block layers per step — the layer-zone concept maps onto this.
- exp-087/088: RAND early burst (9-19 SYK-near at step 512 RAND) reflects structural conformal attractors in random weights. Relevant to H_rand_flat interpretation.
- BCFT protocol: Following exp-007, exp-025, exp-049 measurement protocol (log-log OLS, R²>0.90, Δ∈[0.20,0.30]).

---

## Commit timestamp
This document committed to git before any model download. Hex: see git log.
