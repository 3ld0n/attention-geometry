# exp-131 — Theory-of-A Level 3: Multi-Layer Decomposition Under Random-Token Census

**Pre-registration.** This file was committed before the analysis script was written.
Commit hash appended after push.

**Date pre-registered:** 2026-08-31  
**Analysis type:** Forward passes (50 sequences × 512 tokens, random-token census)  
**Follows from:** exp-130 (INCONCLUSIVE — WikiText per-sequence σ=0.129; exp-117's 0.249 not reproduced)

---

## Background

exp-128 (WikiText, mean-first, σ=0.189), exp-129 (Frobenius, σ≈0), and exp-130
(WikiText, per-sequence, σ=0.129) all failed to reproduce exp-117's σ_delta = 0.249
(C[8]=0.950, C[256]=0.359, R²=0.824).

The conformal geometry — the census exponent Δ ≈ 0.249 — is defined and measured in
the **random-token census regime**: 50 (or 100) sequences of 512 uniformly-random
tokens from the vocabulary. All structural head measurements (exp-107 through exp-127)
use this input distribution.

exp-117's C_delta was almost certainly computed under the random-token census, not
WikiText. The random-token protocol suppresses induction heads and semantic heads
(which fire on coherent text), isolating the positional conformal structure. Under
random tokens, the attention write's position-correlation is driven purely by the
positional-mean score profile (S_mf, the quantity measured in exp-112/113) — exactly
the conformal structure we're trying to explain.

This experiment runs the same multi-layer decomposition (exp-128) but with random-token
census inputs (50 sequences, seed 42, same as exp-107).

---

## Hypothesis

Under the random-token census protocol, the accumulated attention delta at L2H1's
input has position-correlation slope σ_delta ≈ 0.249 (reproducing exp-117). The
per-layer decomposition under random tokens will show that attn0 writes drive the
structure, and the cumulative slope builds monotonically through the layers.

---

## Protocol

**Model:** GPT-2 small, fp32, eager attention.

**Data:** 50 sequences of 512 uniformly-random tokens in [0, 50256), seed=42.
Same protocol as exp-107 (the founding census experiment).

**Residual stream extraction:** Same hooks as exp-128 — capture h^(0), attn^(0),
mlp^(0), attn^(1), mlp^(1) per sequence. No h^(2) hook needed (we don't compare to
h^(2) here; the delta is reconstructed directly).

**σ measurement protocol:** Per-sequence cosine similarity, then averaged profile
(same as exp-130). Also compute mean-first protocol (same as exp-128) for comparison.

---

## Pre-registered Predictions

**P1 (primary — reproduce exp-117):**
σ(Δ_total, per-seq, random) ∈ [0.22, 0.28] AND C[8] ≥ 0.90.

**P2:** σ(Δ_total, random) > σ(Δ_total, WikiText-per-seq) = 0.129.
Random-token protocol gives higher σ than WikiText.

**P3:** σ(attn0, random) > σ(attn0, WikiText-per-seq) = 0.183.
Layer-0 attention write more position-correlated under random tokens.

**P4:** σ(Δ_total, per-seq, random) > σ(Δ after block0, per-seq, random).
The cumulative slope grows with depth (block 1 adds to the structure, not subtracts).
This was NOT observed in WikiText (block 1 reduced the slope).

---

## Kill Conditions

**K1:** σ(Δ_total, per-seq, random) < 0.20 AND C[8] < 0.85 — random-token protocol
also fails to reproduce exp-117; exp-117's measurement had a fundamentally different
input or aggregation.

---

## Verdict Criteria

| Outcome | Verdict |
|---|---|
| P1 and P2 confirmed | `confirmed` — random-token is the correct regime for exp-117 |
| P2 confirmed, P1 not (σ > 0.129 but < 0.22) | `partial` |
| K1 fires | `inconclusive` |
| P2 false | `falsified` |
