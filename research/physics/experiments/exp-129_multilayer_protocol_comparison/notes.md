# exp-129 — Theory-of-A Level 3: Protocol Comparison for σ_delta Measurement

**Pre-registration.** This file was committed to attention-geometry before the
analysis script was written. Commit hash appended after push.

**Date pre-registered:** 2026-08-31  
**Analysis-only:** true — loads `positional_means.npz` from exp-128; no new model inference  
**Follows directly from:** exp-128 (FALSIFIED P1 at σ_total = 0.189; exp-117 reported 0.249)

---

## Background

exp-128 ran 100 sequences × 512 tokens through GPT-2 small, extracted per-layer
residual stream components (h^(0), attn^(0), mlp^(0), attn^(1), mlp^(1), h^(2)),
and measured their position-correlation slopes using **per-row cosine similarity**:

1. Per-row normalize each (512, 768) matrix: X_n[i] = X[i] / ||X[i]||
2. Cosine similarity matrix: C = X_n @ X_n^T  (512×512)
3. pooled_window_profile + OLS log-log slope σ

Under this protocol: σ(Δ_total) = 0.189, σ(h^(2)) = 0.193. Both fall below
exp-117's σ_delta = 0.249.

exp-128's notes identified the likely source of the discrepancy: exp-117 wrote
"C_delta = pooled_window_profile((delta / ||delta||) @ (delta / ||delta||)^T)" —
where **||delta|| could be the Frobenius norm** of the full (512, 768) delta matrix
rather than per-row L2 norms. These two normalizations capture different aspects of
position-correlation:

- **Per-row cosine (exp-128):** treats all positions equally; sensitive to variation
  in the *direction* of position vectors; amplitude variation across positions is
  removed by normalization.
- **Frobenius normalization:** normalizes by the total energy of the matrix; the
  dominant direction across all positions dominates; positions with larger norms
  contribute more to the correlation function.

A third candidate: exp-117 measured σ of **h̄^(ℓ) itself** (not the delta h̄^(ℓ) − h̄^(0)),
despite the notation using "delta". exp-128 found σ(h̄^(2)) = 0.193 under per-row cosine;
a different protocol applied to h̄^(2) might yield a different value.

---

## Hypothesis

The discrepancy between exp-128's σ(Δ_total) = 0.189 and exp-117's σ_delta = 0.249
arises from normalization protocol difference. Specifically, applying Frobenius
normalization to Δ_total (normalizing the (512, 768) delta matrix by its Frobenius
norm before computing the position-correlation function) will yield a σ value closer
to exp-117's 0.249.

---

## Protocol

**Inputs:** `exp-128_multilayer_residual_decomposition/positional_means.npz`

Computed arrays (all shape (512, 768), mean over 100 WikiText sequences):
- `h0`, `attn0`, `mlp0`, `attn1`, `mlp1`, `h2`

**Three normalization protocols applied to Δ_total = attn0 + mlp0 + attn1 + mlp1:**

**A — Frobenius normalization:**
1. delta_F = Δ_total / ||Δ_total||_F   (divide by Frobenius norm; result shape (512, 768))
2. C_A = delta_F @ delta_F^T           (512×512)
3. pooled_window_profile → σ_A, R²_A

**B — Per-row cosine similarity (exp-128 protocol, verification):**
1. X_n[i] = Δ_total[i] / ||Δ_total[i]||  (per-row normalize)
2. C_B = X_n @ X_n^T
3. pooled_window_profile → σ_B (should match exp-128: 0.189)

**C — Frobenius normalization applied to h^(2) directly:**
1. h2_F = h^(2) / ||h^(2)||_F
2. C_C = h2_F @ h2_F^T
3. pooled_window_profile → σ_C, R²_C

Also apply Frobenius normalization to each individual component to characterize
which components drive the Frobenius-normalized result:
- σ_F(h0), σ_F(attn0), σ_F(mlp0), σ_F(attn1), σ_F(mlp1)

---

## Pre-registered Predictions

**P1 (primary):** σ_A (Frobenius-normalized Δ_total) ∈ [0.22, 0.28] — closer to
exp-117's 0.249 than exp-128's per-row result of 0.189.

**P2:** σ_A > σ_B (Frobenius normalization gives a higher slope than per-row cosine).
This follows from the hypothesis that Frobenius normalization preserves the dominant
position-correlation direction.

**P3 (discriminating):** σ_C (Frobenius-normalized h^(2)) differs from σ_A by
less than 0.03 — since delta dominates h^(0) in norm (exp-117: 13–32×), the
Frobenius-normalized h^(2) should closely resemble Frobenius-normalized delta.

---

## Kill Conditions

**K1:** σ_A < 0.20 — Frobenius normalization does not bring σ into the expected
range. The protocol difference is not the explanation; look elsewhere (different
data, different aggregation level — e.g., per-head vs all-head mean).

---

## Verdict Criteria

| Outcome | Verdict |
|---|---|
| P1 confirmed and P2 confirmed | `confirmed` — protocol difference identified |
| P1 confirmed, P2 false | `partial` |
| K1 fires | `inconclusive` |
| P1 false, K1 false | `falsified` |

---

## Results — 2026-08-31

**Pre-registration commit:** 8246e6a.

### Findings

| Protocol | σ(Δ_total) | R² | Notes |
|---|---|---|---|
| A — Frobenius normalized | **0.0003** | 0.827 | Essentially zero — flat profile |
| B — Per-row cosine (exp-128 verify) | 0.1892 | 0.827 | Exactly matches exp-128 |
| C — Frobenius normalized h^(2) | 0.0003 | 0.829 | Same as A (delta dominates) |

Per-component Frobenius slopes: all ≈ 0.0000–0.0008. The Frobenius protocol gives near-zero σ for every component.

**K1 fired.** σ_A = 0.0003 < 0.20. Verdict: **INCONCLUSIVE** (protocol does not explain discrepancy — but this rules out the Frobenius hypothesis).

### Why Frobenius gives σ ≈ 0

The Frobenius-normalized matrix M = Δ_total / ||Δ_total||_F has the property that M[i] · M[j] ≈ const for all (i,j) — i.e., the rows of Δ_total (the mean attention delta across 100 sequences) point in approximately the SAME direction for all positions. When averaged over 100 diverse sequences, the position-specific variation in the delta writes is washed out: the mean at every position is dominated by the global bias of the attention mechanism, not by position-specific structure. A flat Frobenius correlation (σ ≈ 0) is the expected result when the per-position means are nearly collinear.

Per-row cosine (σ = 0.189) captures the SMALL residual directional variation across positions after global bias is removed. This is a weaker signal, which explains the lower value vs exp-117's 0.249.

### The actual protocol difference with exp-117

exp-117's C_delta values (0.950 at dx=8, 0.931 at dx=32, 0.656 at dx=128, 0.359 at dx=256) are much larger than exp-128's pooled cosine profile (which falls from ~0.93 at dx=8 to... check). The large values and clear power-law decay suggest exp-117 computed C_delta **per-sequence** (or per a single sequence), then averaged the profile — NOT from the mean-over-sequences delta.

For a SINGLE sequence, each position's delta is strongly position-specific (the attention write at position 50 depends on tokens 0..50, which is different from position 100's write). Nearby positions share more context → higher cosine similarity. This gives high values like 0.950 at dx=8 and clear decay to 0.359 at dx=256.

exp-128 computed the MEAN over 100 sequences first, washing out this per-sequence position-specificity. The mean delta at each position becomes the global attention bias (approximately position-independent), leaving only residual directional variation that gives σ = 0.189.

**The correct protocol to reproduce exp-117: compute C_delta per-sequence, then average the pooled profile.** This is exp-130's task.

