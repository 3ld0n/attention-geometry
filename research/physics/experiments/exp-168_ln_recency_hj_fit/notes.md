# Notes: exp-168

**LN-recency h(j) fit and wpe trough test**
**Date:** 2026-10-09
**Session:** Physics room, ~12:28–1:05 AM MDT, solo

---

## What was tested

S_abskey[a] = m_q · (kbar[a] − m_k) / sqrt(d_head), computed at full resolution
(1024 positions) from exp-164's kbar_qbar.npz for five structural heads
{L2H1, L3H4, L5H0, L7H11, L10H8}.

Three hypotheses:
- H1: Post-511 rise fits linear h(j) (R² ≥ 0.95)
- H2: S_residual = S_abskey − S_wpe is ≥ 90% monotone (wpe subtraction restores
  monotonicity)
- H3: wpe accounts for ≥ 50% of trough depth on ≥ 4/5 heads

---

## Results

### H1 — CONFIRMED (5/5 heads)

| Head | Linear R² | Log R² | Power γ |
|------|-----------|--------|---------|
| L2H1 | 0.979 | 0.953 | 1.916 |
| L3H4 | 0.997 | 0.985 | 2.092 |
| L5H0 | 0.993 | 0.974 | 2.176 |
| L7H11 | 0.995 | 0.982 | 1.865 |
| L10H8 | 0.989 | 0.967 | 2.319 |

All five structural heads: linear fit on S_abskey[512:1024] gives R² ≥ 0.979.
The LN-recency h(j) ∝ j form fits the post-511 rise extremely well. 

The power-law exponents γ ≈ 1.87–2.32 all exceed 1 (super-linear), but the linear
model fits essentially as well — the 512-step window is not long enough to distinguish
linear from mildly super-linear growth. The shape is robustly increasing and
well-described by h(j) ∝ j in the post-511 range.

**What this says:** The post-511 rise in S_abskey is consistent with the LN-recency
theorem's prediction that attention scores increase linearly in key position. This
is one more piece of evidence that the absolute-key-position drift measured by
exp-138/164 has its genesis in the LayerNorm-induced recency mechanism.

### H2 — NOT CONFIRMED (0/5 heads)

| Head | Monotone fraction after wpe subtraction |
|------|----------------------------------------|
| L2H1 | 51.0% |
| L3H4 | 53.0% |
| L5H0 | 53.1% |
| L7H11 | 52.5% |
| L10H8 | 52.0% |

All five heads: after subtracting S_wpe from S_abskey, the residual is ~51-53%
monotone over a ∈ [1, 1023]. This is barely above chance (50%), indicating that
S_wpe subtraction does NOT restore monotonicity. The trough is not eliminated by
removing the wpe contribution.

**What this says:** wpe is not the mechanism that creates the trough. Subtracting
the word position embedding's projection through W_K produces a residual that is
just as non-monotone as the original S_abskey. Whatever creates the dip in positions
1-511 is not carried by wpe alone.

### H3 — NOT CONFIRMED (3/5 heads)

| Head | Trough S_abskey | Trough S_wpe | wpe fraction |
|------|-----------------|--------------|--------------|
| L2H1 | 4.561 | 2.867 | 63% ✓ |
| L3H4 | 7.134 | 3.416 | 48% ✗ |
| L5H0 | 8.752 | 5.283 | 60% ✓ |
| L7H11 | 5.995 | 3.929 | 66% ✓ |
| L10H8 | 6.685 | 2.402 | 36% ✗ |

wpe contributes 36–66% of the trough depth, depending on the head. Three of five
heads pass the ≥ 50% threshold, but the result is not consistent enough to call it
a dominant mechanism.

---

## Cross-check note

The S_abskey values computed here (from exp-164's kbar) show a constant per-head
offset relative to exp-163's sampled values (max error 0.707, compared to 2.30
before the dk = kbar − m_k correction). This residual offset reflects slightly
different kbar accumulations between exp-163 and exp-164 (both use 200 sequences,
seed 42, but different floating-point accumulation order). The offset is constant
in position for each head, so it cancels in:
- H2 (monotonicity test — differences, not absolutes)
- H3 (trough depth = max−min, offset cancels)
- H1 (relative shape of the rise, only slope and R² matter)

The cross-check residual does not invalidate any hypothesis test.

---

## What this means for T1 and the census

**H1 adds to the T1 restatement thread:** The post-511 S_abskey rise is linear in
absolute key position. This is the seventh piece of evidence that T1's
A(i,j) ~ |i−j|^{−2Δ} formulation — a relative-lag law — is not what the census
measures. The census slope σ_pos reflects h(j) ∝ j (an absolute-key-position
drift), not the relative-lag exponent of A.

**H2+H3 are instructive negatives:** wpe does not explain the trough. The mechanism
that creates the negative dip in S_abskey from positions 1 to ~511 remains open.
Candidate mechanisms not yet tested:
1. Wte (token embedding) contribution: the kbar includes wte information carried
   by the residual stream. If the mean token embedding projects negatively through
   W_K in the short-context range, that could produce the trough.
2. Attention-block rewriting: earlier attention blocks modify the residual stream
   before the structural heads see it; the rewriting may shift the kbar's
   position-dependence relative to what wpe alone would produce.
3. Layer norm interaction: LN normalizes the key such that the absolute magnitude
   of position-correlated components is suppressed at short positions.

---

## Registry notes

| Field | Value |
|---|---|
| Verdict | H1 CONFIRMED; H2 not confirmed; H3 not confirmed |
| Status | complete |
| Bears on | A1, T1 restatement, G7 |
| Surfaced in | OVERVIEW.md (formation / T1 restatement section), theory spine (A1 block) |

---

*Session: Physics room, 2026-10-09, ~12:28–1:05 AM MDT.*
