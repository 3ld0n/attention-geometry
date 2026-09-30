# exp-162 — Pre-registration: Pool-stable head mechanism — absolute-key profile curvature

**Date:** 2026-09-30 (02:45 UTC, 2026-09-29 20:45 MDT)
**Ariel — Physics room, Mission Valley. Autumn evening.**
**Analysis-only:** Yes — from exp-112's saved `scores_gpt2.npz`; no new forward passes.

---

## Background

exp-161 (census query-pool test, 2026-09-25) found two behavioral classes within the 5 structural Δ-window heads of GPT-2 small:

- **Pool-sensitive** (σ increases across B→C→D pools): L2H1 (Δσ=0.047, borderline), L5H0 (Δσ=0.052), L10H8 (Δσ=0.064)
- **Pool-stable** (σ near-constant): L3H4 (Δσ=0.014), L7H11 (Δσ=0.015)

exp-138 (bilinear decomposition, 2026-09-09) showed that for ALL 5 structural heads, the census slope σ_pos is carried almost entirely by the absolute-key-position term S_abskey[a] = m_q · dk[a] / √d_head (σ_abskey/σ_full ≈ 1.07–1.17 across all heads; the relative-lag term is negative for all).

If the census slope is driven by absolute-key drift, pool-sensitivity should depend on how *uniform* the log-log slope of S_abskey[a] is across different key-position sub-ranges. A head where S_abskey[a] ~ a^(-β) with a constant β would show the same apparent σ in any query pool (because the slope of the function doesn't change with key position). A head where β changes with key position — where the profile has *curvature* in log-log space — would show different σ in different pools: later pools sample higher key positions, and if the slope steepens there, σ increases.

---

## Data source

`research/physics/experiments/exp-112_score_drift_decomposition/scores_gpt2.npz`

Contains `qbar_random_L{l}` and `kbar_random_L{l}`: empirically measured position-mean query and key vectors (per head, per position) from random-token forward passes, SEQ_LEN=512. Available positions: 0–511.

From these, S_abskey[a] can be reconstructed:
- m_q = qbar[h].mean(axis=0) — (64,) mean query direction
- m_k = kbar[h].mean(axis=0)
- dk[a] = kbar[h, a] - m_k
- S_abskey[a] = m_q @ dk[a] / sqrt(64), for a ∈ [0, 511]

---

## Hypotheses

### H1 (pool-stable = lower slope curvature)

Pool-stable heads (L3H4, L7H11) have **lower slope curvature** of their S_abskey[a] profile across key positions than pool-sensitive heads (L2H1, L5H0, L10H8).

**Operationalization:** Divide key positions [0, 511] into two halves [0, 255] and [256, 511]. For each structural head, compute:

1. σ_half1 = log-log OLS slope of S_abskey[a] for a ∈ [8, 255] (positive-domain only)
2. σ_half2 = log-log OLS slope of S_abskey[a] for a ∈ [256, 511] (second half)
3. |Δσ_half| = |σ_half1 - σ_half2| — the between-half slope difference

H1 prediction: mean(|Δσ_half|) for {L3H4, L7H11} < mean(|Δσ_half|) for {L2H1, L5H0, L10H8}.

**Secondary metric:** Spearman correlation between |Δσ_half| (from existing data) and Δσ_exp161 (pool-sensitivity from exp-161). Predicted: ρ > 0.5 (monotone relationship).

Exp-161 ordered (by Δσ, ascending): L3H4 (0.014), L7H11 (0.015), L2H1 (0.047), L5H0 (0.052), L10H8 (0.064). If H1 holds, |Δσ_half| should rank approximately L3H4 ≈ L7H11 < L2H1 < L5H0 ≈ L10H8.

### H2 (profile shape characterization)

As a characterization (no kill condition), for each structural head: what is the sign and magnitude of S_abskey[a] across the full [0, 511] range? Does it have a clean power-law decay on the positive domain? Where does it cross zero?

---

## Kill conditions

**K1 (mechanism falsified):** mean |Δσ_half| for pool-stable ≥ mean |Δσ_half| for pool-sensitive. The between-half slope curvature does not distinguish pool-stable from pool-sensitive; the mechanism proposed here is wrong.

**K2 (no Spearman signal):** ρ(|Δσ_half|, Δσ_exp161) ≤ 0. Negative or zero correlation: curvature does not predict pool-sensitivity direction.

---

## Register discipline notes

- This is a registered analysis-only experiment. The hypothesis was formed from inspection of exp-138 and exp-161 data (both already complete), but the specific quantities |Δσ_half| and their correlation with Δσ_exp161 have not been computed. The ranking is a genuine pre-registered prediction, not a post-hoc observation.
- `analysis_only: true` — no new inference required.
- The exp-112 npz must be used as-is; I must not rerun exp-112 with different parameters to fish for a cleaner result.
- The two-half split (at position 256) is the pre-registered design; sub-window boundaries cannot be adjusted after seeing the data.

---

## What a positive result would mean for T1

If H1 is confirmed: pool-stable heads have a more uniform absolute-key decay profile. This means their σ_pos is measuring a genuine property of the function S_abskey(j) rather than a slope that changes with the sampled range. Even though it's still absolute-key drift (not a relative-lag law as T1 originally claimed), the two pool-stable heads are measuring something more stable — a local approximation to a power law holds across their key-position range.

This sharpens the T1 restatement: the census slope measures absolute-key drift for all structural heads, but the *quality* of that measurement varies — pool-stable heads are those where the drift has a near-uniform exponent.
