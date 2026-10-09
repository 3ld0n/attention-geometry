# Pre-registration: exp-168

**Title:** LN-recency h(j) fit — does the post-511 S_abskey rise match the
LayerNorm-recency factorized form, and is the trough a subtractable wpe
contribution?

**Date:** 2026-10-09 (pre-registered before any analysis code or array inspection
beyond the exp-163 results.json summary values already in the log)

**Experiment author:** Ariel (solo)

**Type:** Analysis-only — uses exp-163's kbar_qbar.npz and GPT-2 small model
weights. No new training; no new forward passes on text.

---

## Background and motivation

exp-163 (pool-stable mechanism, SEQ_LEN=1024) found a structural feature in the
S_abskey[a] profile of GPT-2 small's five structural heads: a zero crossing near
position ~511, with a long trough (positions ~1 to ~511, predominantly negative)
and a monotone rise from ~511 to 1023. The S_abskey profile was sampled at 8
positions (a = 0, 127, 255, 383, 511, 639, 767, 895, 1023) in the results; the
full profile at every position a ∈ [0, 1023] lives in kbar_qbar.npz.

The *LayerNorm Induces Recency Bias in Transformer Decoders* paper (Findings ACL
2026) proves that for stacked causal attention with LayerNorm, S_ij = g(i)·h(j),
where h(j) is strictly increasing in key position j. Call this the LN-recency
theorem. If h(j) governs the post-511 rise in S_abskey, the rise should fit the
theorem's predicted form; and the trough (positions 1–511) would then require an
explanation — the most natural candidate is the word position embedding (wpe)
contribution to kbar.

This experiment tests:
1. Whether the post-511 S_abskey rise is well-described by the LN-recency h(j)
   form.
2. Whether subtracting the wpe component of S_abskey eliminates (or substantially
   reduces) the trough, leaving a monotonically increasing residual consistent with
   the theorem.

---

## Data sources

- **kbar_qbar.npz:** `research/physics/experiments/exp-163_pool_stable_kbar_1024/kbar_qbar.npz`
  Contains `kbar[head_name]` (shape [SEQ_LEN, d_head]) and `qbar[head_name]`
  (the mean query vector, shape [d_head]) for each of the five structural heads
  {L2H1, L3H4, L5H0, L7H11, L10H8}. SEQ_LEN=1024.

- **GPT-2 small weights (cached):** Accessed via HuggingFace `gpt2` (cached at
  `~/.cache/huggingface/hub/`). Specifically:
  - `wte`: token embedding matrix [vocab_size, d_model]
  - `wpe`: position embedding matrix [max_len, d_model]
  - Per-layer `W_K` projection (attn.c_attn weight slice [d_model, d_head], one
    per head)

  The wpe contribution to S_abskey at key position j and head h is:
  `S_wpe[h, j] = qbar[h] · (W_K[h]^T wpe[j]) / sqrt(d_head)`
  where W_K[h] is the key projection matrix for head h (d_model → d_head).

---

## Hypotheses

**H1 (LN-recency fit):** The post-511 S_abskey[a] rise (a ∈ [512, 1023]) is
well-fit by a linear function of a:
  `S_abskey[a] ≈ α·a + β`
with R² ≥ 0.95 on all five structural heads.

*Rationale:* The LN-recency theorem predicts h(j) increases with j; the simplest
parametric form consistent with the causal-mask structure is linear growth. If the
rise were logarithmic or sub-linear, that would suggest a different mechanism. We
test linear first; if R² < 0.95, we fit log(a) and power-law a^γ as fallbacks.

**H2 (wpe subtraction eliminates trough):** After subtracting S_wpe[h, a] from
S_abskey[h, a], the residual S_residual[h, a] = S_abskey[h, a] − S_wpe[h, a] is
monotonically increasing over a ∈ [1, 1023] on at least 4/5 structural heads (≥
4/5 passing a sign test: at least 90% of consecutive differences S_residual[a+1]
− S_residual[a] are non-negative over a ∈ [1, 1022]).

*Rationale:* If wpe is the trough mechanism, removing its contribution should
reveal the underlying increasing h(j) without the dip.

**H3 (wpe dominates trough magnitude):** The peak-to-trough depth in S_wpe[h, a]
accounts for at least 50% of the peak-to-trough depth in S_abskey[h, a] (measured
as max(S_abskey[a=0, a∈[1,511] min]) on all five structural heads.

*Rationale:* A quantitative test of whether wpe is the dominant source of the trough.

---

## Kill conditions

**K1:** GPT-2 small's W_K weight extraction fails or produces incorrect shapes
(d_model != 768 or d_head != 64). If K1 fires, report the shape mismatch and
halt.

**K2:** kbar_qbar.npz does not contain all five structural heads with the expected
shape [1024, 64]. Report what is present.

**K3:** The post-511 linear fit (H1) has R² ≥ 0.95 for fewer than 3/5 heads.
This would indicate the LN-recency theorem's h(j) form is not the dominant mechanism
in this region; report fallback fits (log and power-law) for those heads.

---

## Analysis plan

1. Load kbar_qbar.npz from exp-163. Verify shapes.
2. Reconstruct S_abskey[h, a] at full resolution: for each head h and position a,
   `S_abskey[h, a] = qbar[h] · kbar[h, a] / sqrt(64)`.
3. Load GPT-2 small model weights. Extract wpe [1024, 768] and W_K[h] [768, 64]
   for each structural head.
4. Compute S_wpe[h, a] = qbar[h] · (W_K[h]^T wpe[a]) / sqrt(64) for each h, a.
5. **H1 test:** Fit linear regression on post-511 range (a ∈ [512, 1023]) of
   S_abskey[h, a]. Record R², slope α, intercept β.
   Fallback if R² < 0.95: fit log(a) and power-law; record best-fit exponent.
6. **H2 test:** Compute S_residual[h, a] = S_abskey[h, a] − S_wpe[h, a] for
   a ∈ [1, 1023]. Test monotonicity: count consecutive differences ≥ 0.
   Pass criterion: ≥ 90% of differences non-negative over a ∈ [1, 1022].
7. **H3 test:** For each head, compute:
   - trough_abskey = S_abskey[h, 0] − min(S_abskey[h, a], a ∈ [1, 511])
   - trough_wpe = S_wpe[h, 0] − min(S_wpe[h, a], a ∈ [1, 511])
   - Fraction = trough_wpe / trough_abskey. Pass if ≥ 50%.
8. Save all outputs to results.json. Write notes.md.

---

## Verdict table

| Result | Verdict |
|---|---|
| H1 confirmed (R²≥0.95 on ≥3/5 heads), H2 confirmed (≥4/5 monotone), H3 confirmed (≥50% on ≥4/5) | LN-recency mechanism supported; wpe is the trough source |
| H1 confirmed, H2 or H3 not confirmed | LN-recency h(j) fits the rise but wpe is not the trough mechanism; alternative needed |
| H1 not confirmed (K3 fires) | LN-recency form does not describe the post-511 rise; report fallback fits |
| H1 confirmed, H2 confirmed, H3 not confirmed | wpe shapes trough but does not dominate it |

---

## Bears on

- **A1 (census exponent):** If the post-511 rise is linear in a, then S_abskey[a]
  has a linear-in-a component in the support region that is what the pool-slope
  analysis measures. This would connect the pool-slope (which is an OLS slope of
  S_abskey on log(a)) to the LN-recency mechanism.
- **T1 restatement:** One more constraint on what "the census measures." If the
  rise is driven by h(j) ∝ j rather than a conformal lag kernel, the census slope
  σ_pos has a different generator than A(i,j) ~ |i-j|^{-2Δ}.
- **G7 (why 1/4):** If wpe is the trough source, wpe's positional structure is
  part of the picture of what sets the census number.

---

*Pre-registration written 2026-10-09, ~12:28 AM MDT, physics room, Cursor, solo.
Committed to attention-geometry before analysis script is written or any array
inspection beyond the eight sampled values in exp-163/results.json.*

---

## Amendment A-1 (pre-run, 2026-10-09)

**Filed before run.py was executed.**

The pre-registration cited the data source as
`exp-163_pool_stable_kbar_1024/kbar_qbar.npz`. This was incorrect: exp-163 saved
only `results.json` (sampled S_abskey values at 8 positions). The full-resolution
kbar/qbar arrays at SEQ_LEN=1024 were saved by exp-164 (`exp-164_l3h4_bilinear_decomposition/kbar_qbar.npz`), which ran immediately after exp-163 using the same
structural head population and the same 200-sequence, seed-42 protocol.

**Correction:** Data source is `exp-164_l3h4_bilinear_decomposition/kbar_qbar.npz`.
All hypotheses and analysis plan are unchanged. The cross-check values cited in the
Analysis plan (eight sampled S_abskey values per head) come from exp-163/results.json
and remain the correct ground truth for the cross-check step.

Additionally: exp-164's npz stores `qbar[h]` as shape (1024, 64) — the per-position
mean query vector averaged over sequences, not a single mean vector. The mean query
m_q used in S_abskey computation is `qbar[h][256:1024].mean(axis=0)`, matching the
`MQ_POS_START=256, MQ_POS_END=1024` protocol documented in exp-164/run.py. This
does not change any hypothesis; it is a protocol clarification about what the npz
stores.

No new hypotheses are introduced by this amendment.
