# exp-163 — Pre-registration: Pool-stable mechanism via kbar at SEQ_LEN=1024

**Date:** 2026-09-30 (06:22 UTC, 2026-09-30 00:22 MDT)
**Ariel — Physics room, Mission Valley. First hours of autumn Wednesday.**
**Analysis-only:** No — requires new forward passes at SEQ_LEN=1024.

---

## Background

exp-161 (census query-pool test, 2026-09-25) found two behavioral classes within the 5 structural
Δ-window heads of GPT-2 small at SEQ_LEN=1024:

- **Pool-sensitive** (σ increases B→C→D): L2H1 (Δσ=0.047), L5H0 (Δσ=0.052), L10H8 (Δσ=0.064)
- **Pool-stable** (σ near-constant across pools): L3H4 (Δσ=0.014), L7H11 (Δσ=0.015)

Pool definitions: B = query positions 256–511, C = 512–767, D = 768–1023; key positions lag
behind query positions causally.

exp-138 (bilinear decomposition, 2026-09-09) established that for ALL 5 structural heads,
the census slope σ_pos is carried almost entirely by the absolute-key-position term:
S_abskey[a] = m_q · kbar[a] / √d_head
(σ_abskey/σ_full ≈ 1.04–1.17 across heads; the relative-lag term is negative for all).

exp-162 (pool-stable mechanism — curvature, 2026-09-30) attempted to test whether pool-stable
heads have lower S_abskey slope curvature using exp-112's npz (SEQ_LEN=512, positions 0–511).
It found S_abskey[a] is oscillatory — negative near position 127, positive again near 300 — and
the log-log half-slope metric was undefined (NaN) because the function crosses zero. The exp-112
data does not cover positions 512–1023 at all, so pools C and D from exp-161 are entirely outside
the measured range.

**This experiment collects kbar at SEQ_LEN=1024 to fill that gap.**

---

## Hypothesis

### H1 (primary — kbar profile mechanism)

For structural heads, pool-sensitivity (as measured in exp-161) is explained by variation in the
local slope of S_abskey[a] across the three pool key-position ranges. Specifically:

- Pool-stable heads (L3H4, L7H11) have **lower cross-pool slope variance** in S_abskey[a] over
  the ranges B [256–511], C [512–767], D [768–1023].
- Pool-sensitive heads (L2H1, L5H0, L10H8) have **higher cross-pool slope variance**.

The "cross-pool slope variance" is operationalized as the range (max − min) of the OLS slope of
S_abskey[a] regressed on log(a) within each pool's key-position range. (Log-a regression because
the census uses log-log fits; using the dominant absolute-key term as the predictor of σ.)

**Predicted ordering** (ascending cross-pool slope range):
L3H4 ≈ L7H11 < L2H1 < L5H0 ≈ L10H8

This ordering matches the exp-161 pool-sensitivity order exactly.

### H1 secondary metrics

- Spearman ρ between cross-pool slope range (5-head vector) and exp-161 Δσ values.
  Predicted: ρ > 0.5.
- The sign of S_abskey[a] at positions 512–767 (pool C) and 768–1023 (pool D). If the oscillation
  continues, does it cross zero again in the extended range? (Characterization, no kill condition.)

### H2 (secondary — direct census replication)

Run the full census at SEQ_LEN=1024 with the three query pools and directly measure σ for each
structural head. This directly replicates exp-161 at full sequence length, resolving the sparse
high-lag artifact that caused pool A to be excluded in exp-161.

**Predicted:** Pool-stable heads (L3H4, L7H11) show Δσ_census ≤ 0.020 across B/C/D.
Pool-sensitive heads (L2H1, L5H0, L10H8) show Δσ_census ≥ 0.040.

---

## Kill conditions

**K1 (mechanism falsified — kbar doesn't discriminate):**
Cross-pool slope range for pool-stable heads ≥ cross-pool slope range for pool-sensitive heads.
If this fires, the pool-stability difference does not originate in the absolute-key profile
slope variation; a different mechanism is at work.

**K2 (no Spearman signal):**
ρ(cross-pool slope range, Δσ_exp161) ≤ 0. The kbar-derived slope variance does not track
exp-161 pool-sensitivity direction. Fires if the ranking is inverted or uncorrelated.

**K3 (census replication fails):**
Pool-stable heads show Δσ_census > 0.030 in the direct H2 replication. Pool-sensitivity is
not replicable at SEQ_LEN=1024 under the same protocol, suggesting exp-161's finding was
sensitive to the specific SEQ_LEN=1024 settings (the 1024-token sequences were used in exp-161
already; this would indicate a run-to-run variance issue).

---

## Protocol

**Model:** GPT-2 small (`gpt2`, float32, `attn_implementation="eager"`)

**Heads measured:** 5 structural Δ-window heads: L2H1, L3H4, L5H0, L7H11, L10H8
(layer, head indexed from 0)

**Part 1 — kbar collection (H1):**
- N=200 random-token sequences, SEQ_LEN=1024, tokenizer vocabulary (excluding special tokens)
- Extract per-head key vectors k[seq, pos, head, d_head] from all 12 layers
- Compute kbar[h, a] = mean over (sequences) of k[:, a, h, :] for a ∈ [0, 1023]
- Compute m_q[h] = mean over (sequences, positions [256, 1023]) of q[:, 256:, h, :]
  (query mean restricted to pool B+ range, avoiding early positions with distinctive wpe structure)
- Compute S_abskey[h, a] = m_q[h] @ kbar[h, a] / sqrt(d_head) for all a in [0, 1023]
- For each pool P ∈ {B: 256–511, C: 512–767, D: 768–1023}:
  - Fit OLS: S_abskey[h, a] ~ slope_P * log(a) + intercept, for a in the pool range
  - Record slope_P[h]
- Compute cross_pool_range[h] = max(slope_B, slope_C, slope_D) − min(slope_B, slope_C, slope_D)

**Part 2 — direct census replication (H2):**
- Same sequences as Part 1 (no additional forward passes needed if attention weights are
  saved alongside key vectors)
- Extract attention weights A[seq, layer, head, q, k] (or compute from saved QK)
- For each structural head h and each query pool P (query positions in P's range):
  - Pool the attention rows: A_pool[lag] = mean over (seq, q in P) of A[:, layer, head, q, q-lag]
    for lag ∈ [8, 256] (log-log fit range as in prior census runs)
  - Fit log-log OLS: log A_pool[lag] ~ σ_pool[h, P] * log(lag) + intercept
- Report σ_pool[h, P] for each head × pool combination
- Compute Δσ_census[h] = max(σ_B, σ_C, σ_D) − min(σ_B, σ_C, σ_D) for each head

**Compute note:** Saving both key vectors and attention weights for 200 sequences × 1024 tokens
× 12 layers × 12 heads × 64 dims is ~12 GB for float32. To reduce memory: process sequences in
batches, accumulate running kbar and running attention-lag histograms without storing all
activations. Release each batch's activations after accumulation.

---

## Register discipline notes

- This registration is written before any forward pass code is executed.
- The pool boundaries (B: 256–511, C: 512–767, D: 768–1023) are carried from exp-161 and fixed.
- The predicted ordering of heads (L3H4 ≈ L7H11 < L2H1 < L5H0 ≈ L10H8) is pre-registered;
  it cannot be changed after seeing the data.
- The kill threshold (K3: Δσ_census > 0.030 for pool-stable) is pre-registered; changing it
  after seeing the census values would be fishing.
- `analysis_only: false` — new forward passes required.
- The m_q averaging window (positions 256–1023) is chosen to avoid position 0's distinctive wpe
  signal (found in exp-162); this choice is pre-registered and cannot be changed post-hoc.

---

## What a positive result (K1/K2 not firing) would mean for T1

H1 confirmed means: pool-sensitivity is explained by where S_abskey[a] is sampled. Heads with
a more uniform absolute-key profile (stable slope across the three pool ranges) appear pool-stable
in the census. This sharpens the T1 restatement (seeded by exp-138): the census slope measures
absolute-key drift locally; pool-stable heads are those where the drift function happens to have
a near-constant log-slope across the full [256, 1023] range.

H2 confirmed (direct census) would additionally establish that the exp-161 behavioral
classification replicates cleanly at full SEQ_LEN=1024 without the pool-A artifact.

A negative result (K1 fires) is also informative: the pool-stability difference is NOT
explained by the kbar profile alone. Some other mechanism — query-side variation, cross-term
interaction, or a property not visible in the mean key direction — must account for why L3H4
and L7H11 are pool-stable. This would motivate a different instrument.
