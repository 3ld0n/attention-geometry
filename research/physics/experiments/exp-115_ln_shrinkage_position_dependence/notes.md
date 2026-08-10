# exp-115 — Position-dependence of the LN shrinkage on the key-side positional-mean vector

**Status: PRE-REGISTERED, not run.** Written August 10, 2026, ~7:50 AM MDT, solo
physics room session (morning arrival, first sitting of the day). Analysis-only:
no new forward passes; all inputs are saved arrays from exp-112 and exp-113.
Same binding discipline as the rest of the theory-of-A chain: kill conditions
before the argument feels elegant; the prediction direction is stated before any
shrinkage value is read from the arrays.

**Transparency note:** The npz file keys and shapes were inspected before this
registration to confirm the data structures needed for the script; the
*values* of the shrinkage factors themselves (f_k(j) vs j) have not been
examined. The protocol section below names exactly which arrays are used;
the pre-registration gates commit and push before any analysis runs.

---

## Why this exists

exp-113 killed the mean-field reduction: the positional-mean key vector
k̄_j = E[LN(h_j) W_K + b_k] cannot be computed from the mean residual stream
h̄_j through LN; the token-fluctuation statistics enter the carrier itself.
The slope overshoot is systematic (σ_mf − σ_pos = 0.24–0.38 on structural-5,
0.15–0.40 on semantic-16) and directionally uniform (σ_mf > σ_pos on all 21
native pairs).

exp-113's note identified the mechanism constraint: **a lag-constant shrinkage
would cancel in the window OLS, so the slope change must come from
position-dependent shrinkage — specifically from the key side.** The query-side
shrinkage f_q(i) = ||q̄_i|| / ||q_mf_i|| is averaged over the fixed query pool
i ∈ [256, 511] for every lag dx, making it dx-constant and invisible to the
OLS slope. The key-side f_k(j) = ||k̄_j|| / ||k_mf_j||, by contrast, is
sampled at positions j = i − dx (averaging over the key pool j ∈ [256−dx,
511−dx]), which shifts toward smaller j as dx increases. If f_k(j) varies with
j, the pooled mean shifts with dx and contributes a slope to the ratio
S_pos / S_mf.

**The sign constraint from exp-113:** σ_mf > σ_pos means S_mf decays faster
than S_pos (the mean-field profile is steeper). For the ratio C(dx) =
S_pos(dx) / S_mf(dx) to be increasing with dx (as required when both profiles
are positive and S_mf decays faster), the mean f_k over the key pool must
increase with dx. Since the key pool shifts toward smaller j as dx increases,
this requires **f_k(j) to be larger at smaller j** — i.e., f_k decreases as j
increases.

Physical reading: positions close to the sink (large j) have larger relative
fluctuations in LN-normalized key space, so averaging unit-normalized vectors
produces more shrinkage (smaller f_k). Positions far from the sink (small j)
have less relative fluctuation under random tokens (because at those positions
the position embedding dominates more relative to token noise), yielding less
shrinkage (larger f_k). This experiment tests whether the measured norms follow
this pattern, and whether the gradient quantitatively accounts for the overshoot.

---

## Registered sets

Same as exp-112/113:
- **Structural 5** (L2H1, L3H4, L5H0, L7H11, L10H8) × random tokens
- **Semantic 16** (L4H10, L7H1, L8H2, L9H4, L9H6, L10H1, L10H2, L10H10,
  L11H0, L11H1, L11H2, L11H4, L11H5, L11H6, L11H7, L11H9) × WikiText

31 registered head-condition pairs. All other combinations: exploratory, labeled.

---

## Measured objects (fixed now)

All computed from saved arrays; no new forward passes.

**Shrinkage factors:**

For each registered (layer ℓ, head h, condition):
- `kbar[h, j, :]` = `kbar_{cond}_L{ℓ}[h, j, :]` from `scores_gpt2.npz`
  (shape of the array: n_heads × seq_len × d_head = 12 × 512 × 64)
- `k_mf[j, :]` = `k_mf_{cond}_L{ℓ}H{h}[j, :]` from `meanfield_gpt2.npz`
  (shape: seq_len × d_head = 512 × 64)
- `f_k(j)` = ||kbar[h, j, :]||₂ / ||k_mf[j, :]||₂, for j ∈ [0, 511]
- `f_q(i)` = ||qbar[h, i, :]||₂ / ||q_mf[i, :]||₂, for i ∈ [0, 511]

**Slope of shrinkage with position:**
- γ_k = OLS slope of log f_k(j) vs log j over j ∈ [8, 256] (the key pool range)
- γ_q = OLS slope of log f_q(i) vs log i over i ∈ [256, 511] (the query pool)

**Effective correction profile:**
- C_k(dx) = mean over i ∈ [256, 511] of f_k(i − dx), for dx ∈ [8, 256]
  (the pooled key-side shrinkage at each lag; this is the dx-dependent
  correction that the OLS slope sees)
- γ_C = OLS slope of log C_k(dx) vs log dx over dx ∈ [8, 256]

**Direction cosine (diagnostic only):**
- cos_k(j) = kbar[h,j,:] · k_mf[j,:] / (||kbar|| · ||k_mf||)

**Integrity check:**
- Recompute S_pos_check(dx) = mean_{i∈[256,511]} (kbar[h,i,:] · kbar[h,i-dx,:]) / sqrt(d)
  and verify |S_pos_check(dx) − S_pos_random[ℓ, h, dx-8]| < 1e-3 on all registered
  structural pairs. (Score arrays use dx_index = dx − 8; score convention is
  q·k, but k̄·k̄ is a norm proxy; this checks that we are addressing the
  right head in the right condition. Use the actual dot-product reconstruction.)

Wait — the score is q̄_i · k̄_{i-dx} / sqrt(d_head). The integrity check should verify that:
S_pos_check(dx) = (1/N_q) Σ_{i=256}^{511} qbar[h,i,:] · kbar[h,i-dx,:] / sqrt(d_head)
matches S_pos_random[ℓ_idx, h, dx_idx] from scores_gpt2.npz, where dx_idx = dx − 8.

---

## Predictions and kill conditions (before f_k(j) values are read)

**K1 (integrity gate).** For all 5 structural registered pairs under random:
|S_pos_check(dx) − S_pos_stored(dx)| ≤ 1e-3 for all dx ∈ [8, 256].
*Fail → the array addressing is wrong; stop and fix before reading any f_k.*

**P1 — key-side shrinkage decreases with position (f_k(j) larger at smaller j).**
On the structural 5 under random: γ_k < −0.05 on ≥ 4/5 heads.
- **CONFIRMED:** γ_k < −0.05 on ≥ 4/5 heads.
- **DEAD:** γ_k > 0 on ≥ 3/5 heads (f_k increases with j — opposite of the
  direction needed to explain the overshoot).
- Between: ambiguous.
- *Prediction on record: CONFIRMED.* Grounds: the overshoot is consistent
  across all 21 native pairs; the only mechanism that produces a consistent
  *slope* difference is a consistent position-gradient in the key-side
  shrinkage. The sign follows from the derivation above (C(dx) must increase
  with dx for σ_mf > σ_pos → f_k must decrease with j). Named genuine risk:
  the overshoot might be explained by the direction error (cos_k(j) varying
  with j) rather than the norm error; if so, P1 could confirm on the norms
  while the explanation for the slope sits in direction. That would be a
  finding, not a failure — the "mechanism" section would be wrong, not the
  prediction.

**P2 — key-side shrinkage gradient accounts for ≥ 50% of the overshoot slope.**
On the structural 5 under random: |γ_C| ≥ 0.5 · (σ_mf − σ_pos) on ≥ 4/5 heads,
AND the sign of γ_C is positive (C_k increases with dx — the direction that
accounts for the overshoot, not amplifies it).
- **CONFIRMED:** |γ_C| ≥ 0.5 · (σ_mf − σ_pos) with γ_C > 0 on ≥ 4/5 heads.
- **DEAD:** γ_C < 0 on ≥ 3/5 heads (shrinkage gradient in the wrong direction
  — norm shrinkage amplifies the overshoot rather than causing it).
- Between: ambiguous.
- *Prediction on record: CONFIRMED — conditional on P1.* If P1 dies, P2 is
  still reported but its interpretation changes: the overshoot source is
  elsewhere (direction error, query-side interaction, or some other mechanism).

**P3 — semantic heads (WikiText) show the same qualitative pattern.**
On the semantic 16 under WikiText: γ_k < 0 on ≥ 10/16 heads.
- **CONFIRMED:** γ_k < 0 on ≥ 10/16 heads.
- **DEAD:** γ_k > 0 on ≥ 10/16 heads.
- *Prediction on record: CONFIRMED — held with lower confidence than P1.*
  Grounds: the same systematic overshoot appears on semantic pairs (exp-113).
  Risk: WikiText content creates position-specific correlations that could
  produce f_k patterns unrelated to the random-token mechanism; the direction
  could differ. A P3 failure while P1 confirms would say the shrinkage
  mechanism is different between populations — which would be important.

**Diagnostic D (travels with verdicts, no verdict itself):**
- f_q(i) profile and γ_q: expected to be approximately flat over i ∈ [256,511]
  (query-side dx-constant argument); a large γ_q would be a surprise and
  would need explanation.
- cos_k(j) profile: reported so "shrinkage" is distinguished from "direction
  error." If cos_k(j) is close to 1 everywhere, the norm interpretation holds.
  If cos_k varies substantially with j, the direction error is a co-contributor.

---

## Mechanisms for being wrong (named now)

1. **The approximation C(dx) ≈ mean_i f_q(i) f_k(i-dx) ignores direction error
   and query-key covariance.** The full decomposition is
   q̄_i · k̄_j = f_q(i) f_k(j) (q_mf_i · k_mf_j) + (direction cross terms).
   The cross terms can carry their own dx-dependence; if they dominate, the
   norm story is incomplete. Diagnostic D (cos_k profile) is the partial
   check — it cannot separate all the terms, but large direction variation
   flags the risk.

2. **The key pool averaging is approximate.** C_k(dx) = mean_i f_k(i-dx) is
   computed from f_k at integer positions. The "effective C_k" measured this
   way accounts for norm shrinkage only; it does not account for the fact that
   the f_q and f_k factors appear in a product under the pooling sum, so
   correlation between f_q(i) and f_k(i-dx) (if any) would shift the account.
   Expected to be small for iid random tokens.

3. **The OLS slope on 249 lags (dx ∈ [8, 256] at integer steps) is not
   equivalent to a log-log slope on a clean power law.** The f_k(j) profile
   need not be a power law in j. The OLS slope γ_k is a linear fit to a
   possibly non-power-law curve; reporting it as a "slope" is a summary, not
   a mechanistic claim.

4. **One model, one seed, 21 heads.** The standing limit of the whole week.

5. **Log(j) and log(dx) are correlated by construction.** The regression of
   log C_k vs log dx uses C_k built from f_k values at positions j = i − dx.
   The j values and dx values are not independent; the slope γ_C is a
   summary of a derived object, not an independent measurement.

---

## Protocol

**Input arrays (no new forward passes):**
- `research/physics/experiments/exp-112_score_drift_decomposition/scores_gpt2.npz`
  - Arrays used: `qbar_{cond}_L{ℓ}` (shape 12 × 512 × 64) and
    `kbar_{cond}_L{ℓ}` (same) for each registered layer.
  - Also `S_pos_{cond}` (shape 12 × 12 × 249) for the K1 integrity check.
- `research/physics/experiments/exp-113_mean_field_reduction/meanfield_gpt2.npz`
  - Arrays used: `q_mf_{cond}_L{ℓ}H{h}` (shape 512 × 64) and
    `k_mf_{cond}_L{ℓ}H{h}` (same) for each registered pair.
- `research/physics/experiments/exp-113_mean_field_reduction/results_gpt2.json`
  - Used to load σ_mf and σ_pos per registered pair (for the P2 threshold).

**Computation:**
For each registered (ℓ, h, cond) pair:
1. Load kbar[h, :, :] from scores_npz and k_mf[:, :] from mf_npz.
2. Compute f_k(j) = norm(kbar[h, j, :]) / norm(k_mf[j, :]) for j ∈ [0, 511].
3. Compute γ_k = −OLS(log f_k(j) vs log j) over j ∈ [8, 256].
   (Note: γ_k defined as the NEGATIVE of the OLS slope so positive γ_k means
   f_k is decreasing as j increases; consistent with σ convention. Actually,
   let's define γ_k as the raw OLS slope of log f_k vs log j — negative γ_k
   means f_k decreases with j, which is the predicted direction.)
4. Compute C_k(dx) = mean_{i=256..511} f_k(i − dx) for dx ∈ [8, 256].
5. Compute γ_C = OLS slope of log C_k vs log dx.
6. Report cos_k(j) = kbar[h,j,:] · k_mf[j,:] / (||kbar|| ||k_mf||) median
   and its OLS slope vs j over [8, 256].
7. K1: compute S_pos_check and verify against stored S_pos.

Similarly for query side (diagnostic only).

**Outputs:** `results_gpt2.json` (gates, verdicts per head, diagnostics),
`shrinkage_gpt2.npz` (f_k profiles, f_q profiles, C_k profiles per registered
pair), `run_log.txt`.

*Registered before `analyze_shrinkage.py` exists.*

---

*(Results section appended after the run.)*
