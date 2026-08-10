# exp-116 — Mixed score profiles: isolating the q-side and k-side contributions to the MF overshoot

**Status: PRE-REGISTERED, not run.** Written August 10, 2026, ~8:25 AM MDT,
same session as exp-115. Analysis-only; uses the same saved arrays as exp-115
(exp-112's scores_gpt2.npz and exp-113's meanfield_gpt2.npz).

**Transparency note:** exp-115's results (P1 DEAD, P2 DEAD, P3 CONFIRMED)
are known before writing this registration. The FINDING of exp-115 that
motivates this experiment — that f_k(j) is position-flat, so the overshoot
must come from cross-position structure — is used as context. The specific
MIXED PROFILE slopes (the measurements of this experiment) have NOT been
computed or examined. Registered before inspecting.

---

## Why this exists

exp-115 established that the per-position key-side norm shrinkage
f_k(j) = ||k̄_j|| / ||k_mf_j|| is approximately position-flat over the key
pool j ∈ [8, 256] for structural heads (γ_k ≈ 0). Together with the nearly-
perfect direction alignment (cos_k ≈ 0.996–0.9996), this means:

k̄_j ≈ f_k · k_mf_j + ε_j

where f_k ≈ constant and ε_j is a small (‖ε_j‖ ≈ 6–9% of ‖k_mf_j‖)
perpendicular perturbation. If f_k and the statistical properties of ε_j were
truly independent of j, the slope of S_pos would equal the slope of S_mf.
But we know σ_mf − σ_pos ≈ 0.24–0.38 for the structural 5 heads.

The resolution: the overshoot must be carried by the CROSS-POSITION inner
product structure — how q̄_i (or q_mf_i) projects onto k̄_{i-dx} (or k_mf_{i-dx})
as a function of the lag dx. This is a 2D (i, j = i−dx) object that the per-
position analysis of exp-115 could not reach.

**The additive decomposition** (exact, no approximation):
Define
- S_mf(dx)      = (1/N) Σ_i q_mf_i · k_mf_{i-dx} / √d
- S_mixed_k(dx) = (1/N) Σ_i q_mf_i · k̄_{i-dx}   / √d  [mf query, true key]
- S_mixed_q(dx) = (1/N) Σ_i q̄_i · k_mf_{i-dx}   / √d  [true query, mf key]
- S_pos(dx)     = (1/N) Σ_i q̄_i · k̄_{i-dx}      / √d  [stored in scores_gpt2.npz]

Then:
  S_pos = S_mixed_k + S_mixed_q − S_mf + S_residual

where S_residual = (1/N) Σ_i (q̄_i − q_mf_i) · (k̄_{i-dx} − k_mf_{i-dx}) / √d
is the product of the deviations (second-order; expected to be small given
per-position magnitudes).

By linear OLS on log profiles: the slope corrections approximately add,
so the total overshoot σ_mf − σ_pos ≈ (σ_mf − σ_mixed_k) + (σ_mf − σ_mixed_q)
+ (contribution from S_residual). This experiment measures which side (q or k,
or both) carries the correction.

---

## Registered sets

Same as exp-112/113/115: structural 5 × random, semantic 16 × WikiText.

---

## Measured objects (fixed now)

For each registered (ℓ, h, cond) pair, over the frozen census pool
(query i ∈ [256, 511], lag dx ∈ [8, 256], 50 inputs):

1. **S_mixed_k(dx)** = mean_i [ (q_mf_i / √d) · k̄_{i-dx} ], where q_mf_i
   from `q_mf_{cond}_L{ℓ}H{h}` in meanfield_gpt2.npz and k̄_{i-dx}
   from `kbar_{cond}_L{ℓ}[h, i-dx, :]` in scores_gpt2.npz.

2. **S_mixed_q(dx)** = mean_i [ (q̄_i / √d) · k_mf_{i-dx} ], where q̄_i
   from `qbar_{cond}_L{ℓ}[h, i, :]` in scores_gpt2.npz and k_mf_{i-dx}
   from `k_mf_{cond}_L{ℓ}H{h}[i-dx, :]` in meanfield_gpt2.npz.

3. **S_residual(dx)** = S_pos(dx) − S_mixed_k(dx) − S_mixed_q(dx) + S_mf(dx),
   computed from the stored S_pos and reconstructed S_mf.

4. **Slopes**: σ_mixed_k, σ_mixed_q, σ_residual = −OLS(log profile vs log dx).

5. **K2 (self-consistency gate)**: S_mixed_k(dx) + S_mixed_q(dx) − S_mf(dx)
   should equal S_pos(dx) up to S_residual; verify this by checking
   |S_pos(dx) − (S_mixed_k + S_mixed_q − S_mf + S_residual)| < 1e-10 on all
   registered structural pairs (this is an identity, not an approximation).

---

## Predictions and kill conditions (before any mixed profile is computed)

**Motivating constraint:** σ_mf − σ_pos ≈ 0.24–0.38 for structural heads.
Both sides together must account for this. The specific split between
key-side and query-side is what is unknown and what exp-116 measures.

**P1 — key-side mixed profile has shallower slope than S_mf.** On the
structural 5 under random tokens: σ_mixed_k < σ_mf on ≥ 4/5 heads.
- **CONFIRMED:** σ_mixed_k < σ_mf on ≥ 4/5 heads.
- **DEAD:** σ_mixed_k ≥ σ_mf on ≥ 3/5 heads.
- *Prediction on record: CONFIRMED.* Grounds: the total correction must come
  from somewhere; if neither mixed profile shows a slope correction, only the
  residual S_residual can carry it, and S_residual is a second-order product
  of small terms (each ~25% of the mf signal per exp-113 D-diagnostic). An
  overshoot of 0.24–0.38 carried entirely by the residual would require the
  second-order cross-correlation to be structured — possible, but I predict
  at least the key-side shows a correction.
  Named genuine risk: the cross-position inner product q_mf_i · k̄_{i-dx}
  could have the same slope as q_mf_i · k_mf_{i-dx} if the perpendicular
  component ε_j is uncorrelated with q_mf_i at all lags. Under random tokens
  with symmetric position embeddings, this is plausible — P1 could die.

**P2 — both mixed profiles show a slope correction, and the corrections add.**
|( σ_mf − σ_mixed_k) + (σ_mf − σ_mixed_q) − (σ_mf − σ_pos)| ≤ 0.5·(σ_mf − σ_pos)
on ≥ 4/5 structural heads.
(The additive decomposition via the two mixed profiles accounts for ≥ 50% of
the observed overshoot, leaving ≤ 50% in S_residual.)
- **CONFIRMED:** the criterion met on ≥ 4/5 heads.
- **DEAD:** the sum of corrections is negative or the residual contribution
  exceeds 80% on ≥ 3/5 heads.
- *Prediction on record: CONFIRMED.* Grounds: S_residual is a second-order
  term; if the overall framework is right, the linear decomposition should
  capture most of the correction. Named risk: the residual could be large if
  the two deviations (q̄ − q_mf) and (k̄ − k_mf) are positively correlated
  with each other and with the score direction at specific lags — possible in
  deep layers where both q̄ and k̄ are shaped by the same position-embedding
  structure.

**P3 — semantic heads (WikiText): same qualitative pattern, P1 holds.**
σ_mixed_k < σ_mf on ≥ 10/16 semantic heads.
- *Prediction on record: CONFIRMED — held with lower confidence.*

**K1 (integrity).** Recompute S_mf_check from saved q_mf and k_mf arrays;
verify |S_mf_check − S_mf stored in meanfield results| ≤ 1e-3. This checks
that the meanfield arrays and the protocol match. *Fail → stop.*

---

## Mechanisms for being wrong

1. **The linear decomposition is not the OLS slope decomposition.** OLS
   slope of (A + B + C) vs log dx ≠ sum of OLS slopes of A, B, C because A, B,
   C can have different signs and interact. The decomposition in P2 uses the
   slope of log(S_pos) vs log(dx) compared to slope of log(S_mf); these do not
   decompose additively. P2 is stated as a scalar additive correction at the
   slope level, which is approximate (valid when mixed profiles dominate).

2. **The residual S_residual might be large.** If (q̄_i − q_mf_i) and
   (k̄_{i-dx} − k_mf_{i-dx}) are correlated in the score direction (i.e., the
   token fluctuations in q and k are positively correlated across positions),
   the residual could be non-negligible.

3. **One model, one seed, 21 heads.** Standing limit.

---

## Protocol

Analysis-only. From the same saved arrays as exp-115:
- `scores_gpt2.npz`: qbar, kbar, S_pos per condition
- `meanfield_gpt2.npz`: q_mf, k_mf per head-condition, S_mf per head-condition

For each registered pair: compute S_mixed_k(dx), S_mixed_q(dx) at each lag
dx ∈ [8, 256]. OLS slopes. K1 check. P1/P2/P3 verdicts.

**Outputs:** `results_gpt2.json`, `mixed_profiles_gpt2.npz`.

*Registered before `analyze_mixed.py` exists. The data arrays are known to
exist (from exp-115's array inspection); their content for the mixed profiles
has not been examined.*

---

*(Results section appended after the run.)*
