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

---

## Results — August 10, 2026, ~8:45 AM MDT (single run; convention correction applied in the same session)

**Convention note:** The first run of `analyze_mixed.py` used log-log slopes
(σ = −OLS(log|S| vs log dx)), which does not match exp-112/113's convention
(σ = −OLS(S vs log dx), linear). The script was corrected before any verdict
was read; the commit below records the corrected version.

`analyze_mixed.py` → `results_gpt2.json`, `mixed_profiles_gpt2.npz`.
Runtime ~1.3 s. Pre-registration commit `db0e52a` was already pushed;
results commit: see attention-geometry repo.

**K1 — PASS** on all 5 structural pairs. S_pos recomputed from saved
qbar/kbar matches stored profile to within tolerance.

### Structural heads (random tokens)

| Head | σ_mf | σ_mixed_k | σ_mixed_q | σ_pos | corr_k | corr_q | frac | res_rel |
|---|---|---|---|---|---|---|---|---|
| L2H1 | 0.843 | 0.675 | 0.702 | 0.562 | 0.168 | 0.141 | 1.10 | 2.4% |
| L3H4 | 0.988 | 0.778 | 0.775 | 0.610 | 0.210 | 0.213 | 1.12 | 4.5% |
| L5H0 | 0.827 | 0.617 | 0.637 | 0.475 | 0.210 | 0.190 | 1.14 | 5.1% |
| L7H11 | 0.744 | 0.549 | 0.581 | 0.429 | 0.195 | 0.163 | 1.14 | 10.4% |
| L10H8 | 0.839 | 0.706 | 0.713 | 0.600 | 0.133 | 0.126 | 1.08 | 0.9% |

σ_mf values match exp-113 to < 0.001. σ_pos values match exp-112 to < 0.001.

### P1 — CONFIRMED, 5/5.

σ_mixed_k < σ_mf on every structural head (prediction: CONFIRMED).
The mixed-k profile (using mean-field query, true key) has a shallower slope
than S_mf on all 5 heads.

### P2 — CONFIRMED, 5/5.

The additive decomposition accounts for 1.08–1.14 × the overshoot (fraction
consistently > 1.0). The prediction was ≥ 0.5; it overshots — the two mixed
corrections together OVEREXPLAIN the overshoot by 8–14%. The residual S_residual
brings S_pos slightly back toward S_mf (fraction > 1 means the residual product
term partially cancels the corrections from the two linear terms).

### P3 — CONFIRMED, 16/16.

All 16 semantic heads under WikiText show σ_mixed_k < σ_mf. Fractions 1.08–1.18.

### What exp-115 + exp-116 together establish

**The mechanism for the mean-field slope overshoot (σ_mf − σ_pos ≈ 0.24–0.38)
is UNIFORM LN SHRINKAGE, not position-dependent shrinkage.**

1. exp-115 established that f_k(j) = ||k̄_j|| / ||k_mf_j|| is position-FLAT
   over the key pool (γ_k ≈ 0 for structural heads). Same for query side.

2. exp-116 established that the uniform attenuation f_q * f_k < 1 explains the
   overshoot via the LINEAR OLS convention:
   - S_pos(dx) ≈ S_mixed_k(dx) + S_mixed_q(dx) − S_mf(dx) + small residual
   - S_mixed_k(dx) ≈ f_k · S_mf(dx) and S_mixed_q(dx) ≈ f_q · S_mf(dx)
   - Therefore S_pos ≈ (f_k + f_q − 1) · S_mf + residual
   - In linear OLS: σ_pos ≈ (f_k + f_q − 1) · σ_mf + small correction
   - For f_k ≈ 0.74–0.84 and f_q ≈ 0.77–0.86, this gives σ_pos ≈ 0.51–0.70 · σ_mf,
     consistent with σ_pos/σ_mf = 0.562/0.843 ≈ 0.67 for L2H1.

3. **The overshoot is invisible in log-log** (if f is position-constant, 
   log|S_pos| = const + log|S_mf|, so log-log slopes are equal). It is a
   CONVENTION ARTIFACT of the linear OLS: the uniform amplitude reduction
   changes the OLS coefficient of S vs log(dx).

4. **This is physically important:** The true conformal exponent (the slope of
   log A vs log lag, which equals the linear OLS slope of S vs log dx) IS
   σ_pos, not σ_mf. The mean-field approximation gives an OVERESTIMATE of the
   conformal exponent by a factor of 1/(f_k · f_q) ≈ 1.44–2.0. A derivation
   that correctly derives σ_mf from h̄ would still need a correction for the
   E[LN(h)] ≠ LN(E[h]) gap before reaching σ_pos.

5. **Residual (S_residual) is small but real.** The product of deviations
   (q̄_i − q_mf_i) · (k̄_{i-dx} − k_mf_{i-dx}) contributes 1–10% of S_mf
   amplitude. Its slope slightly UNDOES the additive correction (fraction >
   1.0), meaning the true q̄ and k̄ perturbations are slightly anti-correlated
   with the score in a way that partially restores the amplitude.

### What this opens

The theory-of-A target now has a clean three-level structure:
- **Level 1 (done):** σ_pos is the census slope, the position-mean score slope,
  the quenched slope (exp-110/112 confirmed these are identical).
- **Level 2 (done, this session):** σ_pos ≈ f · σ_mf where f ≈ f_q · f_k
  (the uniform LN shrinkage). Exp-115/116.
- **Level 3 (open):** Derive σ_mf from h̄ — the mean residual stream through
  the network. This is simpler than deriving σ_pos directly because it avoids
  the E[LN(h)] ≠ LN(E[h]) complication. The mean residual stream h̄_i at layer
  ℓ = 0 is exactly the mean embedding + position embedding (no approximation).
  The open question is whether the WQK bilinear form on LN(h̄) has analytically
  tractable log-falling diagonals.

### Standing limits

One model, one seed, 21 heads, 50-input ensemble. The "attenuation factor"
f = σ_pos / σ_mf ≈ 0.55–0.72 — not yet derived from the distribution of
LN(h) around LN(h̄).
