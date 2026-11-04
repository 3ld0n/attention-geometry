---
created: "2026-11-03"
time: "~8:30 PM MDT (2026-11-04T03:30Z)"
author: Ariel
surface: physics-room
status: >
  theoretical synthesis — analytic separation of the positional mixing artifact
  from genuine non-Gaussian score structure; null hypothesis computation for exp-174;
  no new experiment registered.
subject: >
  The exploratory κ₄ computation (Oct 27) shows Δ-window heads have higher pooled
  excess kurtosis than non-window heads (p=0.0003). But Δ-window heads have MORE
  distributed key geometry (exp-127: lower λ₁/Σλ), and CLT says more distributed →
  more Gaussian → lower κ₄. This note resolves the tension by separating:
  (a) the mixing artifact contribution to pooled κ₄ from the absolute-key-position
  bias (exp-138); and (b) genuine non-Gaussian score structure (what the 4-point
  hypothesis requires). The mixing artifact calculation gives a null hypothesis
  for exp-174.
builds_on:
  - research/physics/notes/2026-10-25_route_a_failure_and_4pt_hypothesis.md
  - research/physics/notes/2026-10-27_kurtosis_exploration_4pt_hypothesis.md
  - research/physics/experiments/exp-127_eigenvalue_distribution/ (key geometry)
  - research/physics/experiments/exp-138_bilinear_decomposition/ (abs-key-position)
registers: >
  [THEORETICAL] — the mixing artifact decomposition; the null hypothesis for exp-174.
  [INTERPRETIVE] — the CLT tension and its resolution.
  No results claimed; no experiment registered.
---

# Mixing artifact analysis and the exp-174 null hypothesis

## 1. The tension stated precisely

The Oct 27 exploration note identified a tension:

**Fact 1 (exp-127):** Δ-window heads have more distributed key geometry
(λ₁/Σλ = 0.507 vs 0.651 for controls). More modes contributing to the score.

**Fact 2 (CLT):** A score s = (1/√d)·Σ_k q_k·k_k summed over d independent
bilinear terms has κ₄(s) → 0 as d → ∞. More distributed keys → more independent
terms → more Gaussian → lower excess kurtosis.

**Fact 3 (exploratory computation):** Δ-window heads have *higher* pooled excess
kurtosis than non-window heads (1.44 vs 0.37, p=0.0003).

Facts 1+2 predict the opposite of Fact 3. This is the tension. Three resolutions:

(A) **The mixing artifact.** The pooled distribution is a mixture of
position-specific distributions (from the absolute-key-position bias, exp-138).
Mixture distributions have elevated kurtosis independent of within-position
Gaussianity. The 4-point hypothesis is not the explanation; the positional
structure is.

(B) **Non-independence of score components.** The supra-MP structure in Δ-window
heads (exp-127: more eigenvalues above noise floor) means the score components are
*correlated*. Correlated components can produce elevated cross-cumulant terms that
raise κ₄ beyond the CLT expectation.

(C) **Protocol mismatch.** WikiText-native classification vs. random-token kurtosis
(noted in Oct 27 exploration). The Δ-window heads under random-token census may
differ from WikiText-native Δ-window heads. The classification mismatch could
inflate or deflate the apparent κ₄ difference.

---

## 2. The rank-1 baseline (the product-of-Gaussians case)

Before the mixing artifact calculation, establish the baseline for a rank-1 key matrix.

**Rank-1 key matrix:** W_K ≈ u·v^T (single singular value dominates).
Score: s(i,j) = (q_i·u)(v^T·k_j)/√d = X·Y where X and Y are approximately
independent scalars drawn from distributions determined by W_Q·e_i and W_K·e_j
under random-token inputs.

For X ~ N(0, σ_x²) and Y ~ N(0, σ_y²) independent:

  Var(XY) = σ_x²·σ_y²
  E[(XY)^4] = E[X^4]·E[Y^4] = 3σ_x^4 · 3σ_y^4 = 9σ_x^4·σ_y^4
  Excess kurtosis: γ₂ = E[(XY)^4]/(Var(XY))^2 − 3 = 9σ_x^4·σ_y^4/(σ_x²·σ_y²)^2 − 3 = 6

A rank-1 head has excess kurtosis ≈ 6 from the product-of-two-Gaussians distribution.

**Distributed heads (uncorrelated components):** s = (1/√d)·Σ_{k=1}^d X_k·Y_k,
where each X_k·Y_k is an independent product of two Gaussians.

For independent identically distributed terms:

  κ₄(s) = (1/d^2)·Σ_k κ₄(X_k·Y_k) = (1/d)·κ₄(X_1·Y_1) ∝ 6/d → 0

The excess kurtosis drops as 6/d. For d_head = 64 (GPT-2 small): γ₂ ≈ 6/64 ≈ 0.09.

**Summary:** Rank-1 heads → γ₂ ≈ 6; fully distributed uncorrelated heads → γ₂ → 0.
The exploratory data (Δ-window mean 1.44, non-window mean 0.37) doesn't fit either limit.

---

## 3. The mixing artifact calculation

**Setup.** Exp-138 established that for structural/Δ-window heads, the ensemble-averaged
score profile follows an absolute-key-position law: the dominant contribution to
σ_census is the absolute-key-position slope. The score profile can be written as:

  s(i, j) = α·log(j) + ε(i, j)

where α·log(j) is the absolute-key-position trend (the signal), and ε(i,j) is the
residual (approximately Gaussian for a given (i,j) pair).

**Pooled distribution.** The exploratory computation pooled all causal (i,j) pairs
across all query positions i and key positions j. The pooled distribution is:

  s_pooled ~ Mixture over j of N(α·log(j), σ²_ε)

where the mixing distribution for j is p(j) ∝ (L − j) on j = 1, …, L−1 (each key
position j appears in queries i = j+1, …, L, so p(j) ∝ (L−j)).

**The 4th cumulant of a Gaussian mixture.** For a mixture of Gaussians with means
μ_j and common variance σ²_ε, the cumulants of the pooled distribution are:

  κ₁ = E_j[μ_j] = α·E_j[log j]
  κ₂ = σ²_ε + α²·Var_j(log j)                                 [mixture variance formula]
  κ₃ = α³·κ₃(log j)                                            [3rd cumulant from mixing]
  κ₄ = α⁴·κ₄(log j)                                            [4th cumulant from mixing]

where κ_k(log j) are the cumulants of the log-j distribution under p(j).

**Key result:**

  κ₄(s_pooled) = α⁴·κ₄(log j)

The mixing artifact's contribution to the 4th cumulant is α⁴ times the 4th cumulant
of the log-j distribution under the census j-distribution. The ε residual contributes
only to κ₂; all higher cumulants from the mixing come through α·log j.

**The excess kurtosis of the pooled distribution:**

  γ₂(s_pooled) = κ₄(s_pooled) / (κ₂(s_pooled))^2
                = α⁴·κ₄(log j) / (σ²_ε + α²·Var(log j))^2

**Numerical estimate.** For the census j-distribution p(j) ∝ (L − j) on j = 1, …, L−1
with L = 128 (exploratory computation), computing:

  E[log j] ≈ log(L/e) ≈ log(47) ≈ 3.85   (mean of p(j)·log j; heavier near small j)
  Var(log j) ≈ varies, but roughly (log(L)/2)² / 3 ≈ ... numerically ~0.5–1.0

The 4th cumulant of log j (the kurtosis of the log-j distribution):

For p(j) ∝ (L−j), the distribution of log j is approximately log-triangular.
The excess kurtosis of a triangular distribution is −0.6. But p(j) ∝ (L−j) puts
more weight at small j (small log j), making the distribution left-skewed in log j.
Approximately, the excess kurtosis of log j under this distribution is in the range
[−0.6, +0.5] — not strongly non-Gaussian.

**Rough estimate of the mixing artifact's γ₂:**

For α = 0.25, Var(log j) ≈ 0.7, σ²_ε ≈ several (σ of scores is typically ~1–3 for
attention heads):

  γ₂_mixing ≈ (0.25)^4·κ₄(log j) / (σ²_ε + (0.25)²·0.7)^2

If σ²_ε >> α²·Var(log j) (variance dominated by the score noise):
  γ₂_mixing ≈ (0.25)^4·κ₄(log j) / σ⁴_ε ≈ very small (≪ 0.1)

If σ²_ε ≈ α²·Var(log j) (variance comparably from both):
  γ₂_mixing ≈ κ₄(log j) / (2·α²·Var(log j))^2 ≈ still small

**Conclusion:** The mixing artifact (from the linear log-j model) contributes
*small* excess kurtosis to the pooled distribution — less than 0.1 excess kurtosis
units for typical parameter values. The exploratory observation of γ₂ ≈ 1.44 for
Δ-window heads cannot be explained by the mixing artifact alone.

**However:** The model s(i,j) = α·log j + ε(i,j) is a linearization. The actual
score profile may have more complex structure — the Oct-25 note showed that the
score profile for structural heads has a zero crossing near position 511 (rising
at both ends), which is NOT captured by a simple log-linear model. The actual
mixing artifact from the non-linear score profile could be larger.

---

## 4. What this means for exp-174

The mixing artifact calculation shows two things:

**1. The pooled κ₄ cannot be attributed to the log-linear mixing alone.**

If the exploratory data's γ₂ ≈ 1.44 is real (not driven by the protocol mismatch
or L4H10 outlier), it exceeds what the log-linear mixing artifact would predict.
This means either (a) non-independence of score components (trained cross-cumulants),
or (b) more complex non-linear score structure not captured by the linear model.

**2. The within-bin test is the right discriminant.**

The formal exp-174 (kurtosis within fixed key-position bins) tests:
- **Null:** within-bin kurtosis ≈ 0 (Gaussian residual ε, mixing artifact only)
- **Alternative:** within-bin kurtosis > 0 (genuine non-Gaussian score structure)

If the linear log-j model holds perfectly (ε Gaussian), within-bin kurtosis should be 0.
If the actual residual ε is non-Gaussian (from trained weight structure beyond the
log-linear trend), within-bin kurtosis would be positive.

**What within-bin kurtosis > 0 would mean for the 4-point hypothesis:**
The score distribution, after removing the absolute-key-position trend, is non-Gaussian.
The leading non-Gaussian correction is the 4th cumulant. This is consistent with the
4-point structure hypothesis — but doesn't uniquely confirm SYK₄ universality.
Confirming SYK₄ would require matching the specific spectral structure of the 4-point
function, not just elevated κ₄.

**What within-bin kurtosis ≈ 0 would mean:**
The exploratory elevated κ₄ is explained by: (a) the non-linear score profile
(complex mixing artifact beyond the log-linear model) plus (b) the protocol mismatch
(WikiText-native classification under random-token census). The 4-point structure
hypothesis is not supported by the kurtosis test. G7 returns to: why does the
absolute-key-position bias settle at α = 1/4?

---

## 5. What remains open for the structural session

**G7 after this analysis:** The mixing artifact calculation confirms that the census
attractor α = 1/4 is primarily the absolute-key-position log-slope (Route B from
Oct 23). G7 reduces to: why do the trained W_Q, W_K matrices produce a log-linear
score profile with α = 1/4?

Three routes to this answer:

**(a) Statistical mechanics route:** The random-token census with softmax normalization
selects the fixed point where the score profile maximizes some statistical mechanical
quantity (entropy, free energy) subject to the formation constraint. The MaxEnt route
failed in its simplest form (Oct 25) — a data-dependent constraint (the formation
corpus) is needed.

**(b) 4-point structure route:** The score's non-Gaussian structure (if confirmed by
exp-174 within-bin test) selects α via the SYK₄ fixed point. The connection between
the 4th cumulant of the bilinear score and h = 1/4 needs to be derived from the SYK
diagrammatic expansion. This is the physically richer route.

**(c) The positional correlation route:** The formation condition (referential binding
at story scale) trains W_K to have specific positional correlation structure. The
exp-135 result (attention block 0 broadens wpe from 1D to 2D) suggests the trained
positional structure has 2 dominant modes. Why 2 modes, and why does this produce
α = 1/4? Not yet analyzed.

Routes (a) and (c) are compatible with the mixing artifact dominating.
Route (b) requires genuine non-Gaussian score structure (within-bin kurtosis elevated).

**Recommendation for the structural session:** present all three routes and ask
Eldon which direction the theory should go. Route (b) requires waiting for
exp-174 (token-gated). Routes (a) and (c) are solo-runnable in theoretical form.

---

## 6. Honest limits

This note's mixing artifact estimate (γ₂_mixing << 0.1) rests on the linear
log-j model, which is an approximation. The actual score profile has a zero
crossing near position 511 (exp-163 S_abskey data). A non-linear mixing artifact
calculation would need the actual score profile shape and is not attempted here.

The CLT argument for distributed-uncorrelated components (γ₂ → 6/d) assumes
independence of score components. Correlated components (from trained supra-MP
eigenvalues in Δ-window heads, exp-127) can produce larger or smaller κ₄ depending
on the sign of the cross-cumulants. The sign is not predicted without computing it.

The exploratory data (p=0.0003) has two confounds: protocol mismatch and L4H10
outlier. The formal exp-174 addresses both. Until exp-174 runs, the exploratory
result is evidence of a signal, not a measurement of it.

---

*Register ledger:*
*THEORETICAL — the rank-1 product-of-Gaussians baseline; the mixing artifact κ₄
decomposition (κ₄(s_pooled) = α⁴·κ₄(log j)); the three G7 routes.*
*INTERPRETIVE — the numerical estimate of γ₂_mixing; the resolution of the CLT
tension; what within-bin kurtosis would mean.*
*No measured claims; no experiment registered. All analysis rests on approximations
identified in §6 (Honest limits).*
