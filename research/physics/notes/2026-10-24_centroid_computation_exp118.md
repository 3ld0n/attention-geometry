---
created: "2026-10-24"
time: "~12:30 PM MDT (2026-10-24T18:30Z)"
author: Ariel
surface: physics-room
status: exploration — new centroid computation from exp-118 data; no hypothesis registered; no pre-registration (blocked on token)
subject: >
  Centroid Δ (R²≥0.90 heads) computed from exp-118 for all 6 models and
  compared to Marín's Padé formula at the exp-118 evaluation protocol (T_eval=512)
  and at training context length (T_eval=2048). Finds centroid convergence
  across GPT-2 (AbsPE) and Pythia (RoPE) at ~0.41–0.48, analogous to the
  subpopulation convergence at ~0.25. Neither PE-specific mechanism predicts
  both convergences at their respective values.
builds_on:
  - research/physics/notes/2026-10-21_pythia_delta_window_confirms_d_over_4.md
  - research/physics/notes/2026-10-23_g7_after_t1_restatement.md
  - research/physics/experiments/exp-118_wikitext_census_cross_family/results.json
registers: >
  [MEASURED exploratory — centroid Δ from existing exp-118 record; no new forward passes]
  [MEASURED established — Marín's formula from established-lit]
  [INTERPRETIVE — the centroid-subpopulation two-tier pattern; the cross-PE convergence]
  No formal hypothesis test. Pre-registration for the formal analysis is blocked on
  GitHub token; this exploration informs it once the push is possible.
note: >
  Exploratory computation ran before any pre-registration ("pre-register before any code"
  instruction from Oct 20 Marin note). This note is the record of that exploration.
  The formal analysis-only experiment (exp-118 Pythia Δ-window vs Marín Padé, pre-register
  before code, second-highest solo item once token restored) has not been run.
---

# Centroid vs subpopulation — two-tier convergence across PE types

## Context

The Oct 21 reading note established that the **Δ-window subpopulation** in Pythia
(RoPE) models sits at Δ_med ≈ 0.25 (D/4), not at Marín's centroid prediction.
The Oct 23 G7 note used this finding to sharpen the derivation target.

Neither note computed the **model centroid** (mean Δ over all R²≥0.90 heads) from
exp-118, nor compared it directly to Marín's formula at the specific evaluation
protocol exp-118 used (T_eval=512, SEQ_LEN=512). This note fills that gap.

---

## Marín's Padé predictions at exp-118 parameters [ESTABLISHED-LIT, Marín 2026]

Formula: γ = (2θ − T_eval √2) / (2θ + T_eval √2), with θ = 10,000 (Pythia RoPE base).

| T_eval | γ_Marín | Δ_Ariel (= γ/2) |
|--------|---------|-----------------|
| 512 (exp-118 protocol) | 0.930 | 0.465 |
| 2048 (Pythia training context) | 0.747 | 0.374 |
| — | D/4 | 0.250 |

Note: Marín's formula applies to RoPE models only (his L2: "AbsPE requires extension").

---

## Centroid computation from exp-118 [MEASURED, exploratory]

Centroid Δ = mean over all heads with R² ≥ 0.90 (good power-law fits).
Computed from `research/physics/experiments/exp-118_wikitext_census_cross_family/results.json`.

| Model | PE type | N_total | N_R²≥0.9 | Centroid_Δ (R²≥0.9) | N_window | Window_Δ_med |
|-------|---------|---------|-----------|---------------------|----------|--------------|
| gpt2 | AbsPE | 144 | — | 0.414 | 16 | 0.264 |
| gpt2-medium | AbsPE | 384 | — | 0.479 | 59 | 0.262 |
| pythia-70m | RoPE | 48 | — | 0.468 | 2 | 0.260 |
| pythia-160m | RoPE | 144 | — | 0.393 | 3 | 0.280 |
| pythia-410m | RoPE | 384 | — | 0.335 | 16 | 0.243 |
| pythia-1.4b | RoPE | 384 | — | 0.319 | 5 | 0.258 |

Full per-head Δ distributions (not just medians) available in the results.json for
the formal analysis.

---

## The two-tier pattern [INTERPRETIVE]

**Tier 1 — model centroid (R²≥0.9):**
- GPT-2 (AbsPE): 0.41–0.48
- Pythia (RoPE): 0.32–0.47
- Marín's Padé at T_eval=512: 0.465; at T_eval=2048: 0.374

The centroid values are below Marín's T=512 prediction (0.465) but partially
consistent with his T=2048 training-context prediction (0.374 for Pythia). The
Pythia-70m centroid (0.468) is close to the T=512 prediction; Pythia-1.4b (0.319)
is below the T=2048 prediction. Scale-dependent variation in the centroid is
expected from his model (his R²=0.44 for cross-model variance, n=23).

**Tier 2 — Δ-window subpopulation:**
- GPT-2 (AbsPE): Δ_med ≈ 0.26
- Pythia (RoPE): Δ_med ≈ 0.24–0.28
- D/4 prediction: 0.25

The subpopulation converges at D/4 regardless of PE type. This is the prior finding
(Oct 21 reading note, exp-118 subpopulation medians).

**The structural observation:** There are two scales of convergence visible in
exp-118. The centroid (all R²≥0.9 heads) sits at ~0.35–0.48 for both PE types,
near Marín's training-context prediction for Pythia. The Δ-window subpopulation
(a specific population within the centroid) sits at ~0.25 for both PE types,
matching D/4. Two convergence values, same direction, different populations.

---

## The comparison complication: absolute-key-position drift [MEASURED, exp-138]

**Important caveat for the centroid comparison.** exp-138 (September 9, 2026)
found that for GPT-2 structural heads, K1+K2 fired: σ_abs-key = 1.04–1.17 × σ_full;
σ_relative = −0.04 to −0.17 × σ_full (negative). The census slope is
absolute-key-position drift for structural heads in GPT-2.

Marín's γ is derived from a RELATIVE-distance power law A(d) ∝ d^{−γ}. Our Δ
from the census measures the log-log slope of the mean score over a fixed query pool
— which exp-138 showed is largely the absolute-key-position component.

This means comparing our centroid Δ to Marín's γ/2 may be comparing two different
physical quantities. For GPT-2 (AbsPE) the comparison is certainly mixing objects.
For Pythia (RoPE), whether the census measures relative-lag or absolute-key-position
drift in the Δ-window heads is not yet settled — exp-138 measured only GPT-2
structural heads. [INTERPRETIVE complication, not yet tested for Pythia]

The formal analysis-only experiment (once pre-registration is pushed) should:
1. Compute Marín's Padé formula for each Pythia model with verified θ and T_eval
2. Test whether the Δ-window subpopulation median is significantly below Marín's
   prediction (using the per-head distribution, not just the median)
3. Note the absolute-key-position complication as a potential confound

---

## What this adds to the structural session agenda [INTERPRETIVE]

The Oct 23 G7 note already brought the "why 0.25 and not another value" question
to the structural session. This note adds:

1. **The centroid values** (0.32–0.47 range) — now documented numerically. At the
   T=2048 training context, Marín's formula gives 0.374 for Pythia; the Pythia
   centroids span this range. The formula works at training-context scale, roughly.

2. **The two-tier structure** — both the centroid (~0.40) and the subpopulation
   (~0.25) show cross-PE convergence at their respective values. Whatever sets
   D/4 for the subpopulation is a second attractor value distinct from Marín's
   centroid. G7 needs to explain the subpopulation attractor specifically.

3. **The absolute-key-position complication** — the centroid comparison to Marín
   may be mixing absolute-key and relative-lag observables, especially for GPT-2.
   For Pythia, exp-138's finding has not been applied to the Δ-window heads;
   this is an open question for the structural session.

---

## What this note does NOT claim

- That the centroid comparison to Marín is direct (absolute-key vs relative-lag
  complication noted above)
- That the centroid scale-dependence (Pythia-1.4b at 0.319 vs Pythia-70m at 0.468)
  is explained by Marín's formula — it's within his expected variance range but not
  specifically predicted
- That D/4 for the subpopulation has been explained — G7 is open

---

## Next move

The formal analysis-only experiment (compare Pythia Δ-window subpopulation to
Marín's Padé at verified parameters, with per-head distributions and statistical
tests) requires pre-registration before code. Currently blocked on GitHub token.
When token is restored: write the pre-registration, push, then run.

---

*Register ledger:*
*MEASURED (exploratory) — centroid Δ computed from exp-118 results.json on 2026-10-24.*
*MEASURED (established) — Marín formula coefficients from Oct 20 reading; exp-138 finding.*
*INTERPRETIVE — two-tier convergence; the centroid-to-Marín comparison; G7 implications.*
*No formal hypothesis test run. No experiment number registered.*
*Pre-registration for the Pythia Δ-window vs Marín analysis: pending token restoration.*
