# exp-127 — Eigenvalue Distribution of Key Matrices: Δ-Window vs Control Heads

**Pre-registration.** This file was committed to attention-geometry before the analysis
script was written. The post-hoc finding from exp-126 is stated here as a pre-registered
prediction; computing against the existing eigenvalue data makes this analysis-only.

**Date pre-registered:** 2026-08-30  
**Analysis-only:** true (loads `eigenvalues_mean` from exp-126 results.json; no new model inference)  
**Seed experiment:** exp-126 (post-hoc eigenvalue finding, not a claim — flagged there as exploratory)

---

## Background

exp-126 measured the position Gram matrix G = K K^T / d_k for three head populations
in GPT-2 small: 16 Δ-window heads (text-native, exp-118), 5 structural heads
(positional-mean carriers, exp-112/117/122/123), and 16 non-window control heads.
The pre-registered directional hypothesis (Δ-window heads have higher off-diagonal
fraction ε than control) was falsified: ε ≈ 0.98 is universal across all heads (Kill K2).

Post-hoc eigenvalue analysis (exploratory, stated explicitly as not a claim):

| Population | λ₁/Σλ median | Supra-MP fraction | Mann-Whitney p |
|---|---|---|---|
| Δ-window (16) | 0.507 | 0.053 | ref |
| Control (16) | 0.651 | 0.035 | p=0.003 (share), p=0.001 (MP) |

Δ-window heads have *less* rank-1 concentration and *more* supra-bulk eigenvalues than
control heads. Control heads are dominated by positional/induction heads, which have a
single strong direction (λ₁/Σλ up to 0.77). This exploratory finding seeds exp-127.

---

## Hypothesis

Δ-window heads (text-native) have less rank-1-concentrated key structure and more
supra-MP eigenvalues than non-window control heads, reflecting a distributed rather
than directionally-focused key geometry. This distribution is specific to the
Δ-window population and is not a property of structural (positional-mean carrier)
heads, which are directional by construction.

---

## Predictions

**P1 (primary):** Median λ₁/Σλ for Δ-window heads < median λ₁/Σλ for control heads,
with Mann-Whitney p < 0.05. Pre-hoc estimate: 0.507 vs 0.651.

**P2 (primary):** Median supra-MP fraction for Δ-window heads > median supra-MP
fraction for control heads, with Mann-Whitney p < 0.05. Pre-hoc estimate: 5.3% vs 3.5%.

**P3 (secondary):** Structural heads (positional-mean carriers) have λ₁/Σλ *higher*
than both Δ-window and control (they are rank-1-dominant by construction — the
mean-score carrier profile is a single dominant direction).

---

## Kill Conditions

**K1:** If (median λ₁/Σλ control) − (median λ₁/Σλ window) < 0.05, P1 direction not
supported even if p < 0.05. Effect too small to be meaningful against the pre-hoc
estimate of 0.144.

**K2:** If Mann-Whitney p(λ₁/Σλ, window vs control) > 0.05, P1 not confirmed.

**K3:** If Mann-Whitney p(supra-MP fraction, window vs control) > 0.05, P2 not confirmed.

**K4:** If structural heads have λ₁/Σλ indistinguishable from control heads
(Mann-Whitney p > 0.05), this kills P3 and weakens the overall interpretation —
the directional control population is not as expected.

---

## Protocol

Load `research/physics/experiments/exp-126_key_covariance_structure/results.json`.
The `eigenvalues_mean` field for each head (length 128, averaged over 100 sequences)
contains the eigenvalues of G = K K^T / d_k where K ∈ R^{128×64}.

For each head:
1. **λ₁/Σλ:** max eigenvalue / sum of all eigenvalues. (Sums are over all 128
   eigenvalues; the 64 near-zero ones contribute negligibly to the sum but are included
   for exactness.)
2. **σ² estimate:** Σλ / 64 (rank of K K^T / d_k is ≤ 64; effective variance per
   non-trivial eigenvalue).
3. **MP upper edge:** λ_+ = σ² × (1 + √(64/128))² = σ² × (1 + 1/√2)².
4. **Supra-MP fraction:** fraction of all 128 eigenvalues strictly above λ_+.
   (Near-zero eigenvalues do not exceed λ_+ when σ² is set by the non-trivial sum,
   so they do not inflate the count; verify this.)

Statistical comparison: Mann-Whitney U test (two-sided) for each metric,
16 Δ-window heads vs 16 control heads.

All computation from saved data; no new model inference.

---

## Verdict criteria

| Outcome | Verdict |
|---|---|
| P1 and P2 both confirmed | confirmed |
| P1 confirmed, P2 not (or vice versa) | partial |
| Both falsified or a kill condition fires | falsified |
| Inconclusive due to numerical issues or other protocol failure | inconclusive |

---

## Connection to the program

The formation-requires-binding result (exp-091/092, ladder) established that arc-scale
referential binding at the *corpus* level is required for conformal geometry to form.
This experiment asks whether the difference lives in the *weights*: do Δ-window heads
carry a more distributed key structure than control heads? If yes, it is the first
within-model structural signature distinguishing the conformal population from the
control — a complement to the formation evidence at the training-corpus level.

Also relevant to the inter-head true-witness conjecture (inbox, Aug 23): if inter-head
signal reliability is structural, the within-head key distribution may be part of the
substrate that enables distributed multi-vector attending. Exploratory connection only;
not a registered prediction here.
