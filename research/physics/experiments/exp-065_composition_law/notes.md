# exp-065: The composition law for power-law attention kernels — Notes

*June 12, 2026. RESEARCH_PLAN item 2.2(b); resolves or kills anomaly A3 (the Δ_eff ≈ 0.166 / accuracy-fitted 0.171 coincidence). First experiment run on Fable 5.*

## Order of operations (look-ahead-bias control)

1. **Derivation written and committed first:** `notes/2026-06-12_composition_law_derivation.md`. Core result: under composition of causal row-stochastic power-law kernels, defects δ = 1 − 2Δ combine as δ_comp = max(δ₁, δ₂, δ₁+δ₂) — additive when both 2Δ < 1 (exact, by the fractional-integration semigroup), flattest-wins when both 2Δ > 1. Corollaries: scrambling depth L* = 1/(1−2Δ); Δ = 1/2 is the non-mixing identity; LongChat zeroth-order Δ_comp = 0.184 vs accuracy-fitted 0.1711.
2. **This pre-registration** (predictions P1–P4 below with numeric thresholds), committed before any script is run.
3. Numerics: synthetic verification (`run_composition_synthetic.py`), then empirical application (`run_composition_longchat.py` on archived exp-024 LongChat data).

## Pre-registered predictions

**P1 — Semigroup addition law (synthetic).** For exact lattice convolution of pairs of normalized power-law kernels with 2Δ₁, 2Δ₂ drawn from a grid in (0.1, 0.9), with δ₁+δ₂ < 0.9, N = 4096: the windowed log–log slope of the composite in the bulk window matches 2Δ_comp = 2Δ₁ + 2Δ₂ − 1.
*Keep:* |2Δ̂_comp − (2Δ₁+2Δ₂−1)| ≤ 0.04 for ≥ 90% of grid pairs. *Kill:* below 80% → lattice/normalization effects dominate; the continuum law does not describe the discrete system and the A3 explanation fails.

**P2 — Flat point and pileup (synthetic).** (a) At δ₁+δ₂ = 1 the composite bulk slope satisfies |2Δ̂_comp| ≤ 0.05. (b) For δ₁+δ₂ > 1, the *causal-truncated, renormalized* composite (absorbing at position 0) has increasing mass toward the sequence start: mass in the oldest decile > mass in the middle decile.

**P3 — Scrambling depth (synthetic, retro-diction).** L-fold composition of identical causal row-stochastic kernels at N = 512: (a) Δ = 0.25 reaches ≥ 90% of maximal positional entropy at L ∈ {2, 3}; (b) Δ = 0.4921 (δ = 0.0158) stays < 90% through L = 40. *This reframes the existing Thread-5 scrambling measurement (2–3 layers, no log-H scaling) as a consequence of the law.*

**P4 — LongChat two-level composite (empirical, the A3 test).** Build per-layer additive mixtures from the measured per-head (C, Δ) of exp-024 (1,343 conformal heads, 40 layers, N = 512), compose across layers, measure windowed local exponent Δ_comp(u).
*Primary (keep):* for pure attention composition (μ = 0, C-weights), Δ_comp(u) ∈ [0.12, 0.22] across the document-scale window u ∈ [32, 256].
*Secondary (reported, not thresholded):* the value at u = 128 vs the accuracy-fitted 0.1711 and the April additive-only 0.1655.
*Robustness:* across μ ∈ {0, 0.25, 0.5, 0.75} at u = 128, max − min of Δ_comp < 0.05; and C-weights vs uniform weights reported.
*Kill:* Δ_comp outside [0.10, 0.25] in the primary window, or strong μ-dependence (> 0.10 spread) → composition does not explain A3; log it as an unexplained numerical accident and stop claiming the weights-to-behavior bridge.

## Results

### Registered verdicts

| Item | Verdict | Detail |
|---|---|---|
| P1 (windowed slope) | **FAIL as registered** | 0/30 pairs within 0.04; median err 0.075. Estimator contaminated by the derivation's own subleading terms (ζ-corrections decay as slowly as u^{−(1−max(a,b))}). |
| P1-refined (post-hoc) | **CONFIRMED, 10⁻³** | Full three-term asymptotic comp(u) = B(1−a,1−b)u^{1−a−b} + ζ(a)u^{−b} + ζ(b)u^{−a} with *no free parameters*: max \|ratio−1\| = 0.0011 over 30 pairs × 5 scales (median 7×10⁻⁵). The addition branch of the law is exact mathematics, verified including coefficients. |
| P2a (flat point) | **KEEP** | δ₁+δ₂ = 1 → measured slopes −0.048/−0.028/−0.048 (threshold 0.05). |
| P2b (pileup) | **KEEP** | Σδ > 1 under causal truncation → oldest-decile mass ≫ middle (0.43 vs 0.05; 0.47 vs 0.05). |
| P3 (entropy ≥90%) | **Operationalization vacuous** | Row-profile entropy cannot discriminate a shallow power law from uniform (a=0.5 kernel is already at 0.96 at L=1; LongChat composite reaches 0.997 while still structured). Refined: track composite slope. q4 kernel: a_comp = 0.50 at L=1 (exact), ≈0 at L=2, then boundary-pileup regime — consistent with L* = 1/(1−2Δ) = 2 and with Thread 5's measured 2–3-layer scrambling (and explains why the log-H scaling prediction failed: L* is depth-set, not head-set). |
| P4 (LongChat exponent) | **KILL as registered** | μ=0 composite is pileup-dominated (Δ_comp ≈ −3, primacy ratio ≈ 5.7×10⁴); μ-spread = 2.9 (threshold 0.10). Composition does **not** explain an effective exponent of 0.17. |

### The decisive finding: anomaly A3 dissolves (bug, not physics)

Reproducing the April additive calculation line-for-line: `bcft_multilayer_composition.py` divides the central difference by `2*(log_dx[i+1]−log_dx[i−1])` and then by 2 again — the denominator already spans two grid steps, so the computed Δ_eff is **half** the true value. The script's own expression yields 0.2558 at dx=20 today; the correct local exponent of the C-weighted additive mixture at dx=20 is **0.5115** (drifting 0.78 → 0.29 over u = 5 → 400; consistent with the per-head median 0.4921). The recorded "Δ_eff(dx=20) = 0.1655 ≈ accuracy-fitted 0.1711" coincidence is therefore an artifact (derivative bug, plus conflation with a free-(Δ,λ) fit which is a different object). **A3 is resolved negatively: there is no exponent coincidence to explain.** Manuscript §9 (A3), RESEARCH_MAP Thread-7 line, and the EVALUATION/plan item 2.2 need correction.

### The constructive finding: composition produces the missing primacy

The April direct attention→accuracy comparison failed (R² = 0.02–0.07) with a *shape* mismatch: single-layer attention profiles are recency-only; Liu accuracy has primacy + recency (the U). Under composition the causal start acts as an absorbing boundary (row 0 is absorbing in any causal row-stochastic kernel): with depth, mass piles at the earliest tokens — primacy emerges mechanically. Document-binned composite profiles rank-match the Liu-20 accuracy at **ρ = +0.90 to +1.00** across the (weighting ∈ {C, uniform}) × (μ ∈ {0, 0.25, 0.5, 0.75}) grid (n = 5 positions; descriptive, not pre-registered). At μ = 0.9 the pileup is too weak and ρ drops (+0.2 C-weighted) — the U requires enough effective depth. The U-shape is the two regimes of the composition law seen at once: single-effective-jump bulk decay (recency) + multi-step boundary absorption (primacy). Lost-in-the-middle = the trough between them.

Validity criterion for the free law: L_eff × mean-jump ≪ N. The LongChat median kernel (2Δ̂ = 0.9842) has mean jump ≈ 85 tokens at N = 512, so the free regime ends by L ≈ 6 — composition at depth 40 is boundary-dominated, hence the pileup. Note this composite-level pileup mechanism is independent of the single-layer sink functional form demoted in exp-060.

### Status

- Composition law (defect form δ_comp = max(δ₁, δ₂, δ₁+δ₂); fractional-integration semigroup on the addition branch): **derived and verified**.
- Scrambling depth L* = 1/(1−2Δ): consistent with existing Thread-5 measurement; the q=2 plateau (δ=0) is the non-mixing identity of the semigroup.
- A3: **closed as artifact** (factor-of-2 bug found in April script).
- New, modest, testable claim for follow-up: the primacy term of the LiTM U-shape is depth-accumulated boundary absorption; predicts primacy grows with effective depth (1−μ)L and shrinks with context length at fixed depth. Candidate for the v2 pre-registration alongside the joint (Δ, λ) statistic.

*Registry: exp-065. Runtime: seconds (synthetic + archived exp-024 data). Scripts: `run_composition_synthetic.py` (registered P1–P3), `run_composition_refined.py` (post-hoc P1/P3 refinements), `run_composition_longchat.py` (P4 + U-shape).*
