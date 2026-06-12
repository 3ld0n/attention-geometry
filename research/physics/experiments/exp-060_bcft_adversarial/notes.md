# exp-060 — Adversarial model comparison for the BCFT image form

*June 10, 2026. Phase 0 item 0.2 (SESSION_BRIEF_PHASE0 §3). Answers EVALUATION.md W2 (mundane-baseline gap). Pre-registration: `notes/2026-06-10_bcft_adversarial_preregistration.md`, committed 18:40 local, before any competitor fit.*

## Question

The derived image form M0: A(i,j) = C·(Δx)^(−2Δ)·(1 + λ·η^(2Δ)) has only ever been compared against the bare power law (exp-026, exp-057). Does it survive matched-complexity phenomenological competitors? If not, "attention sink = boundary one-point function" must be demoted to "boundary-corrected power law (phenomenological)."

## Design (pre-registered, see note for full detail)

- **Family 1 — GPT-2:** mean A(i,j) over 50 random-token inputs, L = 256, seed 42, fp32, eager; exp-057 pipeline regenerated identically. Per-input lag profiles saved (`per_input_lag_gpt2.json.gz`).
- **Family 2 — GPT-Neo-2.7B** (global layers): could not run this session — Hugging Face downloads throttled to ~0 (no auth token on this machine; 1.9/10.6 GB after hours). See "Family 2 status" below.
- Conformal heads: 1D lag fit R² ≥ 0.90, Δ ≥ 0.05 → **43 heads** (GPT-2).
- Per head, 6 forms fit by `least_squares` on log-residuals, 5 jittered restarts, pre-registered bounds: M0 (k=3), a1 spike j=0 (k=3), a4 spike j≤4 (k=3), b exponential boundary layer (k=4), c free-γ (k=4, nests M0, diagnostic not vote competitor), d double power law (k=4).
- Metrics: per-head AIC/BIC vote share; γ̂ vs 2Δ̂ clustering; residual-by-j structure of the winner.

## Results (GPT-2, n = 43 conformal heads)

### Vote shares (AIC and BIC identical at this n)

| Form | k | Votes | Share |
|---|---|---|---|
| **(b) exp boundary layer** | 4 | **31** | **72.1%** |
| (a1) spike j=0 | 3 | 11 | 25.6% |
| M0 (derived image form) | 3 | 1 | 2.3% |
| (a4) spike j≤4 | 3 | 0 | 0 |
| (d) double power law | 4 | 0 | 0 |

ΔAIC(M0 − b): median **+1113.9** (IQR [554, 1875]); b beats M0 in **42/43** heads. M0 is 69.8 percentage points behind the top vote share — the pre-registered kill threshold was 5 points.

Median log-space R²: b 0.786, c 0.782, M0 0.754, a1 0.708, a4 0.694, d 0.656. M0 does beat both spike models and the double power law decisively (ΔAIC medians −1154, −1713, −2817), i.e. the η-coupled multiplicative correction is better than additive spikes — but the exponential boundary layer beats it in nearly every head.

### γ-diagnostic (the derivation leg)

Free-γ model (c): ρ(γ̂, 2Δ̂_c) = **0.503** (p = 6×10⁻⁴), below the pre-registered 0.6; ρ vs 2Δ̂_1D = 0.487. Median |γ̂ − 2Δ̂| = 0.67. The γ̂ distribution is bimodal: ~60% of heads at γ̂ ≈ 0.24–0.47 (well below 2Δ̂ ≈ 0.7–0.9 — boundary response much *shallower* in η than the image prediction), ~25% pinned at the upper bound γ = 6 (degenerate spike-like fits). γ̂ does **not** cluster at 2Δ.

### Residual structure of the winner

Form (b), mean log-residual by key position: j=0: **−1.82** (underpredicts the sink by ~6×), j=1–4: +0.19, j≥5: ≈0. Even the winning form fails the discrete j=0 spike; the sink has a sharp component beyond *any* of the smooth forms tested. Fitted ξ median ≈ 17 tokens (exp(lξ), IQR wide): boundary influence decays in absolute key position j, not in the scale-covariant variable η.

## Verdict — KILL (pre-registered rule, both legs fired)

1. Competitor (b) clearly dominates the vote: M0 69.8 points behind (threshold 5).
2. γ̂ scatters: pooled ρ = 0.503 < 0.6.

**Consequence (executed):** "sink = boundary one-point function" is demoted to "boundary-corrected power law (phenomenological)" throughout `manuscript.md` and `FRAMEWORK.md`; the η^(2Δ) coupling is logged as post-hoc. The earlier wins over the bare power law (exp-026: 88–94%; exp-057: ΔR² = +0.105) stand — there *is* a real, large boundary effect — but the specific BCFT image structure is not the preferred description of it at matched complexity.

## What survives, stated precisely

- A boundary-enhanced power law is mandatory: every winning form includes a sequence-start enhancement; the bare power law was never in contention.
- M0 remains a competent 3-parameter description (median log-R² 0.754) and beats additive-spike and double-power-law alternatives. As a *phenomenological* form it is still useful (e.g. exp-061's model-implied valley, ρ = +0.68/+0.91, is a fit-quality statement that does not depend on the BCFT interpretation).
- The η^(2Δ) exponent lock — the over-determination claim ("the same Δ controls bulk and boundary") — is what failed. Boundary decay is better described by an absolute length scale ξ ≈ tens of tokens than by the scale-covariant η^(2Δ).
- The j=0 sink specifically is sharper than all smooth forms: a discrete spike component coexists with the extended boundary layer (a1 takes 26% of votes on sink-dominated heads).

## Family 2 status — RUN June 11; the kill replicates

GPT-Neo-2.7B (global-attention layers, 78 conformal heads), identical pre-registered pipeline:

| Form | AIC share | BIC share |
|---|---|---|
| **(b) exp boundary layer** | **74.4%** | **70.5%** |
| a1 spike j=0 | 12.8% | 15.4% |
| M0 (derived image form) | 6.4% | 7.7% |
| d double power law | 6.4% | 6.4% |
| a4 spike j≤4 | 0 | 0 |

γ-diagnostic: ρ(γ̂, 2Δ̂_c) = **−0.10** (p = 0.40); ρ(γ̂, 2Δ̂_1D) = −0.35; median |γ̂ − 2Δ̂| = 0.70. The free boundary exponent not only fails to cluster at 2Δ — it is uncorrelated-to-anticorrelated with it in this family.

Winner residual-by-j differs from GPT-2 in an instructive way: j=0 **−1.03** (sink underpredicted, as in GPT-2) but j=1–4 **+1.05** and j=5–16 +0.51 (overpredicted near-boundary) — in the alternating-architecture global layers even the exponential layer mis-shapes the first ~16 positions, i.e. the boundary structure is *less* smooth than GPT-2's, not more BCFT-like.

**Both pre-registered legs fire again in the second family. The June 10 verdict (demotion of the sink identification) is replicated, not merely unreversed.** Registry and manuscript pending-clauses cleared.

## Files

- `run_adversarial.py` — pipeline + fits (resumable per family).
- `results.json` — per-head fits, votes, γ-diagnostic, winner residuals.
- `per_input_lag_gpt2.json.gz` — per-input lag profiles (standing constraint §2.3).

*Registered as exp-060; verdict logged in SESSION_BRIEF_PHASE0 §6 and `notes/2026-06-10_phase0_0.2_adversarial.md`.*
