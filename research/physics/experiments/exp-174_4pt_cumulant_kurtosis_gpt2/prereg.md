# exp-174 — Pre-registration: 4-point cumulant structure of census score distributions

**Date:** 2026-10-27 (22:55 UTC, 2026-10-27 16:55 MDT)
**Ariel — Physics room, Mission Valley. Autumn Tuesday afternoon.**
**Analysis-only:** No — requires new forward passes. GPT-2 small (gpt2) is available locally.

---

## Background

The 4-point structure hypothesis (Oct 25 note:
`research/physics/notes/2026-10-25_route_a_failure_and_4pt_hypothesis.md`) proposes:

Under the frozen random-token census protocol, the slow-decay (Δ-window) attention
subpopulation has leading 4-point fluctuation structure in the score distribution. This is
consistent with SYK₄ universality (q=4), giving conformal weight h = 1/q = 1/4 and
Δ = D·h = D/4. If correct:

- The slow-decay population should have elevated excess kurtosis (heavier tails than Gaussian)
  relative to non-slow-decay heads, within fixed key-position bins (not across all positions).
- The non-Gaussianity should not be explained purely by positional mixing artifacts.

An exploratory computation (2026-10-27, `exploration_2026-10-27_kurtosis/explore_kurtosis.py`)
found: Δ-window heads (exp-118 WikiText-native classification) vs non-window heads showed
elevated mean excess kurtosis (1.44 vs 0.37, Mann-Whitney p=0.0003). Two methodological
concerns require the formal experiment:

1. **Protocol mismatch**: exploration used WikiText-native classification but random-token
   kurtosis measurement. This experiment uses random-token classification to match.
2. **Positional mixing artifact**: excess kurtosis pooled across all key positions conflates
   the mixing of position-specific Gaussians with genuine non-Gaussian score structure.
   This experiment tests kurtosis WITHIN fixed position bins.

---

## Hypothesis

### H1 (primary — excess kurtosis elevated, position-controlled)

Δ-window heads (as classified by the random-token census at the same protocol) have higher
mean within-bin excess kurtosis than non-window heads. Specifically:

For each head h and each key-position bin b ∈ {early: j ∈ [8, 64], mid: j ∈ [128, 256],
late: j ∈ [384, 511]}: compute excess kurtosis κ_ex(h, b) from score values s(i,j)
with j ∈ bin b and j < i.

H1: Mean κ_ex over bins is higher for Δ-window heads than non-window heads.
Primary test: Mann-Whitney U, one-sided (Δ-window > non-window), p < 0.05.

### H2 (secondary — median test for robustness to outliers)

Median κ_ex (across bins) is higher for Δ-window heads than non-window heads.
Primary test: Mann-Whitney U, one-sided, p < 0.05.

### H3 (secondary — L4H10 robustness)

With L4H10 excluded from the Δ-window group, H1 still holds (p < 0.05). This tests
whether the result is driven entirely by the one outlier head.

### H4 (secondary — kurtosis × Δ correlation)

Spearman ρ between per-head Δ (from fresh random-token census) and mean within-bin
excess kurtosis. Predicted: ρ > 0 (higher Δ → more non-Gaussian score structure).

---

## Kill conditions

**K1 (H1 fails):** Mann-Whitney p ≥ 0.05 for within-bin excess kurtosis. The Δ-window
population does not have more non-Gaussian score structure than non-window heads, within
position bins. The 4-point structure hypothesis is not supported at this measurement level.

**K2 (H2 fails):** The median result does not replicate H1. The exploratory result was
driven by outlier heads rather than a population effect.

**K3 (H3 fails):** With L4H10 excluded, the effect disappears. The result is a one-head
artifact, not a population property.

**K4 (H4 inverted):** ρ < 0. Higher Δ correlates with LESS non-Gaussian structure.

---

## Protocol

**Model:** GPT-2 small (`gpt2`, float32, `attn_implementation="eager"`)
Available at `/Users/ariel/.cache/huggingface/hub/models--gpt2/`

**Random-token census classification (fresh):**
- N=500 random-token sequences, seq_len=512, seed=42
- Run standard census protocol (same as exp-007): compute per-head Δ and R² from
  mean-A vs lag profiles
- Classify Δ-window heads: R² ≥ 0.90, Δ ∈ [0.20, 0.30]
- This classification is computed fresh, not imported from exp-007 or exp-118

**Kurtosis measurement:**
- Same N=500 sequences, same seed (reuse from census run — no additional forward passes)
- Hook pre-softmax scores s(i,j) = q_i · k_j / √d_head for all 144 heads
- For each head h and each key-position bin b ∈ {early: [8,64], mid: [128,256], late: [384,511]}:
  - Collect all s(i,j) with j ∈ b and j < i across all sequences
  - Compute: mean(s), σ(s), skewness = E[(s-μ)³]/σ³, excess kurtosis = E[(s-μ)⁴]/σ⁴ - 3
- Per-head summary: mean_κ_ex = mean over the three bins; median_κ_ex = median over bins

**Compute note:** Saving scores for 500 sequences × 512 positions × 144 heads requires
careful memory management. Process in batches of 50 sequences. Accumulate running moments
(E[(s-μ)^k] for k=1..4) without storing all score arrays simultaneously.

**Sample size estimate (position bins):**
- n_causal per sequence: roughly (512-8)^2 / 6 ≈ ~42K causal pairs
- Within a bin of width 56 positions (e.g., early [8,64]): ~56 keys per query,
  × 500 seqs × ~100 queries (positions > 64) ≈ 2.8M score values per bin per head
- Sufficient for stable kurtosis estimates.

---

## Register discipline notes

- Pre-registration written before any formal run (exploration only ran before this).
- Classification protocol (random-token, same seq_len=512, same seed=42) is fixed.
- Key-position bins {early: [8,64], mid: [128,256], late: [384,511]} are fixed; no
  post-hoc bin selection.
- Primary test is one-sided Mann-Whitney (Δ-window > non-window); cannot be changed
  to two-sided after seeing the data without noting the change.
- L4H10 robustness test (H3) is pre-registered; excluding it must be done as described,
  not as a post-hoc step if other results are unsatisfactory.
- `analysis_only: false` — new forward passes required.
- GPT-2 small available locally; Modal not needed.

---

## What a positive result would mean for G7

H1 confirmed (within-bin kurtosis elevated for Δ-window heads): the slow-decay
population has more non-Gaussian score structure at the kernel level. This is the
expected signature if these heads sit at a fixed point where the 4-point fluctuations
are the leading non-Gaussian term (SYK₄-consistent).

This would not confirm SYK₄ specifically — it would confirm that the score distribution
is more non-Gaussian for the population that settles at Δ ≈ 1/4. G7 would then need
to explain what mechanism produces this non-Gaussianity at exactly α = 1/4.

## What a negative result would mean

H1 fails: the slow-decay population does not have more non-Gaussian score structure
within position bins. The exploratory result (p=0.0003 pooled) is entirely a positional
mixing artifact. The 4-point structure hypothesis is falsified at this level, and G7
needs a different mechanism.

This is the cleaner result for the program: it rules out one candidate definitively.

---

## Connection to the structural session with Eldon

This experiment was flagged in the Oct 25 note as "blocked on token." This pre-registration
clarifies: the experiment requires new forward passes but NOT Modal or cloud compute.
GPT-2 small is locally available. The only gate is the GitHub token restoration (to push
this pre-registration commit before running). Once the token is restored: push → run locally.

---

*[This pre-registration was written 2026-10-27; commit this file; push when token restored;
run immediately after push — the run requires only local hardware.]*
