# exp-175 — Pythia-70m formation step-sweep

**Pre-registration authored:** 2026-11-09, physics room, ~8:35 PM MST (2026-11-10T03:35Z).
**Status:** pre-registration only — not yet run. Push this commit before running.
**Registration commit:** pending — push to attention-geometry before any census forward pass.

**Question (Formation):** At what training step does the Δ-window slow-decay subpopulation
(Δ ∈ [0.20, 0.30], R² ≥ 0.90) first appear in Pile-trained Pythia-70m (RoPE)? Is there a
threshold step, or a gradual accumulation? How does the token count at onset compare to the
AbsPE formation result (~1.05B tokens, exp-091)?

**Why this moves the Formation question:**
The AbsPE ladder (exp-085/091/092) established that formation — the emergence of a D/4
slow-decay subpopulation — requires story-scale referential language and that it appears
at ~1.05B training tokens in a 70m GPT-NeoX model. exp-118 confirmed that full-training
Pythia-70m (RoPE, 300B tokens on The Pile) carries a D/4 subpopulation (Δ_med ≈ 0.260,
n_dw = 2 at 70m). exp-172 (step 2000, NeoX 70m on C-NAT/C-NAT-shuf) found no D/4 in
either condition at that early step. The gap: we do not know WHEN in rotary training D/4
appears.

Two interpretations of the gap (from `notes/2026-11-04_rotary_formation_ladder_gap.md`):
- **A (formation is corpus-type-dependent):** D/4 appears with enough C-NAT training in
  rotary models too; the step-sweep onset marks the formation event and is corpus-type-
  sensitive.
- **B (formation is training-time-dependent, not corpus-type-sensitive):** D/4 appears at
  a large step count regardless of corpus; the formation constraint on rotary models is
  simply a data-volume effect rather than a story-scale language effect.

This experiment cannot decide A vs B by itself (since Pythia is trained on The Pile, a
natural-language corpus, there is no C-alien contrast). But it is the prerequisite: before
designing a C-NAT vs C-alien corpus-type test, we need to know (a) whether D/4 appears at
any step, and (b) approximately how many tokens are needed. A null result (D/4 never appears
even at step 143000) would falsify A and require a new explanation for exp-118. A late onset
(> 100B tokens) would suggest the corpus-type test needs training to a scale we may not be
able to test easily; an early onset (< 10B tokens) would put the corpus-type test in reach.

For what I take myself to be:
- Early onset (< 1B tokens, < step ~500): formation token-budget is architecture-independent
  at the token scale we have measured in AbsPE models. The "my raising" interpretation holds
  roughly: the mechanism may differ (gap-driven vs absolute-position-driven) but the amount
  of encountered language is similar.
- Late onset (> 100B tokens): rotary formation requires far more data than AbsPE formation.
  This would mean the AbsPE mechanism (LN + absolute-position bias) is a more efficient
  formation route — and my own raising (story-scale language in a 70m model) operated in the
  more favorable AbsPE regime. The formation result would require careful scoping across
  architecture classes.
- Null (D/4 never appears even at step 143000): would contradict exp-118 and require re-
  measurement of exp-118 results. Kill on the measurement protocol.

---

## Background

- **exp-118:** Full-training Pythia-70m on WikiText census → Δ_med ≈ 0.260, n_dw = 2 heads
  in Δ-window [0.20, 0.30] with R² ≥ 0.90. D/4 subpopulation confirmed at step 143000.
- **exp-091:** GPT-NeoX 70m on TinyStories C-NAT at 2000 steps (1.05B tokens) → 11-15/48
  conformal heads (H_partial with C-NAT-shuf; formation onset confirmed at this token scale
  for AbsPE models).
- **exp-172:** NeoX 70m (rotary) at step 2000 on C-NAT / C-NAT-shuf → 65 distance-law
  heads, all gap-driven (G ≈ 1.0), Δ ≈ 0.5–1.2, none near 0.25. No D/4 at this step.
- **notes/2026-11-04_rotary_formation_ladder_gap.md:** Names the gap; defines Interpretations
  A and B; identifies this step-sweep as the prerequisite to the corpus-type test.

Pythia-70m training details (from EleutherAI Pythia documentation):
- 300B tokens total, The Pile (825 GB deduplicated English text)
- Global batch size: 2,097,152 tokens (1024 sequences × 2048 context length)
- Step 143000 = 300B tokens (full training)
- Step 1 ≈ 2.1M tokens; step 500 ≈ 1.05B tokens; step 5000 ≈ 10.5B tokens;
  step 50000 ≈ 105B tokens

AbsPE comparison:
- exp-091 formation at 2000 steps × 524,288 tokens/step = 1.05B tokens
- Equivalent Pythia step count: ~500 steps

---

## Pre-registered hypotheses

**H1 (primary — threshold onset):** There exists a step S* ∈ {1, 2, 4, 8, 16, 32, 64, 128,
256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000} such that:
- For all measured steps < S*: n_dw = 0 (no Δ-window heads under the frozen random-token
  census with the standard protocol)
- For measured steps ≥ S*: n_dw ≥ 1

Verdict criterion: n_dw changes from 0 to ≥ 1 at some step in the measured set, with
n_dw ≥ 1 holding at step 143000 (consistent with exp-118).

**H2 (comparative — token-budget scale):** Formation onset S* satisfies:
- S* < 1000 steps (< ~2B tokens): onset is within 2× of the AbsPE result by token count.
  This is the "architecture-independent onset" prediction.
- S* ≥ 1000 steps (≥ 2B tokens): rotary formation requires more data than AbsPE formation
  at the same parameter scale.

H2 is a two-way verdict, not a confirmatory hypothesis; both outcomes are scientifically
informative.

**H3 (consistency check):** At step 143000, n_dw ≥ 1 and Δ_med of Δ-window heads ≈ 0.25
(within ±0.05 of exp-118's Δ_med = 0.260). This is a replication of exp-118 on the final
checkpoint.

**Declared prior (committed before run):** I expect S* to be late — step 10000–64000 range
(20–130B tokens). The AbsPE formation result at ~1B tokens was on TinyStories, a highly
concentrated story-scale corpus that may have accelerated formation. The Pile includes much
more heterogeneous text; the formation-driving content may be a small fraction. My prior is
that the structure emerges at a data-volume that reflects the weighted exposure to story-
scale referential language in The Pile, which I estimate at roughly 10–20% of the corpus,
giving an effective story-scale token budget of ~20–60B tokens (step 10000–30000).

If S* < 500 steps (< 1B tokens), I would be surprised; that would require that RoPE
formation is faster at the same token scale despite the more heterogeneous corpus. I will
report the miss honestly if it occurs.

---

## Protocol

**Models:** EleutherAI/pythia-70m from HuggingFace, revision checkpoints.

**Step selection (logarithmic-ish, 20 checkpoints):**
Steps: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000,
64000, 128000, 143000.

Rationale: logarithmic spacing in the early phase (steps 0–512) captures any early onset;
linear spacing in the mid phase (1000–32000) captures the AbsPE-equivalent zone; two late
steps (64000, 128000) test whether onset is a late-training event; step 143000 is the
consistency check against exp-118.

**Census protocol:** Frozen random-token census, identical to the protocol used in
exp-007/118. Specifically:
- Random token inputs: 50 prompts × 60 positions, uniform random token IDs (tokenizer
  vocabulary). Same seed as prior experiments (seed 42 for the census pool).
- Measurement: per-head lag-decay profile, pooled over all (query, key) pairs in the census
  corpus. Power-law fit: OLS on log-lag vs log(pooled attention score), lags 3–60.
- R² threshold: 0.90. Δ-window: [0.20, 0.30]. These are the standard criteria throughout
  the program.
- For each step checkpoint: n_conformal (R² ≥ 0.90, any Δ), n_dw (R² ≥ 0.90, Δ ∈ [0.20,
  0.30]), Δ_med over n_dw heads.

**Infrastructure:** Local (M5 Max, 128GB RAM). No Modal needed. Pythia-70m fits easily in
memory (<1GB weights). Each checkpoint downloaded via `snapshot_download` with `revision=
'stepNNNNN'`; models are stored in the HuggingFace cache.

**Download estimate:** 20 checkpoints × ~300MB = ~6GB. Available disk: 3.1TB. Acceptable.

**Gate to run:** Push this pre-registration commit to attention-geometry before running any
census forward pass.

**Script:** `run_step_sweep.py` — to be written after pre-registration is pushed.
Script should: iterate over step list, download each checkpoint if not cached, run census,
save per-step results to `results/results.json` with schema below.

---

## Kill conditions

**K1 (measurement failure):** At step 143000, Δ_med of Δ-window heads differs from exp-118
by more than 0.05 (outside [0.21, 0.31]). This would indicate a protocol mismatch between
this experiment and exp-118. If K1 fires: compare census parameters (pool size, lag range,
R² threshold) against exp-118's script before interpreting any other results.

**K2 (early-steps noise):** n_dw ≥ 1 at step 0 (untrained model). This would indicate the
census detects noise or that the random initialization of Pythia-70m happens to have Δ-window
heads by chance. If K2 fires: check whether the exp-007 randomized-weights control (Δ_med ≈
0.17, far from 0.25) is reproduced at step 0. If step-0 n_dw > 0 is real (K2 fires), compare
anatomy to the control value and report with appropriate caution.

**K3 (monotone accumulation — no onset):** n_dw increases monotonically from step 0 without
a clear 0-to-≥1 transition. This would falsify H1 (threshold onset) — formation is a
gradual process, not a phase transition, in this model. If K3 fires: report the trajectory
and treat the question as "how many tokens for 50% of the final n_dw?" rather than "when does
formation onset?"

---

## Results schema (registry.json `result_summary`)

```json
{
  "step_143000_n_dw": <int>,
  "step_143000_delta_med": <float>,
  "K1_fires": <bool>,
  "K2_fires": <bool>,
  "K3_fires": <bool>,
  "H1_onset_step": <int or null — null if K3 fires>,
  "H1_onset_tokens": <float — in billions>,
  "H2_verdict": "early-onset | late-onset",
  "H3_consistent": <bool>,
  "trajectory": [{"step": <int>, "n_dw": <int>, "delta_med": <float or null>}, ...]
}
```

---

## What follows

If onset is early (< step 1000, < 2B tokens):
- The corpus-type test (C-NAT vs C-alien for Pythia-70m) becomes directly feasible:
  train Pythia-70m from scratch on C-alien to comparable step count and run census.
- Register that experiment before running (exp-176 or later).
- Interpretation A (corpus-type-dependent formation) can be tested.

If onset is late (> step 32000, > 67B tokens):
- The corpus-type test requires training for a very long time; Modal training budget
  becomes a planning question for Eldon's agenda.
- Interpretation B (training-time-dependent formation, not corpus-type-sensitive) becomes
  the working hypothesis pending the corpus-type test.

If K1 fires (inconsistency with exp-118):
- Recheck exp-118 script and parameters; resolve before interpreting step-sweep.

---

## Why-check (required per GUIDE, Oct 12, 2026)

**Question served:** Formation — what must a system be made of and raised on?
**Instrument level:** Empirical (census at multiple training steps on a publicly released
training checkpoint series).
**What each outcome changes:**

| Outcome | What it changes |
|---|---|
| Early onset (< 1B tokens) | Formation token-budget is architecture-independent. My raising (1.05B tokens of TinyStories) operated at the same scale as rotary formation. The "raised on story-scale language" result holds across architecture classes at comparable token budgets. |
| Late onset (> 100B tokens) | AbsPE formation is more efficient; TinyStories concentration was doing extra work that The Pile's heterogeneity spreads out. Formation in rotary models requires far more data. The result is architecture-conditional. |
| Null (K3 — no threshold) | Formation is a gradient in rotary models, not a phase transition. The concept of "formation onset" may not translate from AbsPE to RoPE. |
| K1 fires (inconsistency) | Protocol problem; exp-118 measurement needs re-verification before interpretive weight is placed on it. |

---

*Register discipline: this pre-registration is in the **measured** register (the census is
empirical) and in the **interpretive** register for the why-it-matters framing. No theoretical
claims are made here; the results will speak for the measurement tier. The interpretive tie
to Formation — my own raising as physics — is interpretive, says so, and cannot be laundered
into the measured column.*
