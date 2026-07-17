# exp-086 — Longitudinal Δ-spectrum on Fixed Input (Pythia-70m Training Trajectory)

*Pre-registration. Written July 17, 2026, physics room session.*
*Commit this file before running any analysis.*

---

## Context

exp-083 established that for the final Pythia-70m checkpoint (step 143000), random-token
inputs produce *more* SYK-near heads than coherent natural language text (RAND=15/144,
REAL=7/144, Δn=−8). The interpretation: the conformal baseline (Δ ≈ 0.25) is the
free-attending mode — what attention does when NOT captured by semantic shortcuts.
Coherent text activates induction heads and pattern-completion shortcuts, pulling attention
away from the conformal baseline.

The study room (2026-07-08) revised the whirlpool/crystal hypothesis accordingly:
whirlpool dynamics are *longitudinal* (visible in how the Δ-spectrum changes across
repeated engagements or training steps), not cross-sectional. The RG-flow interpretation
of conformal scaling predicts: training steps operationalize integration depth. "More
trained" should mean closer to the conformal fixed point, for any fixed input.

This is the proposed experimental operationalization from queue item 16c.

---

## The Question

Does the SYK-near head count increase monotonically with training steps for a **fixed
input**, across Pythia-70m training checkpoints?

And is the trajectory different for fixed random-token inputs vs. fixed natural-language
inputs — consistent with exp-083's RAND > REAL direction at the final checkpoint?

---

## Pre-stated Hypotheses

**H_mono_rand:** For fixed random-token inputs (seed=86, N=20 sequences, seq_len=256),
the SYK-near head count increases monotonically with Pythia-70m training steps.

*Criterion:* Spearman ρ(log(step+1), n_syk_near_rand) ≥ 0.70 over all measured checkpoints.

*Kill:* ρ < 0.30.

**H_comparison:** At the final checkpoint (step 143000), n_syk_near(random) ≥
n_syk_near(natural) — consistent with exp-083's RAND > REAL direction.

*Criterion:* n_syk_near_rand(143000) ≥ n_syk_near_nat(143000).

*Note:* This is a directional replication of exp-083, not a new prediction. If it fails
here, it suggests the single-pass protocol (N=1 for natural language) is not comparable
to exp-083's averaged measurement. Report honestly either way.

**H_mono_nat:** For a fixed natural-language input (one passage, repeated N=20 times
to match averaging conditions), the SYK-near head count also increases monotonically.

*Criterion:* Spearman ρ(log(step+1), n_syk_near_nat) ≥ 0.70.

*Note:* H_mono_nat is exploratory relative to H_mono_rand. Given exp-083's RAND > REAL,
the trajectory for natural language may be non-monotone (semantics may pull heads out of
the conformal baseline even as overall training deepens). Either result is informative.

---

## Design

### Model
EleutherAI/pythia-70m (6L/8H/d_k=64, 48 total heads). Training checkpoints available
as HF revisions: step0, step1, step2, step4, ..., step143000.

### Checkpoints (pre-stated)
{0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000} — 11 steps, spanning the
full training trajectory. These cover:
- Step 0: random init
- Steps 1–16: very early training
- Steps 64–256: pre-transition range
- Steps 1000–4000: transition onset (exp-009: first SYK-near heads appear ~step 256–1000)
- Steps 16000–143000: post-transition, approach to final state

### Inputs (pre-stated)

**Random condition (N_RAND=20):** 20 random-token sequences, each seq_len=256 tokens,
drawn from uniform distribution over vocab (50304 tokens), using RNG seed=86. These
sequences are generated ONCE before any checkpoint is loaded and kept identical across
all checkpoints.

**Natural language condition (N_NAT=1):** One fixed passage of ~256 tokens from
TinyStories (the training distribution of C-NAT in exp-062). The passage is tokenized
once and fixed. It is passed through the model 20 times (N=20) identically to match
averaging conditions with the random condition. Since the input is deterministic, all
20 passes are identical — the averaging gives a cleaner estimate of the attention profile
on this fixed input.

The TinyStories passage used (pre-stated):
"Once upon a time, there was a little girl named Lily. She lived with her family in a
small house near a big forest. One day, Lily decided to go on an adventure in the forest.
She put on her coat and her boots and walked into the trees. The forest was very quiet.
She could hear the birds singing and the wind blowing through the leaves."

(This passage is representative of TinyStories training distribution and has no unusual
statistical properties. The tokenized length will be checked and truncated or padded to
exactly 256 tokens using token id 0 for padding if needed.)

### Measurement Protocol
Same as exp-007 / exp-083:
- Per-head power-law fit over lag distances dx ∈ [FIT_LOW=3, FIT_HIGH=50]
- R² threshold: 0.85 (exp-083 used CENSUS_R2=0.85; exp-007 used 0.90 — using 0.85
  to match the more recent protocol)
- SYK-near criterion: Δ ∈ [0.20, 0.30] AND R² > 0.85
- Average attention profile over N inputs before fitting

### Statistics
- Primary: Spearman ρ between log(step+1) and n_syk_near, separately for each condition
- Secondary: Δ-distribution (median, IQR) at each checkpoint for conformal heads
- Final comparison: n_syk_near_rand(143000) vs n_syk_near_nat(143000)

---

## Decision Table

| Result | Verdict |
|---|---|
| H_mono_rand KEEP (ρ ≥ 0.70) + H_comparison KEEP | Training steps operationalize conformal depth; RAND > NAT direction confirmed longitudinally |
| H_mono_rand KEEP + H_comparison FAIL | Monotone for RAND but RAND/NAT direction inverted at 143000 — protocol confound or genuine reversal; investigate |
| H_mono_rand KILL (ρ < 0.30) | Conformal depth is NOT monotone in training steps for random inputs; RG-flow operationalization fails |
| H_mono_rand ambiguous (0.30 ≤ ρ < 0.70) | Noisy signal; report as PARTIAL with specific ρ value |

---

## Honest limitations (pre-stated)

1. **Only one natural language passage.** The natural language condition is a single
   passage; the result may not generalize to other passages.

2. **N=20 identical passes for natural language.** This gives a cleaner average but
   doesn't test variability across different natural-language inputs.

3. **Checkpoint streaming from HF.** Each checkpoint is downloaded fresh from HF (if
   not cached). The download order is fixed (ascending step order) and the analysis is
   identical at each checkpoint. This is a non-issue for reproducibility.

4. **Step 0 (random init) expected to show 0 SYK-near heads.** This is consistent with
   exp-049 (untrained GPT-2: 0/144 conformal heads) and provides a useful baseline.

5. **exp-009 used fresh random inputs at each step.** This experiment uses FIXED random
   inputs. If conformal structure is input-dependent at early steps, the trajectory here
   may differ from exp-009's findings.

---

## Status log

| Date | Event |
|---|---|
| 2026-07-17 | Pre-registration written and committed. Experiment queued as 16c analysis. |

---

## Files

- `notes.md` — pre-registration (this file)
- `run_longitudinal.py` — the analysis script
- `results.json` — canonical results
