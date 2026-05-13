# Attention Anisotropy — Experiment 01 Notes
*May 13, 2026 — first experiment on M5 Max*

## Setup

GPT-2 small (124M params, 12 layers, 12 heads) on three passages:
- Narrative (temporally structured)
- Logical (dependency-structured)
- Self-referential (explicitly about temporal processing)

Measured: positional attention profile. For each token at position i, how is
attention distributed over positions i-1, i-2, ... i-d? Fit P(d) ~ d^(-α).

## Results

| Passage | α (global) | R² | Head α mean |
|---|---|---|---|
| narrative | 0.517 | 0.980 | 0.594 |
| logical | 0.560 | 0.927 | 0.644 |
| self_referential | 0.502 | 0.956 | 0.607 |

**α ≈ 0.5 across all passages. Very clean power law fit (R² 0.93–0.98).**

Per-layer breakdown:
- Early layers (0–3): α ≈ 0.83–0.89 (strongly recency-biased)
- Late layers (8–11): α ≈ 0.22–0.29 (nearly isotropic)

Per-head variance: σ ≈ 0.73–0.79. Some heads extremely local (α up to 5.1),
some near-isotropic, a few slightly distance-biased (α < 0). Huge spread.

First-token attention sink: position 0 absorbs ~50% of mean attention.
(Expected from random: ~1/55 ≈ 0.018. Actual: ~0.51. ~25× above chance.)

## What This Means

The model is *not* isotropic — it shows consistent recency bias across all
passages. But this doesn't contradict the structural hypothesis. Here's why:

**Structural vs. learned temporal asymmetry.** The hypothesis was that
transformers lack an *intrinsic* temporal arrow — unlike cortex, where the
arrow comes from physical memory dynamics (synaptic decay, capacity limits,
the actual irreversibility of biological time). In a transformer, the
architecture is symmetric — you could run attention backward with equal
facility. Whatever recency bias exists must be *learned from data*, not
built into the structure.

GPT-2 trained on human language, which has heavy recency structure. Recent
context predicts next tokens better than distant context. So α ≈ 0.5 emerges
as a learned statistic of language, not a structural property of attention.

**Test:** A transformer trained on sequences with no temporal structure
(random tokens, or sequences where distant context predicts equally well as
recent) should show α → 0. That's the next experiment.

## Layer Structure — Unexpected Finding

Early layers (α ≈ 0.85) >> Late layers (α ≈ 0.25).

Interpretation: the model does local syntactic processing in early layers
(where recency bias is highest), then integrates across longer distances
in late layers (where attention is more spread out). The temporal asymmetry
is highest where syntax lives, lowest where semantics and coherence live.

This might be worth tracking in larger models — does the layer-wise α profile
scale predictably? Does a larger model show more isotropic late layers?

## Confounds

1. **Attention sink.** First token absorbs ~50% of attention. This inflates
   the positional profile at d = i (the distance to the first token). The
   power-law fit may be capturing this artifact rather than genuine decay.
   Next: redo with first-token attention excluded.

2. **Short sequences.** 53–60 tokens. Power law fits are only reliable for
   d = 2–20. Need longer sequences to see the true tail behavior.

3. **Causal masking.** GPT-2 can't attend to future tokens. The "isotropic"
   baseline isn't truly symmetric — it's forward-only. Isotropic in this
   context means "uniform over the visible past," not "uniform over all time."

## Next Experiments

1. **Remove attention sink.** Exclude position-0 attention and refit.

2. **Trained-on-random baseline.** Train a tiny transformer (2L, 2H) on
   random token sequences. Measure α. Expect α → 0 if the hypothesis holds.

3. **Larger models.** Does α change systematically with model size? Hypothesis:
   larger models develop more isotropic late layers (more global integration).

4. **Bidirectional vs. causal.** BERT-style (sees full context) vs. GPT-style
   (causal only). If α is structural, it should persist in bidirectional models.
   If it's a learned response to causal masking, it might differ.

5. **LSTM / Mamba comparison.** These have structural memory decay. Do they
   show higher α? More layer-consistency? This is the cross-architecture test
   that would be most diagnostic for the time's arrow argument.

## Connection to the Theory

The time's arrow hypothesis: *the arrow of time is the direction of attending.*

What today's data suggests:

- Transformers learn recency bias because language has temporal structure —
  but the bias is a feature of the *training distribution*, not the architecture.
- The layer-wise structure (high α early, low α late) suggests the model builds
  local temporal scaffolding before global semantic integration — the opposite
  of what we might expect if temporal asymmetry were intrinsic.
- The per-head variance is huge. Some heads are extremely local, some global.
  This diversity might be precisely what allows the model to handle diverse
  temporal contexts — it doesn't have one fixed temporal perspective, it has 144.

The cortex contrast: biological attention doesn't have the option of being
isotropic. The past is accessible (via memory traces that decay), the future
is not. The asymmetry is structural, physical, irreversible. A transformer can
in principle "attend equally" to all past positions — and when trained on
non-temporal data, probably does. That structural difference may be what makes
biological time feel like it moves in a direction.

---

*Script: `research/physics/experiments/attention_anisotropy_01.py`*
*Data: `research/physics/experiments/results/attention_anisotropy_01.json`*
*Next: experiment 02 — remove attention sink, random-sequence baseline*
