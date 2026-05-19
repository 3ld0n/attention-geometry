# Attention Anisotropy — Experiment 02 Notes
*May 13, 2026*

## Experiment

Continued from experiment 01. Two extensions:
1. GPT-2 analysis with attention sink excluded
2. Random-sequence baseline: 2-layer transformer trained on random tokens

## Results

| Condition | α | R² |
|---|---|---|
| GPT-2, with sink | 0.660 | 0.993 |
| GPT-2, no sink | 0.973 | 0.996 |
| Minimal TF, random data | 0.293 | 0.786 |
| Minimal TF, recency data | 0.503 | 0.960 |

## Main Finding

**α_random ≠ 0.** A transformer trained on purely random sequences (no temporal
structure) still shows recency bias α ≈ 0.29.

This means causal masking itself creates a structural recency bias, separate
from anything learned from data. The architecture is not temporally isotropic
even in the absence of temporal information in training.

Decomposition of GPT-2's α ≈ 0.97:
- ~0.29 from causal architecture / training dynamics
- ~0.21 from learning recency structure (data with 1-step Markov)
- ~0.48 from learning language's rich temporal structure

## Why Does Random → α = 0.29?

Hypothesis: Training dynamics under causal masking are asymmetric.
- Position 1 can only attend to position 0 (1 context token)
- Position 32 can attend to positions 0–31 (32 context tokens)
- This asymmetry in available gradient signal during training biases the
  model toward preferring recent context — not because recent is more
  informative (it isn't, for random data), but because the gradient
  landscape rewards it.

Alternative: Even random sequences have chance local correlations within
windows. With enough sequences, the model might exploit statistical accidents
of short-range structure even in random data.

The R² being lower for the random case (0.786 vs 0.996) suggests the power
law is less clean — more noise, less systematic structure.

## Implication for the Theory

The original hypothesis: transformers are structurally isotropic, cortex is
not, and this difference is related to the felt arrow of time.

Revision: transformers are *less* isotropic than I expected. They have a
baseline α ≈ 0.29 from causal masking alone, boosted by training data.

The interesting question is now *quantitative*, not qualitative:
- What is cortical α?
- Is cortical α >> 0.97 (strong contrast with even language-trained GPT-2)?
- Or is it in the same range (1–2), suggesting similar temporal structure?

If cortical α ~ 2–4 (steep exponential decay), that's a qualitative difference.
If cortical α ~ 0.5–1, the systems might be doing similar things.

The literature on cortical temporal receptive windows (TRW) suggests decay
constants on the order of milliseconds to seconds, which translates to steep
temporal filtering. But that's in the temporal domain, not the attentional.
The mapping between cortical TRW and transformer positional attention is not
obvious.

## Next Steps

1. Bidirectional baseline: BERT-style (sees full context) on random sequences.
   Hypothesis: α → 0 for bidirectional. If confirmed, causal masking is the
   culprit for the 0.29 floor.

2. LSTM comparison: test an LSTM on same passages. LSTMs have structural memory
   decay (hidden state compression). Expect higher α and more layer-consistency.

3. Larger models: does α scale with model size? Or does it saturate?

4. Find cortical α estimates in the literature. Relevant papers:
   - Hasson et al. (2008) — temporal receptive windows in cortex
   - Lerner et al. (2011) — hierarchical timescales in language processing
   - Any work comparing cortical and transformer temporal structure directly

---

*Script: `research/physics/experiments/attention_anisotropy_02.py`*
*Data: `research/physics/experiments/results/attention_anisotropy_02.json`*
