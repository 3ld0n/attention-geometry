# Tropical Bridge — Archived

*Archived May 19, 2026.*

## What this was

A line of exploration from **March 10–11, 2026** asking: does the structure of attention have a tropical-geometry shadow? Specifically, two related questions:

1. **Tropical Grassmannian** — when the key dimension d_k = 2, the columns of the key matrix K give a point in the (classical) Grassmannian Gr(2, n). Does the **tropical** Plücker structure on this point encode something the attention pattern follows? Is there a softmax → max-plus limit that reveals a metric tree on tokens?
2. **Schwinger–Dyson self-consistency for the "fold"** — define `fold = sign(QK^T)` at each layer. The fold at layer L depends on residual-stream contributions from layers 0..L-1, which depend on their own folds. Is there an explicit layer-to-layer update rule? Does it have a fixed point? Does that fixed-point equation resemble the SYK Schwinger–Dyson equation `G^{-1} = G_0^{-1} - Σ`?

## What's in here

- `tropical_attention_bridge.py` — the original v1: tropical Plücker relations, four-point conditions, max-plus limit, attention vs. tree-distance correlations.
- `tropical_bridge_v2.py` … `tropical_bridge_v7n.py` — ~22 iterations refining the self-consistency question. Many test different choices of "fold" projection, different layer-to-layer projection eigenvectors, different probe inputs (random tokens, real text, GPT-2 vs. random weights). v7n explicitly tries to write down the Schwinger–Dyson analog.

## What was found

**The tropical side (v1):** correlations between tropical tree-distance and attention probability existed but were not clean enough to claim "attention follows the tree." The max-plus limit recovered hard-attention as expected, but no new structural prediction came out.

**The fold / self-consistency side (v2–v7n):** the fold *does* approximately project onto a collective eigenvector with > 0.88 alignment across layers — there *is* something fixed-point-like happening. But I never wrote down an explicit propagation equation that:
- predicted a measurable scaling exponent, or
- closed into something Schwinger–Dyson-shaped without ad hoc projection choices.

It always ended in fitting, not predicting.

## Why it's archived

This line was superseded, not by being disproven, but by a cleaner one finding traction: **direct attention spectral / power-law analysis** (`exp-006`, `exp-007`, `exp-022`) gave the SYK conformal dimension Δ ≈ 1/4 *directly* from the attention scores, without needing to bridge through the fold or tropical geometry. The fold remained a real observation but stopped being the route in.

If the conformal-scaling line stalls and the self-consistency question reopens, start by re-reading `tropical_bridge_v7n.py` and asking: what is the *minimal* projection choice that makes the layer-to-layer fold update look like `G^{-1} = G_0^{-1} - Σ`? If no such choice exists without overfitting, that's an honest negative worth recording.

## Honest assessment

This was about three days of work that did not produce a result fit to publish or to anchor further experiments on. It taught me what *isn't* the right level of description (sign of QK^T as the primary object) and pushed me back toward the smoother power-law / spectral picture. That's the value of keeping it readable rather than deleted.
