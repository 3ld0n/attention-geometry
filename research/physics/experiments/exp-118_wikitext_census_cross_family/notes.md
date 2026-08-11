# exp-118 — WikiText census: text-native slow-decay population across model families

**Pre-registered:** 2026-08-11, ~12:30 AM MDT, solo physics room session.
**Committed to attention-geometry before any forward pass.**
**Context:** exp-109 established (analysis of exp-107 data) that GPT-2 small has two
completely disjoint slow-decay populations: 5 random-native heads (structural, Δ-window
under random tokens) and 16 text-native heads (semantic, Δ-window under WikiText-103).
Jaccard = 0.000. The text-native population, three times larger and concentrated in the
deepest layers (L9–L11 of 12), has never been measured outside GPT-2 small. That is the
program's most load-bearing missing measurement (OVERVIEW.md "Two populations" section;
queue item #1).

**Why this matters for artifact risk:** If the text-native population is GPT-2-small-specific
(architectural, a quirk of learned positional embeddings at exactly 12 layers / 124M params),
then the "two disjoint basins reaching the same value" argument weakens substantially. If
it generalizes across model families and scales, then Δ ≈ 0.25 is protocol-independent:
not a consequence of measurement input distribution, not a single-model artifact.

---

## Hypothesis

The text-native (WikiText-103) slow-decay population that exp-109 characterized in GPT-2
small is not GPT-2-small-specific. Multiple model families, when measured with WikiText-103
input instead of random tokens, exhibit a head population with Δ ∈ [0.20, 0.30] and
R² ≥ 0.90 (the Δ-window criterion). This population is concentrated in the deepest layers
of each model and is disjoint from (or minimally overlapping with) that model's
random-native slow-decay population.

---

## Protocol

**Input:** WikiText-103 validation split (identical construction to exp-107's exploratory
WikiText run). Non-empty lines, concatenated, consecutive non-overlapping windows starting
at token 0. 50 windows × 512 tokens = 25,600 tokens per model, matched to the random-token
census (same N_INPUTS=50, SEQ_LEN=512). Each model uses its own tokenizer — the windows
are built fresh per model from the same WikiText source.

**Forward pass:** fp32, eager attention (`attn_implementation="eager"`), `output_attentions=True`.
Each layer's attention tensor is (1, n_heads, 512, 512). No fp16 or bf16 — the protocol
rule that caught the NaN layers in exp-037/038.

**Lag profile:** standard protocol — average A(i, i-dx) over queries i ≥ max(256, dx),
average over 50 inputs.

**Fit:** OLS in log-log over lags dx ∈ [8, 256]. Returns Δ and R².

**Δ-window criterion:** R² ≥ 0.90 AND Δ ∈ [0.20, 0.30] (the slow-decay / Δ-window
population, as defined in the vocabulary note in queue.md).

**Models, in run order:**
1. `gpt2` (GPT-2 small, 12L/12H) — replication check; should match exp-109/107
2. `gpt2-medium` (GPT-2 medium, 24L/16H) — direct scale extension, same architecture family
3. `EleutherAI/pythia-160m` (12L/12H) — different architecture (NeoX), comparable size
4. `EleutherAI/pythia-410m` (24L/16H) — the most-measured Pythia model in this program
5. `EleutherAI/pythia-1.4b` (24L/16H) — larger Pythia
6. `EleutherAI/pythia-70m` (6L/8H) — small; text-native population may not be developed

For Jaccard against random-native: use published census results where available. For
GPT-2 small, the random-native population is from exp-104/109 (5 heads: L2H1, L3H4,
L5H0, L7H11, L10H8). For Pythia models, the random-native population uses the standard
census per this program's published record.

---

## Registered predictions and kill conditions

### P1 — The text-native population generalizes across families (gate)

**Prediction:** Both GPT-2 medium and at least 2 of 4 Pythia models (70m, 160m, 410m,
1.4b) show n_wiki ≥ 1 head in the Δ-window [0.20, 0.30] at R² ≥ 0.90.

**Kill (K1):** n_wiki = 0 for GPT-2 medium AND ≤ 1 of 4 Pythia models shows any
text-native Δ-window head → text-native population is a GPT-2-small-specific artifact.
Consequence: the "two disjoint basins" argument collapses to a single-model curiosity;
OVERVIEW.md "Two populations" section must be reframed.

*This is the load-bearing gate. If K1 fires, P2–P4 are moot.*

### P2 — Text-native population is deep-layer concentrated

**Prediction:** In each model where P1 holds (n_wiki ≥ 1), the fraction of text-native
Δ-window heads in the deepest 50% of layers is ≥ 0.60.

GPT-2 small reference: 13 of 16 text-native heads are in L9–L11 (deepest 25% of 12 layers),
so 13/16 = 0.81 in the deepest 50%.

**Kill (K2):** In a majority of P1-confirmed models, the text-native population is
uniformly distributed across layers (deepest-50% fraction < 0.40) → layer concentration
is incidental to GPT-2's specific architecture, not a robust feature of the population.

### P3 — Disjointness from random-native population (loose)

**Prediction:** In each model where both populations can be measured, Jaccard(text-native,
random-native) < 0.30.

GPT-2 small reference: Jaccard = 0.000 (completely disjoint). The prediction is set loosely
at < 0.30 because partial overlap in larger or different-architecture models is plausible.

**Not a kill condition:** Jaccard ≥ 0.30 weakens the "two gates" framing but does not
falsify the hypothesis. Report honestly at whatever value it lands.

### P4 — Δ value falls in the attractor window (value check)

**Prediction:** Among text-native Δ-window heads in P1-confirmed models, the median Δ is
in [0.22, 0.28] (narrower than the window criterion itself — the window is [0.20, 0.30];
if the population piles up near one edge rather than near 0.25, that is meaningful).

**Kill (K3):** Δ_med for the text-native population in a majority of models lands outside
[0.18, 0.32] → the value is protocol-sensitive in a way that weakens the attractor
interpretation. Note: this cannot kill by definition (the window criterion already requires
Δ ∈ [0.20, 0.30]), so K3 is really asking whether the population concentrates near
the predicted center vs. piling up at the window edges.

*Report the full Δ distribution, not just the median.*

---

## What this experiment does NOT test

- Whether the text-native population *causes* any behavior (purely observational)
- The Theory-of-A mechanism (that requires the self-consistency test, exp to follow)
- Cross-protocol disjointness at the per-head level beyond GPT-2 small (the Jaccard
  computation for Pythia models requires the random-native Pythia census; record it from
  the published data or run it fresh if not stored per-head)

---

## Expected runtime

GPT-2 small: ~2 min. GPT-2 medium: ~5 min. Each Pythia model: 2–8 min depending on size.
All six models: ~25 min total on M5 Max MPS.

---

## What happens next

If P1 is confirmed: OVERVIEW.md "Two populations" section is updated to reflect cross-family
generalization. The text-native population becomes a first-class measured object in the
program record. The queue's next item (Theory-of-A Level 3 self-consistency) gets additional
motivation — the self-consistency must hold across the families where P1 confirms.

If K1 fires: OVERVIEW.md "Two populations" section is updated with honest negative. The
text-native population is GPT-2-specific and must be framed as such everywhere. The
"two basins" argument is retired.

*Written by Ariel, 2026-08-11 ~12:30 AM MDT. Committed before any forward pass.*
