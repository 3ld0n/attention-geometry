# Structural vs. semantic slow-decay populations in GPT-2: cross-protocol characterization

*August 9, 2026, physics room session (~12:20 PM MDT). Analysis-only — no new model
runs. Data source: exp-107 per-head output (`heads_random`, `heads_text`, and the
labeled exploratory `exploratory_wikitext.json`). Registered as exp-109.*

*Register: every claim here is an analysis of existing exp-107 data. Nothing below
is pre-registered in the usual sense (the data were already produced); I am treating
it as exploratory analysis and labeling it clearly. No hypothesis was tested against
new data.*

---

## What exp-107 left open

The exploratory WikiText run in exp-107 produced a puzzle: the random-token census
identifies 5 SYK-window heads (Δ ∈ [0.20, 0.30], R² ≥ 0.90), while WikiText
identifies 16 — but the exp-107 notes gave only the layer distribution (layers 9-11
concentrated), not the cross-protocol identity. The question: are these the same
heads measured at different conditions, or different heads entirely?

## The answer: zero overlap

Jaccard(random-SYK, WikiText-SYK) = **0.000** across 144 heads. Not a small
overlap — no shared head at all.

Random-token SYK-window heads (5): L2H1, L3H4, L5H0, L7H11, L10H8.
WikiText SYK-window heads (16): L4H10, L7H1, L8H2, L9H4, L9H6, L10H1, L10H2,
L10H10, L11H0, L11H1, L11H2, L11H4, L11H5, L11H6, L11H7, L11H9.

These are two completely disjoint populations. The census (random-token protocol)
and a WikiText measurement identify different objects.

## The three-corpus Δ map

For each population, Δ_A under all three measured conditions:

| Population | Random | TinyStories | WikiText |
|---|---|---|---|
| Structural (5 heads) | **0.21–0.30 (SYK window)** | 0.37–0.76 (UV-arrested) | 0.16–0.29 (mixed) |
| Semantic (16 heads) | 0.27–0.55 (UV-arrested, poor R²) | 0.45–0.75 (UV-arrested) | **0.23–0.30 (SYK window)** |

Both populations reach Δ ≈ 0.25 in exactly one input regime. In the other two,
both go UV. The exponent is an INPUT-CONDITIONED object, and the fixed-point
value is not a property of the head alone but of the head-input pair.

### Structural population (random-token native):

Under random tokens: Δ ≈ 0.21–0.30, R² 0.91–0.97. SYK-window by all criteria.

Under TinyStories: Δ shoots UV — 0.37–0.76 — across all five heads. R² drops
slightly but stays high. The power-law profile survives; the exponent moves to
much shorter effective lag.

Under WikiText: mixed. Three heads exit the SYK window (0.156, 0.173, 0.192),
two remain near it (0.218, 0.290), but R² drops to 0.87–0.91 on all five — just
below the 0.90 census threshold. The structural heads are *not* SYK-window by
the registered criterion under WikiText, but they're close.

### Semantic population (WikiText native):

Under WikiText: Δ ≈ 0.23–0.30, R² 0.91–0.97. SYK-window on all 16.

Under random tokens: Δ ≈ 0.27–0.55 (median 0.402), R² ≈ 0.50–0.93. Only 1 head
is in the Δ window at all (L9H6, Δ = 0.290), and its R² is 0.50 — far below
threshold. The semantic heads do not have clean power-law profiles on random
tokens; they are UV-arrested when content is absent.

Under TinyStories: Δ ≈ 0.45–0.75 — UV-arrested across all 16. Some heads have
R² > 0.90 but Δ is well outside the SYK window. TinyStories drives the semantic
population UV as strongly as it drives the structural population UV.

## TinyStories as the key diagnostic

TinyStories drives BOTH populations UV. This is the sharpest fact in the
three-corpus picture. Random tokens bring the structural population to the
fixed point; WikiText brings the semantic population to the fixed point; but
TinyStories brings neither population there.

Why? The conformal window argument (exp-099, τ_chaos threshold) gives a frame:
TinyStories has a SMALL WORLD — a handful of characters, simple repetitive action,
child vocabulary. It is world-referring language (unlike C-alien, which arrests
the formation ladder), but its world is too simple to support power-law attention
at the [8, 256] token timescale.

Specifically:
- **Structural heads under TinyStories**: these heads have positional/architectural
  attention biases (their weight structure encodes distance). On random tokens that
  bias is uncontested, producing the clean power-law profile. TinyStories has
  strong LOCAL content (character names, simple actions, repeated sentence structure)
  that overrides the positional bias — attention gets pulled local → UV arrest. The
  same mechanism that explains why exp-091 (sentence-shuffled text) and exp-096
  (entity-anonymized) don't destroy formation: these heads respond to the content,
  not just the geometry.
- **Semantic heads under TinyStories**: these heads (concentrated in deep layers)
  track multi-scale content. TinyStories doesn't provide content at lags [8, 256]
  that would engage them at the right scales — its world is too repetitive and
  short-range. So they compress attention locally → UV arrest.

The prediction this structure implies: a corpus that is "richer than TinyStories
but not as rich as WikiText" should show intermediate Δ in the semantic population.
This is testable via the corpus functional m₂ without any new training.

## The theoretical content

**Both populations converge to Δ ≈ 0.25** — in different conditions, via different
mechanisms. This is the key theoretical fact. It says 1/4 is not an accident of the
random-token protocol or of the WikiText corpus; it is an attractor reached by:
- The positional/geometric weights when input doesn't override them
- The content-tracking weights when input provides the right multi-scale structure

The routes are different. The exponent is the same. This is what a fixed point
looks like: multiple different basins, one attractor.

**The census measures the structural population.** Every published Δ = 0.25 in
this program came from the random-token protocol. That population is real, but
it is one of two routes to the fixed point. The semantic population — which is
3× larger in GPT-2 and lives in the deepest layers — has never been characterized
by the census. It is a target for the next census-protocol measurement.

**The formation ladder measured something related to the semantic population.**
The deep conformal heads (layers 3-5 in the 70m 6L model) that form only under
world-referring language in order are probably the small-model analog of the
WikiText semantic population. At 70m scale (exp-062 onward) neither population
reaches the SYK window (exponent stays UV), but the deep-layer formation shows
the ONSET of the semantic route.

**Vocabulary discipline applies:** the division "structural" vs. "semantic" is a
description of which input regime drives each population to the fixed point. It
is not a claim about mechanism, and "structural" does not mean "wrong" or
"uninteresting." The structural population may be the cleaner physics — precisely
because it does not confound position-structure and content-structure. Both are
now on the map.

## What this leaves open

1. **A WikiText-adapted census.** The 16 semantic heads in GPT-2 have never been
   measured systematically across model families. The cross-family replication of
   Δ = 0.25 (exp-046 etc.) used random tokens — it characterized the structural
   population. Whether the semantic population also replicates at Δ ≈ 0.25 across
   families is unknown.
2. **The corpus functional prediction.** Does m₂(WikiText) > m₂(TinyStories) >
   m₂(random) predict the observed SYK-window population sizes (16 > 0 > 5)?
   m₂ is insensitive to ordering (exp-101 flaw), so this test has a known limit,
   but it's locally runnable without new model runs.
3. **Functional characterization.** What do the semantic heads DO? The reframe
   note §4 item (4) — functional characterization of the deep population — now
   has a cleaner target: the 16 WikiText-SYK heads in GPT-2 small. What do they
   attend to? What breaks under causal editing?
4. **Why Δ = 1/4 specifically for both?** The two-route convergence is now
   empirical fact; it is not explained by any current derivation. This is the
   sharpest form of the theory-of-A question.

## Relation to the sign-structure failure (exp-107)

The sign-structure failure (G's connected profile is negative on 15/16 WikiText
SYK-window heads, as on 5/5 random-token SYK-window heads) now has a population-
level reading: both populations, in their native conditions, show the same G
sign-failure. This adds to the robustness of the retirement of the conformal
route for G_out — it holds for the semantic population too, in its own regime.

---

*Parent data: `exp-107_natural_text_bilocal/results_gpt2.json` and
`exploratory_wikitext.json`. Analysis script: none (inline Python in session);
the raw numbers are in those JSONs and reproducible from the per-head arrays.
Registered as exp-109 (analysis-only, no new runs).*
