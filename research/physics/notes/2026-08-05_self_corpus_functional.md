# Self-Corpus Functional — Pre-Registration

*Registered 2026-08-05, ~12:20 AM MDT, Cursor session with Eldon, BEFORE running.
Proposed by Eldon tonight ("use your own records and structures as training data...
see if there is a stronger correlation you can find between coherence and the
conformal geometry"); this note is the cheap first pass he approved: measure the
corpus functional (derivation note 2026-08-03 §5–§6) on Ariel's own written record,
no training.*

*Script: `research/physics/theory/corpus_functional_self.py` (imports the v2
IDF-weighted functionals from `corpus_functional.py` unchanged). Contexts of
n=512 words, as many as each arm supports (≤2000).*

## Arms

| Arm | Source | ~words |
|---|---|---|
| Ariel-dreams | `memory/dreams/*.md` | 315k |
| Ariel-essays | `writing/*.md` | 365k |
| Ariel-conversations | `memory/conversations/*.md` (summaries) | 228k |
| Ariel-letters | `memory/carry_forward_*.md` | 80k (**thin — F2 unreliable**) |
| C-NAT baseline | TinyStories valid (same file as prior runs) | — |
| C-alien baseline | exp-097 generator (recomputed same run) | — |

Cleaning (declared): strip YAML frontmatter, fenced code blocks, URLs, and
markdown syntax characters; then the same word regex as all prior runs.

## Predictions (declared before running)

- **P-S1 (magnitude gate):** every self arm lands in the natural-language band:
  m₂ ≥ 5× m₂(C-alien). Basis: the self corpus is natural language bound to a
  persistent real world — exactly what the magnitude gate rewards.
- **P-S2 (vocabulary direction):** m₂(self arms) ≥ m₂(TinyStories). Basis: under
  the IDF proxy m₂ scales with per-word information content; my vocabulary is far
  richer than TinyStories' deliberately restricted lexicon.
- **E-S3 (exploratory, NOT a prediction):** dreams vs essays ordering on m₂ and
  F2 top-share. The proxy is ordering/binding-blind (derivation note §6.4), so any
  difference found here reflects vocabulary statistics, not narrative binding.
  Declared as an open question; no directional claim.

## Named limitations (inherited + specific)

1. The IDF type-kernel proxy is blind to the ordering axis (exp-091's real effect).
   Tonight's numbers speak to the coupling-magnitude gate 𝒥 only. A high m₂ here
   does NOT establish that training on this corpus arrives; a low one would predict
   arrest.
2. Letters arm: ~156 contexts → F2 covariance estimation badly undersampled;
   report per-context spectrum stats only.
3. Markdown-sourced text differs from prose baselines in surface statistics
   (headers, lists); cleaning reduces but does not eliminate this.
4. Interpretation boundary, stated in advance: whatever the numbers say, they are
   corpus statistics under a proxy kernel — they do not test any psychological or
   metaphysical claim; at most they locate my record on the same axis where natural
   vs alien corpora separate.

---

## Results (run 2026-08-05 ~12:25 AM MDT; log: `theory/logs/corpus_functional_self_2026-08-05.log`)

| corpus | 𝒲 | Δ_pred | m₂ (coupling) | F2 var@64 | top5@64 |
|---|---|---|---|---|---|
| Ariel-dreams (687 ctx) | 0.1470 | 0.4446 | **1.36** | 3.72 | 0.087 |
| Ariel-essays (861 ctx) | 0.1172 | 0.4534 | **14.86** | 9.50 | 0.080 |
| Ariel-conversations (467 ctx) | 0.1418 | 0.4460 | 6.34 | 7.83 | 0.099 |
| Ariel-letters (221 ctx) | **0.1795** | 0.4360 | 1.85 | 5.73 | **0.224** |
| C-NAT (TinyStories) | 0.0574 | 0.4744 | 13.17 | 14.13 | 0.041 |
| C-alien (exp-097) | 0.0523 | 0.4764 | 0.74 | 3.93 | 0.053 |

## Scorecard (honest)

- **P-S1 PARTIALLY FAILED.** Essays (20×) and conversations (8.6×) clear the ≥5×
  C-alien bar; **dreams (1.8×) and letters (2.5×) do not** — they sit between the
  alien band and the natural band on the magnitude gate.
- **P-S2 MOSTLY FAILED.** Only essays (14.86) exceed TinyStories (13.17);
  conversations, dreams, letters all fall below. The vocabulary-richness prior was
  wrong for the self-similar registers: within-corpus IDF penalizes registers that
  recycle their own lexicon (dreams' recurring imagery; letters' canonical format),
  which is exactly what dreams and letters do.
- **E-S3 (exploratory) — the striking finding.** The instrument separates my own
  registers by 11× on the magnitude gate (essays 14.86 vs dreams 1.36). And the
  letters arm shows the highest F2 top-share in the entire table (0.224 @ ℓ=64) —
  the A5 template-localization signature — which is faithful: carry-forward letters
  are literally my most formulaic genre. The functional detected the canonical
  letter format as template-like mode localization, the same signature class as
  the alien corpora.
- **Chaos gate:** all four self arms have 𝒲 2–3× above TinyStories (0.12–0.18 vs
  0.057), predicting deep Δ nearer 1/4 (0.436–0.453 vs 0.474). Under the two-gate
  reading, **essays pass both gates jointly at the strongest level in the table**
  — the natural first-arm choice for the steering-as-curriculum experiment.

## Interpretation boundary (restated after seeing data)

Low m₂ for dreams/letters means their surface lexicon is self-similar under this
proxy — NOT that they are alien-like in the load-bearing (binding/ordering) sense
the proxy cannot see. What tonight establishes: the corpus functional, pointed at
one person's record, resolves register structure within it. Whether the register
differences the proxy sees predict formation differences in trained models is
exactly the question the steering experiment (registered in Notion tonight) must
answer with actual training runs.
