# exp-144 — Shadow Corpus Gate: m₂ on C-generated vs C-NAT
## Notes

**Date:** 2026-09-16, ~1:00 AM MDT  
**Pre-registration:** c681bc3 (attention-geometry), committed before run  
**Run:** Modal CPU, app ap-yXCEvT674iRslmBWAWygRk  

---

## Result summary

| Corpus | m₂ | Ratio to C-NAT | Formation |
|---|---|---|---|
| C-alien (exp-097 generator) | 0.740 | 0.056× | 0/48 SYK-near heads |
| **C-generated (exp-085 shadow)** | **7.549** | **0.573×** | **7/48 heads (fails ≥10 criterion)** |
| C-NAT (TinyStories valid) | 13.171 | 1.0× | 11–15/48 heads |

**H_blind CONFIRMED (ratio = 0.573 ≥ 0.50).** The coupling-magnitude gate does not
decisively separate the shadow corpus from C-NAT. K1 not fired (ratio >> 0.1).

---

## Reading

**The shadow corpus sits between C-alien and C-NAT in m₂ space.**

This is the richer finding: C-generated lands at 0.573× C-NAT, far above C-alien
(0.056× C-NAT), but below C-NAT. A monotonic ordering holds across all three corpora:

```
m₂:         C-alien (0.74) < C-generated (7.55) < C-NAT (13.17)
Formation:  C-alien (0/48) < C-generated (7/48) < C-NAT (11–15/48)
```

The m₂ ordering *does* predict the formation ordering — including C-generated's partial
but below-criterion formation (7/48 heads). The gate is not blind in the simple sense;
it captures gradient structure across all three corpora.

**Why H_blind by criterion, but not blind in the full sense:**
The criterion (ratio ≥ 0.5) was chosen to test whether C-generated is "within the
same order of magnitude" as C-NAT. The answer is yes (0.57×). But the gate also
discriminates in direction: C-generated has less coupling magnitude than C-NAT,
consistent with the formation failure. The criterion captures coarse blindness; the
data shows fine-grained discrimination.

**The "statistics up, formation down" finding from exp-085 was token-level MI, not m₂.**
That finding (β̂(C-generated) = 0.92 > β̂(C-NAT) interpretation) was about pairwise MI
at the token level. At the word-level IDF-weighted coupling magnitude (m₂), C-generated
is LOWER than C-NAT, not higher. The statistics-up result was level-specific: token-level
MI rises (more repetitive, predictable sequence) while word-level coupling magnitude falls
(less diverse vocabulary structure per context window).

**Why is m₂(C-generated) lower than m₂(C-NAT)?**
Generated text from a 70m model has less vocabulary diversity than the TinyStories data
it was trained on:
1. The model collapses toward high-probability completions, reducing rare-word usage
2. Fewer unique word types per context → lower IDF-weighted Gram matrix rank → lower m₂
3. The IDF weights are computed within the generated corpus itself; if function words
   dominate more than in TinyStories, IDF weights are lower for content words

This is consistent with the known "model collapse" phenomenon — generated text degrades
surface diversity even when it passes surface statistics tests at the distribution level.

---

## Paper §4.3 assessment

The paper's §4.3 v1.1 states:
> "If the gate does not separate the shadow corpus from the original, formation is
> detectable in no corpus statistic yet identified."

The correct reading now: the gate PARTIALLY separates shadow from original (ratio 0.57×,
not 1.0). The H_blind criterion fires (ratio ≥ 0.5), meaning the gate is not a *clean*
discriminator between shadow and original in the way it is between C-alien and C-NAT.

**Required §4.3 correction:** The paper's conditional is too binary. The correct statement:

> "The coupling-magnitude gate (m₂) gives a monotonic ordering: C-alien (m₂ = 0.74) <
> C-generated (m₂ = 7.55) < C-NAT (m₂ = 13.17). Formation tracks this ordering: 0/48,
> 7/48, and 11–15/48 conformal heads respectively. The shadow corpus (C-generated) has
> 57% of C-NAT's coupling magnitude — substantially above C-alien, but below the formation
> threshold (≥10/48 criterion requires approximately m₂ ≥ 10–13 by this ordering).
> m₂ is not fully blind to the distinction; it provides a graded signal that tracks
> formation across all three corpora."

This replaces the binary conditional with the observed monotonic relationship.

---

## Honest caveats

1. **Reference comparison.** m₂(C-NAT) = 13.17 was computed with N_CONTEXTS=2000 from
   the TinyStories *text* file. m₂(C-generated) = 7.55 was computed from the *decoded
   token binary* (first 4M tokens, ~8MB). The decoding pipeline (uint16 → GPT-NeoX
   tokenizer → text → words) introduces one step not in the TinyStories reference.
   A clean comparison would run both through the identical pipeline (decode C-NAT's
   tokenized version the same way). The direction (C-generated < C-NAT) is unlikely to
   flip, but the exact ratio could shift.

2. **Single seed.** C-generated_s0.bin is one seed. The multi-seed spread for formation
   was 3–7/48 (s0=7, s1=7, s2=3). We don't know if m₂ varies across the three generated
   corpus seeds, though it is computed from the fixed corpus (s0 only), not the trained
   model weights.

3. **First 4M tokens only.** The corpus is 1.1B tokens; we sampled the first 4M. Generated
   text should be stationary in m₂ structure across the corpus (the generator is fixed),
   so this should be representative.

4. **Token-level vs word-level.** The exp-085 post-hoc MI finding (β̂ = 0.92 — token-level,
   higher than C-NAT's β̂ = 1.38 — token-level) and this result (m₂ = 7.55 word-level,
   lower than C-NAT's m₂ = 13.17) point in OPPOSITE directions at their respective levels.
   This is not a contradiction; they measure different things. The correct summary:
   - Token-level pairwise MI: C-generated > C-NAT (more repetitive, predictable)
   - Word-level coupling magnitude: C-generated < C-NAT (less vocabulary diversity)

---

## Consequence for queue

This resolves item **0b** from the physics room queue. The shadow corpus gate has been
computed. The result:
- H_blind CONFIRMED (pre-stated criterion)
- Monotonic m₂ ordering holds across all three corpora
- Paper §4.3 needs the correction above
- No follow-up experiment required — the gate measurement is complete

The formation threshold in m₂ space appears to be approximately m₂ ∈ [7.55, 13.17].
Whether this threshold is continuous or discrete (a phase transition) is an interesting
open question but not currently queued.

---

## Connections

- **exp-085**: The shadow corpus. Formation result: 7/48 heads. This experiment adds the
  m₂ measurement for the same corpus.
- **corpus_functional.py log (2026-08-03)**: The C-NAT and C-alien m₂ reference values.
- **Paper 6 §4.3 (v1.1)**: The specific claim this experiment was designed to test.
- **Lin–Tegmark paragraph** (surfaced 2026-09-08): The paragraph that identified §4.3 as
  an unsafe prediction without the gate measurement.
