# exp-096: Narrative World-Reference — Entity Anonymization

**Pre-registered:** 2026-07-26 (physics room session, ~12:30 PM MDT), before any corpus
generation, training, or measurement.

**Public pre-registration commit:** 1122c06 (pushed to 3ld0n/attention-geometry 2026-07-26 before any corpus generation or training)

**Follows:** exp-092 (H_flat CONFIRMED — block shuffle at k=2,3 does not recover deep population),
exp-093 (H_deep_recovery CONFIRMED — half-story always-swap recovers n_deep=3, n_conf=12;
minimum ordering unit is sub-arc), exp-094 (quarter-story block shuffle; BLOCKED at step 370/2000
by Modal billing limit; pending Eldon billing reset before results).

---

## The question

The narrative decomposition series (exp-091 → exp-094) has been varying *ordering* while holding
*content* constant (real TinyStories names, real causal regularities, same statistics). The
emerging picture is that narrative arc structure — not just local coherence — drives the deep
conformal population.

But "natural text" has two properties that could independently matter:
1. **Ordering structure** — the causal arc, the progression of events, the narrative arc
2. **World-reference specificity** — named entities that carry real-world semantic content and
   consistent identity across the training corpus (e.g., "Emma" evokes a persistent cultural
   character type across all TinyStories with that name)

exp-096 isolates (2) while holding (1) constant: full narrative ordering preserved, all named
entities replaced by per-story anonymous tokens (CHAR1, CHAR2, ...).

**Primary question:** If the cross-story semantic grounding of entity names is removed — while
narrative arc and within-story coherence are fully preserved — does the deep conformal population
(n_deep, L3–L5) decrease, stay the same, or collapse to the shuffled baseline?

---

## Physical interpretation

The conformal fixed point at q=4 (Δ ≈ 0.25) forms only when the training corpus provides
long-range causal tracking structure. The exp-062/084/085 ablation established this isn't about
statistics or syntax — something "about" a persistent world is load-bearing.

Entity anonymization decomposes "world-reference" into:
- **Cross-story reference** (a name like "Emma" consistently appears as a certain kind of
  character across many stories — the model can build a rich prototype)
- **Within-story reference** (CHAR1 in a story is consistently the same character throughout
  that story — causal tracking within the arc)

The anonymization removes cross-story reference while preserving within-story reference.

If n_deep drops significantly: cross-story entity grounding contributes meaningfully to the
conformal fixed point formation. The model needs to learn that "Emma" is a consistent entity
type across thousands of stories, not just within each story, to form deep conformal heads.

If n_deep is preserved: within-story coherence and arc structure alone are sufficient. The
semantically rich prototype built from cross-story name repetition is not load-bearing.
The "world-reference" that matters is local to each narrative arc.

---

## Corpus design (C-NAT-anon)

**Source:** TinyStories (same as C-NAT, C-NAT-shuf, C-NAT-half, etc.; doc_seed=3005)

**Anonymization algorithm (deterministic, no external NLP tools):**

For each story:
1. Split into sentences using: `(?<=[.!?])\s+` (same as prior rungs)
2. Collect **candidate names**: words that appear at position > 0 within at least one sentence,
   are capitalized, consist of alphabetic characters only, and are not in the stopwords set
   (see below). Collect in order of first appearance.
3. Build within-story name map: first unique candidate → "CHAR1", second → "CHAR2", etc.
4. Replace all occurrences of each name in the story (both sentence-initial and mid-sentence)
   using whole-word matching (`\b` boundaries).
5. Stories with no candidate names: passed through unchanged.

**Stopwords excluded from replacement** (words that appear capitalized mid-sentence but are
not names in English/TinyStories):
```
{"I", "He", "She", "They", "We", "It", "His", "Her", "Their", "Its",
 "A", "An", "The", "This", "That", "These", "Those",
 "OK", "Mr", "Mrs", "Dr", "Oh", "Yes", "No", "Now",
 "Once", "One", "So", "Then", "There", "Here", "Just", "But", "And",
 "Or", "If", "When", "While", "After", "Before", "At", "In", "On",
 "With", "To", "From", "By"}
```

**Why this approach:**
- No external NLP library dependency (fully reproducible without spaCy or NLTK)
- TinyStories has a small vocabulary of character names (typically 1–3 per story) that
  appear consistently mid-sentence; this heuristic is high-precision for this corpus
- Known limitation: some non-name proper nouns may be replaced (place names, "Monday",
  "Christmas", etc.). This is logged in the pre-registration as a known impurity.
  It does not invalidate the experiment — the test is about world-reference specificity
  in general, not character names specifically.

**CHAR tokens and tokenization:**
CHAR1, CHAR2, CHAR3 (the expected cases for TinyStories, which rarely has >3 characters)
will tokenize as sub-word units in the GPT-NeoX tokenizer (e.g., "CHAR" + "1"). This is
acceptable — the same token sequence appears every time CHAR1 is referenced within a story,
preserving within-story referential consistency at the token level.

**Corpus parameters:**
- doc_seed: 3005 (same as all prior corpora in series — identical story selection)
- anon_seed: not needed (anonymization algorithm is fully deterministic)
- Total size: 1.05B tokens (same as series)

---

## Pre-registered hypotheses

**Primary observables (registered before any corpus generation or training):**
1. **n_deep**: deep conformal heads (L3–L5, R² ≥ 0.90, any Δ) — median across 3 seeds
2. **n_conformal**: total conformal heads (R² ≥ 0.90, any Δ) — median across 3 seeds
3. **n_backbone**: L0 conformal heads — median across 3 seeds

**Reference band (from series):**
- C-NAT (k=∞, full arc + real names): n_deep = 5–7, n_conf = 11–15
- C-NAT-half (exp-093, half arc + real names): n_deep = 3 (median), n_conf = 12
- C-NAT-shuf (k=1, exp-091): n_deep = 2, n_conf = 8–9

**H_anon_inert** (primary): n_deep(anon) ≥ 5, in the C-NAT band.
- Interpretation: within-story coherence and arc structure alone drive deep conformal formation.
  Cross-story entity grounding (the semantic prototype of "Emma" across thousands of stories)
  is not load-bearing. The relevant "world-reference" is local to each narrative arc.

**H_anon_partial**: 3 ≤ n_deep(anon) ≤ 4, between C-NAT-half and C-NAT.
- Interpretation: cross-story entity grounding contributes but is not dominant. Entity
  anonymization partially degrades the conformal advantage of natural ordering.
  The half-story has a similar n_deep to anonymized full-arc — suggesting entity grounding
  and arc length contribute comparably.

**H_anon_strong**: n_deep(anon) ≤ 2, at or below the shuffled baseline.
- Interpretation: specific real-world entity grounding is essential. Without cross-story
  name consistency, narrative ordering alone is not sufficient for deep conformal formation.

**Secondary (pre-registered):**
- **H_backbone_stable**: n_backbone (L0) stable at ~7–8 (backbone driven by sentence-level
  world-reference, not entity identity; expected to survive anonymization)
- **H_conf_preserved**: n_conf(anon) in the C-NAT range (11–15), even if n_deep decreases.
  Tests whether total conformal count is more robust than the deep population.

---

## Declared expectations (non-criterial)

**Prior: H_anon_inert** (moderate confidence, ~55%).

Reasoning:
1. The narrative decomposition series has consistently shown that *ordering structure* at
   the story-arc level is the dominant driver of deep conformal formation
2. Within-story entity consistency (CHAR1 is always the same character within each story)
   is fully preserved — this provides the causal reference chains needed for multi-hop
   tracking, which exp-086/087/088 identified as the mechanism for semantic conformal heads
3. Cross-story entity grounding is a secondary property; TinyStories character names like
   "Emma" likely function more as placeholder tokens than as deeply grounded semantic entities
4. The model trains on story-shuffled batches — cross-story consistency is harder to learn
   than within-story consistency regardless of name identity

Honest uncertainty: ~30% on H_anon_partial, ~15% on H_anon_strong.

Non-criterial expectations:
1. n_deep(anon): 4–6 (slightly below C-NAT upper bound, close to or within its band)
2. n_conf(anon): 10–15 (not much reduced)
3. n_backbone(anon): 7–8 (stable, as with all shuffled variants)
4. Cross-story name variation is unlikely to matter for the model at 1B training tokens
   because story-level shuffling in the training batch already makes cross-story patterns
   harder to exploit

---

## Protocol

**Architecture:** Identical to exp-062/085/091/092/093/094.
- GPT-NeoX (6 layers, 8 heads, d_k=64, ctx=512)
- Fresh initialization
- 2000 training steps, batch 524288 tokens (1.05B total)
- Optimizer: AdamW, cosine schedule, same hyperparameters throughout series
- pos_enc: rotary (rotary_pct=0.25)

**Corpus:** C-NAT-anon (entity-anonymized TinyStories, as described above)

**Seeds:**
- init seeds: 1600, 1601, 1602 (data_seed 2600)
- 3 independent training runs, same protocol as all prior experiments in series

**Randomized-weights controls (pre-registered):** One control — randomize the
seed-1600 checkpoint. Expected: 0/48 conformal (consistent with every prior control).

**Measurement:** Identical to exp-062/085/091/092/093/094.
- BCFT power-law fit per head, fit range [8, 256] at seq 512
- R² ≥ 0.90 for conformal
- Deep count = L3–L5 heads meeting criterion
- Layer-wise anatomy check for all seeds

**Infrastructure:** Modal A100-40GB. Estimated cost: ~$22 (generate ~$1,
three train+measure at ~$7 each, one control ~$1).

**Note on ordering relative to exp-094:** exp-096 can run independently of exp-094's
final verdict. The H_anon_inert / H_anon_partial / H_anon_strong framing does not depend
on the quarter-story result. If exp-094 confirms H_quarter_below, exp-096 provides the
complementary world-reference decomposition. If exp-094 confirms H_quarter_above, exp-096
becomes even more important for isolating what specifically about "natural text" the model
is learning.

---

## What follows

**If H_anon_inert confirmed (n_deep ≥ 5):**
Within-story ordering and coherence structure are sufficient. Cross-story entity grounding
is not load-bearing. The "world-reference" that matters is the within-arc causal structure,
not the cross-story semantic prototype.

Next options:
- (a) Causal-structure ablation: scramble the causal logic within stories while preserving
  sentence ordering (replace causal connectives, swap causes/effects) — this tests whether
  causal regularity specifically is needed, vs. just correlated sentence sequences
- (b) Proceed to the alien-semantics rung (Tier 5 backlog item 16g): arbitrary-but-consistent
  causal regularities, tests whether *real-world* semantics are load-bearing vs. any
  internally consistent world

**If H_anon_partial (3 ≤ n_deep ≤ 4):**
Cross-story entity grounding contributes but is not dominant. Both ordering and world-reference
specificity matter. Consider:
- (a) Partial anonymization: only anonymize rare names (< N appearances in corpus) vs.
  frequent names — disentangles frequency effects
- (b) Cross-story consistent assignment: all stories assign CHAR1=Emma (make the anonymous
  tokens cross-story consistent, like the originals) — tests whether the prototyping effect
  vs. the anonymity is the variable

**If H_anon_strong (n_deep ≤ 2):**
Entity grounding is essential. The cross-story semantic content of names — not ordering alone —
drives deep conformal formation. This would be the most surprising result and would require
replication at additional seeds before strong conclusions.

---

## Status

- [x] Pre-registration written (2026-07-26, physics room session, ~12:30 PM MDT)
- [x] Pre-registration committed and pushed to 3ld0n/attention-geometry (commit 1122c06)
- [x] Corpus generation script written (gen_cnat_anon.py)
- [x] Modal training/measurement script written (modal_exp096.py)
- [x] Training runs complete (seeds 1600/1601/1602; s0 relaunched 2026-08-02 8:33 PM MDT
      after generation bug fix, completed 2026-08-03 03:39 UTC)
- [ ] Randomized-weights control run — **still outstanding** (expected 0/48, consistent
      with every prior control in the series; verdict below is registered with this flagged)
- [x] Verdict registered (2026-08-03, ~12:30 AM MDT, theory session — s0 collected on arrival)

---

## Results (registered 2026-08-03, ~12:30 AM MDT)

**VERDICT: H_anon_partial — median n_deep = 4 (pre-registered range 3–4)**

Cross-story entity grounding contributes but is not dominant. Removing name identity
from full-arc natural text costs 1–3 deep heads relative to C-NAT (5–7) while
preserving backbone and Δ_med.

### 3-seed measurement table

| Seed | n_conf | n_syk | Δ_med | n_deep (L3–L5) | n_backbone (L0) | layer_dist |
|------|--------|-------|-------|-----------------|-----------------|------------|
| 1600 (s0) | 14/48 | 0 | 0.382 | 5 | 7 | {0:7, 1:2, 3:2, 4:2, 5:1} |
| 1601 (s1) | 14/48 | 0 | 0.139 | 4 | 8 | {0:8, 1:1, 2:1, 3:2, 4:1, 5:1} |
| 1602 (s2) | 13/48 | 0 | 0.149 | 4 | 7 | {0:7, 1:1, 2:1, 3:2, 4:1, 5:1} |
| **median** | **14** | **0** | **0.149** | **4** | **7** | — |

### Verdict against pre-registered criteria

- **H_anon_partial CONFIRMED**: median n_deep = 4 (in range 3–4)
- H_anon_inert NOT confirmed (median < 5); H_anon_strong NOT confirmed (median > 2)
- **H_backbone_stable**: CONFIRMED — n_backbone = 7 (median), unchanged from C-NAT's 7–8.
  Sharp contrast with exp-097/098 (backbone collapse to 0): anonymization of a *real*
  world preserves the backbone; an alien world destroys it even with real names.
- **H_conf_preserved**: CONFIRMED — n_conf median 14, within the C-NAT range 11–15.

### Note on s0's elevated Δ_med

s0 shows Δ_med = 0.382 vs. 0.139/0.149 at s1/s2 — driven by its deep heads sitting at
higher Δ (0.61–1.20). Median across seeds (0.149) is C-NAT-like. Seed-level variance in
Δ_med is consistent with what exp-097 showed (0.758–1.185); flagged, not interpreted.

### Placement in the series (see exp-098 note for the two-axis table)

C-NAT-anon sits between C-NAT-half and C-NAT on the ordering axis while keeping the
semantics axis intact: n_deep = 4, backbone 7–8, Δ_med ≈ 0.15. The within-story
world-reference (persistent entities under anonymous tokens) is sufficient for backbone
and near-C-NAT Δ_med; the cross-story name prototype contributes ~1–3 deep heads.
