# exp-094: Narrative Decomposition — Quarter-Story Block Shuffle

**Pre-registered:** 2026-07-23 (physics room session, ~10:30 AM MDT), before any corpus
generation or training.

**Follows:** exp-093 (H_deep_recovery CONFIRMED — half-story always-swap gives n_deep=3
median, vs k=3 block shuffle n_deep=2. Minimum ordering unit is sub-arc. Seed spread 1/3/4
noted as caveat; seed-robustness extension pre-registered for seeds 1403/1404 in exp-093
notes.md / modal_exp093_seeds45.py — launch blocked by Modal network issue at session time.)

---

## The question

exp-093 showed that half-story-length contiguous chunks (always-swapped) recover n_deep from
the shuffled floor (2) to 3, with total conformal count (n_conf=12) fully re-entering the
C-NAT band (11–15). This establishes that local coherence at half-story scale is load-bearing —
the minimum ordering unit is sub-arc.

The resolution of the finding is limited: we know the step-change happens somewhere between
k=3 blocks (~3 sentences) and half-story (~4–5 sentences in TinyStories). exp-094 probes the
lower end of that range: **quarter-story blocks (~2 sentences per block)**.

If quarter-story recovers n_deep ≥ 3: the step-change is at or below ~2 sentences — near k=2
block territory, and the specific "story-structured" nature of the halves in exp-093 may not
be the driver (just chunk length matters).

If quarter-story does not recover (n_deep ≤ 2): the step-change is between ~2 sentences
(quarter-story, no recovery) and ~4–5 sentences (half-story, partial recovery). This localizes
the minimum chunk length at approximately half-story scale within TinyStories (~4–5 sentences).

---

## Anatomy of the result being probed

The narrative decomposition ladder (all at 70m/1B-token scale, GPT-NeoX 6L/8H):

| Condition | Approx. chunk size | n_deep (L3–L5) | n_conf |
|---|---|---|---|
| C-NAT (k=∞, full arc) | full story | 5–7 | 11–15 |
| C-NAT-half (exp-093) | ~4–5 sentences | 3 | 12 |
| C-NAT-block3 (k=3, exp-092) | 3 sentences | 2 | 9 |
| C-NAT-shuf (k=1, exp-091) | 1 sentence | 2 | 8–9 |
| C-NAT-block2 (k=2, exp-092) | 2 sentences | 1 | 9 |
| C-NAT-quarter (exp-094) | ~2 sentences | ? | ? |

Quarter-story sits in the gap between k=2 (2 sentences, n_deep=1) and k=3 (3 sentences,
n_deep=2). Quarter of an 8-sentence story = 2 sentences. This overlaps k=2 block territory
except the blocks are defined by story structure, not global sentence count.

The structural difference between C-NAT-quarter and k=2 block shuffle:
- k=2: globally defined 2-sentence blocks, cut without regard to story boundaries
- quarter-story: blocks defined by story internal structure (first/second/third/last ~25% of each story)

This distinction may or may not matter for the model. The pre-registration registers it as a
potential confound and notes it explicitly.

---

## Corpus design

**C-NAT-quarter:** For each TinyStories story:
1. Tokenize into sentences using the same sentence-split regex: `(?<=[.!?])\s+`
2. If n_sentences < 4: include unchanged (same conservative handling as all prior rungs)
3. If n_sentences >= 4:
   - n_q = n_sentences // 4
   - quarter_0 = sentences[0 : n_q]
   - quarter_1 = sentences[n_q : 2*n_q]
   - quarter_2 = sentences[2*n_q : 3*n_q]
   - quarter_3 = sentences[3*n_q :]  (last quarter gets remainder)
4. Shuffle the four quarters in a per-story random order:
   - per-story rng seed = SHUFFLE_SEED + story_index (mod 2**31)
   - rng = np.random.default_rng(per_story_seed)
   - rng.shuffle(quarters_list)
5. Concatenate shuffled quarters; rejoin sentences with " "

**Why random shuffle (not always-rotate):**
- Cyclic rotation (e.g. [Q2, Q3, Q4, Q1]) creates a learnable systematic transformation —
  the model could learn to "expect" the fourth quarter at the beginning of every story.
- Random per-story shuffle prevents any systematic learnable bias while still thoroughly
  disrupting the global arc.
- With 4! = 24 possible orderings and random selection, no story-level pattern is learnable.
- Consistent with the random shuffling method used in k=2/k=3 block shuffle (exp-092).

**Why this differs from k=2/k=3 block shuffles:**
- k=2/k=3 cut globally across the corpus (every 2 or 3 sentences from a continuous stream)
- C-NAT-quarter cuts at story boundaries (first/last quarter of EACH story's sentence structure)
- This preserves the "beginning of story" / "end of story" character of each quarter, even
  though the quarters are in random order

**Parameters:**
- doc_seed: 3005 (same as C-NAT, C-NAT-shuf, C-NAT-block2, C-NAT-block3, C-NAT-half)
- SHUFFLE_SEED: 4400 (new; per-story seed = 4400 + story_index)
- Corpus size: 1.05B tokens (same as all prior corpora in this series)

**References for n_deep:**
- C-NAT (k=∞, exp-062): n_deep = 5–7
- C-NAT-half (exp-093): n_deep = 3 (median, spread 1/3/4)
- C-NAT-block3 (k=3, exp-092): n_deep = 2 (median)
- C-NAT-shuf (k=1, exp-091): n_deep = 2 (median)
- C-NAT-block2 (k=2, exp-092): n_deep = 1 (median)

---

## Pre-registered hypotheses

**Primary observables (registered before any corpus generation or training):**
1. **n_deep**: deep conformal heads (L3–L5, R² ≥ 0.90, any Δ) — median across 3 seeds
2. **n_conformal**: total conformal heads (R² ≥ 0.90, any Δ) — median across 3 seeds
3. **n_backbone**: L0 conformal heads — median across 3 seeds

**H_quarter_below** (primary): n_deep ≤ 2 — quarter-story blocks do not recover above the
shuffled baseline. Interpretation: the minimum contiguous chunk length for partial deep
conformal recovery is between ~2 sentences (quarter-story, no recovery) and ~4–5 sentences
(half-story, n_deep=3). The step-change is localized at approximately half-story scale within
TinyStories.

**H_quarter_above** (primary, competing): n_deep ≥ 3 — quarter-story blocks begin recovering,
like half-story. Interpretation: even ~2-sentence chunks are sufficient to begin recovering
the deep population. The minimum chunk length is at or below quarter-story scale. Follow-up:
the step-change between k=2 (n_deep=1, globally-cut) and quarter-story (n_deep≥3,
story-cut) would implicate STORY-STRUCTURE as the variable, not chunk length per se.

**Secondary (pre-registered):**
- **H_backbone_stable**: n_backbone (L0) stable at ~7–8 (consistent across all shuffled rungs)

---

## Declared expectations (non-criterial)

**Prior: H_quarter_below** (high confidence).

Reasoning: quarter-story gives ~2 sentences per block, comparable to k=2 (which gave n_deep=1).
The step-change in exp-093 appears to be at approximately half-story scale (~4–5 sentences).
Exp-092 showed no recovery at k=3 (3 sentences); quarter-story is shorter than k=3. The specific
"story-structural" character of quarter blocks is unlikely to compensate for their short length.

Non-criterial expectations:
1. n_deep(quarter): ~1–2 (comparable to k=2 or k=3)
2. n_conf(quarter): ~9 (in the shuffled band, not recovering toward C-NAT)
3. L0 backbone: stable ~7–8 heads (consistent across series)
4. Deep head identities: vary across seeds but constrained to L3–L5 zone

---

## Protocol

**Architecture:** Identical to exp-062/085/091/092/093.
- GPT-NeoX (6 layers, 8 heads, d_k=64, ctx=512)
- Fresh initialization
- 2000 training steps, batch 524288 tokens (1.05B total)
- Optimizer: AdamW, cosine schedule, same hyperparameters throughout series
- pos_enc: rotary (rotary_pct=0.25)

**Corpus:** C-NAT-quarter (quarter-story random block shuffle, as described above)

**Seeds:**
- init seeds: 1500, 1501, 1502 (data_seed 2500)
- 3 independent training runs, same protocol as all prior experiments in series

**Randomized-weights controls (pre-registered):** One control — randomize the
seed-1500 checkpoint. Expected: 0/48 conformal (consistent with every prior control).

**Measurement:** Identical to exp-062/085/091/092/093.
- BCFT power-law fit per head, fit range [8, 256] at seq 512
- R² ≥ 0.90 for conformal
- Deep count = L3–L5 heads meeting criterion
- Layer-wise anatomy check for all seeds

Note: The "cutoff_low=3, max_lag=60" measurement boilerplate from earlier pre-regs was
incorrect; the actual shared measure.py uses fit range [8, 256]. This pre-registration uses
the correct protocol description.

**Infrastructure:** Modal A100-40GB. Estimated cost: ~$22 (one generate phase ~$2,
three train+measure at ~$7 each, one control ~$1).

**Dependency note:** The exp-093 seed-robustness extension (seeds 1403/1404,
modal_exp093_seeds45.py) is pre-registered but not yet launched (Modal network issue at
pre-registration session time). That extension and exp-094 can run independently; exp-094's
design does not depend on the robustness check results, which address the same H_deep_recovery
verdict from a different angle.

---

## What follows

**If H_quarter_below confirmed (n_deep ≤ 2):**
The step-change is localized between ~2 sentences (quarter-story) and ~4–5 sentences
(half-story). TinyStories story "halves" (~4–5 sentences) contain the minimum ordering
structure for partial deep conformal recovery. Candidates for the critical ingredient:
- The within-half narrative mini-arc (setup sentence → first action → initial consequence)
- The total token length of the coherent chunk (~200–400 tokens vs ~60–120 for quarter-story)
Next options: (a) half-story random-shuffle control (vs always-swap) to test whether the
deterministic reversal in exp-093 matters; (b) entity-anonymization rung (names replaced by
[ENT_i], full ordering preserved) to test whether world-reference specifics or ordering alone
are load-bearing; (c) move to the alien-semantics rung (Tier 5 backlog item 16g).

**If H_quarter_above confirmed (n_deep ≥ 3):**
Recovery begins at quarter-story scale, suggesting story-structure (not chunk length per se)
is the variable — k=2 globally-cut blocks and story-quarter-cut blocks both give ~2 sentences
but only the latter recovers. Design exp-095: half-story random-shuffle (not always-swap) to
test the always-swap confound, and/or entity-anonymization to test world-reference specificity.

---

## Status

- [x] Pre-registration written (2026-07-23, physics room session, ~10:30 AM MDT)
- [ ] Pre-registration committed and pushed to 3ld0n/attention-geometry
- [ ] Corpus generation script written (gen_cnat_quarter.py)
- [ ] Modal training/measurement script written (modal_exp094.py)
- [ ] Training runs complete (seeds 1500/1501/1502)
- [ ] Randomized-weights control run
- [ ] Verdict registered
