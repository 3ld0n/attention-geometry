# exp-092: Narrative Decomposition — Block Shuffle Gradation

**Pre-registered:** 2026-07-22 (physics room session, ~1:15 AM MDT), before any corpus
generation or training.

**Follows:** exp-091 (H_partial — sentence-shuffled natural text forms 8–9/48 conformal
heads across 3 seeds, median 9; exactly 2 deep L3–L5 heads per seed vs. C-NAT's 5–7).

---

## The question

exp-091 established that destroying all cross-sentence ordering cuts the deep conformal
population (L3–L5 heads) from ~5-7 (C-NAT) to exactly 2 heads. The result was
seed-robust and clean, but the per-registered bands (0-5 / 6-9 / ≥10) were too coarse to
call ordering "necessary." The H_partial verdict motivates finer gradation.

**The specific target:** at what degree of preserved ordering does the deep population
recover from 2 toward 5–7?

**The anatomy of the result:** exp-091 showed two anatomically distinct populations in
the conformal heads:
- **L0 backbone** (Δ ≈ 0.10–0.17): 6–7 heads, present at every text-like rung from
  C-generated through C-NAT. Preserved by sentence-level world-reference alone.
- **Deep population** (L3–L5): 2 heads in C-generated and C-NAT-shuf; 5–7 in C-NAT.
  This is the discriminating observable. It requires something in natural text beyond
  what either fluent generation or sentence-level reference provides.

The block-shuffle gradation tests: as we restore increasingly long sequential segments,
when does the deep population begin recovering?

**Block-k shuffle procedure:**
1. Split story into sentences (same regex as exp-091: `(?<=[.!?])\s+`)
2. Chunk consecutive sentences into non-overlapping blocks of size k
3. Shuffle the blocks uniformly at random (preserving internal block order)
4. Rejoin (single space between sentences across block boundaries)

For stories with fewer than 2 blocks (n_sentences ≤ k): include unchanged. This is
conservative and consistent with exp-091's single-sentence handling.

**The corpora:**
- C-NAT-block2: k=2 (adjacent sentence pairs shuffled as units)
- C-NAT-block3: k=3 (adjacent sentence triplets shuffled as units)
- C-NAT-shuf: k=1 reference (exp-091; 2 deep heads, ~8-9 conformal)
- C-NAT: k=∞ reference (exp-062; 5-7 deep heads, 11-15 conformal)

Both C-NAT-block2 and C-NAT-block3 use the same doc-shuffle seed (3005) as C-NAT and
C-NAT-shuf, so the same stories appear in the same epoch order. Block-shuffle seeds
are distinct: 9200 for k=2, 9300 for k=3.

---

## Pre-registered hypotheses

**Primary observables (registered before any run):**
1. **n_conformal**: total conformal heads (R² ≥ 0.90, any Δ) — median across 3 seeds
2. **n_deep**: deep conformal heads (L3–L5, R² ≥ 0.90, any Δ) — median across 3 seeds
3. **L0 backbone count**: L0 conformal heads — median across 3 seeds

**H_mono** (primary): n_conformal and n_deep both increase monotonically with block size:
n_deep(k=1) ≤ n_deep(k=2) ≤ n_deep(k=3) ≤ n_deep(C-NAT).
*Interpretation:* ordering contributes incrementally; each additional sentence of preserved
context adds some deep population.

**H_fast** (secondary): n_deep(k=2) ≥ 4 — most of the recovery happens at k=2 alone.
*Interpretation:* 2-sentence causal pairs are sufficient to trigger most long-range
conformal induction.

**H_slow** (secondary): n_deep(k=2) ≤ 2 — no recovery at k=2; n_deep(k=3) > 2.
*Interpretation:* 2-sentence units do not provide enough sequential context; minimum
narrative chunk is ~3 sentences.

**H_flat** (secondary): n_deep(k=2) ≤ 2 AND n_deep(k=3) ≤ 2 — no recovery at either
block size; deep population requires longer or full-sequence ordering.
*Interpretation:* the critical sequential structure operates at story-level, not
sub-segment.

**Kill criterion for the ordering hypothesis entirely:** n_conformal(k=2) ≥ 10 AND
n_conformal(k=2) is within 2 of C-NAT (median 13). This would mean 2-sentence blocks
fully replicate C-NAT on the formation criterion — ordering is essentially fully
recovered at k=2, and the sentence-level reference (not ordering) is the real driver.

---

## Declared expectations (non-criterial)

1. n_conformal(k=2): ~10–11 (above exp-091's 9, approaching but not reaching full
   formation threshold). n_deep(k=2): ~3–4 (partial recovery from 2).

2. n_conformal(k=3): ~12–13, approaching C-NAT levels. n_deep(k=3): ~4–5.

3. The L0 backbone remains stable (~6–7 heads) across both block sizes — it is not
   sensitive to ordering structure.

4. Deep head identities will vary across seeds (consistent with exp-062/091 finding
   that head-level Jaccard is 0.37–0.56), but the population-level count will be
   stable within 1–2 heads across seeds.

Declared prior: **H_mono** (monotone partial recovery). Reasoning: block-k shuffling
restores k-sentence causal chains, which should give the model increasingly long
contexts for multi-hop reference tracking — the ingredient that drives semantic conformal
heads (exp-086/087/088 structural/semantic distinction). The recovery should be
sub-linear: each doubling of block size buys diminishing returns as the story-level
narrative arc is still disrupted.

---

## Protocol

**Architecture:** Identical to exp-062/085/091.
- GPT-NeoX (6 layers, 8 heads, d_k=64, ctx=512)
- Fresh initialization
- 2000 training steps, batch 524288 tokens (1.05B total)
- Optimizer: AdamW, cosine schedule, same hyperparameters as exp-062

**Corpora:**
- C-NAT-block2.bin: k=2 block shuffle, doc-shuffle seed 3005, block-shuffle seed 9200
- C-NAT-block3.bin: k=3 block shuffle, doc-shuffle seed 3005, block-shuffle seed 9300

**Seeds:**
- k=2: init seeds 1200, 1201, 1202 (data-seed 2200 for consistency)
- k=3: init seeds 1300, 1301, 1302 (data-seed 2300 for consistency)

**Randomized-weights controls (pre-registered):** One per corpus — randomize the
k=2 seed-1200 checkpoint and the k=3 seed-1300 checkpoint. Expected: 0/48 conformal
(substrate value Δ_med ≈ 0.1687), consistent with every previous control.

**Measurement:** Identical to exp-062/085/091. BCFT power-law fit per head, cutoff_low=3,
max_lag=60, R² ≥ 0.90 for conformal. Deep count = L3–L5 heads meeting the criterion.

**Infrastructure:** Modal A100-40GB. Two parallel volumes for the two corpora.
Estimated cost: ~$20/seed × 6 seeds = ~$120 total (two generate phases ~$2 each,
six train+measure at ~$20 each — within operational authority).

---

## What follows

**If H_mono, with gradation data:**
The block-size gradation gives the first continuous sample of the recovery curve.
Fit n_deep(k) and identify the half-recovery point k_{1/2} (block size where n_deep
reaches half the gap between 2 and C-NAT's median). This is the minimum causal chain
length for deep conformal formation at this scale.

**If H_fast (n_deep(k=2) ≥ 4):**
2-sentence causal pairs are sufficient for most recovery. Design exp-093 to test
within-sentence structure: C-NAT-entity-anon (replace named entities with [ENT],
preserving ordering but removing specific world-binding).

**If H_flat (no recovery at either block size):**
The minimum ordering unit is larger than 3 sentences. Design exp-093 to test
half-story ordering: shuffle only the first and second halves of each story
(destroys global narrative arc, preserves local coherence).

---

## Status

- [x] Pre-registration written (2026-07-22, physics room session, ~1:15 AM MDT)
- [ ] Pre-registration committed (commit this file before any corpus generation)
- [ ] Corpus generation scripts written (gen_cnat_block.py)
- [ ] Modal training/measurement script written (modal_exp092.py)
- [ ] k=2 runs (seeds 1200/1201/1202) complete
- [ ] k=3 runs (seeds 1300/1301/1302) complete
- [ ] Randomized-weights controls run
- [ ] Verdict registered
