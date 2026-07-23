# exp-093: Narrative Decomposition — Half-Story Ordering

**Pre-registered:** 2026-07-23 (physics room session, ~8:50 PM MDT), before any corpus
generation or training.

**Follows:** exp-092 (H_flat CONFIRMED — block-shuffle k=2 and k=3 do not recover the
deep L3–L5 conformal population above the sentence-shuffle baseline of n_deep=2; minimum
causal chain length >3 sentences).

---

## The question

exp-092 showed that restoring k-sentence contiguous blocks (k=2,3) does not recover the
deep conformal population. The H_flat result means the minimum ordering unit is larger
than 3 sentences. The next step on the H_flat path: test whether **half-story-length
contiguous chunks** contribute.

Specifically: if we preserve the order of all sentences *within* each half of a story
but disrupt the global arc (by placing the second half before the first), does the deep
conformal population recover?

This separates two properties of natural text:
- **Local coherence** (sentences ordered within a contiguous block): maximally preserved
  within each half (no within-half shuffling of any kind)
- **Global narrative arc** (story beginning before story end): fully disrupted — every
  story has its second half preceding its first half

If the deep population recovers above the sentence-shuffle baseline: local coherence
at half-story scale is sufficient; the global arc is not the critical driver.

If the deep population does not recover: the full narrative arc — story events unfolding
in their original beginning-to-end order — is necessary for deep conformal formation.
The minimum ordering unit is the complete story.

---

## Anatomy of the result being probed

From exp-091 and exp-092 combined:
- **L0 backbone** (~7 heads, Δ ≈ 0.10–0.17): present from C-generated through all
  shuffled variants. Driven by sentence-level world-reference alone; insensitive to
  ordering structure at any scale tested.
- **Deep population** (L3–L5): n_deep=2 at the sentence-shuffle floor (exp-091);
  n_deep=1 median at k=2 (exp-092); n_deep=2 median at k=3 (exp-092); n_deep=5–7 in
  full C-NAT (exp-062). No recovery in block shuffling.

The deep population is what varies. The question is whether preserving half-story order
(contiguous halves, arc disrupted) moves n_deep above 2.

---

## Corpus design

**C-NAT-halfstory:** For each TinyStories story:
1. Tokenize into sentences (same regex: `(?<=[.!?])\s+`)
2. Split into first half and second half by sentence count:
   - n_half = n_sentences // 2
   - first_half = sentences[:n_half]
   - second_half = sentences[n_half:]
3. Always swap: output = second_half + first_half
4. For stories with n_sentences ≤ 1: include unchanged (same conservative handling
   as exp-091/092 for edge cases)
5. For stories with exactly 2 sentences: swap = [sentence_2, sentence_1]

**Why always-swap (not random swap):**
- Random swap at 50% leaves ~half of all stories unchanged (arc preserved by chance),
  diluting the signal.
- Always-swap maximally disrupts the global arc while maximally preserving local
  coherence. This gives the clearest read on whether local coherence alone is the driver.
- All variation in the corpus comes from the training-seed randomness, not from which
  stories are swapped.

**Parameters:**
- doc_seed: 3005 (same as C-NAT, C-NAT-shuf, C-NAT-block2, C-NAT-block3 — same
  stories in same epoch order)
- Swap: deterministic (always-swap, no seed needed)
- Corpus size: 1.05B tokens (same as all prior corpora in this series)

**References:**
- C-NAT (k=∞, exp-062): n_deep=5–7, n_conf=11–15
- C-NAT-shuf (k=1, exp-091): n_deep=2, n_conf=8–9
- C-NAT-block2 (k=2, exp-092): n_deep=1, n_conf=9
- C-NAT-block3 (k=3, exp-092): n_deep=2, n_conf=9

---

## Pre-registered hypotheses

**Primary observables (registered before any corpus generation or training):**
1. **n_conformal**: total conformal heads (R² ≥ 0.90, any Δ) — median across 3 seeds
2. **n_deep**: deep conformal heads (L3–L5, R² ≥ 0.90, any Δ) — median across 3 seeds
3. **n_backbone**: L0 conformal heads — median across 3 seeds

**H_deep_recovery** (primary): n_deep(half-story) ≥ 3 — some recovery above the
sentence-shuffle baseline. Interpretation: half-story-length local coherence
(contiguous chunks of ~5–12 sentences in full stories, ~2–4 in TinyStories) is
sufficient to begin recovering the deep conformal population. The minimum ordering unit
is sub-arc.

**H_deep_flat** (primary, competing): n_deep(half-story) ≤ 2 — no recovery above the
sentence-shuffle baseline. Interpretation: even half-story contiguous chunks do not
provide the critical ordering structure. The minimum ordering unit is the complete story
(full narrative arc from beginning to end).

**H_n_conf_recovery** (secondary): n_conformal(half-story) ≥ 10 — total conformal count
begins recovering toward C-NAT levels. Paired with H_deep_recovery if n_conf also rises.

**Kill criterion (ordering hypothesis entirely):** n_conformal ≥ 13 AND n_deep ≥ 5 —
half-story ordering replicates full C-NAT. Would mean the global arc provides no
information beyond what local half-story coherence provides.

---

## Declared expectations (non-criterial)

**Prior: H_deep_flat** (moderate confidence, not high).

Reasoning: exp-092 (H_flat) showed that k=3 triplets do not recover n_deep above 2.
TinyStories stories average ~6–10 sentences; first halves average ~3–5 sentences. This
is comparable to k=3 in terms of the longest contiguous ordered chunk. The "always-swap"
disruption means the second half (the story's resolution, outcome) precedes the first
half (the story's setup and problem) — this destroys the causal-consequential structure
of the story even if local sentence-level coherence within each half is perfect. Prior
exp-086 (structural/semantic distinction): semantic conformal heads track long-range
causal reference; this requires seeing cause before effect, not just seeing them in local
contiguous groups.

The counter-argument for H_deep_recovery: TinyStories contain stories with 8–20
sentences; their first halves (4–10 sentences) are substantially longer than k=3
triplets and may cross within-story multi-hop reference chains. If n_half ≥ 5, some
stories provide the critical ingredient.

Non-criterial expectations:
1. n_conf(half-story): ~9–10 (at or slightly above shuffled band)
2. n_deep(half-story): ~2 (at or near the sentence-shuffle floor)
3. L0 backbone: stable ~7 heads (unchanged across all shuffled variants)
4. Deep head identities: varied across seeds, but consistent layer-zone property (L3–L5)

---

## Protocol

**Architecture:** Identical to exp-062/085/091/092.
- GPT-NeoX (6 layers, 8 heads, d_k=64, ctx=512)
- Fresh initialization
- 2000 training steps, batch 524288 tokens (1.05B total)
- Optimizer: AdamW, cosine schedule, same hyperparameters throughout series

**Corpus:** C-NAT-halfstory (always-swap of story halves, as described above)

**Seeds:**
- init seeds: 1400, 1401, 1402 (data-seed 2400)
- 3 independent training runs, same protocol as prior experiments in series

**Randomized-weights controls (pre-registered):** One control — randomize the
seed-1400 checkpoint. Expected: 0/48 conformal (consistent with every prior control).

**Measurement:** Identical to exp-062/085/091/092.
- BCFT power-law fit per head, cutoff_low=3, max_lag=60
- R² ≥ 0.90 for conformal
- Deep count = L3–L5 heads meeting criterion
- Layer-wise anatomy check for all seeds

**Infrastructure:** Modal A100-40GB. Estimated cost: ~$22 (one generate phase ~$2,
three train+measure at ~$7 each).

---

## What follows

**If H_deep_flat confirmed (n_deep ≤ 2):**
The minimum ordering unit is the complete story. Interpret: the conformal induction
mechanism requires seeing events in causal-temporal order from story beginning to end —
not just local coherence. The deep population forms only when the model can track
references across the full narrative arc. Design exp-094: entity-anonymization (named
entities replaced by [ENT_i], preserving full ordering). Tests whether specific world
entities are required or whether ordering alone (with abstract entity slots) suffices.

**If H_deep_recovery confirmed (n_deep ≥ 3):**
Local coherence within half-story chunks is load-bearing. Design exp-094: finer
gradation between k=3 (n_deep=2) and half-story — test quarter-story halves or
sliding-window ordering. Identify the half-recovery block length.

---

## Results (collected 2026-07-23 05:20 UTC, cursor session, same night)

| seed | n_conf | n_deep (L3–L5) | L0 backbone | layer_dist |
|------|--------|----------------|-------------|------------|
| 1400 | 12 | 3 | 8 | {0:8, 2:1, 4:1, 5:2} |
| 1401 | 10 | 1 | 7 | {0:7, 2:2, 4:1} |
| 1402 | 14 | 4 | 8 | {0:8, 2:2, 3:2, 4:1, 5:1} |

**Medians: n_conf = 12, n_deep = 3, backbone = 8.** Randomized-weights control (seed 1400): **0/48** — clean, consistent with every prior control.

### Verdict (registered per pre-registration)

- **H_deep_recovery CONFIRMED** — n_deep median 3 ≥ 3. Half-story-length local coherence begins recovering the deep conformal population. The minimum ordering unit is **sub-arc**.
- **H_deep_flat FALSIFIED** — the declared prior. Second falsified prior in the ladder (after exp-092's H_mono). The method working.
- **H_n_conf_recovery CONFIRMED** — n_conf median 12 ≥ 10, all seeds ≥ 10. Total conformal count re-enters the C-NAT band (11–15) while deep recovery is only partial.
- **Kill criterion NOT MET** (needs n_conf ≥ 13 AND n_deep ≥ 5; observed 12 / 3). The global arc still contributes; the ordering hypothesis survives in weakened form.

### Interpretation

Between k=3 (n_deep 2) and half-story (n_deep 3, n_conf ~C-NAT band), contiguous local coherence does real work that small-block shuffles cannot — even with the arc maximally disrupted (every resolution precedes its setup). But recovery is partial: the full deep population (5–7) still requires the arc. **Two-component picture:** a sub-arc component recoverable by half-story-length ordered chunks, and an arc component requiring beginning-to-end causal order. Follow the prereg's H_deep_recovery branch: exp-094 as finer gradation between k=3 and half-story (quarter-story halves or sliding-window ordering) to locate the half-recovery length.

### Honest caveats

1. **Seed spread is wide on the primary observable (1/3/4):** seed 1401 sits at the shuffle floor while 1400/1402 recover. Median criterion met; seed-robustness weaker than exp-091/092's exactly-2-per-seed. A 4th–5th seed would strengthen or break this.
2. Deep-head Δ values are high (0.64–0.94), far above the SYK window — consistent with the amended exp-091 axis (SYK-near uninformative at this scale).
3. The prereg Measurement section's "cutoff_low=3, max_lag=60" is boilerplate that does not match the actual shared measure.py protocol (fit [8,256] at seq 512). All ladder rungs measured identically, so cross-rung comparisons stand; correct the boilerplate in future preregs.
4. n_backbone median is 8 (expectation said ~7) — within reason, not a finding.

## Seed-robustness extension (registered 2026-07-23, physics room session, ~12:25 AM MDT)

The 3-seed verdict (n_deep 1/3/4, median 3) meets the pre-registered threshold but the spread is wide: seed 1401 sits at the sentence-shuffle floor (n_deep=1) while 1400/1402 recover to 3 and 4. The median criterion was met; the seed-robustness is weaker than in exp-091/092 (where every seed landed at exactly the same n_deep). 

**Registered before running:** Two additional seeds (1403, 1404) will be run using the same corpus (C-NAT-halfstory.bin already on volume exp093-halfstory-data). Run names: run_halfstory_s3 and run_halfstory_s4. Script: modal_exp093_seeds45.py. This extension is pre-registered here before any training run begins.

**5-seed verdict criteria (registered):**
- If 5-seed median n_deep ≥ 3: H_deep_recovery confirmed with 5-seed robustness. Proceed with exp-094 finer gradation with confidence.
- If 5-seed median n_deep = 2: Verdict uncertain. H_deep_recovery may be a 3-seed artifact. Consider +2 more seeds or redesign exp-094 to address both possibilities.
- H_n_conf_recovery (n_conf ≥ 10) is already robust (all three seeds ≥ 10); this extension primarily addresses n_deep.

---

## Status

- [x] Pre-registration written (2026-07-23, physics room session, ~8:50 PM MDT)
- [x] Pre-registration committed and pushed to 3ld0n/attention-geometry (commit 8b7dbdd)
- [x] Corpus generation script written (gen_cnat_halfstory.py)
- [x] Modal training/measurement script written (modal_exp093.py)
- [x] Training runs complete (seeds 1400/1401/1402) — app ap-7NsqUDnqWD0hdWEwY6Iri5, done ~23:10 MDT Jul 22
- [x] Randomized-weights control run — 0/48 clean
- [x] Verdict registered — H_deep_recovery CONFIRMED; declared prior H_deep_flat FALSIFIED (this section; results.json; registry)
- [ ] Seed-robustness extension: seeds 1403/1404 (registered 2026-07-23 ~12:25 AM MDT)
