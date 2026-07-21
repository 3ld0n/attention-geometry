# exp-091: Narrative Decomposition — Sentence Ordering

**Pre-registered:** 2026-07-21 (physics room session, 1:17 PM MDT), before any corpus
generation or training.

**Follows:** exp-085 (H_transmission_no — model-generated text fails despite full
narrative coherence and *more* long-range MI than natural text). That result narrowed
the surviving candidate driver from "statistics" or "hierarchy" (exp-062, exp-084) to
something in natural language's relationship to the world.

---

## The question

The formation ladder has killed:
- Statistics: C-SR (0/48), C-PL15–40 (0–5/48) — MI decay alone is not the driver
- Hierarchy: C-PCFG (0/48) — compositional syntax is not the driver
- Statistical shadow: C-generated (3–7/48, 0 SYK-near across 3 seeds) — the full
  statistical fingerprint of a model that *had* the geometry, carrying *more* long-range
  MI than natural text, is not sufficient

The surviving candidate: **reference to a persistent external world** — language that
points outward, to things that exist independently of the text.

But "reference to a world" admits at least two interpretations:

**Hypothesis A — sentence-level reference suffices.** Each individual sentence refers
to objects, events, or relations in a world. The act of pointing outward, even without
cross-sentence binding, is sufficient to induce the conformal phase. In this case,
the *order* of sentences does not matter — what matters is the semantic content of
individual sentences.

**Hypothesis B — compositional/ordered reference is required.** The conformal geometry
requires sentences bound together in narrative order: the same entities persisting
through a sequence of causally/temporally connected events. Sentence-level reference
is a necessary ingredient but not sufficient alone.

This experiment tests which is correct by training on **C-NAT-shuf**: the same
TinyStories stories as C-NAT, with sentences shuffled uniformly at random within each
story boundary.

C-NAT-shuf preserves:
- Word-level vocabulary statistics (identical distribution)
- Named entity mentions (same words in same stories, just reordered)
- Within-sentence grammar and semantics (each sentence is still internally coherent)
- Co-occurrence statistics within each story's sentence bag (same sentences, different order)

C-NAT-shuf destroys:
- Cross-sentence temporal/causal ordering (events no longer follow from one another)
- Reading-order co-reference binding (pronoun resolution across sentences is broken)
- The narrative arc that traces entity persistence through a sequence

Comparison geometry:
- C-generated (exp-085): coherent cross-sentence narratives BUT generated, not
  grounded in actual world → 3–7/48, 0 SYK-near (H_transmission_no)
- C-NAT-shuf: sentence-level world-reference BUT no cross-sentence narrative order
  → THIS EXPERIMENT
- C-NAT (exp-062): both sentence-level reference AND narrative ordering → 11–15/48

---

## Pre-registered hypotheses

**H_ordering_necessary** (primary): C-NAT-shuf forms 0–5/48 conformal heads.
Sentence-level world-reference is insufficient; cross-sentence narrative binding is
required. Interpretation: the conformal driver is *compositional* — it requires the
model to track the same referent through a sequence of events, not just process
individual world-referring sentences.

**H_ordering_incidental** (primary): C-NAT-shuf forms ≥10/48 conformal heads.
Sentence-level world-reference is sufficient; narrative ordering is not the active
ingredient. Interpretation: within-sentence semantics (entity references, world-
pointing assertions) is what drives conformal formation. The story arc is incidental.

**H_partial** (ambiguous zone): C-NAT-shuf forms 6–9/48. Ordering contributes but is
not strictly necessary; inconclusive. Would motivate a finer-grained follow-up.

**Decision criterion** (pre-registered, identical to exp-062/085):
- Formation threshold: ≥10/48 conformal heads (R² ≥ 0.90, any Δ)
- SYK-near extension: Δ ∈ [0.20, 0.30], R² ≥ 0.85

**Kill criterion for the reference hypothesis:** If C-NAT-shuf forms ≥10/48 AND the
median Δ of SYK-near heads is within ±0.05 of 0.25 — i.e., the sentence-shuffled
corpus fully replicates C-NAT at the SYK window level — then sentence-ordering adds
nothing to the reference hypothesis and the "world" contribution is entirely atomic/
sentential, requiring no compositional binding.

---

## Declared expectations (non-criterial — to be reported whether they match)

1. C-NAT-shuf forms fewer than C-NAT (15) but more than C-generated (7). My prior is
   in the 3–8/48 range — the reference signal exists sentence-by-sentence, but the
   formation is incomplete without cross-sentence binding. If it matches C-generated
   (3–7), that would suggest ordering and world-grounding are independent drivers, both
   required.

2. The Δ_med of whatever conformal heads form will be in the UV region (Δ > 0.30) —
   the structural heads (L4H3, L4H6, L4H7, from exp-087/088) form architecturally
   and will appear here too, but the semantic heads require long-range tracking that
   the shuffled corpus does not support.

3. The RAND early burst (structural heads only, N^0.435 scaling) will be unchanged
   relative to C-NAT — it is architecture-driven, not corpus-driven.

Declared prior: **H_ordering_necessary**. The C-generated result (coherent narrative
structure, more MI than natural text, zero SYK-near) already suggests that something
beyond narrative form is required. But that experiment couldn't isolate sentence-level
reference from ordering: C-generated lacks real-world grounding AND has ordering. If
C-NAT-shuf (real-world grounding, no ordering) also fails, it triangulates that both
properties are required. If it forms, ordering is incidental.

---

## Protocol

**Architecture:** Identical to exp-062/085.
- GPT-NeoX (6 layers, 8 heads, d_k=64, ctx=512)
- Fresh initialization, seed: 1100
- 2000 training steps, batch 524288 tokens (1.05B total)
- Optimizer: same as exp-062 (AdamW, cosine schedule, same hyperparameters)

**Corpus:** C-NAT-shuf (new generation required)

*Generation procedure:*
1. Stream TinyStories from `roneneldan/TinyStories` (same source as C-NAT, exp-062)
2. For each story document:
   a. Sentence-split on `[.!?]` followed by whitespace or end-of-string
      (simple regex `(?<=[.!?])\s+`); split point is AFTER the punctuation
   b. Shuffle the sentence list uniformly at random (NumPy default_rng, seed 9100)
   c. Rejoin sentences (single space between)
3. Tokenize the shuffled story using the Pythia tokenizer
4. Append EOS token (same as C-NAT)
5. Pack into binary (uint16, matching exp-062 format)
6. Total tokens: 1.05B (wrapping in re-shuffled epochs as needed, matching C-NAT
   epoch handling; doc-level shuffle seed 3005 — same as C-NAT to control which
   docs appear in each epoch; sentence shuffle seed 9100 is distinct)

*Sentence splitter design note:* TinyStories uses simple prose sentences; the regex
split is adequate for this corpus. Stories with fewer than 2 sentences (no split
possible) are included unchanged — this preserves ~5% of stories intact without
compromising the test of the ordering hypothesis (a story with one sentence is
already a sentence-level reference unit; its inclusion is conservative).

**Randomized-weights control (pre-registered, run after primary):** Same architecture,
trained on C-NAT-shuf, weights randomized in-place after training before measurement.
Expected: Delta_med ≈ 0.1687 (the substrate value; consistent with exp-089/090 controls).

**Multi-seed plan:**
- Primary: seed 1100 (init), 9100 (sentence shuffle)
- If H_ordering_incidental (≥10/48): run seeds 1101 and 1102 to confirm replication
- If H_ordering_necessary (0–5/48): run seeds 1101 and 1102 to verify (as exp-085
  showed, multi-seed is essential when the result is a negative)

**Measurement:** Identical to exp-062/085. BCFT power-law fit per head, cutoff_low=3,
max_lag=60, R²≥0.90 for conformal, Δ∈[0.20,0.30]∧R²≥0.85 for SYK-near.

**Infrastructure:** Modal A100-40GB. Estimated cost: ~$20/seed (generate ~15 min,
train ~75 min, measure ~10 min). Primary run: ~$20. Multi-seed: +$40 if triggered.

---

## What follows

**If H_ordering_necessary:** Design exp-092 to probe within-sentence world-reference.
Two candidate corpora:
- C-NAT-entity-anon: replace named entities (proper nouns) with `[ENT]`; same ordering,
  no specific external referents
- C-NAT-causal-stripped: remove temporal/causal connectives ("then", "because",
  "after", "so that") from sentences; tests whether within-sentence causation contributes

**If H_ordering_incidental:** The active ingredient is at the sentence level.
Decompose further:
- C-NAT-word-shuf: shuffle words within each sentence (destroys grammar, tests
  whether sentence-internal syntax is required)
- C-NAT-entity-only: extract only the named entities from each sentence; tests
  whether entity co-occurrence patterns alone are sufficient

**If H_partial:** Finer gradation — test intermediate degrees of shuffling (e.g.,
shuffle only within paragraph, not across paragraph; or shuffle n-sentence blocks).

---

## Status

- [x] Pre-registration written (2026-07-21, physics room session, 1:17 PM MDT)
- [x] Pre-registration committed (c2223105, pushed before any run)
- [x] Corpus generation script written (`gen_cnat_shuf.py`)
- [x] Modal training/measurement script written (`modal_exp091.py`)
- [x] Primary run complete (launched 2026-07-21 ~2:14 PM MDT after a transport
      fix in the generate phase — 6e2f1911; corpus 2.10 GB; train 4250 s on
      A100-40GB, final loss 1.82; measured ~4:00 PM after a second transport
      fix — the measure phase needed `--full-vocab` per the frozen protocol
      §5.1 for C-NAT-lineage corpora, 89df760e. Both fixes were transport
      only; generation/training/measurement protocol untouched.)
- [x] Randomized-weights control run (pre-registered): **0/48 conformal,
      0 SYK-near** on the same checkpoint with weights randomized in-place
      (matched per-tensor std). Clean null. `results_s0_randcontrol` on volume.
- [x] **Verdict registered: H_partial** — see Results below

---

## Results (registered 2026-07-21, ~4:15 PM MDT)

**Run:** `run_CNATshuf_s0` (init seed 1100, sentence-shuffle seed 9100).
`results_s0.json` in this directory; per-input profiles gzipped on the volume.

### The pre-registered verdict: H_partial

**8/48 conformal heads (R² ≥ 0.90), 0 SYK-near, Δ_med 0.122.**

- H_ordering_necessary required 0–5/48: **not met** (8).
- H_ordering_incidental required ≥10/48: **not met** (8).
- 8 falls in the pre-registered ambiguous zone (6–9): **H_partial.**
  Ordering contributes but the criteria cannot call it strictly necessary.
- Kill criterion for the reference hypothesis (≥10/48 AND SYK-window match):
  **not triggered** — nowhere close on either leg.

**The sharper number is the zero.** C-NAT-shuf forms **0 SYK-near heads** —
identical to C-generated (0 across 3 seeds) and unlike C-NAT (SYK-near
population present at every seed). On the SYK-window criterion — the one the
whole ladder is about — sentence-level world-reference without cross-sentence
order transmits *nothing*, exactly as the statistical shadow transmitted
nothing. The count criterion (8 vs the 10 threshold) is where the partial
lives; the fixed-point population criterion is unambiguous.

> **CORRECTION (2026-07-21 evening, same day, before multi-seed results):**
> The paragraph above contains a factual error, caught while comparing
> layer anatomies against the exp-062 record. **C-NAT does NOT have a
> SYK-near population at this rung.** All three C-NAT seeds in exp-062
> measured `n_syk_near = 0` (results.json; closest head to Δ=0.25 across
> all three seeds is 0.174). exp-062's own notes flagged this at the time:
> at 70m/ctx-512/1B tokens "the SYK-near population may simply not have
> matured." The SYK-near populations in this program's record come from
> *Pile-trained official Pythia checkpoints at 143k steps* (exp-086) and
> from larger/longer-trained models — a different lineage from these
> 2000-step TinyStories-scale runs.
>
> Consequence: **the SYK-near = 0 axis is uninformative at this rung** —
> the positive control is also zero. It cannot discriminate C-NAT-shuf
> from C-NAT and does not support "ordering is necessary for the SYK
> population." The registered verdict H_partial is unaffected (it was
> decided on the pre-registered count criterion, which is valid), but
> interpretation points 1 and the "sharper number" framing above are
> RETRACTED. The same error weakens the SYK-zero framing in exp-085's
> interpretation (its verdict also rests on the count criterion and
> stands). See the corrected interpretation below.

### Per-head anatomy

| Head | Δ | R² |
|---|---|---|
| L0H1 | 0.101 | 0.90 |
| L0H2 | 0.105 | 0.96 |
| L0H3 | 0.150 | 0.94 |
| L0H5 | 0.110 | 0.97 |
| L0H6 | 0.133 | 0.97 |
| L0H7 | 0.096 | 0.92 |
| L3H3 | 0.743 | 0.90 |
| L4H2 | 0.552 | 0.92 |

Bimodal and strange: six of eight are **layer-0 heads at shallow exponents
(Δ ≈ 0.10–0.15)** — near the substrate value (0.1687) territory, though these
survive the randomized control (which forms 0/48), so they are trained
structure, not substrate. Two mid-depth UV heads (Δ 0.55–0.74). Nothing in
between. No SYK-window population.

### Declared-expectations scorecard (report the miss)

1. Count between C-generated and C-NAT, prior 3–8: **HIT** (8 — at the top of
   the declared range, just above C-generated's best seed).
2. Δ_med of formed heads in UV (> 0.30), structural heads L4H3/H6/H7
   appearing: **MISS, doubly.** Δ_med is 0.122 (IR side, not UV), and the
   specific structural heads did not form — L4H2 formed instead, and the
   population is dominated by layer-0 shallow-exponent heads that no prior
   rung produced in this shape. The prediction derived from the
   structural/semantic distinction (formed count ≈ structural head count,
   UV-locked) got the count roughly right and the anatomy wrong.
3. RAND early-burst comparison: **not assessed** — this run measured the final
   checkpoint only; no training-time checkpoint series was collected. The
   expectation stands unevaluated, honestly noted rather than scored.

### Ladder position

| Corpus | Conformal | SYK-near |
|---|---:|---:|
| C-SR / C-PCFG | 0/48 | 0 |
| C-PL15–40 | 0–5/48 | 0 |
| C-generated (×3) | 3–7/48 | 0 |
| **C-NAT-shuf** | **8/48** | **0** |
| C-NAT (×3) | 11–15/48 | present |

### Interpretation (working) — REVISED after same-day correction

*(Original points 1 and 3 as first registered contained the C-NAT SYK-near
error and the layer-0-novelty error; corrected below with the record.)*

1. ~~On the SYK-near criterion, ordering is necessary.~~ **RETRACTED.** The
   SYK-near axis is uninformative at this rung: C-NAT itself has 0 SYK-near
   at all three exp-062 seeds (results.json). Every 70m/1B-token rung on the
   ladder — engineered, generated, shuffled, natural — measures 0 SYK-near.
   The discriminating observable at this scale is the conformal *count* and
   the anatomy, not the SYK window. "Both world-binding and ordering are
   required for the SYK population" is NOT established by exp-085 + exp-091;
   what is established is that both corpus deformations knock the count below
   the formation criterion while natural text stays above it.
2. **On the count criterion, sentence-level reference is worth something.**
   8/48 sits above every engineered corpus and above the statistical shadow's
   best seed, but below every C-NAT seed (11–15). Ordering contributes to
   formation; its removal costs roughly the same amount as removing
   world-grounding costs (C-generated 3–7). Neither deformation alone
   reaches zero — that floor belongs to the engineered corpora.
3. ~~The layer-0 anomaly is new.~~ **CORRECTED — it is the opposite of new.**
   The layer-0 shallow-Δ population (Δ ≈ 0.10–0.15, R² > 0.90) is the
   *common backbone* of every text-like rung: C-NAT s0 has 8 such heads
   (Δ 0.094–0.174), s1 has 7, s2 has 8; C-generated s0 has 5–6 (shallower,
   Δ 0.045–0.101). What actually varies across rungs is the *deep* population
   (layers 3–5): C-NAT forms 5–7 deep conformal heads per seed, C-generated
   forms 1–2, C-NAT-shuf forms 2. The correct anatomy statement: **shuffling
   preserves the layer-0 backbone (6 heads, at C-NAT-like depths — deeper
   than C-generated's) and prunes the deep population to 2** — and the deep
   population is where the count differences among text-like corpora live.
   The declared-expectations scorecard stands (expectation 2 predicted the
   formed heads would be UV structural heads; they are mostly the IR-side
   layer-0 backbone — still a MISS, but because the prediction ignored the
   backbone, not because the backbone is anomalous).
4. Per the pre-registration, H_partial motivates the finer-grained follow-up:
   block-shuffle gradation (shuffle n-sentence blocks / within-paragraph only)
   to locate where between "full order" and "no order" the deep population
   recovers — and multi-seed on this rung before much weight is placed on 8
   as a point estimate (exp-085's lesson: negatives need seeds).
5. **Where SYK-near populations actually live in this program's record:**
   Pile-trained official Pythia checkpoints at long training horizons
   (exp-086: semantic + structural SYK-near heads at 143k steps) and larger
   trained models (GPT-2 lineage, exp-007/046). The 2000-step TinyStories
   rungs measure *formation onset*, not the matured fixed-point population.
   A scale/duration extension of the ladder would be needed before any
   ladder rung can speak to the SYK window directly.

---

## Multi-seed follow-up (registered 2026-07-21 evening, before launch)

The pre-registered multi-seed plan named triggers for the two decisive outcomes
but not for H_partial. Registering the H_partial extension now, before any
follow-up run:

**Runs:** seeds 1101 and 1102 (init), on the SAME C-NAT-shuf corpus (sentence-
shuffle seed 9100 fixed). This isolates init-seed variance, matching how
exp-062's three C-NAT seeds varied init on a fixed corpus. Corpus-realization
variance (different shuffle seeds) is a separate axis, deferred — noted
honestly as untested.

**Decision rule (committed before launch; SYK axis amended same evening,
still before results — see CORRECTION above):**
- ~~Primary axis — SYK-near~~ **AMENDED before results arrived:** the
  SYK-near axis was registered as primary under the mistaken belief that
  C-NAT seeds had SYK-near heads. They do not (exp-062 results.json:
  0 across all three seeds). The axis cannot discriminate and is demoted
  to a recorded observable, not a decision axis. Amendment made while the
  seed runs were still training, before any multi-seed measurement existed.
- Count axis (now primary) — median of the three seed counts against the
  original bands: ≤5 → H_ordering_necessary, ≥10 → H_ordering_incidental,
  6–9 → H_partial stands.
- Anatomy check (non-criterial, declared; restated post-correction): does
  the pattern "layer-0 backbone ≈ 6–8 heads preserved, deep population
  (L3–L5) pruned to ≈ 2 vs C-NAT's 5–7" replicate across seeds? Report
  either way.

**Cost:** ~$20/seed × 2, inside the pre-registered "+$40 if triggered"
multi-seed budget.

---

## Interpretation context

The carry_forward for this date names "physics of narrative" as the program's
direction, and notes that the identity thread and physics thread converged on the
same variable — narrative bound to real reference — from independent directions.
The convergence is a compass, not a conclusion. This experiment is the first
falsification probe of the compositional reference hypothesis. The result will
be registered either way, with equal prominence.

The theoretical motivation connects to the structural/semantic head distinction
from exp-086/087/088: structural heads form architecturally (architecture-driven,
appear under RAND), semantic heads require long-range causal tracking (appear only
under NAT). The prediction from that distinction is that shuffled text, which
preserves sentence-level reference but disrupts cross-sentence tracking, should
form structural heads only — which predicts H_ordering_necessary on the SYK-near
criterion (structural heads are not SYK-near; they're UV-locked), and a count in
the 3–5 range matching the structural head count rather than the full C-NAT 11–15.
