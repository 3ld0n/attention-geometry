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
- [ ] Pre-registration committed (commit hash to be added here)
- [ ] Corpus generation script written (`gen_cnat_shuf.py`)
- [ ] Modal training/measurement script written (`modal_exp091.py`)
- [ ] Primary run complete
- [ ] Verdict registered

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
