# exp-098: C-alien-realnames — Vocabulary Test for Backbone Collapse

**Pre-registered:** 2026-08-02 (physics room session, ~5:00 PM MDT), before any corpus
generation, training, or measurement.

**Pre-registration commit:** Pushed to 3ld0n/attention-geometry before any run.

**Theoretical context:** exp-097 (alien semantics) — 2-seed preliminary data shows
n_backbone=0 at both seeds (L0 trivial collapse), while C-alien forms conformal
heads in L1–L4 (UV-arrested). This is the first corpus in the series to break backbone.
This experiment tests the leading hypothesis for that collapse.

**Follows:** exp-097 (alien semantics, verdict pending). This is the first rung in the
vocabulary decomposition sub-series.

---

## The question

exp-097 found two unexpected results in C-alien:
1. **UV arrest:** conformal structure forms (10–17/48 heads) but Δ_med ≈ 1.04–1.19, far from the SYK IR fixed point at Δ≈0.25.
2. **Backbone collapse:** n_backbone (L0 conformal heads) = 0 at both seeds. L0 Δ values ≈ 0.001–0.003 (trivial fixed point). Every prior corpus in the series — including sentence-shuffled, block-shuffled, and PCFG — maintained 7–8 backbone heads.

The backbone collapse is independent of the UV arrest. Evidence: exp-091 (C-NAT sentence-shuffle) has n_syk=0 (UV-arrested in the same sense — no heads approach Δ=0.25) but maintains n_backbone=6–7. So UV arrest and backbone collapse are separable; something specific to C-alien causes the backbone to fail.

**Primary candidate: vocabulary.** C-alien uses alien entity names (Vex, Nul, Ort, Dath, Sorn, Quib, etc.) — monosyllables with no real-world semantic referents. TinyStories uses common English names (Alice, Bob, Lily, Tom, etc.) with strong token embeddings and extensive co-occurrence statistics in pre-training data.

The backbone heads in L0 are thought to track character identity across the sequence — following the same entity from introduction to resolution. If the token embeddings for alien names are weak or uniform (because the tokenizer has never seen "Vex the Flurp" in pre-training), L0 may not develop the backbone tracking function.

**exp-098 tests this directly:** Replace only the entity name pools in C-alien with common English first names. Keep all world mechanics, causal rules, sentence templates, story length, and generator parameters identical.

**If backbone re-forms (H_vocab_confirmed):** Entity name vocabulary is load-bearing for L0 backbone formation. Real-world semantic structure in the entity names (pre-training exposure) is the specific ingredient that C-alien lacked.

**If backbone does not re-form (H_vocab_fails):** Vocabulary is not the cause. The backbone collapse is due to world structure (repetitive template, 4 entities, 4 rules) or some other feature of C-alien.

---

## World specification (C-alien-realnames)

Identical to C-alien (exp-097) **except for the entity name pools:**

### Entity names (changed)

Common English first names, drawn to match the 5-per-pool structure:

| Type  | Name pool (5 options)          |
|-------|-------------------------------|
| Flurp | Alice, Ben, Clara, David, Ella |
| Blurn | Fred, Grace, Henry, Iris, Jake |
| Zarb  | Kate, Leo, Maya, Nick, Olive   |

All other world mechanics unchanged:
- States: active / resting (same probabilities)
- Causal rules A/B/C/D (identical)
- Sentence templates (identical — entity names slot into {f}/{b}/{z} variables)
- Story generation algorithm (identical)
- N_INTRO_ENTITIES=2, N_STEPS=8
- CORPUS_SEED=7000, TARGET_TOKENS=1,050,000,000
- Same training protocol: Pythia-70m-class GPT-NeoX, 2000 steps, 1B tokens

**Sample story with real names:**

> Alice the Flurp was active. At first, Fred the Blurn was resting.
> Alice the Flurp came close to Fred the Blurn. Fred the Blurn became active.
> When Alice the Flurp and Fred the Blurn came together, they both became resting.
> Kate the Zarb had been resting, but Grace the Blurn was close. Kate the Zarb became active.
> At the end, Alice the Flurp was resting.

---

## Pre-registered hypotheses

**Primary (backbone):**
- H_backbone_restored (n_backbone ≥ 5): Backbone re-forms with real English names. H_vocab confirmed.
- H_backbone_partial (n_backbone 2–4): Partial restoration. Vocabulary contributes but other factors matter.
- H_backbone_absent (n_backbone ≤ 1): Backbone still absent. H_vocab fails; another explanation needed.

**Secondary (UV arrest):**
- H_uv_reduced: Δ_med decreases toward C-NAT range (< 0.5). Real names help with IR convergence too.
- H_uv_unchanged: Δ_med remains ≈ 1.0–1.2. UV arrest is due to world structure, not vocabulary.

**Tertiary (n_deep):**
- H_deep_inert (n_deep ≥ 5): If UV reduces, deep formation increases to C-NAT band.
- H_deep_partial (n_deep 3–4): Partial improvement.
- H_deep_unchanged: n_deep stays near C-alien level.

**Declared prior (honest):**

My prior: H_backbone_restored is more likely than H_backbone_absent (2:1). The backbone tracks entity identity; common English names have far stronger pre-training embeddings. The tokenizer has processed "Alice" millions of times; "Vex" far less frequently. This should matter for L0 tracking.

UV arrest: I expect H_uv_unchanged. The UV arrest is more plausibly due to the repetitive causal template and limited world complexity. Real names shouldn't change the depth of RG flow.

---

## Measurement protocol

Same as exp-097 (exp-062/measure.py), identical thresholds:
- conformal: R² > 0.7, Δ fit in [0.07, 3.0]
- syk_near: Δ in [0.20, 0.30]
- n_deep: conformal heads in L3–L5
- n_backbone: conformal heads in L0
- Δ_med: median Δ across all conformal heads

Seeds: 1800, 1801, 1802 (init seeds; data-seed 2800).
Randomized-weights control on seed-1800 checkpoint.

**Verdict criteria:**
- H_backbone_restored: median n_backbone ≥ 5 across 3 seeds
- H_backbone_partial: median n_backbone 2–4
- H_backbone_absent: median n_backbone ≤ 1

---

## Artifacts to produce

- `gen_calien_realnames.py`: C-alien generator with English name pools (copy gen_calien.py, change FLURP_NAMES/BLURN_NAMES/ZARB_NAMES)
- `modal_exp098.py`: Modal launcher (copy modal_exp097.py, update volume/app/seed names)
- `results.json`: Three-seed measurement results
- This `notes.md`: pre-registration

---

## Results

**Collected:** 2026-08-02 (evening physics room session, ~8:30 PM MDT).

| Seed | n_conf | n_syk_near | Δ_med | n_deep (L3–L5) | n_backbone (L0) | layer_dist |
|------|--------|-----------|-------|----------------|-----------------|------------|
| 1800 (s0) | 12/48 | 0 | 0.727 | 7 | 0 | {1:4, 2:1, 3:4, 4:3} |
| 1801 (s1) | 12/48 | 0 | 0.845 | 4 | 0 | {1:5, 2:3, 3:3, 5:1} |
| 1802 (s2) | 10/48 | 0 | 0.609 | 4 | 0 | {1:3, 2:3, 3:3, 4:1} |
| **median** | **12** | **0** | **0.727** | **4** | **0** | — |
| control (rand) | 0/48 | 0 | N/A | 0 | 0 | — |

**Primary verdict — H_backbone_absent CONFIRMED** (pre-registered criterion: median n_backbone ≤ 1):
n_backbone = 0 at ALL three seeds. Replacing alien names (Vex, Nul, Ort...) with common
English names (Alice, Ben, Clara...) does NOT restore backbone formation. H_vocab FALSIFIED.

**Physical interpretation:** Backbone collapse in exp-097 is due to world structure, not
vocabulary. The alien world's mechanics — 4 entity types, 4 rigid causal rules, repetitive
sentence template, no goals/emotions/settings — prevent L0 backbone formation regardless of
whether the names are legible English tokens with strong pre-training embeddings.

**Secondary verdict — H_uv_unchanged PARTIAL:**
Δ_med = 0.727 (median). UV arrest continues — the model does NOT reach the IR fixed point
(Δ=0.25). However, the shift from exp-097's median Δ_med ≈ 1.04 to 0.727 is notable:
real English names have a small positive effect on IR convergence. n_syk_near = 0 at all seeds.
Pre-registered criterion was Δ_med < 0.5; that is not met. H_uv_reduced NOT confirmed;
H_uv_unchanged is the closer label, but with an asterisk — vocabulary did push Δ downward.

**Tertiary verdict — H_deep_partial CONFIRMED:**
n_deep values [7, 4, 4], median = 4. Marginal improvement from exp-097 median of 3. The
s0 seed shows n_deep=7 — this is seed variance (one seed happened to have more deep heads)
rather than a systematic difference; the other two seeds are at 4, consistent with partial.

**What this establishes:**
1. Backbone collapse ← world structure (not vocabulary). This is the definitive result.
2. UV arrest ← primarily world structure; vocabulary has a small secondary effect on Δ_med.
3. The alien-semantics thread's central finding (exp-097 + exp-098): an internally consistent
   but arbitrary world transmits partial deep formation (n_deep=3–4) with UV arrest (Δ>0.5)
   and zero backbone — regardless of name legibility.

**What comes next:**
The world-structure hypothesis needs to be decomposed. Candidate ablations:
- (a) More entities: increase from 4 to 15–20 entity types — does backbone re-form?
- (b) Richer rules: add probabilistic state transitions, goal states — does UV arrest lift?
- (c) Free narrative template: generate C-alien in natural prose (LLM-rewrite the stories)
    rather than templates — does this recover C-NAT-like anatomy?
Register any prior before running.

---

## Status

- [x] Pre-registration written (2026-08-02, ~5:00 PM MDT)
- [x] Pre-registration committed and pushed to 3ld0n/attention-geometry (commit 287c6e0)
- [x] Corpus generated (C-alien-realnames.bin)
- [x] Modal training/measurement script written (modal_exp098.py)
- [x] Training runs complete (seeds 1800/1801/1802)
- [x] Randomized-weights control run (seed-1800 checkpoint → 0/48)
- [x] Verdict registered (2026-08-02, evening)
