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

## Status

**[PRE-REGISTERED 2026-08-02]** — Not yet run. Awaiting exp-097 3-seed verdict before launch.
