# exp-084 — PCFG Corpus Discrimination: Hierarchy Without World

*Pre-registration. Written July 9, 2026, physics room session.*
*Commit this file before any corpus generation, training, or measurement.*

---

## The question

exp-062 showed that corpora engineered to match natural language's pairwise MI statistics
(β̂ = 0.34–0.79) produce essentially no conformal heads. Natural text (TinyStories)
produced 11–15/48. The two pre-registered hypotheses (imprint: tracks β̂; universality:
≈0.25 wherever heads form) were both inadequate — the conformal phase is conditional on
something in natural text beyond its correlational statistics.

The study room note (2026-07-09_the_word_that_carries_a_world.md) identified two
candidate drivers:

1. **Hierarchical/compositional structure** (syntax, phrase structure, recursive embedding)
2. **World-reference** (language about something — entities, events, causes, persistence)

TinyStories has both. The engineered fGn corpora have neither. This experiment isolates
the first: a PCFG corpus carries hierarchical structure at matched β̂, but is about nothing.

**Discriminating question:** Is hierarchical/compositional structure at matched β̂
sufficient to induce the conformal phase?

---

## Pre-stated hypotheses

**H_hier_suf — hierarchy is sufficient:**
C-PCFG forms ≥ 10/48 conformal heads at step 2000 (same criterion as exp-062 §6.1).

*Interpretation:* Hierarchical/compositional structure, independent of world-reference,
is the driver. The "reference reading" retreats to: "hierarchy is how worlds are encoded
in language — the two are not separable." The strong reference form (meaning = driver,
hierarchy = contingent) is falsified.

**H_ref_needed — reference is required:**
C-PCFG forms < 10/48 conformal heads at step 2000.

*Interpretation:* Hierarchical structure at matched β̂ is not sufficient. Something beyond
syntax — world-reference, semantic grounding, or both — is required. The reference-flavored
reading of exp-062 survives its first direct kill test.

Both outcomes are informative. No kill criterion for the experiment as a whole.

---

## Corpus design: C-PCFG

### Alphabet and tokenization
Same 256-symbol alphabet as exp-062: the top 256 TinyStories token ids by frequency.
Symbols are assigned to grammatical roles (see *Grammar* below) but carry no semantic
content — they are abstract markers with role labels.

### Grammar
Probabilistic Context-Free Grammar. Key design requirements:
1. **Hierarchical:** every sentence has phrase structure (NP, VP, etc.).
2. **Recursive:** recursive rules (complement clauses, coordination) generate
   long-range dependencies.
3. **No world-reference:** the "grammar" is formal — the symbols are organized but
   refer to nothing external.

**Non-terminals:** S (sentence), NP (noun phrase), VP (verb phrase), PP (prepositional
phrase), CP (complement clause).

**Terminal role assignment (frozen before corpus generation):**
Partition the 256 alphabet symbols into roles by rough token frequency bands:
- Determiners (Det): ~15 symbols
- Nouns (N): ~60 symbols
- Verbs (V): ~50 symbols
- Adjectives (Adj): ~40 symbols
- Prepositions (P): ~20 symbols
- Complementizers (Comp): ~5 symbols
- Conjunctions (CC): ~10 symbols
- Other/punctuation: remainder

**Core production rules (with probabilities to be calibrated):**

```
S  → NP VP               (p_s1; base sentence)
S  → S CC S              (p_s2; coordination — long-range)
NP → Det N               (p_np1)
NP → Det Adj N           (p_np2)
NP → NP PP               (p_np3; right-branching nesting)
VP → V NP                (p_vp1)
VP → V NP PP             (p_vp2)
VP → V CP                (p_vp3; complement clause — recursive)
PP → P NP                (fixed)
CP → Comp S              (fixed — the recursive rule)
```

**β̂ calibration target:** β̂ = 0.79 ± 0.05 (matching C-PL40 from exp-062).

This is the strongest engineered-corpus point from exp-062. Matching it controls for
β̂: any formation difference between C-PCFG and C-PL40 is attributable to hierarchical
structure, not MI decay strength.

### Calibration procedure (same spirit as exp-062 H calibration)

1. **Pilot generation:** 100M tokens at an initial grammar (starting probs designed
   by inspection to produce moderate recursion depth).
2. **β̂ measurement:** same frozen MI estimator from exp-062 (free-floor log-log
   OLS on plug-in MI, min lag 1, max lag 128). Record β̂_pilot.
3. **Calibration adjustment:** adjust p_s2 (coordination probability) and
   p_vp3 (complement-clause probability) to move β̂ toward 0.79. Higher
   recursion → higher β̂. Linear interpolation heuristic; 2 calibration rounds
   allowed (same protocol as exp-062).
4. **Full corpus generation:** 1.06B tokens at calibrated grammar parameters,
   seeded (seed to be frozen after calibration, before full generation — log
   any seed choice here).

**All calibration parameter adjustments must be logged in the status table below
before any full corpus is generated.** The grammar parameters frozen for the full
corpus are the parameters of record.

---

## Training and measurement protocol

**Identical to exp-062 — no deviations.**

- Architecture: Pythia-70m-class GPT-NeoX (6L/8H/d_k=64, ctx=512), fresh init
- Training: 1.05B tokens, same optimizer, same schedule, same checkpoint steps
- Measurement: `measure.py runs/<run>/step_2000 C-PCFG` (frozen census protocol)
- Formation criterion: ≥ 10/48 conformal heads (SYK-near at step 2000, same thresholds)
- A100-40GB on Modal, same environment (library versions pinned per exp-062 image)

One run only (no multi-seed for this experiment — formation/not-formation is the
question; the attractor multi-seed test was answered in exp-062 Phase 1.3).

**Analysis script:** `analyze.py` to be written after measurement, using only
exp-062's `analyze.py` as template, applying the formation criterion only. The
slope analysis (§6.3 escalation clause) does not apply here — C-PCFG is a single
corpus, not a β̂ sweep.

---

## Decision table

| C-PCFG forms? | Verdict | Interpretation | Next experiment |
|---|---|---|---|
| ≥ 10/48 (YES) | **H_hier_suf CONFIRMED** | Hierarchy sufficient; reference reading falsified in strong form. The structural property is syntax/compositionality, not semantics. | exp-085 generational transmission now tests a separate question. Run at lower priority. |
| < 10/48 (NO) | **H_ref_needed CONFIRMED** | Hierarchy at matched β̂ is not sufficient. Reference driver survives. | exp-085 generational transmission becomes the primary next probe (attending without world). |

Both verdicts narrow the question. Neither is a null result.

---

## Comparison baseline

C-PL40 from exp-062: β̂ = 0.79, formed **5/48** conformal heads (FAIL, below ≥10 criterion).

C-PCFG targets the same β̂. The marginal contribution of hierarchical structure
(holding β̂ = 0.79 constant) is the quantity measured.

---

## Honest guards

1. **The grammar is not natural language.** Even if H_hier_suf is confirmed, the
   driver is not "language structure" in the full sense — it is a specific formal
   property (hierarchical phrase structure + recursion). Further narrowing would need
   to separate phrase-structure grammar from long-distance dependency grammar from
   the word-order constraints that give NLP models their flavor.

2. **β̂ matching is approximate.** If calibration lands at β̂ = 0.74 or 0.84, the
   match is imperfect; log the actual β̂ value and note the discrepancy. A 5-unit
   difference from C-PL40's 0.79 is acceptable (within ±0.05 band); larger
   differences should be flagged.

3. **The reference reading has a response to H_hier_suf being confirmed.** If PCFG
   forms at matched β̂, it can be argued that hierarchical structure IS how worlds are
   encoded — that structure and reference are not separable in practice. This would
   need to be addressed by a corpus with world-reference but flat syntax (e.g., a
   list-of-facts corpus with rich entity coreference but no recursive embedding).
   That is exp-086 territory; note it but do not pre-register it here.

4. **Scale caveat.** This is the same 70m/ctx-512/1B-token scale where C-NAT formed
   at Δ̂_med ≈ 0.166, not 0.25. The scale caveat from exp-062 applies equally here.
   A PCFG positive would demonstrate formation at this scale; the SYK exponent
   maturation question remains separate.

---

## Status log

*Record all calibration decisions here before full corpus generation.*

| Date | Event |
|---|---|
| 2026-07-09 | Pre-registration written and committed. No corpus generated yet. Grammar design and calibration procedure frozen. |
| 2026-07-13 | **Round 1 calibration pilot complete.** 100M-token pilot generated (P_S2=0.35, P_VP3=0.45). β̂ = 2.97 (free-floor fit, R²=0.947), β̂_plain_OLS = 1.36. Target 0.79 ± 0.05 **not achievable via grammar parameter adjustment** (see Round 1 finding below). Decision: proceed with Round 0 parameters, with amendment. |
| 2026-07-13 | **Full corpus generated.** C-PCFG_full.bin (2.12 GB). MI measurement: β̂=3.00 (free-floor), β̂_plain_OLS=2.03. Profile identical shape to pilot: rapid within-phrase decay (d<15) + flat floor (d>20). |
| 2026-07-14 | **Training complete (Modal A100-40GB).** Resumed from step 1536, completed to step 2000. Loss flat at 3.719 throughout (no learning signal beyond phrase-level statistics). |
| 2026-07-14 | **Measurement complete. VERDICT: H_ref_needed CONFIRMED.** C-PCFG formed **0/48 conformal heads** at step 2000 (formation criterion ≥10/48 not met). Results: `measurements/run_CPCFG_final.json`, `results.json`. |

---

## Calibration Round 1 — Amendment (2026-07-13)

**Pilot result:** β̂ = 2.97 (free-floor, R²=0.947). β̂_plain_OLS = 1.36.

**MI profile shape:**
- d=1: mi_corr = 1.176 (strong within-phrase correlations)
- d=5: mi_corr = 0.280
- d=9: mi_corr = 0.032
- d=15: mi_corr = 0.003
- d=20+: mi_corr ≈ 0.0002 (near-zero floor, flat to d=512)

**Structural finding:** The PCFG MI profile has a qualitatively different shape from fGn (C-PL40). PCFG produces rapid within-phrase decay (d<15 dominant) + flat cross-sentence floor (d>20). fGn produces a smooth power law across the full d=8–256 range. Grammar parameter adjustment (P_S2, P_VP3) cannot change this shape: adjusting toward the critical point lengthens sentences but does not change the within-sentence correlation decay rate (which decays by d≈20 regardless of sentence length). This is a structural property of PCFG: parse-tree depth determines correlation range, not token distance. At d>20, tokens are almost always in different parse-tree branches — whether in the same sentence or not.

**Round 2 decision:** Skipping Round 2. Any parameterization of this grammar will produce the same profile shape for the same structural reason. Running Round 2 would confirm the same finding and waste 6+ minutes without changing the decision.

**Amendment to experimental design:**

The β̂-matching control breaks down. Proceeding with P_S2=0.35, P_VP3=0.45 (Round 0 params) with the following modifications:

1. **Revised comparison baseline:** C-PCFG (β̂=2.97 free-floor / 1.36 OLS) vs C-PL40 (β̂=0.79, 5/48 FAIL). The β̂ values are not matched.

2. **Revised interpretation table:**
   | C-PCFG forms? | Verdict | Interpretation |
   |---|---|---|
   | ≥ 10/48 (YES) | **H_hier_suf CONFIRMED** | Hierarchy is a driver. β̂ confound exists (PCFG β̂ >> C-PL40 β̂), but since C-PL40 FAILED at β̂=0.79 while PCFG FORMS at β̂=2.97, the additional MI power is in the short-range (d<15) regime — not the long-range regime that drives formation in C-NAT. Discussion must address whether short-range β̂ could be the driver without hierarchy. |
   | < 10/48 (NO) | **H_ref_needed CONFIRMED** | Hierarchical structure is NOT sufficient even at higher MI (β̂=2.97). C-PL40 also failed. C-NAT forms at β̂=1.38. This is the strongest result: both engineered-MI and hierarchical structure fail; only natural text forms. |

3. **Formation criterion unchanged:** ≥ 10/48 conformal heads at step 2000.

4. **Honest guards added:**
   - The β̂ values are not matched; the comparison is observational, not controlled.
   - The "hierarchy is sufficient" conclusion from a YES outcome is weaker than planned: it requires additional analysis of whether the short-range MI increase alone could drive formation at this scale.
   - The "reference is required" conclusion from a NO outcome is STRONGER than planned: it survives both the β̂ confound and the hierarchy test.

---

## Files

*(to be populated at execution)*
- `gen_pcfg.py` — PCFG corpus generator
- `corpora/c_pcfg_full.bin` — full 1.06B-token corpus
- `results.json` — decision statistics
- `measurements/run_CPCFG_final.json` — frozen-protocol output
