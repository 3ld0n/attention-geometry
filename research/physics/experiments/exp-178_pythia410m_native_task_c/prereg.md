# Pre-registration: exp-178
# Pythia-410m text-native Δ-window heads: native-tokenizer Task C battery
# Date: 2026-11-14
# Room: Physics (solo)
# Registered BEFORE run.py is written

---

## Background

exp-177 (2026-11-13) tested whether the functional signature of text-native Δ-window
heads in GPT-2 small (exp-159, September 24) replicates in Pythia-410m. The result was
H_rep FALSIFIED: only H_graded confirmed (CV = 0.897); H_positive, H_content_selective,
and H_recency all dead. The main finding: only 2/16 heads showed content-retrieval
function (L12H13: mean ΔP_C = −0.100, 12/20 items; L3H14: −0.054, 10/20 items). The
remaining 14 heads were neutral or positive on individual W_K ablation.

**The design issue raised by exp-177:** The Task C battery used in exp-177 was designed
for the GPT-2 BPE tokenizer. In Pythia's GPT-NeoX tokenizer, 14/20 target words
tokenize as multiple tokens. The exp-177 multi-token sum protocol (summing log-probs
of all constituent tokens) was declared in the pre-registration and is a valid
adaptation, but it changes the measurement: ablation must degrade ALL constituent
tokens coherently for the item to contribute a negative ΔP_C.

exp-177 included a 6-item single-token sub-analysis (the 6 targets that happen to be
single tokens in GPT-NeoX). The sub-analysis found mean ΔP_C = +0.014 (positive) for
the 6 single-token items — the same null direction as the full 20-item result. This is
evidence that the multi-token protocol is not the cause of the null, but 6 items is a
small sample with low statistical power.

This experiment tests that interpretation directly: a fresh 20-item Task C battery with
all targets verified as single tokens in GPT-NeoX, designed with the same content
structure (entity-property binding retrieved through long intervening filler), tested
on the same 16 text-native Δ-window heads in Pythia-410m.

---

## Population under study

The 16 text-native Δ-window heads in Pythia-410m, as identified by exp-118's
WikiText-native census (Δ ∈ [0.20, 0.30], R² ≥ 0.90):

| Index | Layer | Head |
|---|---|---|
| 1 | 0 | 3 |
| 2 | 1 | 8 |
| 3 | 2 | 8 |
| 4 | 3 | 2 |
| 5 | 3 | 7 |
| 6 | 3 | 14 |
| 7 | 4 | 12 |
| 8 | 7 | 11 |
| 9 | 8 | 15 |
| 10 | 9 | 15 |
| 11 | 12 | 3 |
| 12 | 12 | 13 |
| 13 | 14 | 15 |
| 14 | 19 | 15 |
| 15 | 21 | 11 |
| 16 | 22 | 1 |

Same population as exp-177. Δ_med = 0.2426, source: exp-118.

---

## Battery design

20 new items in `task_c_native_items.json`, designed 2026-11-14 as part of this
pre-registration. All 20 target tokens are verified single-token in the GPT-NeoX
tokenizer before this file is committed (token IDs listed in the JSON description).

**Battery structure:** Each item has:
- `setup`: one sentence establishing entity → property binding (e.g., "The floor of
  the hall was marble.")
- `filler`: ~100 words of unrelated natural-environment prose that separates setup
  from retrieval
- `cue`: a sentence fragment that expects the target property as the next token (e.g.,
  "Everyone remembered that the floor of the hall was")
- `target_token`: the property word as it appears in the GPT-NeoX tokenizer

Targets are drawn from the same semantic category as the original Task C (architectural
materials, surface properties, structural forms) so the content class is comparable
to exp-159 and exp-177.

---

## Instruments and hypotheses

The same three instruments as exp-177 and exp-159, run on the new native battery.

### Instrument 1 — Individual-head contribution to native Task C

**Protocol:** 20-item native battery. Baseline forward pass, then per-head W_K ablation
(zero the W_K slice of `attention.query_key_value.weight` for that head) + sham (same-
Frobenius-norm random W_K). Log-prob of target_token (single token; `token_logprob`
directly) under ablation and sham versus baseline. ΔP_C = ablation − baseline, per head.

**Primary hypothesis H_graded:**
CV of |ΔP_C_i| across 16 heads ≥ 0.5. (Threshold carried from exp-159 and exp-177.)

**Secondary hypothesis H_positive:**
≥ 10/16 heads produce mean ΔP_C < −0.05 on the native battery.

**Kill conditions:**
- K1: CV < 0.5 — contributions approximately equal; H_graded dead.
- K2: Fewer than 10/16 ablations produce mean ΔP_C < −0.05.

**Replication-of-null hypothesis H_null_rep:**
If H_positive fires (K2 fires, ≤ 2 heads degrade), the architecture-conditionality
finding from exp-177 replicates with better-calibrated stimuli. If H_positive
confirms (≥ 10/16 heads degrade), the exp-177 null was a task-design artifact.

### Instrument 2 — Attention distribution

Same as exp-177: Spearman ρ between |ΔP_C_i| and mean attention-on-target.

**Hypothesis H_content_selective:** ρ ≥ 0.40.
**Kill condition K3:** ρ < 0.40.

### Instrument 3 — Position probe

Same three-condition design as exp-159/exp-177: target setup placed at beginning,
middle, or end of the combined context (filler rearranged around setup). Report mean
log-prob by condition. Declared prior: H_recency (monotone end > begin, |diff| ≥ 0.10
nats) — same declared prior as exp-177, which found a U-shape (H_recency dead, U-shape
pattern for Pythia's 2048-token context). If U-shape appears again, H_recency is dead
and H_lim fires; if recency gradient appears, it revises exp-177's finding.

**Kill condition K4:** No monotone ordering or |end − begin| < 0.10 nats.

---

## Declared prior

**H_arch (architecture-conditional null replicates):** Based on exp-177's 6-item
single-token sub-analysis (mean ΔP_C = +0.014), the absence of content-retrieval
function in Pythia-410m's text-native population is not an artifact of the multi-token
protocol. A native 20-item battery will also show ≤ 3 heads with mean ΔP_C < −0.05
(consistent with exp-177's 2/16).

This is the declared prior. If H_positive fires (≥ 10/16 heads degrade), the declared
prior is wrong and the architecture-conditionality conclusion must be revised.

---

## What each outcome changes

**If H_null_rep (K2 fires, ≤ 3 heads):**
The architecture-conditionality finding is robust to task design. The text-native
Δ-window population's functional character is genuinely different in RoPE vs. AbsPE
architectures — it is not an artifact of mis-calibrated stimuli. The scope of the
functional characterization in exp-159 is explicitly scoped to AbsPE/learned-PE
architecture classes.

**If H_design fires (H_positive confirmed, ≥ 10/16 heads degrade):**
The exp-177 null was a task-design artifact (multi-token protocol, not genuine
architectural difference). The functional signature — graded tier structure, majority
of heads carrying content-retrieval function — is architecture-independent. The scope
limit is retracted; exp-177's declared prior was wrong in the right direction (the
declared prior was false but the detection worked).

---

## Stopping rule

All three instruments run in one session. If fewer than 5 items show any
between-condition variation in Instrument 1, expand to 30 items using the same design
(append 10 new single-token-target items before resuming) — same pre-registered
amendment pathway as exp-177.

---

## Relation to prior work

- exp-118 (2026-08-11): identifies the 16 text-native Δ-window heads in Pythia-410m
- exp-159 (2026-09-24): functional characterization in GPT-2 small (the reference)
- exp-177 (2026-11-13): cross-family replication in Pythia-410m; H_rep FALSIFIED
- This experiment (exp-178): native-battery discrimination test

---

## Why-check

**Question:** Geometry — does the semantic population's functional character survive
the architecture shift when tested with properly calibrated stimuli? Also touches
Formation: is content-retrieval function a universal property of training on natural
language, or is it architecture-conditional?

**What each outcome changes:**
H_null_rep → the architecture-conditionality of function is established with better
evidence; the scope limit on the three-population anatomy's functional layer is firm.
H_design fires → the architecture-conditionality was an instrument artifact; the
functional signature is architecture-independent and the scope limit retracts.
In both cases: what I take the semantic population to be changes. The null in exp-177
either is confirmed as real or is corrected as artifactual.

---

*Pre-registration complete. run.py to be written and committed after this file is
pushed to attention-geometry. token verification run on 2026-11-14 before commit.*
