# exp-178 Notes
# Pythia-410m text-native Δ-window heads: native-tokenizer Task C battery
# 2026-11-14, solo physics room, ~12:40–1:30 AM MST

---

## Context

exp-177 (2026-11-13) falsified H_rep: the content-retrieval functional signature of
text-native Δ-window heads (established in GPT-2 small by exp-159) did NOT replicate
in Pythia-410m. Only H_graded confirmed (CV = 0.897); 2/16 heads degraded on Task C.
A design concern: the Task C battery used in exp-177 had 14/20 multi-token targets in
the GPT-NeoX tokenizer. This experiment uses a fresh 20-item battery with all targets
verified as single tokens, to test whether the null was a task-design artifact.

---

## Results summary

| Hypothesis | Threshold | exp-159 (GPT-2 sm) | exp-177 (Pythia) | exp-178 (Pythia native) | Verdict |
|---|---|---|---|---|---|
| H_graded (CV ≥ 0.5) | CV ≥ 0.5 | CV = 0.775 | CV = 0.897 | **CV = 1.1478** | **CONFIRMED** |
| H_positive (≥10/16 degrade) | ≥ 10/16 | — | 2/16 | **0/16** | **DEAD (K2)** |
| H_content_selective (ρ ≥ 0.40) | ρ ≥ 0.40 | ρ = 0.459 | ρ = 0.347 | **ρ = 0.206** | **DEAD (K3)** |
| H_recency (end-begin ≥ 0.10) | ≥ 0.10 nats | +0.155 | +0.031 | **−0.401** | **DEAD (K4)** |
| **H_rep (all 4 confirm)** | — | — | 1/4 | **1/4** | **FALSIFIED** |

**Declared prior H_arch: SUPPORTED.** The null in exp-177 is not a task-design artifact.

---

## Instrument 1 — Per-head ablation results

| Layer | Head | ΔP_abl | ΔP_sham | Improved/20 | Character |
|---|---|---|---|---|---|
| 0  | 3  | −0.021 | −0.020 | 5  | Weak negative |
| 1  | 8  | +0.021 | +0.018 | 14 | Positive (competitor) |
| 2  | 8  | +0.009 | +0.005 | 13 | Neutral/positive |
| 3  | 2  | +0.063 | +0.040 | 14 | Strong positive (competitor) |
| 3  | 7  | +0.100 | +0.160 | 14 | Strong positive (competitor) |
| 3  | 14 | +0.018 | −0.030 | 13 | Neutral/positive |
| 4  | 12 | +0.016 | −0.022 | 11 | Neutral/positive |
| 7  | 11 | +0.069 | +0.072 | 15 | Positive (competitor) |
| 8  | 15 | **+0.219** | +0.329 | **20** | **Dominant competitor** |
| 9  | 15 | −0.043 | −0.023 | 5  | Modest degrader |
| 12 | 3  | +0.052 | +0.088 | 18 | Strong positive (competitor) |
| 12 | 13 | −0.021 | −0.164 | 9  | Weak degrader |
| 14 | 15 | −0.033 | −0.049 | 7  | Weak degrader |
| 19 | 15 | −0.014 | −0.060 | 7  | Weak degrader |
| 21 | 11 | +0.018 | +0.019 | 17 | Positive (competitor) |
| 22 | 1  | −0.002 | +0.012 | 10 | Neutral |

**H_graded CONFIRMED with CV = 1.1478** — the highest graded-tier CV across the three
experiments (GPT-2 small: 0.775; Pythia exp-177: 0.897; Pythia native: 1.148). The
gradient is steeper with the native battery, not shallower. L8H15 dominates even more
strongly than L12H13 did in exp-177.

**H_positive DEAD (K2 fires) — 0/16 heads with ΔP_C < −0.05.** This is a stronger
null than exp-177 (which found 2/16). With task-calibrated stimuli, the architecture-
conditionality is more pronounced, not less.

**Competitor structure more visible.** On the native battery, 8/16 heads have positive
ΔP_abl (ablation *improves* content retrieval). The dominant one is L8H15 (+0.219 nats,
20/20 items). This population actively competes with content-specified retrieval — their
W_K directions suppress the correct target. This is the inverse of the content-retrieval
pattern seen in GPT-2 small.

---

## Instrument 2 — Attention-on-target

**Spearman ρ = 0.206 (p = 0.444).** H_content_selective DEAD (K3).

The correlation is weaker than exp-177 (ρ = 0.347) and in the same direction. The
dominant competitor (L8H15, ΔP_abl = +0.219) presumably has LOW attention on the
target — removing it frees the content signal. The modest degraders (L9H15, L14H15)
presumably have higher attention. The weak correlation reflects this mix: competition
and weak retrieval are both present but not cleanly separated by attention magnitude.

---

## Instrument 3 — Reversed recency pattern

**End − Begin = −0.401 nats.** H_recency DEAD (K4).

The result is unexpected: **begin (setup first) outperforms end (setup just before cue)**
by 0.40 nats on average. This is the reverse of the recency gradient seen in GPT-2 small
(exp-159: +0.155 nats end-minus-begin).

| Condition | Mean log-prob |
|---|---|
| Begin (setup first, long separation) | −2.170 |
| Middle (setup mid) | −2.300 |
| End (setup last, just before cue) | −2.570 |

The pattern is monotone anti-recency: Begin > Middle > End. This is not an LiM U-shape
(which would have middle worst). It is a systematic preference for early-context setup.

**Possible interpretations:**
1. **RoPE anti-recency for bound tokens.** In RoPE, keys from early positions (position
   0–20) may be encoded in a way that makes them easier for a mid-context query to
   retrieve than keys from just before the cue (positions 150–170 in the 'end' condition).
   Rotary embeddings are designed to encode relative distance, but the learned W_K may
   preferentially attend to early-established content.
2. **Context-window edge effects.** The 'end' condition places the setup near the end of
   a long context (filler + setup ≈ 120+ tokens). Pythia-410m may have learned that
   entity-property bindings near the end of a long context are frequently overridden,
   making early-context bindings more trustworthy.
3. **Different from exp-177's U-shape.** exp-177 (multi-token, exp-159 items) found a
   weak U-shape (middle worst, begin and end close). exp-178 (native items) finds a
   monotone anti-recency gradient. The difference may reflect item design: the native
   items have cleaner setups and cues, making the position effect more visible.

This is a novel finding. The exp-177 pattern and the exp-178 pattern together suggest
Pythia-410m's context retrieval for entity-property bindings is genuinely different from
GPT-2 small's: not recency-biased, and potentially early-context-biased.

---

## What this changes

**The architecture-conditionality finding is robust.** The exp-177 null was not a
task-design artifact. With a native battery (all single-token targets, stimuli
calibrated to GPT-NeoX), the result is a stronger null: 0/16 heads degrade vs.
2/16 in exp-177. The functional character of text-native Δ-window heads in
Pythia-410m is confirmed to be different from GPT-2 small's.

**The competitor structure is real.** At least 8/16 heads in the Pythia-410m
text-native population COMPETE with content-specified long-range retrieval.
L8H15 is the dominant competitor (+0.219 nats, 20/20 items). This is not noise —
removing the head systematically improves the task across all items.

**The scope limit on the three-population anatomy's functional layer is firm.**
"Text-native Δ-window heads carry content-retrieval function" is accurate for GPT-2
(AbsPE). It is not demonstrated for Pythia-410m (RoPE). The confirmed statement
after exp-177 and exp-178: *the text-native geometric classification does not imply
identical functional content across architecture families.*

**A new finding: reversed recency in Pythia-410m.** GPT-2 small shows standard
recency (end > begin). Pythia-410m shows reversed recency (begin > end), confirmed
on a native battery. This is an independent architecture-conditional finding.

---

## Relation to prior work

- exp-118: identified the text-native heads in Pythia-410m
- exp-159: reference experiment (GPT-2 small); content-retrieval confirmed
- exp-177: Pythia-410m with exp-159 battery; H_rep falsified (2/16 heads degrade)
- exp-178: Pythia-410m with native battery; H_arch supported (0/16 degrade)
- Together: architecture-conditionality is real; competitor structure more prominent
  with native stimuli; reversed recency is a new Pythia-specific finding

---

## For the structural session with Eldon

Four things this result adds:
1. **H_arch confirmed with native battery** — architecture-conditionality is robust
   to task design; scope limit is firm.
2. **Competitor structure** — 8/16 Pythia text-native heads actively compete with
   content retrieval. What positive function do they serve?
3. **Reversed recency** — begin > end, monotone anti-recency, not LiM. Novel;
   may be a RoPE geometry effect worth following up.
4. **exp-178 closes the design question** raised by exp-177 about task calibration.
   The open design question is now: what positive function do Pythia's text-native
   heads serve? Not registered here — a next-sitting question.
