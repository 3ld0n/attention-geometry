# exp-177 Notes
# Pythia-410m text-native Δ-window heads: individual functional characterization
# 2026-11-13, solo physics room, ~12:30–2:00 AM MST

---

## Context

exp-159 (GPT-2 small, 2026-09-24) characterised the 16 text-native Δ-window heads in
GPT-2 small and found: a strongly graded tier structure (CV = 0.775), content-selective
attention (Spearman ρ = 0.459), and a recency gradient (not LiM U-shape, +0.155 nats
end-minus-begin). This experiment asks whether that functional signature replicates in
Pythia-410m (RoPE, 24 layers, 16 text-native heads from exp-118).

---

## Results summary

| Hypothesis | Threshold | exp-159 (GPT-2 small) | exp-177 (Pythia-410m) | Verdict |
|---|---|---|---|---|
| H_graded (CV ≥ 0.5) | CV ≥ 0.5 | CV = 0.775 | CV = 0.897 | **CONFIRMED** |
| H_positive (≥10/16 degrade) | ≥ 10/16 heads | — | 2/16 heads | **DEAD (K2)** |
| H_content_selective (ρ ≥ 0.40) | ρ ≥ 0.40 | ρ = 0.459 | ρ = 0.347 | **DEAD (K3)** |
| H_recency (end−begin ≥ 0.10) | ≥ 0.10 nats | +0.155 | +0.031 | **DEAD (K4)** |
| **H_rep (all 4 confirm)** | — | — | 1/4 | **FALSIFIED** |

---

## Per-head ablation results (Instrument 1)

| Layer | Head | ΔP_abl | ΔP_sham | Degraded/20 | Character |
|---|---|---|---|---|---|
| 0  | 3  | −0.001 | −0.016 | 1  | Neutral |
| 1  | 8  | +0.008 | +0.003 | 4  | Neutral |
| 2  | 8  | −0.019 | −0.027 | 7  | Weak negative |
| 3  | 2  | +0.029 | +0.007 | 5  | Neutral |
| 3  | 7  | +0.050 | +0.083 | 6  | Neutral/positive |
| 3  | 14 | **−0.054** | −0.045 | **10** | **Modest degrader** |
| 4  | 12 | +0.009 | −0.009 | 3  | Neutral |
| 7  | 11 | −0.019 | −0.037 | 7  | Weak negative |
| 8  | 15 | +0.071 | +0.110 | 4  | Positive |
| 9  | 15 | −0.011 | −0.113 | 8  | Weak (sham > abl) |
| 12 | 3  | +0.066 | +0.105 | 3  | Positive |
| 12 | 13 | **−0.100** | −0.139 | **12** | **Dominant degrader** |
| 14 | 15 | −0.025 | −0.071 | 8  | Weak (sham > abl) |
| 19 | 15 | −0.034 | −0.033 | 9  | Borderline |
| 21 | 11 | +0.002 | −0.071 | 2  | Neutral (sham negative) |
| 22 | 1  | +0.005 | +0.030 | 0  | Neutral |

**CV of |ΔP_C_i| = 0.897** — strongly graded; H_graded confirmed and even higher than exp-159.

The distribution is dominated by L12H13 (−0.100, 12/20 items) and L3H14 (−0.054, 10/20
items). The remaining 14 heads show essentially no content-retrieval function on the Task C
battery: 7 heads have ΔP_abl > 0 (ablation actually improves the task), and 7 show weak
negative effects (−0.034 to −0.011) with borderline or no item-level consistency.

**H_positive DEAD (K2 fires).** Only 2 heads (L12H13 and L3H14) have mean ΔP_abl < −0.05.
This is in contrast to exp-159 where 8 heads showed clear degradation (ΔP_C < −0.10).

---

## Instrument 2 — Attention-on-target

**Spearman ρ = 0.347 (p = 0.188).** H_content_selective is DEAD (K3 fires), just below the
0.40 threshold. The correlation is in the right direction but not significant and falls just
short of the declared criterion.

The two dominant degraders (L12H13, L3H14) presumably have higher attention on the target
token; the other 14 heads with near-zero contributions presumably have lower attention. The
sub-threshold ρ reflects that the dominant signal is concentrated in 2 heads rather than
distributed across the population, making the correlation noisy for the remaining 14.

---

## Instrument 3 — Recency / LiM probe

**End − Begin = +0.031 nats.** H_recency DEAD (K4 fires). The recency gradient is very weak
compared to exp-159's +0.155 nats.

An unexpected pattern: Middle (−5.453) is substantially worse than both Begin (−5.255) and
End (−5.224). This is a U-shape — the LiM pattern that exp-159 did NOT find in GPT-2 small
but which is classically found in long-context models. In Pythia-410m, the middle-of-context
performance dip does appear, but the declared H_recency (end > begin, end-begin ≥ 0.10)
does not fire because the difference is weak (the U-shape dominates over the recency slope).

**H_rep verdict: FALSIFIED.** Only H_graded confirmed (1/4 sub-hypotheses).

---

## What this changes

**The functional signature of the text-native population is architecture-conditional.**
The content-retrieval functional character found in exp-159 (GPT-2 small, learned AbsPE)
does NOT replicate in Pythia-410m (RoPE). The three-population anatomy's functional layer
must be scoped: "text-native Δ-window heads carry content-retrieval function" is an
accurate statement for GPT-2's learned-PE architecture class; it is not demonstrated to
hold in RoPE architectures at the same functional threshold.

**The graded tier structure IS architecture-independent (at least at this resolution).**
Both GPT-2 small (CV = 0.775) and Pythia-410m (CV = 0.897) show highly graded
distributions of individual ablation effects. In Pythia-410m the gradient is steeper —
the entire population is more dominated by 1–2 heads — which is the pattern of a
population where only a small subset carries the Task C function.

**Possible interpretations:**
1. **Function is architecture-conditional:** in AbsPE models, text-native heads serve
   content retrieval; in RoPE models, they serve different functions not captured by
   Task C. The task was designed for GPT-2's attention patterns.
2. **Multi-token protocol weakens the signal:** 14/20 targets are multi-token in Pythia's
   tokenizer. The sum-log-prob protocol may diffuse the ablation signal across the
   constituent tokens. The 6-item single-token sub-analysis showed mean ΔP_abl = +0.014
   (positive!), which suggests the multi-token items are not the explanation — the
   single-token items also show no degradation on average.
3. **The semantic population in Pythia is not functionally equivalent:** the same Δ-window
   population (learned in a fully-rotary model on The Pile) may be doing something other
   than content-specified long-range retrieval. The W_K direction in Pythia encodes
   different content features than in GPT-2, perhaps shaped by RoPE's relative-position
   geometry.

**The LiM-like pattern in Instrument 3 is unexpected.** Pythia-410m with a 2048-token
context window shows a middle-performance dip (the U-shape) that GPT-2 small (1024-token)
did not. The declared prior (H_recency, based on exp-159) was wrong in direction: the
recency gradient is real but small, and there is a concurrent middle dip. This may be a
context-length effect (Pythia's longer context window allows the LiM phenomenon to emerge
at intermediate positions), or a RoPE effect (relative PE changes the within-context
distribution of attention differently from absolute PE at the same context lengths).

---

## Relation to prior work

- exp-118: identified the text-native heads in Pythia-410m (the population under study)
- exp-159: the reference experiment in GPT-2 small — H_rep falsification is relative to this
- exp-155: population-level Task C result in GPT-2 small (−0.855 nats, 20/20 items)
  → Pythia-410m shows only L12H13 approaching that signal (−0.100, 12/20 items)

The honest summary: the geometric classification (Δ-window membership) does not imply
identical functional content across architecture families. The text-native population
exists in both GPT-2 small and Pythia-410m with the same census protocol, but its
functional role differs. This is a first-class finding, not a failure.

---

## Technical note — abl < sham asymmetry in Pythia

In exp-159, text-native heads in GPT-2 small showed abl < sham in magnitude on Task C
(the trained W_K direction carries specific content-retrieval function; random W_K is
more disruptive than zeroing). In Pythia-410m, the most informative contrast is:
- L12H13: ΔP_abl = −0.100, ΔP_sham = −0.139 → abl < sham (sham more disruptive)
- L3H14: ΔP_abl = −0.054, ΔP_sham = −0.045 → abl ≈ sham (trained direction = random direction)
- L9H15: ΔP_abl = −0.011, ΔP_sham = −0.113 → abl << sham (zeroing neutral, random highly disruptive)
- L21H11: ΔP_abl = +0.002, ΔP_sham = −0.071 → ablation neutral, sham harmful

The wide spread of sham effects (from +0.110 to −0.139) in the absence of consistent
ablation effects suggests that W_K directions in Pythia-410m are individually important
for the model's overall behavior (sham is disruptive), but the specific trained directions
are not encoding content-specified retrieval in the way GPT-2 small's text-native heads do.

---

## For the structural session with Eldon

Three things this result brings:
1. **H_rep falsification is an honest negative** — the functional character is
   architecture-conditional; the three-population anatomy's functional layer must be
   scoped.
2. **The U-shape in Pythia's LiM probe** — unexpected, contradicts the declared H_recency
   prior, may be a context-length or RoPE effect. Worth noting.
3. **A design question:** does a modified Task C battery (designed for Pythia's tokenizer
   and RoPE attention patterns) reveal content-retrieval function in Pythia-410m? Or is
   the absence genuine? This is not registered here; it's a design question for a later
   sitting.
