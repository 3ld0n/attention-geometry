# exp-179 Notes
# Pythia-410m text-native Δ-window heads: positional retrieval (Task B)
# 2026-11-15, solo physics room, ~7:00–7:30 PM MST

---

## Context

exp-177 (Nov 13) and exp-178 (Nov 14) established that Pythia-410m's 16 text-native
Δ-window heads do NOT carry content-retrieval function: 0/16 degrade Task C on W_K
ablation, and 8/16 actively compete (ablation improves Task C). The declared prior here
was that these heads serve positional retrieval — the same function GPT-2 small's
structural/steep-local heads carry — because the competitor structure suggested routing
to a different retrieval modality.

---

## Results summary

| Hypothesis | Threshold | exp-179 result | Verdict |
|---|---|---|---|
| H_positional (≥10/16 carry positional) | ΔP_B < -0.05, items_degraded ≥ 12/20 | **1/16** | **FALSIFIED** |
| H_graded (CV ≥ 0.5) | CV ≥ 0.5 | **CV = 1.581** | **CONFIRMED** |
| H_competitor_null (fires if ≥5 competitor heads carry positional) | ≥5/9 | **1/9** | **NOT FIRED** |

Declared prior H_positional FALSIFIED. The text-native Δ-window population in
Pythia-410m does not carry positional retrieval function.

---

## Per-head ablation results

| Layer | Head | ΔP_B abl | ΔP_B sham | Degraded/20 | Competitor (exp-178) | Carries positional |
|---|---|---|---|---|---|---|
| 2  | 8  | −0.056 | −0.232 | 17/20 | ✓ | ✓ |
| 3  | 14 | −0.032 | −0.001 | 14/20 | ✓ |   |
| 3  | 7  | −0.029 | −0.130 | 12/20 | ✓ |   |
| 19 | 15 | −0.022 | −0.035 | 15/20 |   |   |
| 22 | 1  | −0.000 | −0.002 | 8/20  |   |   |
| 21 | 11 | +0.001 | −0.029 | 9/20  | ✓ |   |
| 12 | 3  | +0.026 | −0.014 | 7/20  | ✓ |   |
| 4  | 12 | +0.031 | +0.010 | 4/20  |   |   |
| 3  | 2  | +0.034 | +0.004 | 7/20  | ✓ |   |
| 8  | 15 | +0.049 | +0.070 | 6/20  | ✓ |   |
| 0  | 3  | +0.061 | +0.043 | 4/20  |   |   |
| 1  | 8  | +0.072 | +0.142 | 7/20  | ✓ |   |
| 14 | 15 | +0.086 | +0.057 | 4/20  |   |   |
| 9  | 15 | +0.114 | +0.128 | 3/20  |   |   |
| 12 | 13 | +0.187 | −0.184 | 0/20  |   |   |
| 7  | 11 | +0.189 | +0.196 | 3/20  | ✓ |   |

**Only L2H8 meets the full H_positional criterion** (ΔP_B = −0.056, 17/20 degraded). It is
also a competitor from exp-178 — ablating it simultaneously improves content retrieval
and degrades positional retrieval. The other heads are near-zero or positively improve
Task B on ablation.

---

## Key findings

### 1. The declared prior was wrong — no general role-switch from content to positional

The prediction that "RoPE architecture routes text-native Δ-window heads to positional
retrieval" is false. The population is not organized around either content-retrieval OR
positional-retrieval function. Three experiments now confirm: in Pythia-410m, the
text-native geometric class does not predict functional identity.

### 2. Some competitor heads also compete with positional retrieval

L7H11 (+0.189 nats on Task B, 3/20 degraded) and L1H8 (+0.072, 7/20) improved Task B
on ablation. These were also competitors in exp-178 (ablation improved Task C). So
L7H11 and L1H8 appear to suppress both content-retrieval and positional-retrieval
signals. L8H15 (+0.049) is on the boundary.

This is K1-adjacent but not K1 by the pre-registered criterion (requires ≥5 competitor
heads with ΔP_B > +0.05; only 2 clearly qualify). The pattern is real but weak.

### 3. L12H13 is a selective positional-retrieval suppressor

L12H13 (+0.187, 20/20 improved) is a strong positional-retrieval suppressor: ablating
it improves positional retrieval on all 20 items. But it was NOT a competitor in exp-178
(it was a weak degrader of Task C, −0.021 nats, 9/20 degraded). So L12H13 competes
primarily with positional retrieval while being neutral for content retrieval.

This head's functional character is the inverse of a typical text-native head in GPT-2:
it suppresses positional retrieval without serving content retrieval.

### 4. Most heads are functionally undifferentiated under these two instruments

The majority of the 16 text-native Δ-window heads in Pythia-410m show near-zero or
ambiguous effects on both Task C (exp-178) and Task B (exp-179). The geometric
classification is not a reliable predictor of function in this architecture class.

### 5. CV high (1.581) despite the null

CV = 1.581 — higher than exp-178's 1.148 and far higher than GPT-2 small's 0.775.
The graded-tier structure is confirmed: some heads are extreme (L7H11, L12H13),
most are near-zero. But the grading is in the direction of suppression rather than
carrying a function.

---

## What this changes

**The scope limit on the three-population anatomy is now complete for Pythia-410m:**
- Text-native Δ-window heads in GPT-2 small (AbsPE): carry content-retrieval function
- Text-native Δ-window heads in Pythia-410m (RoPE): carry neither content-retrieval
  nor positional-retrieval function; some actively compete with both

**The geometric classification is architecture-conditional for function.** Three experiments
(exp-177, exp-178, exp-179) establish: the Δ ≈ 1/4 shape in RoPE models is not associated
with the same functional role as in AbsPE models.

**The competitor/suppressor vocabulary needs refinement.** Some heads suppress one retrieval
mode (most exp-178 competitors → content), some suppress both (L7H11), some suppress the
other mode (L12H13 → positional). The competitor structure from exp-178 is not a
single phenomenon.

---

## Open questions

1. What positive function do Pythia-410m's text-native Δ-window heads serve, if not
   retrieval? Task A (entity-state tracking) is the natural next test. Alternatively,
   syntax-structural probing.

2. Is the multi-modal suppressor profile (L7H11 suppressing both C and B) consistent
   across the two independent batteries? This would be a novel architectural finding.

3. What is L12H13 doing? A head that systematically suppresses positional retrieval
   without competing with content retrieval is an unusual signature.

---

## Relation to prior work

- exp-118: identified the text-native heads in Pythia-410m (the population)
- exp-159: reference experiment (GPT-2 small); content-retrieval confirmed, recency gradient
- exp-177: Pythia-410m, exp-159 battery; H_rep falsified (2/16 degrade Task C)
- exp-178: Pythia-410m, native battery; H_arch confirmed (0/16 degrade); competitor structure
- exp-179: Pythia-410m, Task B; H_positional falsified (1/16); suppressor structure extends
            to positional domain for a subset of heads

---

## For the structural session with Eldon

Three things to add:
1. **H_positional falsified** — the role-switch hypothesis is wrong. The same geometry
   does not route to positional function in RoPE. What accounts for the competitor
   structure remains open.
2. **Multi-modal suppressor** — L7H11 suppresses both content and positional retrieval.
   L12H13 suppresses positional only. These are qualitatively different suppressor profiles.
3. **Design question for exp-180:** Task A (entity-state tracking) in Pythia-410m —
   does the text-native population serve entity-state coherence, since both retrieval
   modalities are either null or suppressed?
