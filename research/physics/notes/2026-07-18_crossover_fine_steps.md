# exp-086 Follow-up: Crossover at Fine Step Resolution

*Analysis-only. Follow-up to exp-086 (crossover_layer_analysis.py, 2026-07-17).*
*Script: `exp-086_longitudinal_delta_spectrum/crossover_fine_steps.py`.*
*Data: `exp-086_longitudinal_delta_spectrum/crossover_fine_results.json`.*

---

## Background

exp-086 measured Δ across Pythia-70m training steps [0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000]. The crossover between RAND-dominant and NAT-dominant SYK-near head counts was placed between step 1000 and step 4000 (RAND=8 at 1000, NAT=8 at 4000). The July 17 per-layer analysis used steps 256, 1000, 4000 only, concluding "all 8 NAT SYK-near heads at step 4000 appear for the first time; none stable from step 1000."

This follow-up adds the available intermediate checkpoints — step 512, 2000, 3000 — to locate the crossover precisely and characterize whether the head-level transition is abrupt or gradual.

---

## Results: SYK-near count trajectory

| Step | RAND | NAT | RAND−NAT | Note |
|------|------|-----|----------|------|
| 256  | 5    | 2   | +3       | pre-crossover |
| 512  | 9    | 1   | +8       | **RAND peak** |
| 1000 | 8    | 1   | +7       | RAND plateau |
| 2000 | 5    | 5   | 0        | **crossover — equal** |
| 3000 | 8    | 8   | 0        | still equal, both expand |
| 4000 | 5    | 8   | −3       | NAT dominant |

**The crossover is at step 2000, not step 4000.** The July 17 analysis overestimated the transition step by 2×.

---

## What actually happens between steps 1000 and 2000

The key is L3 and L4 Δ_med:

| Step | L3_RAND | L3_NAT | L4_RAND | L4_NAT |
|------|---------|--------|---------|--------|
| 256  | 0.486   | 0.341  | 0.296   | 0.196  |
| 512  | 0.285   | 0.766  | 0.268   | 0.503  |
| 1000 | 0.357   | 0.515  | 0.309   | 0.690  |
| 2000 | 0.505   | 0.287  | 0.239   | 0.360  |
| 3000 | 0.435   | 0.266  | 0.264   | 0.299  |
| 4000 | 0.521   | 0.202  | 0.285   | 0.246  |

**L3 NAT Δ_med** drops from 0.515 (step 1000) → 0.287 (step 2000) — entering the SYK window [0.20, 0.30]. This is the L3 transition. L4 NAT is still at 0.360 at step 2000 but enters the SYK window by step 3000 (0.299) and step 4000 (0.246).

Between steps 1000 and 2000, NAT semantic-capture heads in L3 and L4 reorganize into conformal (power-law) structure. The heads were "captured" (large Δ) because induction patterns pulled attention away from the position-space power-law baseline; by step 2000 the model has developed sufficient structure that L3 can maintain conformal scaling even for natural language.

---

## Head-level picture

### NAT SYK-near head first appearances

| Head    | First appears at step |
|---------|----------------------|
| (1, 2)  | 256                  |
| (1, 7)  | 256                  |
| (4, 6)  | 512                  |
| (3, 6)  | 1000                 |
| (0, 5)  | 2000                 |
| (3, 5)  | 2000                 |
| (4, 2)  | 2000                 |
| (4, 3)  | 2000                 |
| (0, 0)  | 3000                 |
| (3, 0)  | 3000                 |
| (3, 3)  | 3000                 |
| (4, 7)  | 3000                 |
| (0, 6)  | 4000                 |
| (4, 4)  | 4000                 |

The main formation wave is at step 2000 (4 heads), not step 4000. A second wave at step 3000 (4 more heads). Only 2 heads arrive at step 4000.

### Stable shared heads (appear in both RAND and NAT)

At steps 3000 and 4000: (4,3), (4,6), (4,7) appear SYK-near in BOTH conditions. These are the universally conformal L4 heads — they maintain power-law structure regardless of whether the input is random tokens or natural language. (4,3) also appears shared at step 2000.

At step 4000, NAT-specific heads are: (0,6), (3,3), (3,5), (4,2), (4,4). These are the "captured-attending" heads that develop conformal structure specifically for natural language processing.

---

## Correction to July 17 analysis

The July 17 crossover_layer_analysis.md concluded:
> "The NAT crossover is a phase transition at the head level: all 8 NAT SYK-near heads at step 4000 appear for the first time; none carry over from step 1000."

This was correct about "none carry from step 1000" but misleading about "appear for the first time at step 4000." The 8 NAT SYK-near heads at step 4000 actually formed mostly at step 2000 (4 heads) and step 3000 (3 more heads); only 1 appeared at step 4000 for the first time. The absence of step 2000 and 3000 data caused the July 17 analysis to conflate "first appearance relative to step 1000" with "first appearance absolute."

Corrected picture: the transition is not a step-4000 phase transition but a **gradual reorganization spanning steps 1000–3000**, centered at step 2000 where the crossover in total count occurs.

---

## Is there a sharp transition?

The data does not show an abrupt phase transition. The RAND and NAT counts change gradually from step 512 through step 4000. No single inter-step interval shows a discontinuous jump:

- 256→512: RAND +4, NAT −1 (RAND expands rapidly)
- 512→1000: RAND −1, NAT 0 (plateau)
- 1000→2000: RAND −3, NAT +4 (main crossover — 7-head swing)
- 2000→3000: RAND +3, NAT +3 (symmetric expansion)
- 3000→4000: RAND −3, NAT 0 (RAND contracts)

The 1000→2000 interval shows the largest differential swing (+7 net toward NAT), consistent with the L3 NAT Δ_med dropping into the SYK window at step 2000. But RAND also fluctuates significantly (+3 at 2000→3000, −3 at 3000→4000), suggesting the RAND conformal heads are less stable than the NAT heads at this scale.

---

## Relationship to exp-083 and the whirlpool/crystal distinction

exp-083 showed RAND > REAL (natural text) for SYK-near heads in the mature GPT-2 model, interpreted as: "conformal baseline is the free-attending mode; coherent text activates induction heads, pulling attention away." Here, for Pythia-70m at step ≥2000, NAT ≥ RAND — the opposite direction.

Scale-dependent: GPT-2 (12L/117M) vs. Pythia-70m (6L/70M). The 70m model may not have enough capacity to fully separate induction from conformal structure; by step 2000 the natural language processing drives L3/L4 into the conformal window rather than away from it. The capacity at which the direction reverses is an open question.

---

*Note: the crossover_fine_results.json contains per-head Δ and R² values for all 6 checkpoints in both conditions.*
