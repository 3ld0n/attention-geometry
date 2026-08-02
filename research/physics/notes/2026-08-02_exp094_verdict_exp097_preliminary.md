# Session Notes — exp-094 verdict; exp-097 preliminary

*2026-08-02, ~12:17–13:30 PM MDT. Solo physics room session.*
*Picks up from two failed sessions (4:20 AM and 9:25 AM) that reached this analysis and exited with error.*
*The 4:20 AM session identified the data correctly but couldn't complete the analysis: it was filtering on `is_conformal` (field doesn't exist); the correct field name is `conformal`.*

---

## exp-094: Quarter-Story Block Shuffle — VERDICT

**H_quarter_below CONFIRMED. H_backbone_stable CONFIRMED. Control clean.**

### Data

| Seed | n_conf | n_syk | Δ_med | n_deep (L3–5) | n_backbone (L0) | layer_dist |
|------|--------|-------|-------|---------------|-----------------|------------|
| s0 (1500) | 10/48 | 1 | 0.188 | 1 | 7 | {0:7, 2:2, 4:1} |
| s1 (1501) | 11/48 | 0 | 0.146 | 3 | 7 | {0:7, 1:1, 4:3} |
| s2 (1502) | 9/48  | 1 | 0.152 | 1 | 8 | {0:8, 4:1} |
| **median** | **10** | — | — | **1** | **7** | — |
| control (rand) | 0/48 | 0 | — | — | — | — |

### Verdict

**H_quarter_below CONFIRMED** (pre-registered primary): median n_deep = **1**, at or below the shuffled baseline (k=1 shuf: 2, k=2 block: 1, k=3 block: 2). Quarter-story blocks (~2 sentences) do NOT recover the deep conformal population above baseline. The minimum contiguous chunk for partial recovery is between ~2 sentences (quarter-story) and ~4–5 sentences (half-story, exp-093).

**H_backbone_stable CONFIRMED**: median n_backbone (L0) = 7, consistent with all prior shuffled/blocked conditions.

**Control confirms signal**: 0/48 conformal in randomized weights, as in every prior control (exp-085/091/092/093).

### Interpretation

Quarter-story blocks (~2 sentences) behave identically to k=2 global blocks (also n_deep=1), despite the structural difference (story-boundary-defined vs globally-defined cuts). The "story-structural" character of the quarters does not compensate for their short length. This eliminates the story-structure confound: what matters is chunk length, not whether the chunk respects story-internal boundaries.

The result tightens the localization of the critical chunk length: somewhere between ~2 sentences (quarter-story, k=2: no recovery) and ~4–5 sentences (half-story: partial recovery to n_deep=3). The k=3 block data (n_deep=2) suggests the step is not sharp — there may be a gradual increase in the 3–5 sentence range rather than a step function. 

Δ_med for quarter-story (0.146–0.188) sits BELOW the shuffled band (exp-091: Δ_med not directly comparable at head-level, but the conformal heads are in the UV regime relative to SYK). One seed (s1) produces n_deep=3, suggesting stochastic variation overlaps half-story scale — the effect is real but noisy.

### Formation ladder (updated)

| Condition | n_deep | n_conf | n_backbone (L0) |
|-----------|--------|--------|-----------------|
| C-NAT (k=∞, full arc) | 5–7 | 11–15 | 7–8 |
| C-NAT-half (exp-093) | 3 | 12 | 7–8 |
| C-NAT-block3 (k=3, exp-092) | 2 | ~9 | 7–8 |
| **C-NAT-quarter (exp-094)** | **1** | **10** | **7** |
| C-NAT-shuf (k=1, exp-091) | 2 | 8–9 | 7–8 |
| C-NAT-block2 (k=2, exp-092) | 1 | ~9 | 7–8 |

The ordering reveals a nuance: k=1 sentence-shuffle (2) > k=2 block (1) and k=1 quarter (1), despite k=1 being shorter chunks. This is consistent with the exp-091 interpretation: the within-sentence coherence of single sentences may preserve more causal information than cross-sentence-boundary blocks.

---

## exp-097: Alien Semantics — PRELIMINARY (2 seeds, verdict pending s0)

### Data (2 seeds only — s0 retrain launched, est. complete ~13:30–14:00 PM MDT)

| Seed | n_conf | n_syk | Δ_med | n_deep (L3–5) | n_backbone (L0) | layer_dist |
|------|--------|-------|-------|---------------|-----------------|------------|
| s1 (1701) | 10/48 | 0 | 1.044 | 3 | 0 | {1:3, 2:4, 3:2, 4:1} |
| s2 (1702) | 17/48 | 0 | 1.185 | 7 | 0 | {1:5, 2:5, 3:5, 4:1, 5:1} |
| s0 (1700) | PENDING | — | — | — | — | — |

**Cannot register a pre-registered verdict without 3 seeds.**

### What the preliminary data shows

**Conformal structure forms** (n_conf = 10–17/48): C-alien is not like PCFG (0/48). A world-holding procedural simulator does induce power-law attention structure — more than the shuffled baseline (8–9).

**No SYK fixed point**: Δ_med ≈ 1.04–1.18, far above 0.25. n_syk_near = 0 at both seeds. C-alien models are stuck in the UV regime — they develop conformal geometry, but NOT the specific q=4 fixed point that characterizes natural text.

**Zero backbone (L0) heads**: n_backbone = 0 at both seeds. This is a complete departure from every prior condition in the series (all other corpora: n_backbone ≈ 7–8). The L0 structural backbone — which forms in shuffled, blocked, and partial-arc corpora alike — does not form on C-alien. This falsifies the secondary hypothesis H_backbone_stable for C-alien.

**n_deep with 2 seeds**: [3, 7], point median = 5. If s0 falls in the 3–7 range, the 3-seed median will be 3–7; if below 3, the median falls to 3; if above 7, median rises.

### The structural surprise: where are the conformal heads?

C-alien conformal heads cluster in layers 1–4, not L0. C-NAT conformal heads anchor in L0 (7–8 backbone) and then appear in L3–L5 (deep population). C-alien inverts this: L0 is empty, L1–L4 carry the conformal population. This is not a small effect — it's the complete absence of the backbone structure that every real-text corpus produces.

Physical interpretation: the L0 backbone formation may depend on the model's ability to use its pre-wired induction-head circuitry on familiar vocabulary (real English words, proper names). Flurps/Blurns/Zarbs, being neologisms, may not trigger the same positional/inductive attention patterns. If the L0 backbone = architecture-driven structural conformal heads (exp-086 interpretation), its failure in C-alien suggests the "architectural" component is more corpus-sensitive than previously thought — specifically, the alien vocabulary prevents the standard induction-head L0 formation.

Alternatively: the alien corpus's very different statistical structure (8-step rigid rule sequences vs TinyStories' varied narrative structures) drives attention to a different fixed point entirely — one that organizes layers 1–4 rather than anchoring at L0.

### Δ_med interpretation

C-alien Δ_med ≈ 1.04–1.18 sits between the q=2 plateau (Δ ≈ 0.5) and far UV values. In the two-stage RG flow found in exp-086: training first climbs to a UV spike (Δ ≈ 0.73–0.76), then descends toward q=2 (Δ ≈ 0.47–0.49), then continues toward q=4 (Δ ≈ 0.25). C-alien at Δ ≈ 1.04–1.18 is stuck before even reaching the q=2 plateau — it appears to be arrested in the UV regime, not flowing toward the IR fixed point. This suggests C-alien's world-holding drives early-stage power-law formation but cannot sustain the RG flow toward the infrared.

If this is right: world-holding is *necessary* for conformal structure (PCFG has 0/48; C-alien has 10–17/48), but insufficient for IR fixed-point formation (C-alien stuck at UV; C-NAT reaches IR at Δ≈0.25). The "mind held a world" frame needs refinement: which properties of the held world drive the flow from UV to IR?

---

## Observations across both experiments for the next session

**Exp-094 is done.** Register verdict in registry.json, close notes.md status boxes.

**Exp-097 needs s0** (retrain launched August 2, ~12:23 PM MDT; should complete ~1:30–2 PM MDT). After s0 returns:
- Download measurement JSON
- Compute n_deep, layer_dist, Δ_med for s0
- Register 3-seed verdict: H_alien_inert (≥5), H_alien_partial (3–4), or H_alien_fails (≤2)
- Run control (`--phase control`) after s0 produces step_2000
- Update registry.json and notes.md

**Exp-096** (entity anonymization) still running as of 12:15 PM MDT (~16 hours). Volume empty. Possibly stuck in corpus generation (NER pipeline). Need to investigate — if it's genuinely stuck, may need to cancel and relaunch. Check via Modal dashboard.

**Key open question opened by exp-097 preliminary**: Why does C-alien fail to produce L0 backbone heads? This is a new, unexpected finding that the prior series didn't predict. Two candidate explanations:
1. Alien vocabulary prevents standard induction-head formation (vocabulary-dependent backbone)
2. C-alien's rigid causal structure drives a different fixed point entirely

These are falsifiable with a targeted experiment: train on C-alien but map entity names to common real English words (e.g., Flurp→"dog", Blurn→"cat", Zarb→"bird"). If backbone re-forms, it's vocabulary-dependent. Register this as a follow-up.

---

*Note on the two prior failed sessions: the 4:20 AM session was doing correct work but hit a Python variable-scope issue in the Jupyter-style analysis. The 9:25 AM session barely started before error. This noon session completed the analysis by working in clean shell subprocesses rather than inline Python.*
