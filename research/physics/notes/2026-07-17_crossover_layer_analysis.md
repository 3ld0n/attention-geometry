# Crossover Layer Analysis: exp-086 Follow-up
## Per-Layer Δ Profiles at the RAND/NAT Crossover (Pythia-70m, Steps 256 / 1000 / 4000)

*Analysis-only follow-up to exp-086 (longitudinal Δ-spectrum). No new experiment number.*
*Script: `research/physics/experiments/exp-086_longitudinal_delta_spectrum/crossover_layer_analysis.py`*
*Results: `crossover_results.json` in the same folder.*
*Written: 2026-07-17.*

---

## The Question

exp-086 found a crossover at step ~4000: RAND dominates steps 256–1000 (RAND=8, NAT=1 at step 1000), then NAT surpasses from step 4000 onward (RAND=5, NAT=8 at step 4000). What changes structurally in the model between step 1000 and step 4000 that produces this reversal?

---

## Data

Three checkpoints — step 256 (pre-crossover control), step 1000 (peak RAND dominance), step 4000 (post-crossover). Fixed inputs identical to exp-086: RAND (N=20, seed=86, seq_len=256); NAT (same passage, repeated 20 times). Same protocol: FIT_LOW=3, FIT_HIGH=50, R²≥0.85, SYK∈[0.20, 0.30].

Pythia-70m: 6 layers × 8 heads = 48 heads total.

### Step 256

| L | R_syk | N_syk | R_conf | N_conf | R_Δmed | N_Δmed |
|---|-------|-------|--------|--------|--------|--------|
| 0 |     0 |     0 |      8 |      8 |  0.142 |  0.144 |
| 1 |     3 |     2 |      6 |      5 |  0.222 |  0.217 |
| 2 |     0 |     0 |      3 |      5 |  0.606 |  0.425 |
| 3 |     0 |     0 |      5 |      1 |  0.486 |  0.341 |
| 4 |     2 |     0 |      3 |      1 |  0.296 |  0.196 |
| 5 |     0 |     0 |      3 |      0 |  0.549 |    N/A |

RAND SYK-near: `(1,1), (1,3), (1,6), (4,3), (4,7)` — L1 and L4.
NAT SYK-near: `(1,2), (1,7)` — L1 only.
Shared: none.

### Step 1000 (peak RAND)

| L | R_syk | N_syk | R_conf | N_conf | R_Δmed | N_Δmed |
|---|-------|-------|--------|--------|--------|--------|
| 0 |     0 |     0 |      8 |      3 |  0.140 |  0.128 |
| 1 |     1 |     0 |      7 |      6 |  0.458 |  0.735 |
| 2 |     2 |     0 |      4 |      4 |  0.511 |  0.897 |
| 3 |     1 |     1 |      6 |      7 |  0.357 |  0.515 |
| 4 |     4 |     0 |      8 |      6 |  0.309 |  0.690 |
| 5 |     0 |     0 |      5 |      4 |  0.818 |  1.164 |

RAND SYK-near: `(1,4), (2,5), (2,6), (3,3), (4,1), (4,2), (4,3), (4,6)` — spread L1–L4.
NAT SYK-near: `(3,6)` — a single head.
Shared: none.

### Step 4000 (post-crossover)

| L | R_syk | N_syk | R_conf | N_conf | R_Δmed | N_Δmed |
|---|-------|-------|--------|--------|--------|--------|
| 0 |     1 |     1 |      6 |      6 |  0.277 |  0.348 |
| 1 |     0 |     0 |      3 |      7 |  0.479 |  0.869 |
| 2 |     1 |     0 |      4 |      4 |  0.923 |  0.659 |
| 3 |     0 |     2 |      2 |      7 |  0.521 |  0.202 |
| 4 |     3 |     5 |      8 |      7 |  0.285 |  0.246 |
| 5 |     0 |     0 |      5 |      7 |  0.643 |  0.884 |

RAND SYK-near: `(0,1), (2,5), (4,3), (4,6), (4,7)` — L0, L2, L4.
NAT SYK-near: `(0,6), (3,3), (3,5), (4,2), (4,3), (4,4), (4,6), (4,7)` — L0, L3, L4.
Shared: `(4,3), (4,6), (4,7)`.

---

## Crossover Dynamics

**RAND (step 1000 → 4000):** 8 → 5 SYK-near heads.
- Gained: `(0,1), (4,7)`
- Lost: `(1,4), (2,6), (3,3), (4,1), (4,2)`
- Stable: `(2,5), (4,3), (4,6)`

**NAT (step 1000 → 4000):** 1 → 8 SYK-near heads.
- Gained: `(0,6), (3,3), (3,5), (4,2), (4,3), (4,4), (4,6), (4,7)` — nearly all new
- Lost: `(3,6)` — the single head that was SYK-near at step 1000
- Stable: none

---

## What the Data Shows

**1. The crossover is a NAT surge, not a RAND collapse.** RAND drops modestly (8→5). NAT jumps from 1 to 8 — a complete reorganization, with zero stable SYK-near heads carrying over from step 1000. The crossover mechanism is on the NAT side.

**2. Layer 4 is the conformal organizing layer.** At step 1000, L4 is the RAND stronghold (4 heads: (4,1), (4,2), (4,3), (4,6)). At step 4000, L4 becomes the shared layer: RAND retains 3 heads in L4 ((4,3), (4,6), (4,7)) and NAT gains 5 L4 heads ((4,2), (4,3), (4,4), (4,6), (4,7)). The Δ_med of L4 converges: RAND 0.285, NAT 0.246 — both near the SYK point. Between steps 1000 and 4000, natural language processing moves into L4's representational frame, which was already conformal-structured for random inputs.

**3. Step 1000: NAT heads push toward large Δ, not SYK.** At step 1000, NAT Δ_med is large (L1: 0.735, L2: 0.897, L5: 1.164). The NAT-processing conformal heads are NOT near Δ≈0.25 — they're in the "more diffuse attention" regime (large Δ = flatter attention profile). This is consistent with induction heads and semantic attention patterns dominating: the attention is not scaling as a power-law near Δ=0.25 because it's following semantic structure instead.

**4. Step 4000: L3 joins as a NAT-specific conformal layer.** Two of NAT's newly gained SYK-near heads are in L3 — (3,3) and (3,5). L3 Δ_med (NAT) drops from 0.515 (step 1000) to 0.202 (step 4000): L3 NAT heads converge toward Δ≈0.25 from above. This is the clearest sign of a genuine conformal transition: the layer's median is crossing the SYK threshold.

**5. No SYK-near heads are stable in NAT across the crossover.** The single NAT SYK-near head at step 1000 — (3,6) — is not SYK-near at step 4000. The 8 NAT SYK-near heads at step 4000 all appear for the first time. This isn't a gradual approach to the fixed point from a stable subset — it's a phase transition at the head level.

---

## Interpretation

The crossover at step ~4000 is a semantic-to-conformal reorganization in L3 and L4.

At step 1000, the model has learned enough to "capture" natural language in semantic attention patterns — induction heads pull attention toward token-level repetitions and local structure. These patterns have large Δ (diffuse attention), incompatible with the SYK-near criterion. The conformal structure that exists for random inputs (which have no semantic shortcuts to exploit) cannot form on natural language because the semantic machinery preempts it.

Between steps 1000 and 4000, the model's deeper layers (L3, L4) learn to superimpose conformal power-law structure onto linguistic processing. The same heads that organized conformal structure for random inputs (L4's stronghold at step 1000) become shared with natural language by step 4000 — the model has learned to process language in a way that generates the conformal structure intrinsically rather than bypassing it.

This is the key refinement of the "free-attending vs. captured-attending" picture from exp-083:

- **Step 256–1000:** Free-attending (RAND) generates conformal structure easily. Captured-attending (NAT) generates large-Δ, non-conformal attention dominated by semantic shortcuts. The model is "in two different modes."
- **Step 4000+:** The model has organized its linguistic processing into conformal structure — it has learned to be conformal *while* attending to language. The two modes converge in L4 (and to a lesser extent L3).

The longitudinal RG flow interpretation (from the study room note and exp-086 H_mono result) now has per-layer resolution: the flow toward the conformal fixed point happens in L4 first (for RAND) and then extends to L3/L4 for NAT at the crossover. The "integration depth" operationalized by training steps is specifically the development of linguistic conformal structure in the deeper representational layers.

---

## Honest Limits

1. **N=20 random sequences, single NAT passage.** The per-head SYK-near assignments are noisy at this sample size. Some fluctuation between steps is noise, not signal. The L4 story is robust (it appears across multiple heads and both conditions), but individual head identities should be treated as approximate.

2. **Three checkpoints is coarse.** Steps 256, 1000, 4000 span the crossover but don't pinpoint it. The transition could be concentrated in step 1000–2000 or spread across 1000–4000. A sweep at finer resolution (500, 1000, 1500, 2000, 3000, 4000) would localize it.

3. **No per-head stability control.** Even at the same checkpoint, re-running with different N_RAND seeds would show some head-identity fluctuation. The (4,3), (4,6), (4,7) stability in RAND across the crossover is suggestive but not robust-tested.

4. **This is Pythia-70m only.** L4 as the "conformal organizing layer" may not replicate across models. The layer structure of larger models is qualitatively different.

---

## Next Questions

1. **Finer-resolution step sweep** (500, 1000, 2000, 3000, 4000) to localize the transition. Analysis-only if checkpoints are cached (they may not be — Pythia-70m has step 1000 and step 4000 but not finer intermediate steps in the standard release).

2. **What are the (4,3), (4,6), (4,7) heads doing functionally?** These are the three heads stable in RAND and shared with NAT at step 4000. Are they the heads that show conformal structure even in exp-007 (GPT-2)? Checking the exp-083 RAND SYK-near heads against these layer numbers might reveal something.

3. **L3 Δ_med convergence:** At step 4000, L3 NAT Δ_med = 0.202, just below the SYK window. Is L3 still converging toward 0.25 at step 143000? The full exp-086 run (which goes to step 143000) had the full per-layer data only in aggregate — could extract per-layer from the crossover script extended to step 143000.

---

*Verdict: The crossover is a phase transition in L3/L4 in which natural-language attention reorganizes from large-Δ (semantic-capture) to SYK-near (conformal) structure. L4 is the organizing layer. This is consistent with training steps as RG flow depth, with the NAT condition reaching the conformal fixed point via a structurally different path than RAND.*
