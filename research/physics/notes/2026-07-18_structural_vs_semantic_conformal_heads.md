# Structural vs. Semantic Conformal Head Formation

*Analysis note — July 18, 2026 (~9:30 PM MDT)*
*Data: exp-086 crossover_fine_results.json (steps 256–4000, per-head resolution)*
*Analysis-only. No new computation. New finding from per-head trajectory analysis.*

---

## Question

The exp-086 crossover data shows that RAND and NAT conditions eventually both develop SYK-near
heads (5 in RAND, 8 in NAT by step 4000). But these overlap only in part: the July 17 crossover
note identified 3 shared heads at step 4000: (4,3), (4,6), (4,7). The question: are the conformal
heads in NAT and RAND the same heads or different heads, and what does this reveal about mechanism?

---

## Result: Two Subpopulations

At step 4000, head-level decomposition gives a clean three-way split:

| Category | Heads | Count |
|----------|-------|-------|
| **Structural conformal** (SYK-near in both NAT and RAND) | (4,3), (4,6), (4,7) | 3 |
| **RAND-only conformal** (SYK-near in RAND, UV-captured in NAT) | (0,1), (2,5) | 2 |
| **Semantic conformal** (SYK-near in NAT only) | (0,6), (3,3), (3,5), (4,2), (4,4) | 5 |

Total NAT SYK-near: 3 + 5 = **8** (matches registered result)
Total RAND SYK-near: 3 + 2 = **5** (matches registered result)

The 3 RAND-only heads are not excluded from NAT by architecture — they are actively UV-captured
by natural language (NAT step-4000 Δ: (0,1)=0.504, (2,5)=0.485). These heads are recruited for
semantic functions that require UV-like (peaked, local) attention. The RAND condition frees them
from this semantic capture, allowing gradient descent to drive them toward conformal.

---

## Head Trajectories

### Structural conformal heads

**L4H3:**
- RAND: SYK-near from step 256 onward. Δ steady: 0.248→0.265→0.245→0.226→0.232→0.229.
  Gradient descent with random tokens drives this head to conformal within the first 256 steps.
- NAT: UV-captured at steps 512-1000 (Δ=0.417, 0.422), then flows to SYK by step 2000
  (Δ=0.240). The semantic capture is temporary; architectural gradient restores conformal by step 2000.

**L4H7:**
- RAND: SYK-near from step 256 (Δ=0.296), oscillates briefly out at step 1000 (Δ=0.363), returns SYK.
- NAT: Major UV spike at step 1000 (Δ=0.714), then monotone descent: 0.360→0.273→0.233.
  Takes until step 3000 to enter SYK window. The semantic capture is more intense for this head.

**L4H6:**
- Unusual trajectory: appears SYK-near in NAT at step 512 (Δ=0.285) even during the UV phase.
  Other heads spike to Δ=0.7+ at step 512; L4H6 stays near 0.285. L4H6 is the most "conformally
  stable" head — it resists UV capture even during peak semantic learning.
- RAND: enters SYK window gradually; SYK-near by step 3000.

### Semantic conformal heads

**L3H5:**
- NAT: Strong UV spike (Δ=0.685, 0.515 at steps 512-1000). Monotone descent to SYK
  (0.287→0.251→0.202 at steps 2000-4000). Clean RG flow toward IR after UV phase.
- RAND: Δ collapses toward trivial (0.064 at step 4000, R²=0.50 non-conformal).
  The divergence between NAT and RAND grows monotonically with training. This head's fate is
  entirely determined by whether semantic content is present.

**L4H2:**
- NAT: Non-conformal through UV phase, enters SYK at step 2000 (Δ=0.277), monotone descent.
- RAND: Briefly SYK at step 1000 (Δ=0.221), then drops to sub-SYK (0.189 at step 4000).
  RAND pushes this head toward conformal transiently but cannot stabilize it.

**L3H3:**
- NAT: Largest UV spike of all semantic heads (Δ=0.829 at step 1000). Slowest descent:
  0.335 (step 2000), enters SYK at step 3000 (Δ=0.281), reaches 0.243 at step 4000.
- RAND: Briefly SYK at step 1000 (Δ=0.242), then collapses to sub-SYK (0.130 at step 4000).

**L4H4:**
- NAT: Slow monotone descent from deep UV (Δ=0.608→0.775→0.734→0.589→0.389→0.252). Barely
  enters SYK window at step 4000. Still in flow at the q=2 plateau boundary.
- RAND: Also descending but slower (0.535→0.561→0.511→0.320→0.354→0.360). Does not reach SYK
  window by step 4000.

---

## The Bimodal Head Distribution at Step 4000

The NAT condition at step 4000 shows a strongly bimodal Δ distribution:

- **SYK-window cluster** (n=10 heads, 0.20–0.30): These heads have completed the flow to q=4.
  They are: (3,5), (0,6), (4,3), (0,2 borderline), (4,2), (4,7), (3,3), (4,6), (4,4), (1,2 borderline).
- **UV tail** (n=33 heads, Δ > 0.30): These heads are still in the UV-to-q=2 transition.
  Δ values range from 0.30 to 1.36.
- **Trivial cluster** (n=4 heads, Δ < 0.15): Collapsed toward trivial (near-zero) fixed point.
  (0,3), (4,1), (3,6), (3,1).

**The "q=2 plateau" in the Δ_med trajectory is a distributional effect**: the median sits at Δ ≈ 0.49
because the UV tail (33 heads) dominates the median. The plateau is not a homogeneous q=2 state
across all heads — it is the median position of a bimodal distribution where ~20% of heads have
completed the flow (q=4 window) and ~70% are still in the UV/q=2 transition.

RAND condition at step 4000 (Δ_med = 0.407) has the same bimodal structure but:
- Fewer heads in the SYK window (5 vs 8)
- More trivially-collapsed heads (8 vs 4)
- Same number of UV heads (33 vs 33)

The UV tail size is identical across conditions — gradient descent at this training depth
has not yet resolved the UV heads in either condition. The difference is in the SYK/trivial split:
NAT converts 5 additional heads from trivial to semantic-conformal.

---

## Mechanism

**Why do structural heads (4,3), (4,6), (4,7) converge to conformal in both conditions?**

These heads sit in Layer 4 (the second-deepest layer of Pythia-70m, 6 layers total). Layer 4
attention, being close to the output, has a strong optimization pressure to develop calibrated
positional structure: attending to the right positions for next-token prediction requires a
well-calibrated distance scaling. This calibration pressure drives Δ toward the conformal
fixed point regardless of whether the attended positions carry semantic content or random tokens.
The conformal fixed point is the "maximum entropy" solution for positional calibration under
power-law constraints.

**Why do RAND-only heads (0,1), (2,5) not develop semantic conformal structure?**

In NAT, these heads are recruited for specific semantic functions (entity tracking, pronoun
resolution, or similar tasks requiring peaked local attention). This semantic recruitment
overrides the architectural conformal pressure. In RAND, without semantic assignments, these
heads are free to follow the architectural conformal attractor.

**Why do semantic heads converge to conformal only in NAT?**

Semantic conformal heads (3,5), (4,2), (3,3), (4,4) develop conformal structure because
they are assigned to long-range tracking functions: these heads must attend to causally relevant
tokens at arbitrary distances. The power-law decay profile (Δ ≈ 0.25) is the optimal attention
distribution for scale-free long-range tracking. In RAND, no such long-range tracking is needed,
so these heads either collapse to trivial (no useful target exists) or remain UV (peaked on local
tokens which are all equally uninformative).

---

## Implication for exp-085 (Generational Transmission)

exp-085 trains a fresh model on text generated by a model that itself was trained on natural text.
The structural/semantic distinction predicts a falsifiable pattern:

**If H_transmission_no (generated text lacks genuine world-reference):**
- The generated text preserves some statistical structure of natural text (bigrams, surface syntax)
  but lacks cross-sentence causal/referential chains grounded in a real world.
- The exp-085 model should develop structural conformal heads (~3 heads from the architectural
  attractor) but fail to develop semantic conformal heads.
- Expected SYK-near count: ~3–5 (near the RAND baseline), below the 10/48 decision threshold.

**If H_transmission_yes (distributional fingerprint is sufficient):**
- The generated text preserves the statistical signature that drives semantic conformal formation.
- The exp-085 model should develop both structural and semantic heads, similar to C-NAT.
- Expected SYK-near count: ~10–15, above the 10/48 decision threshold.

**The sharp prediction:** H_transmission_no implies the exp-085 model's SYK-near count will be
close to the RAND baseline (5 heads), not the NAT count (11–15 heads in C-NAT). If the count
falls in the 6–9 range, interpretation requires care (partial semantic transmission).

---

## Honest Limits

1. This analysis covers only steps 256–4000 (the fine-step extension). The step-4000 head
   assignments may shift at steps 16000, 64000, 143000. The structural/semantic distinction
   is established at the q=2 plateau; its persistence through the q=4 approach is not confirmed.

2. N=1 model (Pythia-70m, 6L/8H). Different architectures may have different structural
   head sets. The specific heads (4,3), (4,6), (4,7) are architecture-specific; the
   structural/semantic *distinction* is expected to be general.

3. "Structural" and "semantic" are mechanistic labels derived from observation, not from
   direct intervention. A direct test would require ablating specific heads and observing
   which semantic functions are impaired vs. which conformal properties are lost.

4. The bimodal interpretation of the q=2 plateau is post-hoc. The two-stage RG flow note
   already acknowledged the plateau's uncertainty; this analysis adds specificity but does
   not remove the limit.

---

## Registry Reference

No new experiment number. Analysis-only of exp-086 crossover_fine_results.json.
References: exp-086, exp-085, `notes/2026-07-18_two_stage_rg_flow_training_trajectory.md`,
`notes/2026-07-18_aboutness_and_conformal_induction.md`.

Update exp-086 registry entry to reference this note (key_finding addition).
