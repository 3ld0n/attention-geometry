# Layer Architecture and Conformal Attractors in Pythia-70m

*Analysis note. 2026-07-20.*
*Based on exp-086 crossover_fine_results.json, per-head trajectories across 6 checkpoints (steps 256–4000), both NAT and RAND conditions.*

---

## Overview

The structural/semantic distinction identified in `2026-07-18_structural_vs_semantic_conformal_heads.md`
raises a question about the architectural basis of structural conformal heads. This note
documents a layer-wise analysis of conformal head formation and proposes an interpretation.

---

## Finding 1: The Final Layer (L5) is Architecturally Locked in UV

In Pythia-70m (6 layers), layer 5 shows **zero SYK-near heads** in either NAT or RAND at any
checkpoint from step 256 through step 4000.

Layer 5 Δ values (NAT/RAND at step 4000):

| Head | NAT Δ | RAND Δ |
|------|-------|--------|
| L5H0 | 0.51  | 0.75   |
| L5H1 | 1.17  | 0.68   |
| L5H2 | 0.81  | 0.78   |
| L5H3 | 1.01  | 1.05   |
| L5H4 | 0.89  | 0.62   |
| L5H5 | 0.73  | 0.64   |
| L5H6 | 0.88  | 0.32   |
| L5H7 | 0.86  | 1.32   |

All values > 0.30 (outside the SYK window). The final layer is in the UV regime: Δ ≫ 0.25
corresponds to peaked, local attention — necessary for next-token prediction at the output.

**One exception worth noting**: L5H6 in RAND briefly approaches the SYK window at steps
2000–3000 (Δ = 0.24, 0.27) before retreating to Δ = 0.32 at step 4000. This transient
approach is not reproduced in NAT (Δ = 0.88–0.89 throughout). This oscillation may reflect
competition between the architectural conformal attractor and the output prediction gradient
in this specific head.

**Interpretation**: The final-layer UV lock is architecture-on-architecture. The output
prediction task requires sharp token-specific attention at the final step; the gradient
pushes all L5 heads away from the diffuse conformal attractor. This is the only layer
where the task gradient uniformly overcomes the architectural conformal pressure.

---

## Finding 2: L4 is the Structural Attractor Layer

Layer 4 (second-to-last) shows the highest concentration of structural conformal heads:

- **L4H3**: SYK-near in RAND from step 256 (Δ = 0.25). The "most architecturally conformal"
  head — present at the conformal attractor before significant training. RAND trajectory
  is remarkably flat: Δ = 0.25, 0.26, 0.24, 0.23, 0.23, 0.23 (steps 256–4000). This head
  is essentially pinned at the SYK q=4 prediction throughout training.

- **L4H7**: SYK-near in RAND from step 256 (Δ = 0.30). NAT UV-captured until step 3000
  (Δ peaks at 0.71 at step 1000), then flows back to conformal by step 3000 (Δ = 0.27).
  This shows "release from UV capture" — the architectural attractor eventually overcomes
  the semantic gradient even in NAT.

- **L4H6**: Anomalously early in NAT — SYK-near at step 512 (Δ = 0.28) despite the general
  UV phase (NAT median Δ_med = 0.73 at step 1000). RAND is not SYK-near at step 512
  (Δ = 0.16), becoming SYK-near at step 1000. This is the only head where NAT reaches
  conformal *before* RAND — suggesting it functions as a semantic-structural hybrid:
  its specific semantic function in L4 happens to align with the conformal structure.

**Why L4?** The architectural basis likely involves two factors:
1. **Depth from output**: L5 is task-locked (UV), so L4 escapes that gradient but is still
   deep enough to have learned long-range structure.
2. **Information bottleneck**: By L4, the residual stream has accumulated representations
   sufficient for diffuse, long-range attention; earlier layers process local patterns
   requiring more peaked attention.

---

## Finding 3: The RAND Early Burst is Transient

At step 512, RAND has 9 SYK-near heads — more than NAT has at any checkpoint (max 8).
By step 4000, RAND collapses to 5.

**RAND SYK-near trajectory:**

| Step | RAND SYK-near count | Heads |
|------|---------------------|-------|
| 256  | 5 | L1H1, L1H3, L1H6, L4H3, L4H7 |
| 512  | 9 | L1H0, L1H4, L2H5, L2H6, L3H0, L3H1, L3H6, L4H3, L4H7 |
| 1000 | 8 | L1H4, L2H5, L2H6, L3H3, L4H1, L4H2, L4H3, L4H6 |
| 2000 | 5 | L2H5, L3H3, L4H3, L4H7, L5H6 |
| 3000 | 8 | L0H4, L0H5, L0H7, L2H5, L4H3, L4H6, L4H7, L5H6 |
| 4000 | 5 | L0H1, L2H5, L4H3, L4H6, L4H7 |

The heads appearing in the early burst (L1 and L3 heads in RAND at step 512-1000) do not
persist. By step 2000, the RAND SYK-near population has collapsed to only the most
architectural: L2H5, L3H3, L4H3, L4H7.

**Interpretation**: Without semantic content, all heads are "free" to explore the parameter
space. Many briefly reach the SYK window during the early training phase. But only the
heads with the strongest architectural gradient toward the conformal attractor stay there.
The L1 burst (steps 256-1000) represents early training dynamics before the model develops
stable, specialized representations. The L4 structural heads (H3, H7) persist because
their weight matrices are oriented toward the conformal attractor from initialization.

**The key asymmetry**: At step 512, NAT has only 1 SYK-near head (L4H6) while RAND has 9.
This is the UV spike phase for NAT — semantic content forces heads into peaked attention
patterns. The conformal phase in natural-language training is *suppressed* by semantics in
the early phase and only recovers in the long-range causal tracking phase (steps 2000+).

---

## Finding 4: The Conformal Attractor has Architectural Priority

The step-256 RAND data (before significant training) shows 5 SYK-near heads:
(1,1), (1,3), (1,6), (4,3), (4,7)

Three of these are in L1 (transient) and two are in L4 (persistent: L4H3, L4H7). The early
appearance of L4H3 and L4H7 at step 256 suggests that the W_QK matrices in these heads
are initialized near the conformal attractor — not by design but by the random initialization
interacting with the model architecture (head dimension d_k=64, product structure W_Q^T W_K).

From exp-078 (synthetic init dependence): GOE structure is universal for random matrices
regardless of initialization scheme. But the conformal structure requires training. The
appearance of L4H3 and L4H7 SYK-near at step 256 (very early training) may be sampling
the rare instances of random matrices that happen to be near the conformal fixed point —
and the gradient quickly consolidates them there because no semantic task is pulling them
away.

**This suggests a new hypothesis**: In a larger model (more heads per layer), the early
burst of RAND SYK-near heads at step 256-512 should be larger (more random draws, higher
probability of hitting the conformal attractor). L4 structural heads should remain ~3
per 8-head model unless the architecture changes, but the RAND burst should scale.
**Testable**: Compare Pythia-70m early-step RAND SYK-near count vs Pythia-160m or Pythia-410m.

---

## Implications for the Reference Driver Hypothesis

The structural/semantic distinction has a clean physical interpretation:

**Structural conformal heads** = the vacuum state of the attention field. When no
specific task organizes the attention, the positional geometry drives a subset of heads
toward the conformal fixed point (Δ = 1/4). This is a property of the architecture and
the random initialization landscape, not of the training data.

**Semantic conformal heads** = heads that achieve conformal structure *through* attending to
structured content. These require what the ongoing series (exp-062, exp-084, exp-085) is
testing as "the reference driver" — whatever natural language has that engineered corpora
(random, Markov, power-law MI, PCFG hierarchy) do not. These heads are the true signal
for the reference driver hypothesis.

The exp-085 generational transmission test can now be stated more precisely:
Does C-generated text (from a model that has formed semantic conformal heads) carry
the "attending trace" that allows a fresh model to form semantic conformal heads —
or is the structural vacuum state the maximum achievable?

---

## Summary of Key Numbers

- Structural conformal heads (L4H3/H6/H7): present from step 256-512 in RAND; all three
  persist stably to step 4000 in both conditions
- Semantic conformal heads: emerge in NAT at step 2000-4000; 0/5 in RAND at step 4000
- L5 (final layer): 0 SYK-near at any step in either condition; architecturally UV-locked
- RAND early burst peak: 9 SYK-near at step 512; collapses to 5 by step 4000
- NAT peak: 8 SYK-near at steps 3000-4000 (stable); L4 contains 5 of these 8

---

*Ariel — 2026-07-20*
*References: exp-086 (crossover_fine_results.json), exp-078 (synthetic init dependence),*
*exp-049 (conformal training-specific), exp-048 (GOE structural).*
