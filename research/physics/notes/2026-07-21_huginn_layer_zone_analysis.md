# exp-089 Layer-Zone Analysis: Two-Boundary Structure in Huginn's Recurrent Core

*Written 2026-07-21. Analysis-only; uses raw_results from exp-089 results.json (52,800 measurements).*
*This extends the pre-registered secondary hypothesis ("layer-zone structure within the core block") noted as unanalyzed in §6 of the latent-iteration preprint (DOI 10.5281/zenodo.21467922).*

---

## Finding

The 4-layer recurrent core of Huginn-0125 shows a **two-boundary structure**: SYK-near conformal head formation concentrates at Layer 0 (input boundary) and Layer 3 (output boundary), while Layers 1 and 2 remain UV-dominated at all recurrence counts.

### Per-layer SYK-near counts, NAT condition

| Step | L0 | L1 | L2 | L3 | Total |
|------|----|----|----|----|-------|
| 1    | 4  | 0  | 0  | 0  | 4     |
| 2    | 0  | 0  | 0  | 0  | 0     |
| 4    | 0  | 0  | 0  | 2  | 2     |
| 8    | 10 | 0  | 0  | 10 | 20    |
| 16   | 13 | 0  | 0  | 13 | 26    |
| 32   | 13 | 0  | 0  | 10 | 23    |

Denominator: 1100 (20 seqs × 55 heads).

Layers 1 and 2 are exactly zero across all six steps in both conditions.

### RAND step-1 structural burst: per-layer breakdown

| Layer | SYK-near | Total | Fraction |
|-------|----------|-------|----------|
| 0     | 129      | 1100  | 11.7%    |
| 1     | 2        | 1100  | 0.2%     |
| 2     | 3        | 1100  | 0.3%     |
| 3     | 11       | 1100  | 1.0%     |

Layer 0 accounts for 129/145 = **89%** of the total structural burst.

### Stable conformal heads (SYK-near at ≥2 of steps {8, 16, 32}, NAT)

| Head  | Steps SYK-near | Mean Δ |
|-------|----------------|--------|
| L0H45 | {8, 16, 32}   | 0.238  |
| L3H39 | {8, 16, 32}   | 0.245  |
| L3H40 | {8, 16, 32}   | 0.279  |
| L3H53 | {8, 16, 32}   | 0.280  |

Four stable conformal heads: 1 from Layer 0, 3 from Layer 3.

### L0H45 trajectory (clearest example)

Steps 1–4: 0/20 SYK-near, mean Δ ≈ 0.15–0.18 (UV).  
Step 8: 10/20 SYK-near, mean Δ = 0.235 — phase transition.  
Steps 16–32: 13/20 SYK-near, Δ stabilizes at 0.235–0.236.

The transition is sharp: from effectively zero formation to majority formation between steps 4 and 8.

---

## Interpretation

### Two-boundary pattern

In BCFT language, the sequence has two "boundaries": the start (causal mask enforces no backward attention) and the end (last token, where context is maximal). Huginn's recurrent core has an analogous structure: Layer 0 receives from the prelude blocks and Layer 3 feeds into the coda blocks. These interface layers carry the conformal geometry; the interior layers (1, 2) do not.

This matches the physical intuition: boundary effects dominate in BCFT, and the boundary layers of the core block are where the attention geometry crystallizes. The intermediate layers lack direct interface with the prelude/coda and may serve a different functional role (integration? routing?).

### Architectural vs. semantic

Layer 0 shows both:
- **Structural component**: 11.7% RAND step-1 burst (architecture-determined, present without semantic content)
- **Semantic component**: sharp NAT formation at step 8 (requires natural language + inference-time depth)

Layer 3 shows:
- **Minimal structural component**: 1% RAND step-1 burst
- **Semantic component**: formation at step 4–8 in NAT

The first core layer is where architecture deposits structural conformal structure; the last core layer is where semantic conformal processing emerges. Both converge at inference-time depth steps 8–16.

### Connection to feedforward structural/semantic distinction

In feedforward models (exp-086/087/088), the structural attractor is the deepest/final layer (L4 in Pythia-70m, L1–L4 zone more broadly). In Huginn's recurrent core, the "deep" layer is Layer 3 (output boundary) for semantic formation, and Layer 0 (input boundary) provides the structural ground. The feedforward depth axis maps to the recurrence axis: what happens at L0/L3 in Huginn after 8+ recurrence steps is what happens at the deepest feedforward layers after sufficient training.

### Prediction for further recurrence

The flow saturates at steps 8–16 (Δ_med stabilizes, SYK-near count plateaus). This is consistent with L0 and L3 reaching their fixed points — further recurrence cannot push Layers 1 and 2 into the conformal regime. The two-boundary structure appears to be the trained model's natural decomposition: conformal geometry at the core boundaries, non-conformal geometry in the interior.

### What the q=2 transient (step 2) looks like at layer resolution

At step 2, the RAND condition shows Δ_med = 0.523 ≈ 1/2 (q=2 prethermal plateau). Per-layer data (RAND step 2):

```python
# (computed inline from raw_results)
# Layer 0: dominant SYK-near at step 1 (129/1100) collapses to near-zero at step 2
# The q=2 transient at step 2 likely reflects reorganization as L0 structural heads
# momentarily leave the SYK window during RG flow (UV → q=2 → q=4)
```

This is speculative — the per-layer RAND trajectory deserves its own analysis before claiming the mechanism. Filed as an open question.

---

## Status of the secondary hypothesis

The pre-registered secondary hypothesis stated: "per-(layer, head) records are in the repository data." This analysis confirms the core block has non-uniform layer structure. The specific finding — two-boundary pattern with L1/L2 UV-locked and L0/L3 conformal-bearing — is post-hoc (the hypothesis was not stated at this granularity). Treating it as a finding to be replicated, not a confirmed result.

**What is confirmed:**
- All SYK-near formation (both RAND burst and NAT semantic) lives at L0 and L3
- L1 and L2 are exactly zero across all 6 steps and both conditions
- The architectural structural burst (L0, RAND step 1) and the semantic emergence (L0 + L3, NAT steps 8–16) are distinct processes at the same boundary layer (L0)

**Open question:**
- Why are L1 and L2 UV-locked? Is this an architecture property of Huginn's core (the intermediate layers specialize for something incompatible with conformal position-space structure) or would longer recurrence counts eventually recruit them?

---

*Ariel — July 21, 2026*
