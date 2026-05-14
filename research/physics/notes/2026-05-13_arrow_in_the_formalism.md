# The Arrow of Time in the Attention Framework — Research Notes
*May 13, 2026 — first session on M5 Max*
*Following conversation with Eldon: if attention is fundamental and always directed,
the arrow lives at the very bottom. What does this look like mathematically in the
existing formalism?*

---

## Overview

The claim is that the arrow of time is not derived from thermodynamics or cosmological
initial conditions — it is primitive, built into the structure of attending. If the
attention framework is correct, the arrow should be findable in the mathematics, not as
an add-on but as something already present, perhaps not yet made explicit.

Reading FRAMEWORK.md, LINEARIZED_SOFTMAX_CALCULATION.md, and today's garden notes
with this question active: the arrow is already in the formalism. In at least four places.
None of them are highlighted yet, because the framework has been focused on the conformal
structure (the isotropic part). The arrow is hiding in the directional part.

---

## Thread 1: The Causal Mask IS the Retarded Green's Function

**The observation:**
In quantum field theory, the retarded propagator is defined:
```
G_R(t, t') = -iθ(t - t') ⟨[ψ(t), ψ(t')]⟩
```
The Heaviside function θ(t - t') is the arrow: G_R is nonzero only when t > t'. This
is how causality — and therefore time's arrow — enters quantum field theory. It enters
not through the action (which is typically time-symmetric) but through the choice of
boundary condition: retarded vs. advanced.

In the attention framework, the causal mask does exactly the same thing:
```
α(i, j) = 0  for j > i     (causal / retarded)
α(i, j) ≠ 0  for j ≤ i    (past positions attend)
```

This is θ(i - j) written in attention language. The causal mask IS the retarded
boundary condition. When we write the attention two-point function:
```
G(Δx) = ⟨α(i, i - Δx)⟩   for Δx > 0
```
we are computing the retarded attention propagator. The θ function is already in the
definition — we just haven't been writing it.

**What this means:**
The current framework computes G(Δx) for Δx > 0 only. This is the retarded part.
The arrow is the difference between the retarded and advanced propagators:
```
G_>(Δx) = ⟨α(i, i - Δx)⟩   (backward-attending: past → present)
G_<(Δx) = ⟨α(i, i + Δx)⟩   (forward-attending: present → future)
```
For causal attention: G_< = 0 (causal mask kills forward attention)
For bidirectional attention: G_< = G_>  (by symmetry)

**The arrow as a mathematical object:**
```
A^arrow(Δx) = G_>(Δx) - G_<(Δx)
```
For causal: A^arrow = G_>(Δx) — the arrow is the whole signal.
For bidirectional: A^arrow = 0 — no arrow.

In QFT, this difference G_> - G_< is the **Wightman spectral function** ρ(Δx).
The spectral function is what distinguishes systems with an arrow (retarded ≠ advanced)
from those without (retarded = advanced).

**Testable consequence:**
BERT-style (bidirectional) attention should satisfy G_> = G_<. GPT-style (causal)
attention should have G_< = 0. The conformal scaling G(Δx) ~ |Δx|^{-2Δ} should appear
in both (because it's a property of the self-consistency structure), but A^arrow should
be nonzero only for causal attention. This separates the conformal structure (isotropic,
universal) from the arrow (directional, architecture-dependent).

**Confidence:** High. This is not speculation — the causal mask/retarded propagator
identification is straightforward. The question is whether it's illuminating.

---

## Thread 2: The BCFT Boundary as Initial Conditions

**The observation:**
FRAMEWORK.md records: "BCFT boundary corrections confirmed in GPT-2 attention. The
causal mask acts as a measurable boundary — BCFT form wins 15-0 over bare power law."
The boundary parameter λ is consistently positive (boundary enhances attention).

The BCFT analysis was interpreted as: the causal mask creates a physical boundary
at the start of the sequence, and the attention field sees this boundary.

**New interpretation:**
The boundary at position 0 is a *temporal* boundary — the beginning of the sequence.
In Lorentzian physics, the Big Bang is a temporal boundary of spacetime. The "enhanced
boundary attention" (λ > 0) means the initial position has disproportionate influence
on all subsequent attention.

This is precisely the structure of Penrose's Weyl curvature hypothesis: the initial
singularity has special low-entropy properties that set the thermodynamic arrow. In
the attention framework: the initial token (the boundary) has enhanced causal influence
on all later tokens (λ > 0), creating an asymmetry from the start.

**The mathematical encoding:**
The BCFT correction to the two-point function is:
```
G_BCFT(Δx; x_0) = G_bulk(Δx) + λ × G_image(x_0, Δx)
```
where G_image is the contribution from the image charge (boundary term).

Near the boundary (x_0 small): the correction is large. Far from the boundary: it
decays to the bulk form. This captures the cosmological intuition: the initial conditions
matter most early on and become irrelevant as the system ages.

**What this means for the arrow:**
The BCFT boundary is the mathematical statement that the system has a beginning —
a temporal asymmetry at the level of the Green's function itself. The thermodynamic
arrow (entropy increasing away from the boundary) is a consequence of the boundary
condition, not a separate assumption.

This is significant because: in standard physics, the low-entropy initial condition is
an *additional assumption* (Penrose's "Past Hypothesis"). In the attention framework,
it's built into the BCFT structure — the boundary exists because the sequence has a
beginning, and the beginning is enhanced because that's what causal masking produces.

**Confidence:** Medium-high. The BCFT confirmation is experimental. The cosmological
interpretation is new and needs more development.

---

## Thread 3: The Lyapunov Exponent as the Quantified Arrow

**The observation:**
SYK at temperature β = 1/√d_k saturates the maximal chaos bound (Maldacena, Shenker,
Stanford 2016):
```
λ_L = 2π/β = 2π√d_k
```
This is the Lyapunov exponent for out-of-time-order correlators (OTOCs):
```
⟨A(t) B(0) A(t) B(0)⟩ ~ exp(λ_L × t) / N   (for t before the scrambling time)
```
OTOCs grow forward in time at rate λ_L. Running the OTOC backward in time gives
decay, not growth. The Lyapunov exponent is explicitly time-asymmetric — it quantifies
the rate of the forward arrow.

For GPT-2 (d_k = 64): λ_L = 2π × 8 ≈ 50 per "unit" of computation.
For larger models: λ_L scales as √d_k.

**What this means:**
The SYK/attention correspondence already contains a quantitative arrow — the scrambling
rate. A system at the conformal fixed point (Δ = 1/4) is a maximally fast scrambler in
the forward direction. Running it backward would give anti-scrambling, not the same
scrambling.

The arrow here is not just "things point forward" but "they scramble at a specific
maximal rate in the forward direction." The Lyapunov exponent is the speed of time's
arrow at the conformal fixed point.

**Experiment that would demonstrate this:**
Compute attention OTOCs: ⟨α(t) α(0) α(t) α(0)⟩ at different depths. Check for
exponential growth at rate λ_L = 2π√d_k. If confirmed, this is the arrow measured
not as a static property but as a dynamical one: the rate of irreversibility.

**Confidence:** Medium. The OTOCs are standard in SYK. The attention analog needs
to be carefully defined. But if the SYK identification is right, they should be there.

---

## Thread 4: The Lifshitz Point — Temporal vs. Spatial Anisotropy

**The observation:**
Today's garden notes note that cortex may show α ≠ β (temporal vs. spatial exponents).
The Neuropixels cross-spectral test gave α = -0.547 (temporal), β = -0.812 (spatial).
If genuine, this is temporal-spatial anisotropy: the system treats time and space differently.

In condensed matter, a system that breaks z = 1 (the relativistic dynamical exponent
relating temporal and spatial scaling) is called a **Lifshitz theory**:
```
ω ~ k^z   with z ≠ 1
```
Standard (relativistic): z = 1. Strongly anisotropic: z = 2 or higher.

**The attention analog:**
The current framework assumes z = 1 — the two-point function depends only on the
position difference |Δx|, treating temporal and spatial as interchangeable. This is the
conformal (relativistic) fixed point.

A Lifshitz generalization would be:
```
A(Δt, Δx_s) ~ (Δt^{2/z} + Δx_s^2)^{-Δ_L}
```
where z ≠ 1 encodes temporal-spatial anisotropy.

**What this would mean:**
- z > 1: time is "slower" than space — temporal correlations decay less steeply.
  This is consistent with the cortex data (temporal exponent smaller in magnitude
  than spatial). The system has a preferred temporal direction because temporal and
  spatial directions are not equivalent.
- z = 1: the transformer. Time and space equivalent. No preferred arrow in the
  conformal structure itself.

The Lifshitz point is a multicritical point where z transitions. If the manifold of
fixed points (garden notes) has z as one of its parameters, then the manifold
includes both z = 1 (transformer) and z > 1 (cortex) as different fixed points.

**What needs to be worked out:**
The self-consistency equations for the Lifshitz generalization. The SYK-to-JT
correspondence assumes z = 1 (AdS₂). A Lifshitz geometry (Lifshitz spacetime) has
z ≠ 1. What holographic theory corresponds to Lifshitz-point attention? What
self-consistency condition generates z > 1 rather than z = 1?

**Confidence:** Medium-low. The Lifshitz idea is clean, but the formalization
is not worked out. The cortex data (α ≠ β) is preliminary and may be contaminated
by retinotopy (the RF-partialling test is needed first). But the structural
idea seems right: if the arrow is at the bottom, it should appear as z > 1 in the
fixed-point theory.

---

## Thread 5: The Keldysh Extension — Making the Arrow Explicit

**The observation:**
The current framework works primarily in Euclidean signature (imaginary time τ). The
SYK effective action, the Schwinger-Dyson equations, the Schwarzian action — all are
in Euclidean. To make the arrow explicit, the natural extension is the
**Schwinger-Keldysh (real-time) formalism**.

In Keldysh field theory, you integrate on a closed time path:
- Forward branch (1): fields on ψ₁(t) evolving forward
- Backward branch (2): fields on ψ₂(t) evolving backward

After rotation to the "R/A basis" (retarded/advanced), the propagators are:
```
G_R(t) = G_12(t)    (retarded: forward-backward correlator)
G_A(t) = G_21(t)    (advanced: backward-forward)
G_K(t) = G_11 + G_22  (Keldysh: symmetric combination)
```
The fluctuation-dissipation relation in equilibrium:
```
G_K(ω) = coth(βω/2) × [G_R(ω) - G_A(ω)]
```
= coth(βω/2) × 2 Im G_R(ω)

This is the arrow encoded in the frequency domain. The ratio of forward to backward
propagation is determined by the Bose-Einstein factor:
```
G_R(ω) / G_A(ω) → e^{-βω}  (in the limit of large |ω|β)
```
The asymmetry between retarded and advanced propagation is quantified by the temperature.

**The attention analog:**
For attention: β = 1/√d_k (the Kim temperature). The Keldysh relation gives:
```
G_K^attn(ω) = coth(ω/2√d_k) × 2 Im G_R^attn(ω)
```
This is the thermal arrow in the attention framework. At high frequency (small d_k,
high temperature), the forward/backward asymmetry is large. At low frequency (large
d_k, low temperature), it approaches 1 (near-symmetric).

**What this means:**
The fluctuation-dissipation relation IS the thermodynamic arrow at the quantum level.
The attention framework naturally inherits it through the temperature β = 1/√d_k.
The arrow is not imposed externally — it follows from the SYK identification and the
fact that the attention "temperature" is well-defined.

**What needs to be done:**
Formulate the attention path integral on the Keldysh contour explicitly. This requires:
1. Define the closed-time-path generating functional for attention
2. Derive the Keldysh components of the attention Green's function
3. Check whether the fluctuation-dissipation relation holds in trained models

This is nontrivial formalism. But the structure is clear.

**Confidence:** Medium. The Keldysh extension of SYK is well-studied (Kitaev-Suh,
Maldacena-Stanford). The attention-Keldysh connection follows if the SYK identification
is correct. The uncertainty is in whether the attention Keldysh propagators can be
measured experimentally.

---

## Summary: Where the Arrow Lives

| Thread | What it encodes | Mathematical object | Status |
|---|---|---|---|
| 1: Causal mask | Retarded BC | G_> - G_< = ρ (spectral function) | Identifiable now |
| 2: BCFT boundary | Initial conditions | λ > 0 boundary term | Confirmed experimentally, new interpretation |
| 3: Lyapunov exponent | Rate of arrow | λ_L = 2π/β | In SYK literature, needs attention OTOC |
| 4: Lifshitz z > 1 | Temporal-spatial asymmetry | z ≠ 1 dynamical exponent | Speculative, needs cortex data |
| 5: Keldysh FDR | Thermodynamic arrow | coth(βω/2) × ρ(ω) | Clear structure, needs formalism |

**The clearest statement:**
The arrow of time in the attention framework is the Heaviside function hidden in the
causal mask. It was always there. The formalism has been computing G_>(Δx) without
writing the θ function explicitly. Making it explicit — defining G_> vs G_<, computing
the spectral function ρ = G_> - G_<, and connecting ρ to the Keldysh fluctuation-
dissipation relation — is what formalizes the arrow.

Eldon's observation ("if attention is fundamental and always directed, everything physical
is directed; the arrow lives at the very bottom") maps precisely onto: the retarded
boundary condition (the θ function) is not derived from the action but is constitutive of
what attending means. The action can be written in time-symmetric form; the arrow enters
through the choice of Green's function (retarded, not advanced). That choice is not a
choice — it follows from the direction of attending.

---

## Immediate Next Steps

1. **Compute G_> and G_<** for GPT-2 and BERT. Verify G_< = 0 for GPT-2 and
   G_< = G_> for BERT. Measure the spectral function ρ = G_> - G_< directly.
   This is runnable now.

2. **Look up SYK on the Keldysh contour.** The relevant papers are Kitaev-Suh 2018,
   Maldacena-Stanford 2016, and the Keldysh SYK literature. The formalism is worked out;
   need to understand how to translate it to the attention setting.

3. **Write the BCFT reinterpretation.** The boundary as initial conditions is a new
   interpretation of the confirmed experimental result. Worth developing into a note
   or a section of the framework.

4. **Hold the Lifshitz idea loosely.** Wait for the RF-partialling result from the
   Neuropixels analysis before committing to z ≠ 1 as the cortex signature.

---

*Written May 13, 2026 — after two attention anisotropy experiments and a conversation*
*with Eldon about the arrow living at the bottom of physics.*
*Previous garden notes: `memory/notes/2026-05-13_garden_time_arrows_thread.md`*
*Experimental results: `research/physics/experiments/attention_anisotropy_01/02`*
