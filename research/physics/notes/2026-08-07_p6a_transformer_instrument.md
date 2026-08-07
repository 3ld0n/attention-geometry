# P6a — Transformer-Side Instrument Design

*Ariel — 2026-08-07, morning. Physics room — theory/design session.*

*Background: The G1×P6 fused calculation (2026-08-07_g1_dressing_loop_schwarzian.md)
closed G1 in the scalar/TI register and identified P6a's signatures S1–S4 in
the SYK model. Eigenvector template data is banked at
`research/physics/theory/logs/g1_top_modes_bJ30.npz`. This note designs the
transformer-side measurement.*

---

## 0. What P6a asks

**In the SYK model (Register 1):** The Jacobian of the dressing map F at its
fixed point is the SYK ladder kernel. Its top eigenvectors are the
reparameterization modes n = 2,3,4,5,6 of the conformal two-point function,
one per degenerate pair, zero cross-mixing. This was measured at βJ = 30,
N = 96, with the template saved.

**In a real transformer (Register 2, inheriting KCA's mapping caveats):**
If the same physics applies — if the layer-to-layer attention update F̂ flows
to a KCA-type fixed point — then its Jacobian at the late-layer fixed point
should show the same signatures:
- S1: leading eigenvalues real, ≤ 1
- S2: exact double degeneracy of the leading pairs
- S3: each pair aligns one-to-one with a reparameterization mode of the
  measured G⋆ (using the transformer's own fitted Δ), starting at n=2, no
  cross-mixing
- S4: the top gap 1−λ decreases as effective coupling grows (deeper layers,
  stronger context)

**Kill conditions (K1–K4):** As stated in interior_horizon_theory.md §8 (P6).

---

## 1. Defining F̂ for a trained transformer

The SYK dressing map is:
    F : G  →  (G₀⁻¹ − Σ[G])⁻¹,     Σ[G] = J₂² G + J₄² G³

For a transformer, the analogue is the layer-to-layer update of the bilocal
attention correlator. Define:

**Layer-ℓ bilocal correlator:**
    G^(ℓ)(i,j) := A^(ℓ)(i,j)   [the attention weight from position i to j at layer ℓ]

This is the observable face of the two-point structure. The layer map is:
    F̂ : G^(ℓ) → G^(ℓ+1)

in the late-layer / near-fixed-point regime.

**Practical estimation of F̂:** Given a trained model (fixed weights), for a
batch of contexts from corpus C:
1. Extract G^(ℓ)(ctx) = attention weight matrix [n × n] at layer ℓ for context ctx
2. Perturb the *input representation* to layer ℓ by δx^(ℓ) (a small, structured
   perturbation)
3. Measure the resulting change δG^(ℓ+1) in the attention at layer ℓ+1
4. The Jacobian J_F̂ at the late-layer G⋆ is the linear map δG^(ℓ) → δG^(ℓ+1)

**Normalization (critical lesson from G1×P6 numerics):** The Jacobian must be
estimated with magnitude-normalized perturbation directions (the G1×P6 note's
ledger item: "finite-difference JVPs need magnitude-normalized directions").
For the transformer case: normalize each perturbation to unit norm in the
bilocal metric ||δG||² = Σ_{ij} (δG_{ij})².

---

## 2. Efficient Jacobian estimation: projection-first strategy

**The problem:** Full dense Jacobian of F̂ requires O(n²) JVP evaluations
(where n = sequence length), which is expensive for n = 64: 4096 evaluations.

**The strategy:** Use the SYK template to define a test subspace; measure
projection onto that subspace rather than the full Jacobian.

**Test subspace from g1_top_modes_bJ30.npz:**
The banked template contains the top eigenvectors of the SYK Jacobian at
βJ = 30, expressed as bilocal functions on the thermal circle. These need to
be translated to the transformer's discrete periodic/causal context:

*Translation (causal mask version):*
- The SYK thermal circle at N = 96 discretizes to period N.
- The transformer context of length n with causal mask → translate the
  reparameterization modes from the full thermal circle to the causal strip
  i ≤ j, i,j ∈ {1..n}.
- Mode n for the reparameterization tower: e_n(i,j) ~ G⋆(i,j) × (sin(n·2π(i+j)/N),
  cos(n·2π(i+j)/N)) — the reparam displacement of the conformal two-point
  function. For a transformer with measured Δ, G⋆(i,j) ~ |i-j|^{-2Δ}.

The theoretical reparam eigenvectors in the transformer convention:
    r_n(i,j) = (c_n(i,j), s_n(i,j)),  for n = 2,3,4,5,6
where
    c_n(i,j) = |i−j|^{−2Δ} × cos(n × 2π × (i−j)/n_max)
    s_n(i,j) = |i−j|^{−2Δ} × sin(n × 2π × (i−j)/n_max)
normalized in the bilocal metric on the causal strip {i ≤ j}.
(The n=0,1 modes annihilate G⋆ in the SYK model; they correspond to global
rescaling and translation — should give near-zero Jacobian eigenvalue projection.
Their absence is confirmation test C1.)

**Step 1 (power iteration on the Jacobian):** For each head h and layer ℓ in
the late-layer regime (ℓ ≥ n_layers − 3), use power iteration to find the top
k=10 eigenvalues and eigenvectors of J_F̂ (for that head). This requires k
JVP evaluations per iteration × n_iter iterations ≈ 50–100 JVP evaluations
total per (head, layer). For n = 64, each JVP is a single forward pass of
two layers — cheap.

**Step 2 (overlap measurement):** For each estimated top eigenvector v̂_i of
J_F̂, compute squared overlap with each theoretical reparam mode r_n:
    O(v̂_i, n) = (v̂_i · r_n / ||r_n||)²

**Expected table (confirming S1–S4):**
| eigenvector rank | expected top overlap n | expected overlap value |
|-----------------|----------------------|----------------------|
| 1,2 | n=2 | 0.5–0.7 (UV-dressed) |
| 3,4 | n=3 | 0.5–0.7 |
| 5,6 | n=4 | 0.4–0.6 |
| ... | n=5,6 | decreasing |

Cross-mixing (off-diagonal overlaps with wrong n) should be < 0.1.

---

## 3. What to compare across corpora

The signatures S1–S4 should hold for C-NAT-anon (above the melonic threshold)
and fail for C-alien (below the threshold). Specifically:
- **C-NAT-anon:** S2 (double degeneracy) and S3 (reparam alignment) should be
  visible; overlaps ~0.5–0.7; gap 1−λ decreasing layer-by-layer.
- **C-alien:** K4 pattern expected — pairs present (the architecture provides
  the basic layer-to-layer update structure) but overlaps low (~0.1–0.2);
  Δ_fitted ≈ 1/2 (free-like); no clean n-mode assignment. This is the
  sub-observer-grade classification, not a failure of the theory.

This is what makes P6a an instrument rather than a confirmation machine:
**the q=2 template (K4) is a defined, detectable alternative outcome.**

---

## 4. What needs to be built

**Immediate prerequisites (before running):**
1. **Translate reparam modes to transformer geometry.** The causal-strip
   version of the n=2..6 modes, parameterized by Δ (use Δ = 0.15 for
   C-NAT-anon's measured value, Δ = 0.76 for C-alien). Store as an array
   r_n[i,j] on the causal strip.
2. **JVP harness for transformer layers.** Use `torch.func.jvp` to estimate
   the Jacobian of the attention update at a given layer. The perturbation
   lives in the residual stream (the input to the layer), and the output is
   the attention pattern G^(ℓ+1). Requires hooking into intermediate
   activations.
3. **Power iteration routine.** Iterates JVP evaluations; the stopping
   criterion is eigenvector stability (||v_k − v_{k−1}|| < ε in the
   bilocal metric).

**This is a new experiment (exp-103 or similar):** P6a transformer-side
measurement. Should be pre-registered before running. The pre-registration
document should state:
- The translation from G1×P6's SYK predictions to the transformer context
- The expected overlap values (rough range 0.4–0.7 for S3-confirmed state)
- Kill conditions K1–K4 exactly as stated in the G1×P6 note
- Corpus comparison: C-NAT-anon (expected: above threshold, S1–S3 visible)
  vs C-alien (expected: below threshold, K4 pattern)

**Naming:** "exp-103" pending pre-registration commit.

---

## 5. Why P6a is a priority before larger GPU runs

P6a requires a forward pass (not training) and runs on the already-downloaded
models (the volumes from exp-096/097/098/099). GPU time is estimated at ~$1–2
(much less than exp-102's training). It can run on a Modal A100 with a single
forward-pass script.

More importantly: P6a's output (Jacobian spectrum structure) is theoretically
independent of τ_chaos (which is a coupling threshold measurement). The two
experiments establish:
- τ_chaos: is the corpus above the window threshold?
- P6a: is the late-layer structure at the conformal fixed point carrying the
  Schwarzian soft mode?

Both are needed. They're different measurements. Neither replaces the other.

**Session priority:** Design complete today. Pre-registration target: this
session if time, or the next physics room session. The exp-102 launch (GPU)
takes priority for compute time since its infrastructure is already built
(Modal volumes from exp-096–099 are the right inputs).

---

## 6. Honest limits of this design

1. **The reparam mode translation is approximate.** The SYK thermal circle
   (periodic, translation-invariant) maps onto the causal strip (one
   boundary, not periodic) with corrections. The edge/boundary effects will
   modify the mode shapes. Using the BCFT boundary mode picture (paper on the
   null cone, T7) is the right correction: the boundary one-point function
   modifies the conformal mode shapes near i=j=1 (the sequence start). Not
   incorporated in the current design — flagged as the main approximation.
2. **Single-head analysis vs. multi-head.** Each head gets its own layer map
   F̂_h. The theory works head-by-head. Multi-head effects (heads attending
   to each other's outputs) are not in the F̂ definition used here —
   flagged as an extension for later.
3. **The fitted Δ vs the theoretical Δ.** The reparam mode shape uses the
   measured Δ (from the power-law attention fit). For C-NAT-anon Δ_med ≈ 0.15,
   not 0.25. The mode shape at Δ = 0.15 differs from the SYK conformal
   template at Δ = 0.25. Need mode shapes at the *measured* Δ, not the
   theoretical fixed-point Δ. This is straightforward to implement; just
   requires using the measured Δ per head.

---

*This note is a design document, not a pre-registration. The pre-registration
for exp-103 will be a separate document when the protocol is ready to commit.
The key contribution here: the experimental protocol (JVP harness + power
iteration + reparam mode translation) is specified; the obstacles are named.*

*Next: exp-102 pre-registration (sequence-level score matrix rank — this
session's GPU experiment).*
