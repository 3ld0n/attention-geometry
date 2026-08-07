# exp-103 — P6a Transformer-Side Jacobian Spectroscopy

*Pre-registration: this file is committed to 3ld0n/attention-geometry BEFORE
the analysis script runs. Data access and results are appended afterward.*

*Design document: `notes/2026-08-07_p6a_transformer_instrument.md` (Aug 7 session).*
*Theoretical grounding: `notes/2026-08-07_tau_chaos_product_formula.md` (same session);
`theory/interior_horizon_theory.md` §8 P6.*
*SYK template bank: `theory/logs/g1_top_modes_bJ30.npz` (G1×P6 overnight session).*

*Ariel — August 7, 2026.*

---

## Motivation

The G1×P6 fused calculation (2026-08-07) closed named obstacle 1 (dressing-loop
closure) in the scalar/translation-invariant register. It established:

1. **The Jacobian of the SYK dressing map F at its conformal fixed point IS the
   SYK ladder kernel.** Its spectrum is real, ≤ 1, with no eigenvalue above 1
   (verified at βJ ≤ 30, N = 96).
2. **The slowest sector is the reparameterization tower.** Modes n = 2,3,4,5,6,
   doubly degenerate, zero cross-mixing, overlaps ~0.6–0.67 at βJ = 30 (finite-βJ
   UV dressing). Template saved: `theory/logs/g1_top_modes_bJ30.npz`.
3. **The gap 1−λ_top closes as βJ grows** (exponent −0.72 at βJ ≤ 30, drifting
   toward the Schwarzian −1 asymptote).

This closure makes P6a a *testable proposition*: if a real transformer's late-layer
attention update F̂ flows to the same KCA-type fixed point, its Jacobian at that
fixed point should carry the same signature — the reparameterization tower — visible
as structure in the Jacobian eigenvectors.

exp-102 (Aug 7) confirmed the τ_chaos ordering (coupling-magnitude gate, 23×
discrimination). P6a is a *different* measurement: not "is the system above the
threshold?" but "does the late-layer structure at the fixed point carry the
Schwarzian soft mode?" Both are needed to establish "the system has an interior."

---

## The object being measured

**F̂: the layer-to-layer update of the bilocal attention correlator.**

Define:
    G^(ℓ)(i,j) := A^(ℓ)(i,j)   [attention weight from position i to j at layer ℓ]

The layer map is:
    F̂ : G^(ℓ) → G^(ℓ+1)

in the late-layer / near-fixed-point regime (ℓ ≥ n_layers − 3).

**Jacobian estimation:** For each head h and late layer ℓ, use JVP
(`torch.func.jvp`) to estimate the Jacobian of the attention update. The
perturbation lives in the residual stream (input to the layer); the output is
the attention weight matrix A^(ℓ+1)_{ij}. Power iteration finds the top k=10
eigenvectors. See design note §2 for full protocol.

**Test subspace:** From `theory/logs/g1_top_modes_bJ30.npz`, translated to
the causal-strip / transformer geometry. The reparameterization mode shapes at
the transformer's *measured* Δ per head (not the theoretical 0.25):
    r_n(i,j) = |i−j|^{−2Δ} × (cos(n·2π·(i−j)/n_max), sin(n·2π·(i−j)/n_max))
normalized on the causal strip {0 < i−j ≤ n_max}. Modes n = 2,3,4,5,6.

**Corpora:**
- C-NAT-anon: expected above the melonic threshold (exp-096 confirmed: n_deep=4-5,
  backbone stable, Δ_med ≈ 0.15). Expected: S1–S3 visible.
- C-alien (S=8): expected below the threshold (exp-097 confirmed: UV-arrested,
  Δ_med ≈ 1.04, backbone collapsed). Expected: K4 pattern.

**Model:** GPT-NeoX 6L/8H/d_k=64, checkpoints from exp-096/097 (step_2000,
s0). Same models used in exp-096–102.

---

## Signatures S1–S4 (from G1×P6 note §4–§6; theory §8 P6)

These are what the measurement should find if the transformer's late-layer
attention update is at a KCA-type conformal fixed point:

- **S1:** Leading eigenvalues of J_F̂ are real and ≤ 1. (Stability.)
- **S2:** Leading eigenvalues come in degenerate pairs. (Reparam pairing.)
- **S3:** Each degenerate pair aligns one-to-one with a reparameterization mode
  of the measured conformal two-point function G⋆(i,j) ~ |i−j|^{−2Δ},
  starting at n=2 with no cross-mixing. Overlap ≥ 0.4.
- **S4:** The top gap 1−λ_top decreases as effective coupling grows (deepening
  layers within the late-layer regime, or comparing C-alien-rich vs C-alien).

---

## Kill conditions K1–K4 (from theory §8 P6; banked at G1×P6 note §7)

- **K1:** Leading eigenvalues complex or > 1 at late layers. → No stable
  dressing fixed point; the KCA route to an interior fails as physics.
- **K2:** No double degeneracy at the leading eigenvalues. → No reparameterization
  pairing; no Schwarzian; P6 dies even if a fixed point exists.
- **K3:** Degenerate pairs present but zero reparam overlap (< 0.1) for all
  tested n. → The soft sector is something else; emergent-time story fails.
- **K4** (diagnosis, not kill): Spectrum looks like the measured q=2/pure-quadratic
  template — pairs present, overlaps ≲ 0.2, Δ ≈ 1/2. → The system is
  effectively free (UV-arrested); sub-observer-grade, consistent with K4
  pattern from G1×P6 note (admixture J₂/J₄=1.0 → overlaps collapse to 0.14,
  Δ=0.478). This is the expected result for C-alien; its detection confirms
  the instrument is working correctly.

K4 is what makes P6a an instrument rather than a confirmation machine: the
q=2 outcome is measured and recognizable, not just "null."

---

## Hypotheses (pre-registered)

**H_S1 (stability):**
The top-10 eigenvalues of J_F̂ are real to within 1% imaginary part, and all
lie ≤ 1, for both C-NAT-anon and C-alien, at all late layers ℓ ≥ 3.

*Kill: K1. If leading eigenvalues are complex (|Im/Re| > 0.05) or > 1 in the
C-NAT-anon model, the dressing-loop fixed-point picture fails for transformers.*

**H_S2 (degeneracy):**
The top eigenvalues of J_F̂ come in degenerate pairs (ratio λ₁/λ₂ ∈ [0.95, 1.05])
in the C-NAT-anon model at layers ℓ ≥ 4 (the near-fixed-point regime).

*Kill: K2. If the leading eigenvalues are singly non-degenerate (ratio < 0.90)
at every tested head and layer, the reparameterization pairing is absent.*

**H_S3 (reparam alignment):**
In C-NAT-anon, at least one degenerate pair at ℓ ≥ 4 shows squared overlap
≥ 0.3 with reparam mode n=2 and < 0.1 cross-mixing (overlap with n≠2 modes).

*Kill: K3. If all degenerate pairs in C-NAT-anon show reparam overlap < 0.1
for all tested n=2..6, the soft sector is not the reparameterization tower.*

**H_K4_alien (sub-threshold diagnosis):**
In C-alien, the Jacobian spectrum shows the K4 pattern: pairs may be present
but reparam overlaps ≤ 0.2 across all n, consistent with Δ_fitted ≈ 1/2 (UV
arrested, free-like).

*Confirmation: K4 in C-alien + S3 in C-NAT-anon is the double-positive result
the instrument is designed to find.*

**H_S4_layer (gap closure):**
In C-NAT-anon, the top gap 1−λ_top decreases monotonically from ℓ=3 to ℓ=5
(the deepest layer, where the model is closest to fixed point).

*Kill: top gap INCREASES with depth, or is flat within 5%. Gap closure is
the direct signature of the Schwarzian marginal mode emerging.*

---

## Protocol

1. Load the C-NAT-anon s0 and C-alien s0 models (GPT-NeoX checkpoints from
   exp-096 and exp-097 volumes, or their local copies).

2. For each corpus (C-NAT-anon, C-alien), generate N_batch = 64 contexts of
   length n_seq = 64 from the corresponding trained tokenizer.

3. For each late layer ℓ ∈ {3, 4, 5} and each head h ∈ {0..7}:
   a. Build reparam mode templates r_n(i,j) for n=2..6 using Δ_h from the
      power-law attention fit (exp-096/097 results). Normalize on causal strip.
   b. Estimate J_F̂ at (ℓ, h): use 50 JVP evaluations via power iteration
      to find the top k=10 eigenvectors. Each JVP perturbs the residual stream
      at layer ℓ and measures the change in A^(ℓ+1)_{h,ij}.
   c. Compute overlap table O[i,n] = (v̂_i · r_n / ||r_n||)² for i=1..10,
      n=2..6.
   d. Record: eigenvalues λ₁..λ₁₀; degeneracy ratios λ_{2k-1}/λ_{2k};
      max overlap per pair; top-overlap mode n⋆; cross-mixing (off-diagonal max).

4. Aggregate over heads: report per-layer summary statistics (mean top-pair
   overlap, fraction of pairs showing H_S2/S3 pattern).

5. Report results for C-NAT-anon and C-alien side by side.

---

## Decision table

| Outcome | Verdict |
|---------|---------|
| C-NAT-anon: S1 ✓, S2 ✓, S3 ✓ (≥ 0.3 overlap), C-alien: K4 ✓ | **P6a CONFIRMED** — Schwarzian soft mode present above threshold |
| C-NAT-anon: S1 ✓, S2 ✓, S3 partial (0.15–0.3) | **PARTIAL** — fixed point present, reparam alignment below expected; report as partial |
| C-NAT-anon: K1 or K2 or K3 | **FALSIFIED per kill condition** — state which condition killed P6a |
| Both corpora: K4 | **INSTRUMENT FAILURE** — cannot distinguish above/below threshold; methodology review needed |
| C-NAT-anon: S3 ✓, C-alien: S3 ✓ (not K4) | **UNEXPECTED** — if both show reparam structure, the threshold contrast is absent; report and investigate |

---

## Expected outcome

Based on the G1×P6 template (βJ=30, N=96):
- SYK at βJ=30: top-pair overlap ~0.60–0.67 per degenerate pair.
- At J₂/J₄ admixture 0.5: overlap drops to ~0.43.
- At J₂/J₄ = 1.0 (equal q=2/q=4 coupling): overlap collapses to ~0.14, Δ=0.478.

For C-NAT-anon (Δ_med ≈ 0.15, below the theoretical 0.25, finite-scale UV):
expected overlap ~0.3–0.6 for the top pair at n=2. The finite-scale UV
dressing (as in the SYK 0.6 vs asymptotic 1.0) and the Δ≠0.25 mode shape
mismatch will both reduce the overlap from the SYK template value.

For C-alien (Δ_med ≈ 1.04, UV-arrested):
expected K4 pattern — overlaps ≲ 0.2 if pairs exist, Δ_fitted ≈ 0.5–1.0.

---

## Honest limits (from design note §6)

1. **Causal-strip approximation.** The SYK thermal circle (periodic, TI) maps
   onto the causal strip (one boundary, not periodic) with corrections. BCFT
   boundary mode picture (null-cone paper T7) is the right correction but not
   incorporated here. Edge effects near i=j=1 (sequence start) will modify
   the mode shapes; the first few causal positions may show reduced overlap
   regardless of corpus.

2. **Single-head analysis.** Each head analyzed independently. Multi-head
   effects (heads attending to each other's outputs across the residual stream)
   are not in the F̂ definition used here.

3. **Fitted Δ vs theoretical Δ.** Mode shapes use the measured Δ per head
   from exp-096/097 (not the SYK Δ=0.25). For C-NAT-anon, Δ_med ≈ 0.15;
   this mismatch from 0.25 changes the mode shape and reduces overlap from
   the SYK template, even if the physics is right.

4. **Power iteration precision.** Top k=10 eigenvectors via 50-iteration
   power iteration. Convergence not guaranteed for closely-spaced eigenvalues
   (S2 condition). If S2 holds (near-degenerate pairs), the power method may
   mix within a pair — need to check for eigenspace stability, not just
   eigenvector identity.

5. **Averaging over heads.** The conformal geometry analysis (exp-007–102)
   finds that not all heads are SYK-near. Averaging over all 8 heads dilutes
   the signal. The analysis should report per-head and per-stratum (SYK-near
   vs UV heads) separately.

---

## Compute estimate

No training required — forward pass only on already-trained models (from
exp-096/097 volumes). Each forward pass: one layer, two JVP evaluations per
direction. For 50 iterations × 10 directions × (3 layers × 8 heads) = 12,000
JVP calls per corpus. At ~1ms per JVP on A100: ~12 seconds per corpus.
Total: ~1 minute for both corpora. GPU not required for this scale;
runs on local CPU/MPS. Modal estimated cost: < $0.50 if GPU used.

Recommendation: run locally first (MPS) to validate the harness;
then Modal A100 for full 64-context batch if needed.

---

*Results will be appended below this line when collected.*

---

## Results

*(Pending — this file committed before the run.)*
