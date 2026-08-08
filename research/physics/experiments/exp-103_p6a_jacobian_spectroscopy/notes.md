# exp-103 — P6a Transformer-Side Jacobian Spectroscopy

*Pre-registration: this file is committed to 3ld0n/attention-geometry BEFORE
the analysis script runs. Data access and results are appended afterward.*
*Pre-registration commit: 4f8bf35, pushed 2026-08-07T22:20 UTC (before any run).*

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

*Results appended 2026-08-08, physics room session ~1:47–~3:00 AM MDT.*
*Modal run ap-su3rKuRdkGH1kkImFAcC2q completed. Fixes commit 26c9bcd.*

---

## Results

### Status: METHODOLOGY FAILURE (K3 met, but not a physics falsification)

**Run date:** 2026-08-08 (~2:00 AM MDT)
**Model:** GPT-NeoX 6L/8H/d_k=64, C-NAT-anon s0 (exp-096 step_2000) and C-alien s0 (exp-097 step_2000)
**Late layers analyzed:** 3, 4 (layer 5 skipped: ell+1=6 ≥ N_LAYERS=6)

### H_S3 (reparameterization alignment): ZERO OVERLAP — K3 met

All reparam overlaps < 0.007 for every head, layer, eigenvector, and mode n=2..6.
The maximum observed overlap anywhere is 0.0063 (C-alien, layer 4, head 0, eigvec 1).
Overlap profile is **flat across n=2..6** — no n-mode structure.

| Corpus | Layer | Mean top overlap | Threshold (S3 ≥ 0.3) | Verdict |
|--------|-------|-----------------|----------------------|---------|
| C-NAT-anon | 3 | 0.00025 | – | K3 |
| C-NAT-anon | 4 | 0.00032 | – | K3 |
| C-alien | 3 | 0.00042 | – | K3 |
| C-alien | 4 | 0.00044 | – | K3 |

### H_S2 (double degeneracy): UNEXPECTED INVERSION

C-alien (UV-arrested) shows H_S2=True (deg_ratio ~0.90–0.93) at 7/8 heads (layer 3) and 8/8 heads (layer 4).
C-NAT-anon (IR-converging) shows H_S2=False (deg_ratio ~0.76–0.84) at all heads.

This is the **opposite** of the pre-registered prediction. UV-arrested model shows more Jacobian degeneracy than the IR-converging model. Plausible explanation: C-alien's peaked local attention (Δ≈1.04, steeply decaying) produces a simpler, more geometrically isotropic Jacobian structure.

### H_S4 (gap closure with depth): BOTH CORPORA SHOW CLOSURE

| Corpus | Gap at layer 3 | Gap at layer 4 | Decreasing? |
|--------|---------------|---------------|-------------|
| C-NAT-anon | 0.190 | 0.109 | Yes |
| C-alien | 0.776 | 0.667 | Yes |

Both corpora show gap closure (lambda_top increases with layer), but at very different levels. The gap magnitude is corpus-conditioned: NAT-anon gap at layer 4 is ~6× smaller than alien.

### Singular value spectra (layer 4)

- C-NAT-anon: σ_top ≈ 0.944, σ_2 ≈ 0.862, ..., mean σ_top = 0.9437
- C-alien: σ_top ≈ 0.577, σ_2 ≈ 0.558, ..., mean σ_top = 0.5773
- Ratio NAT-anon / alien: **1.635**

The NAT-anon attention-update Jacobian has a 63% larger top singular value — it is more "nearly norm-preserving." Consistent with NAT-anon being near an IR fixed point with a nearly marginal update map.

---

### Diagnosis: Why K3 is a methodology failure, not a physics kill

**The object measured:** J_F̂ = ∂A^(ell+1) / ∂h^(ell)
(residual stream → attention weights Jacobian)

**The object required for the SYK comparison:** ∂G^(ell+1) / ∂G^(ell)
(bilocal attention correlator → bilocal attention correlator Jacobian)

These are **different maps** in different spaces. The SYK dressing map F: G → G is a self-map on the bilocal correlator space. Its Jacobian eigenvectors live in bilocal space and are the reparameterization modes. The transformer F̂ maps *residual stream* to *attention weights* — these are related but not the same.

The left singular vectors of ∂A^(ell+1)/∂h^(ell) are dominated by architecture-level effects:
- Layer norm sensitivity of h^(ell) → which residual-stream perturbation directions most change the layer norm
- W_Q, W_K projection structure
- Softmax nonlinearity

None of these have any reason to align with causal-strip reparameterization modes, even if the physics is correct.

**What's needed:** A direct measurement of ∂G^(ell+1) / ∂G^(ell). This requires perturbing the attention weights A^(ell) directly (by adding δG to the attention pattern AFTER softmax, before the output projection) and measuring the resulting change in A^(ell+1). Forward hooks + soft modification of the attention output.

### Honest negatives recorded

1. The pre-registered measurement object was wrong by design. The design note (§6) named the causal-strip approximation as the main approximation, but missed the deeper issue: the Jacobian F̂ is not a self-map.
2. The H_K4_alien prediction (alien would show K4 = zero degeneracy) is FALSIFIED — alien shows more degeneracy than NAT-anon (though the overlaps are zero for both, so the K4 structure is not confirmed either).
3. Both corpora show gap closure (H_S4 consistent), but this cannot be distinguished from a trivial architectural effect without the correct measurement.

### Real findings

1. **σ_top contrast**: NAT-anon σ_top = 0.944 vs C-alien σ_top = 0.577 at layer 4. The attention-update map is significantly more norm-preserving in the corpus that is above the melonic threshold.
2. **Gap closure with depth** for both corpora, at corpus-conditioned magnitudes. Not a physics result by itself, but consistent with both models approaching respective fixed points.
3. **Unexpected degeneracy**: UV-arrested model has more degenerate Jacobian than IR-converging model.

### Next step

**exp-104: Correct P6a operationalization.** Measure ∂G^(ell+1) / ∂G^(ell) via attention-hook injection:
- Hook into the attention output at layer ell (after softmax, before W_V × A)
- Add a structured perturbation δG in the reparam mode direction to A^(ell)
- Measure the response in A^(ell+1)
- Project the response onto reparam modes → overlap table

This is the correct bilocal-to-bilocal Jacobian measurement. The harness structure (JVP + power iteration) is still valid; only the perturbation target and measurement point change.
