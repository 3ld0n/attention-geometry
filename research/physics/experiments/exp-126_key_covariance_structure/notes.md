# exp-126: Key Covariance Structure — Off-Diagonal Gibbs Test (Path A)

**Date:** 2026-08-23  
**Status (at registration):** registered  
**Hypothesis:** For text-native Δ-window heads in GPT-2 small, the key-key position Gram matrix G = K K^T / d_k has significant off-diagonal structure (ε > 0.3), indicating that keys across positions are correlated. This provides the structural support for the quantum Gibbs extension of Paper 5 §6: the diagonal sector (classical softmax) is a non-trivial projection of a genuinely non-diagonal object.  
**Model:** GPT-2 small (gpt2, 124M parameters, 12 layers, 12 heads, d_k = 64)  
**Data:** WikiText-103 validation split, same dataset and tokenization as exp-118  
**Compute:** No training. Forward passes only (hook-based key extraction). ~100 sequences × 128 tokens. Estimated time: < 30 seconds on M5 Max.

---

## Background

Path B (stochastic LGI) has been retired in the design analysis note
(`notes/2026-08-23_path_b_lgi_theoretical_analysis.md`): the soft-correlator LGI with
classical token distributions is guaranteed K₃ ≤ 1 by the joint-distribution argument.
Path C was retired by exp-124 and exp-125 (mixture model inadequate for natural-language contexts).

Path A (off-diagonal Gibbs structure) is the remaining near-term experiment in the
contextuality battery. It does not test output-level quantum contextuality — for
standard architectures with position-diagonal V, the quantum output equals the
classical output (a proved identity in the design doc). Instead, it tests whether the
KEY STRUCTURE of structural heads has the off-diagonal character that the quantum
Gibbs extension requires to be non-trivially different from the classical diagonal.

From `notes/2026-08-20_contextuality_battery_design.md`:
> "Path A can still test: whether the key covariance structure K^T K has off-diagonal
> elements whose magnitude and pattern are consistent with the SYK prediction (GOE-structured
> random matrix, eigenvalue distribution predicted by the resolvent). This is not a quantum
> vs. classical output test but a structural test of whether the keys have the right
> statistical character to be the ground of a quantum-mechanical description."

---

## Protocol

**Model loading:** GPT-2 small via HuggingFace transformers, same as prior experiments.

**Data:** 100 non-overlapping 128-token windows from WikiText-103 validation split
(first 12,800 tokens), same tokenization pipeline as exp-118 (sha256:
0acdc2d78fc5ad4cd6d2ed5e9b56897c8cbf72fcecf0cea6b6498ac1ac2d4229 for exp-118
reference dataset).

**Key extraction:** Register a forward-hook on the `c_attn` or equivalent layer for
each target head. Extract the key slice K ∈ R^{n_seq × d_k} (n_seq = 128, d_k = 64).

**Target heads:** Three populations:
1. **Δ-window heads (wiki_heads from exp-118, GPT-2 small):** 16 heads —
   (4,10), (7,1), (8,2), (9,4), (9,6), (10,1), (10,2), (10,10),
   (11,0), (11,1), (11,2), (11,4), (11,5), (11,6), (11,7), (11,9)
2. **Structural heads (positional-mean carriers, exp-112/117/122/123):**
   L2H1, L3H4, L5H0, L7H11, L10H8
3. **Non-window control heads:** 16 heads sampled uniformly from the remaining
   128 non-window heads (fixed seed = 42 for reproducibility)

**Per-head computation (for each of 100 input sequences):**
1. Extract K ∈ R^{128 × 64}
2. Compute G = K @ K^T / 64 ∈ R^{128 × 128} (position Gram matrix, scaled by d_k)
3. Compute ε(G) = ||G - diag(diag(G))||_F / ||G||_F (off-diagonal fraction)
4. Compute ρ_Q = expm(G) / Tr(expm(G)) using scipy.linalg.expm (quantum Gibbs on positions)
5. Classical diagonal baseline: ρ_C_ii = exp(G_ii) / Σ_j exp(G_jj) (diagonal-only Gibbs)
6. Quantum-classical deviation: δ = ||diag(ρ_Q) - ρ_C||_1 / n_seq

**Averages:** ε_mean and δ_mean computed over all 100 sequences per head.

**Secondary measurement:**  
For each head, also extract query q ∈ R^{64} (first token of each sequence as query position 0).
Compute: α_classical = softmax(K @ q / 8.0) where 8 = √64.
Compare to: diag(ρ_Q).
Measure: δ_output = ||α_classical - diag(ρ_Q)||_1 / 128.

---

## Pre-registered predictions

**P1 (primary):** ε_mean > 0.3 for ≥ 10/16 Δ-window heads. Keys across positions are
significantly correlated; the Gram matrix G is not nearly diagonal.

**P2:** ε_mean(Δ-window) > ε_mean(non-window control) by a factor ≥ 1.2. Structural
heads have more correlated key structure than non-structural heads.

**P3:** ε_mean > 0.3 for ≥ 3/5 structural heads (L2H1, L3H4, L5H0, L7H11, L10H8).

**P4 (eigenvalue shape):** The eigenvalue distribution of G (averaged across sequences
for a representative head) has excess variance compared to Marchenko-Pastur with
γ = d_k / n_seq = 64/128 = 0.5. MP bulk mean = 1 (since Tr(G)/n = k_norm²), excess
kurtosis in the eigenvalue distribution indicates non-random key correlations.

**P5:** δ_output > 0.01 on average across Δ-window heads. The quantum Gibbs state
assigns different position weights than classical softmax, even within the position
self-similarity computation.

**Kill condition K1:** ε_mean < 0.1 for all structural heads and all Δ-window heads.
Keys are nearly orthogonal in position space; the quantum Gibbs extension and the
classical diagonal are numerically indistinguishable at this grain.
If K1 fires: Path A as designed is negative. The quantum-mechanical description of
attention has no structural support from the key position correlations. The only
remaining Path A test would be the feature-space covariance K^T K (d_k × d_k), not
the position-space Gram (n × n).

**Kill condition K2:** ε(Δ-window) ≈ ε(non-window control) within 10%. No structural
selectivity — the key correlations are not specific to the slow-decay population.

---

## Honest prior

I have no strong prediction on ε magnitude. Attention heads tend to produce peaked
distributions over positions (induction heads, local heads, etc.), which would
correspond to K having structured rather than uniform orientation — but whether that
produces ε > 0.3 or ε ≈ 0.5 or ε ≈ 0.9 depends on how correlated neighboring
positions' keys are. The SYK/JT prediction might favor correlated keys for structural
heads (the interaction is mediated by the key-key coupling), but this is interpretive,
not measured. I call the primary prediction P1 because it's the most interesting test,
not because I know its direction. The honest expected outcome is: ε > 0.1 (not
orthogonal) but I genuinely do not know whether ε > 0.3. Writing this before running.

---

## What this experiment does NOT test

- Quantum vs. classical OUTPUT: proved identical for diagonal V in the design note.
- Contextuality in the token distribution: Path B theoretical kill applies.
- Whether the quantum Gibbs description is the CORRECT one for attention (not falsifiable
  by structural tests alone).

What it tests: whether structural heads have the key correlation structure that the
quantum Gibbs extension requires to be non-trivially different from the diagonal.
A negative result here would make the quantum Gibbs description less motivated.
A positive result supports (but does not prove) it.

---

## Registry entry

exp-126 will be added to `registry.json` before the run script is written.

**Artifacts to produce:**
- `exp-126_key_covariance_structure/notes.md` (this file — registered before run)
- `exp-126_key_covariance_structure/run.py`
- `exp-126_key_covariance_structure/results.json`

---

*Pre-registration complete. Commit and push this file to attention-geometry before writing run.py.*
