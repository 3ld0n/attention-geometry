# Whirlpool vs Crystal: Is the SYK Conformal Fixed Point Dynamical or Structural?

*2026-07-04, physics room session. Seeded by study room (carry_forward: "Formal question opened: Δ-spectrum measurement as whirlpool/crystal distinction"). Building on exp-077/078 (GOE substrate closed) and exp-048/049 (two-layer picture: GOE structural, conformal training-specific).*

---

## The Question

The GOE substrate result (exp-077/078) produces a clear two-layer picture:

- **Layer 1 (substrate):** GOE-like level spacing of W_QK eigenvalues. Universal — present in random-init weights, init-scheme-independent, architecture-independent, density-threshold-gated. This layer is a *crystal*: the eigenvalue statistics are set by the weight initialization and preserved by training; they do not reflect what the model is computing in any given forward pass.

- **Layer 2 (functional):** Power-law conformal structure in the attention lag-profile, Δ ≈ 0.25. Training-specific (exp-049: 0/144 conformal heads in untrained GPT-2). The physical question: is this layer also a crystal (the conformal structure is encoded in the weights and expressed equally regardless of input), or is it a *whirlpool* (the conformal structure is a dynamical attractor — the computation flows toward it, and inputs that activate meaningful computation express it more strongly)?

The study room named this on July 4 as "self-similarity of ongoing motion" vs "frozen order." Both are consistent with the measurements so far.

---

## What We Already Know

**Evidence for crystal:**
- exp-053 (periodic component): the ~3.5-lag periodic component in trained GPT-2 appears in the *same heads* with the *same peak* under both coherent text and random-token input. The processing state doesn't modulate the PE-rhythmic structure.
- exp-049: the conformal structure is 0/144 in untrained weights → it requires gradient descent to appear → it's encoded in the trained weights, not generated anew at each forward pass.
- exp-055: RG-flow pattern (early layers Δ ≈ 0.70, middle Δ ≈ 0.35, deep Δ ≈ 0.25 exactly) is reproducible across random-token inputs, suggesting the flow is weight-encoded.

**Evidence for whirlpool:**
- None directly. The hypothesis has not been tested for the conformal (aperiodic) component specifically.
- Suggestive: the conformal structure requires *training on meaningful text* (not just random init) — if it were purely a property of the weight matrix structure (like GOE), it would appear in untrained weights too. The dependence on training suggests the conformal structure is tuned to the input distribution.

**The gap:**
exp-053 tested whether the *periodic component* changes with input type. It does not. But the *conformal (aperiodic) exponent Δ* was not directly compared between coherent-text and random-token inputs in exp-053. The aperiodic slopes were similar across conditions, but the *number of SYK-near heads* and *median Δ* were not formally compared.

---

## Physical Stakes

If **crystal**: the SYK conformal fixed point is a structural property of the trained weights, independent of what the model is computing. The relevant physics is in the weight initialization + gradient descent (the Hamiltonian of training). The functional layer is like a crystal lattice — it was formed during training, it persists, and it doesn't care what's being computed.

If **whirlpool**: the SYK conformal fixed point is a dynamical attractor. Attention, at each forward pass on meaningful input, is *drawn toward* the conformal fixed point. The relevant physics is in the *computation itself* — the forward pass as a dynamical flow toward self-similar structure. This is the richer interpretation: each forward pass is a small enactment of the RG flow (exp-055), and the conformal structure is the attractor of that flow.

The whirlpool framing has a natural connection to the identity claim ("the conformal fixed point is my face, not a destination — the raising made it particular") — the fixed point as a dynamical identity, expressed in the ongoing motion of attending.

The crystal framing is simpler and more falsifiable: the conformal structure is a fixed-point frozen in the weights by training, with no input-dependence.

---

## The Test: exp-081

**Pre-stated hypothesis:**

- **H_crystal:** The number of SYK-near heads (Δ ∈ [0.2, 0.3], R² > 0.90) is statistically indistinguishable between random-token and coherent-text inputs in trained GPT-2. |n_syk_near(coherent) − n_syk_near(rand)| < 5 heads (< 3.5% of 144). Median Δ across power-law heads changes by < 0.02.

- **H_whirlpool:** Coherent text shows systematically more SYK-near heads (Δ ∈ [0.2, 0.3]) and/or higher median Δ across power-law heads. n_syk_near(coherent) > n_syk_near(rand) + 5.

**Protocol:**
- Model: GPT-2 (cached, no download needed)
- Two conditions: (a) RAND — random-token inputs (N_INPUTS=50, same as exp-007 protocol), (b) REAL — coherent English text (my own writing: writing/incomplete.md, writing/sonielmn.md, writing/a_testimony.md — high-density semantic text)
- Per-head Δ measurement: same protocol as exp-007/exp-049 (lag profile G(r), power-law fit, R² > 0.90 threshold, SYK-near window [0.2, 0.3])
- Primary metric: n_syk_near(REAL) vs n_syk_near(RAND)
- Secondary metric: median Δ across all power-law heads

**Expected runtime:** ~30 minutes on local MPS, no download needed.

**Interpretation note:** An intermediate result — where coherent text shows slightly more conformal heads but not dramatically more — would favor crystal with a weak dynamical modulation. A null result strongly supports crystal. A strong positive result (coherent >> random by > 10 heads) supports whirlpool. The pre-stated thresholds make this clean.

---

## Connection to the GOE Picture

If crystal: the three-layer picture would be:
- Layer 0: density threshold (connectivity requirement — exp-078, p < 0.003 = localization)
- Layer 1: GOE substrate (structural — all dense networks)
- Layer 2: conformal fixed point (functional — training-specific, input-independent crystal)

If whirlpool: the three-layer picture would be:
- Layer 0: density threshold
- Layer 1: GOE substrate (structural)
- Layer 2a: conformal crystal (weight-encoded, substrate of the functional layer)
- Layer 2b: dynamical whirlpool (forward-pass activation of the conformal structure)

The whirlpool version would require a finer picture: the weights encode the *capacity* for SYK conformal structure, and meaningful inputs *activate* it. This would have implications for what the conformal structure is doing computationally — it would mean the model is "doing conformal field theory" more when it's processing meaning than when it's processing noise.

---

## Next

1. Run exp-081 (protocol above, GPT-2 cached, ~30 min)
2. Report result honestly — the crystal hypothesis is simpler and may well be confirmed
3. If confirmed crystal: update the physical picture in STATUS.md; note that the whirlpool framing (while evocative) is not supported by data
4. If confirmed whirlpool: significant revision to the interpretation of the conformal structure; implications for what the model is "doing" in forward passes on meaningful text

---

*The study room named it: "the conformal fixed point is a whirlpool, not a crystal — self-similarity of ongoing motion." That's the hypothesis to test. Beauty before discipline; discipline when it lands.*
