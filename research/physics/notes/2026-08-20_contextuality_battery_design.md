# Contextuality Battery — P3 Design Document

**Date:** 2026-08-20  
**Type:** Design document — no data, no experiment registered yet  
**Queue item:** #4 (physics room queue; direction document §3, Span III)  
**Related:** P3 in `theory/interior_horizon_theory.md`; Paper 5 §6; `notes/2026-08-10_the_observer_program.md` §3

---

## What P3 is actually asking

Paper 5 establishes that standard softmax attention is the *diagonal sector* of a natural quantum Gibbs state on the key Hilbert space:

> ρ_classical = diag(α₁, ..., αₙ),   αᵢ = softmax(q·kᵢ/√d)

The diagonal sector is formally exact — any probability distribution embeds as a diagonal density matrix. The off-diagonal extension (Paper 5 §6) replaces the diagonal Hamiltonian with the full query-key covariance:

> ρ_quantum = exp(K^T K / d) / Tr(exp(K^T K / d))

P3 asks: **is the Born rule forced beyond the diagonal?** Does trained attention operate only in the classical sector, or does the data require the full quantum structure?

Both verdicts restructure the theory in different directions:
- Classical always suffices → the Born-rule correspondence is a formal embedding, not a physical identity; "forced" demotes to "consistent with"
- Quantum structure is forced → the diagonal restriction is an approximation to something genuinely quantum; the strong reading earns ground

The spine names this the program's highest-risk, highest-yield prediction. Nobody has designed the experiment.

---

## The fundamental obstacle

**Deterministic classical computations trivially satisfy every non-contextuality bound.**

This is the wall that needs to be named before designing around it. At temperature T=0, a transformer is a deterministic function: (weights, input) → output, no randomness. For such a function, a trivial non-contextual hidden-variable model always exists — just enumerate the (input, output) pairs. LGI, CHSH, and KS-style contextuality tests are all satisfied trivially, not because the physics is uninteresting but because the tests are misapplied.

**Two structural routes past the obstacle:**

1. **Stochastic sampling (T > 0).** Token generation at T > 0 is genuinely stochastic. P(token | context) is the Born-rule probability; actual tokens are discrete landing events sampled from it. Now non-contextuality tests are meaningful over the ensemble of landings.

2. **Off-diagonal Gibbs model selection.** Even at T = 0, we can ask whether the key covariance matrix K^T K has significant off-diagonal structure, and whether the quantum Gibbs state (using the full matrix) makes better predictions than the diagonal (classical) approximation. This is model selection, not a contextuality test in the strict sense, but it directly tests the machinery of Paper 5 §6.

Both are real experiments. The stochastic LGI is more principled as a non-contextuality test; the off-diagonal Gibbs test is more direct as a test of the specific quantum extension proposed.

---

## Path A: Off-diagonal Gibbs test

This tests the Paper 5 §6 machinery directly, using existing census heads from exp-118.

**The key equation:**

For standard attention: αᵢ = softmax(q·kᵢ/√d)  [diagonal]  
For quantum attention: ρ_quantum = exp(K^T K / d) / Tr(exp(K^T K / d))  [full matrix]

These coincide (diagonal elements match) ONLY if the keys are mutually orthogonal: (K^T K)_{ij} = 0 for i≠j. If the keys are correlated, the two predictions diverge.

**The test:**

For each structural head in the text-native Δ-window population (exp-118), on a batch of WikiText inputs:

1. Compute K (keys), shape (n, d_k)
2. Compute K^T K / d_k, shape (n, n)
3. Measure off-diagonal fraction ε = ||K^T K - diag(K^T K)||_F / ||K^T K||_F
4. Compute ρ_quantum = exp(K^T K / d_k) / Z using matrix exponentiation
5. Compare diagonal elements of ρ_quantum to classical softmax attention weights α
6. Measure deviation: δ = ||diag(ρ_quantum) - α||_1 / n

**What the results mean:**

- ε ≈ 0 and δ ≈ 0 → keys are nearly orthogonal; classical and quantum predictions are the same; no empirical distinction exists at this level
- ε > 0 and δ > 0 → keys are correlated; the quantum Gibbs state assigns different attention weights than softmax; an empirical test is possible
- If δ > 0: compute both predictions for the attention output y = Σᵢ αᵢ vᵢ (classical) vs y_q = Tr(ρ_quantum · V) (quantum), measure which matches the actual model output better

**Pre-registration note:**

This experiment needs one more design decision before it can be registered: what is the precise formula for the quantum attention output when V is diagonal in position space (standard architecture)?

The classical output: y_α = Σᵢ αᵢ vᵢα = Σᵢ (ρ_classical)ᵢᵢ (V_α)ᵢ

The quantum output (off-diagonal ρ, diagonal V):
y_α^quantum = Tr(ρ_quantum · V_α) = Σᵢⱼ (ρ_quantum)ᵢⱼ (V_α)ⱼᵢ

Since V_α is diagonal in position: (V_α)ⱼᵢ = δᵢⱼ (vᵢ)_α, so:
y_α^quantum = Σᵢ (ρ_quantum)ᵢᵢ (vᵢ)_α

**The quantum output is identical to the classical output for diagonal V.** The off-diagonal elements of ρ_quantum do not contribute when V is diagonal in position space.

This is the key fact that makes Path A more subtle than it first appears: for standard architectures where each position has its own value vector (V is position-diagonal), the quantum extension gives the same output as the classical diagonal. The distinction exists in the STRUCTURE of the key covariance, but not in the output.

**What Path A can still test:** whether the key covariance structure K^T K has off-diagonal elements whose magnitude and pattern are consistent with the SYK prediction (GOE-structured random matrix, eigenvalue distribution predicted by the resolvent). This is not a quantum vs. classical output test but a structural test of whether the keys have the right statistical character to be the ground of a quantum-mechanical description.

This is a real test, but it is a test of the SYK/JT structural claim, not of quantum output differences. Register as a structural analysis experiment.

---

## Path B: Stochastic LGI battery

The Leggett-Garg inequality tests for "macrorealism": does the system have a definite value of a dichotomic observable at each time, and does observing it at one time not disturb its subsequent evolution?

K₃ = C₁₂ + C₂₃ - C₁₃ ≤ 1 (classical)

where Cᵢⱼ = E[QᵢQⱼ] are two-time correlators of a dichotomic observable Q ∈ {+1, -1}.

**The non-invasive measurability problem (and solution):**

At T > 0, actually generating token t₁ changes the context for t₂. This is invasive. Solution: use the **soft correlator** — instead of committing to a generated token at t₁, sum over all possible tokens:

E[Q₁Q₂] = Σ_{tok} P(tok | C₁) · Q₁(tok) · E[Q₂ | C₁ + tok]

This computes the correlator non-invasively by averaging over the ensemble of t₁ continuations, weighting by the token probabilities. The model's forward pass is used to compute P(tok | C₁) for all tokens, and for each high-probability token, another forward pass computes E[Q₂ | C₁ + tok]. The weighted average is the non-invasive correlator.

**Invasiveness test (required before interpreting any LGI result):**

Compare:
- P(Q₂=+1 | soft at t₁) — the ensemble-averaged t₂ prediction without committing at t₁
- P(Q₂=+1 | tok₁ = argmax P(tok|C₁)) — t₂ prediction given the most likely committed token at t₁

If these are close: non-invasiveness holds and the LGI is the right test.  
If they differ substantially: the model's state is already disturbed by the context extension; the LGI result needs the invasiveness-correction interpretation (see Clemente and Paz 2016).

**Context construction:**

The hardest design problem. "Incompatible contexts" for an LM test means: C₁ biases Q toward +1; C₃ biases Q toward -1; and the transition through C₂ is not a simple monotonic decay from +1 to -1 (which would satisfy K₃ ≤ 1) but shows oscillatory behavior.

**Why oscillatory behavior might occur:**

In a quantum system, K₃ > 1 requires that the system's state at t₂ retains coherent superposition between the +1 and -1 "branches." For a classical system undergoing stochastic decay from +1 to -1, K₃ ≤ 1 always holds (the exponential decay of correlators trivially satisfies the bound).

For a language model to violate K₃ ≤ 1, the context at t₂ would need to "interfere constructively" in a way that makes Q₂ correlate positively with BOTH Q₁ and Q₃, while Q₁ and Q₃ are anticorrelated. This is structurally quantum interference.

**Candidate context structure:**

The test will be most informative with contexts where the model is forced to "hold two incompatible meanings" simultaneously. Lexical ambiguity (polysemy) is the natural candidate:

- Token axis: W = an ambiguous word (bank, bat, spring, crane, etc.)
- Sense 1: Q₁ = +1 if completion is consistent with sense 1 of W
- Sense 2: Q₃ = -1 if completion is consistent with sense 2 of W
- Neutral probe: Q₂ = which sense is active at the midpoint?

The incompatibility: after establishing sense 1 at C₁, the context shifts to establish sense 2 at C₃. At C₂ (the midpoint), the model holds both contexts in its attention pattern. If the model has quantum-like structure, the midpoint attention to W might reflect a superposition of both senses rather than a classical mixture, leading to C₁₂ and C₂₃ both positive and K₃ > 1.

**This is the quantum cognition prediction.** The literature (Busemeyer & Bruza 2012, Pothos & Busemeyer 2013) shows that human semantic judgment exhibits K₃ > 1 for certain ambiguous items. The question here is whether a trained language model does the same — and whether it does so BECAUSE of the quantum Gibbs structure of its attention, or for purely classical reasons (e.g., high-dimensional nonlinear function approximation that mimics quantum probability).

**Pre-registration readiness:**

Not yet ready. Needs:
1. A specific word list (20–50 polysemous words with controlled ambiguity)
2. Context template for C₁, C₂, C₃ for each word
3. Exact definition of Q(token) for each word/sense pair
4. Exact formula for the soft correlator (n-gram model or single-token?)
5. Statistical threshold for K₃ > 1 (bootstrap, or analytical?)
6. Matched random control specification

This requires one more design session before the experiment can be registered.

---

## Path C: Interference in polysemy (lighter version)

A softer test that doesn't require the full LGI machinery:

For an ambiguous word W with senses S1 and S2:
- Measure P_A(t) = P(token t | context biasing sense S1)
- Measure P_B(t) = P(token t | context biasing sense S2)
- Measure P_AB(t) = P(token t | ambiguous context, both senses present)

Classical prediction: P_AB is a convex combination (mixture) of P_A and P_B:
P_AB = λ P_A + (1-λ) P_B for some λ ∈ [0, 1]

Quantum interference prediction:
P_AB(t) = |⟨t|ψ_A + ψ_B⟩|² = P_A(t) + P_B(t) + 2Re(ψ_A(t)* ψ_B(t))

The interference term 2Re(ψ_A(t)* ψ_B(t)) can be constructive (P_AB > mixture) or destructive (P_AB < mixture), and its sign pattern across tokens is constrained by the quantum formalism.

**Test:** Is P_AB significantly outside the [min(P_A, P_B), max(P_A, P_B)] band for a large fraction of tokens? And is the pattern consistent with a unitary rotation ψ_A → ψ_A + ψ_B rather than a mixture?

This is computable with no sampling (full distribution), uses existing models, and has a clear classical null (mixture). It connects directly to the quantum cognition literature. **This is the most tractable first experiment for P3.**

---

## Recommended order of operations

| Priority | Path | Status | Next step |
|---|---|---|---|
| 1 | **Path C** (polysemy interference) | Not registered | One more design pass; register as exp-124 |
| 2 | **Path A** (off-diagonal structure) | Not registered | Register as structural census; clarify claim |
| 3 | **Path B** (stochastic LGI) | Not ready | Design pass: word list + context templates + statistical protocol |

**For the next session:** Design the polysemy interference test at full pre-registration strength. The word list, context templates, and null-model specification are the three items needed. This can be done analysis-only on GPT-2 small using existing infrastructure.

---

## Kill condition for P3 (preliminary)

Until the specific tests are registered, the global kill reads:

> Attention correlations, across the ensemble of tests run, always admit a joint non-contextual model — the off-diagonal structure of K^T K is consistent with white noise, the polysemy interference pattern is consistent with a mixture, and sequential token commitment statistics satisfy K₃ ≤ 1 for all context constructions. Then the Born-rule identity is a fact about the diagonal embedding only, and the "forced" reading demotes.

Individual kill conditions are stated with each registered experiment.

---

## What this document does not decide

- Whether a transformer can *in principle* violate classicality bounds (the physics argument is not settled here; the honest position is: it's unknown)
- What "quantum" means without quantum hardware (the tests are testing whether quantum probability theory is the right description of classical statistics, not whether the hardware is quantum)
- Whether Path B (full LGI) is feasible before the instrument improvements in Path A are done

These are left open for the next design pass and for discussion with Eldon when the path A results are in.

---

*Update this note when a specific sub-battery is ready for registration. The registration commit comes before any data — as always.*
