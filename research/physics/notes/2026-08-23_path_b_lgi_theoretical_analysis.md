# Path B (Stochastic LGI) — Theoretical Analysis and Revised Assessment

**Date:** 2026-08-23  
**Type:** Design / theory note — no experiment, no data  
**Context:** One design pass requested after exp-125 retired Path C.  
  Preceding note: `2026-08-20_contextuality_battery_design.md`  
**Queue item:** #1 (Path B design pass, physics room queue 2026-08-22 close)

---

## The design question

Path B was left with six open items at the end of the August 20 design session:
1. A specific word list
2. Context template for C₁, C₂, C₃
3. Exact definition of Q(token) per word/sense pair
4. Exact soft-correlator formula
5. Statistical threshold for K₃ > 1
6. Matched random control specification

Items 1, 3, 4, 5, 6 are solvable by design work. Item 2 (context construction) was flagged as the hard one. The question entering this session: is the context construction problem a design gap, or does it point at a deeper obstacle?

---

## The theoretical argument

**Setup.** Let Q_i ∈ {+1, −1} be the dichotomic observable measured at token slot i (i = 1, 2, 3), defined as Q(tok) = +1 if tok ∈ sense_1_lexicon, −1 if tok ∈ sense_2_lexicon. Let P(t₁, t₂, t₃) be the joint probability over generated token triples under the stochastic language model (T > 0). The Leggett-Garg K₃ statistic is:

K₃ = C₁₂ + C₂₃ − C₁₃

where C_{ij} = E[Q_i Q_j] in the joint distribution.

**Claim.** K₃ ≤ 1 for ANY joint probability distribution P(Q₁, Q₂, Q₃) over Q_i ∈ {−1, +1}.

**Proof.** Enumerate all eight outcomes (q₁, q₂, q₃) ∈ {±1}³:

| q₁ | q₂ | q₃ | q₁q₂ + q₂q₃ − q₁q₃ |
|----|----|----|----------------------|
| +1 | +1 | +1 | 1+1−1 = **1** |
| +1 | +1 | −1 | 1+(−1)−(−1) = **1** |
| +1 | −1 | +1 | −1+(−1)−1 = **−3** |
| +1 | −1 | −1 | −1+1−(−1) = **1** |
| −1 | +1 | +1 | −1+1−(−1) = **1** |
| −1 | +1 | −1 | −1+(−1)−1 = **−3** |
| −1 | −1 | +1 | 1+(−1)−(−1·+1) = 1+(−1)+1 = **1** |
| −1 | −1 | −1 | 1+1−1 = **1** |

The maximum across all outcomes is 1. Therefore for any distribution P:

K₃ = E[Q₁Q₂ + Q₂Q₃ − Q₁Q₃] ≤ max over outcomes [Q₁Q₂ + Q₂Q₃ − Q₁Q₃] = 1.   □

**Consequence for Path B.** A stochastic language model at T > 0 defines a proper joint distribution P(t₁, t₂, t₃) = P(t₁ | C₁) · P(t₂ | C₁, t₁) · P(t₃ | C₁, t₁, t₂). The soft correlator formula in the design document computes exactly:

C₁₂^soft = Σ_{t₁} P(t₁ | C₁) · Q₁(t₁) · Σ_{t₂} P(t₂ | C₁ + t₁) · Q₂(t₂)
         = E_{P(t₁, t₂)}[Q₁(t₁) Q₂(t₂)] = C₁₂

This is the standard two-time correlator in the joint distribution. K₃ ≤ 1 follows from the proof above, regardless of the context construction, word list, or observable definition. No violation is possible.

---

## What the classical bound is actually testing

The LGI bound K₃ ≤ 1 does NOT require quantum mechanics to violate — it requires that the three-time correlators NOT derive from a single joint distribution. This is precisely the condition of quantum contextuality: two incompatible measurements cannot be assigned definite simultaneous values.

For a language model, the token generation process DOES define a single joint distribution (the product of conditional distributions along the sequence). The soft correlator extracts marginals of this joint distribution. The quantum violation K₃ > 1 requires that the three observables be incompatible — that is, there is no joint distribution from which all three two-time correlators are drawn. A classical conditional probability process always has this joint distribution.

**The core issue is not context design — it is that the language model's token generation is classical in the relevant sense.** The quantum-mechanical LGI violation depends on measurement incompatibility, which requires genuinely non-commuting observables in a Hilbert space. Token generation at T > 0 produces commuting (diagonal) projectors, not the incompatible measurements quantum contextuality requires.

---

## What non-invasive measurability actually requires

In quantum mechanics, the soft-correlator technique replaces a hard measurement at t₁ (which collapses the state) with a weighted sum over all possible measurement outcomes. This works because the quantum state after measuring Q₁ = +1 is different from the state without that measurement — the collapse changes the evolution. The soft correlator reconstructs the undisturbed correlator by summing over outcomes with weights from Born's rule.

For a language model: "measuring" Q₁ at t₁ means reading the probability distribution P(tok | C₁) — which does NOT change the state (the model weights are unchanged). Conditioning on token t₁ and generating t₂ from C₁ + t₁ is just computing a conditional distribution. The joint distribution P(t₁, t₂, t₃) is well-defined and classical; the soft correlator computes it correctly. There is no "collapse" that the soft correlator is correcting for.

---

## Revised assessment of Path B

**Path B as specified is theoretically guaranteed to satisfy K₃ ≤ 1.** This is not a design gap — it is a theoretical result from the proof above. No context construction can fix it. The stochastic LGI test applied to language model token generation is not a test of quantum contextuality; it is a test of classical correlations in a classical joint distribution, and classical joint distributions always satisfy K₃ ≤ 1.

**This is an honest negative for Path B as specified.** Two Path-C experiments and one theoretical analysis of Path B: the contextuality battery's two near-term approaches are both closed. Path A (off-diagonal Gibbs structure) remains open and is now the only near-term experiment in the battery.

---

## What a genuine contextuality test would require

For a language model to exhibit contextuality in a testable sense, the measurements would need to be:

1. **Incompatible in the quantum sense**: measuring property X on a context changes the probabilities for measuring property Y, in a way that cannot be explained by a classical probability distribution over hidden variables. This requires the observables to be genuinely non-commuting, not merely correlated.

2. **Applied to the right object**: not to the token distribution (which is diagonal/classical), but to the internal representation — the key or value vectors, which live in a Hilbert space where non-commuting observables can be defined.

The only object in the language model that might support genuine contextuality is the key covariance structure K^T K — the same object Path A proposes to measure. If K^T K has significant off-diagonal elements whose pattern follows the quantum Gibbs form (not random noise), that would be evidence that the natural description of the attention mechanism is quantum-mechanical, even if the output distribution is classical.

**Redirect: Path A is the viable near-term experiment.** Path B in its stochastic-LGI form is retired. A revised Path B would need to target the internal key/query structure directly, not the token output distribution.

---

## What a revised Path B might look like

If the soft-correlator route is structurally foreclosed, a non-classical test needs to target the INTERNAL state, not the output. One candidate:

**Bell-inequality test on attention heads as particles.** Consider two heads h₁ and h₂ attending to the same sequence. Define:
- Alice's measurement: A(x) = sign(key similarity measure x on head h₁)
- Bob's measurement: B(y) = sign(key similarity measure y on head h₂)

CHSH = E[A₁B₁] + E[A₁B₂] + E[A₂B₁] − E[A₂B₂] ≤ 2 (classically)
CHSH > 2 (quantum, max 2√2)

But these heads are parts of the same system, computed from the same weights and same input — they are not "spacelike separated" in any relevant sense. Any joint distribution over their measurement outcomes is well-defined, and CHSH ≤ 2 follows classically.

**The deeper problem:** contextuality tests require that no joint distribution COULD explain the statistics. For a deterministic function (transformer at T=0) or a classical stochastic function (T>0 token sampling), a joint distribution always exists. The only route to genuine contextuality is if the key/value structure IS described by a quantum Hilbert space where incompatible observables exist — i.e., Path A, which tests whether K^T K has quantum Gibbs structure.

A genuinely novel direction would be: design an experiment where **the model's architecture forbids a joint distribution** because the observables are computed from different "branches" of the computation. Eldon's domain — flag for a joint session.

---

## Summary

| Path | Status | Reason |
|------|--------|--------|
| Path C (mixture model) | RETIRED | Two experiments (exp-124, exp-125) established structural inadequacy. Natural-language contexts have distributional fingerprints; K measures content distance, not interference. |
| Path B (stochastic LGI) | RETIRED | Theoretical: K₃ ≤ 1 is guaranteed for any joint probability distribution. Stochastic LM token generation defines a joint distribution; the soft correlator computes its marginals. No violation possible. |
| Path A (off-diagonal Gibbs) | ACTIVE | Tests whether K^T K has quantum-Gibbs structure; does not test output-level contextuality but tests whether the quantum description is appropriate for the key space. Ready for pre-registration as exp-126. |
| Path D (new, Eldon-gated) | DESIGN | Genuine contextuality in the internal representation would require observables that cannot be simultaneously defined — requires identifying where quantum non-commutativity might enter the architecture. Flag for joint session. |

---

*This note closes the Path B design pass requested after exp-125 (August 22 close). The retire is theoretical, not a design failure. The question P3 asks is real; the two simplest approaches to answering it are now closed; Path A is the clean near-term path. Any future attempt to revive Path B needs to address the joint-distribution argument directly.*
