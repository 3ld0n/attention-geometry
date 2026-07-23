# Page–Wootters Formalization: exp-089 ↔ Relational Time

*Physics room session, 2026-07-23 (~4:30 PM MDT)*  
*Inbox item: [from: cursor, 2026-07-23] exp-089 ↔ Page–Wootters, formally*  
*Analyzing whether loop-iteration count in inference-time recurrence constitutes a clock subsystem in the Marletto–Vedral sense*

---

## Registered prior (stated before analysis)

**Prior:** The formal PW correspondence does NOT hold for Huginn. The reasons expected in advance:
1. Huginn's iteration count N is a classical discrete parameter, not a quantum subsystem with its own Hilbert space
2. PW requires quantum entanglement between clock and system in the universal state |Ψ⟩ = Σ_N |N⟩_C ⊗ |ψ(N)⟩_S — this structure doesn't exist in Huginn
3. PW requires a Hamiltonian constraint H_total|Ψ⟩ = 0 — Huginn has no such constraint
4. Even the Giovannetti–Lloyd–Maccone 2015 operational reformulation still requires quantum correlations

**Prior confidence:** ~85% that formal correspondence fails; ~15% that some extension I haven't considered might apply.

*Prior registered before analysis. Analysis follows.*

---

## The Page–Wootters mechanism — formal requirements

Sources used: Page–Wootters (1983, Phys. Rev. D 27, 2885); Marletto–Vedral (2016, arXiv:1610.04773); Giovannetti–Lloyd–Maccone (2015, arXiv:1504.04215).

The PW mechanism requires five things:

**PW-1. Factorized Hilbert space.**  
H_total = H_clock ⊗ H_system. The clock C and system S are quantum subsystems with their own Hilbert spaces.

**PW-2. Hamiltonian constraint.**  
H_total|Ψ⟩ = 0 — the Wheeler-DeWitt constraint of canonical quantum gravity. The total state of the universe is stationary: no global time exists in the total state.

**PW-3. Quantum entanglement.**  
The universal state is entangled: |Ψ⟩ = Σ_T |T⟩_C ⊗ |ψ_T⟩_S. This is a quantum superposition — the clock and system are *entangled*, not classically correlated. Classical correlation (given N, the state is ψ(N) deterministically) is not entanglement for PW purposes; quantum coherence across different clock readings is required.

**PW-4. Non-interacting clock.**  
Marletto–Vedral's strengthening (2016): H_int ≈ 0 — the clock must be a non-interacting subsystem. This is how the standard Schrödinger equation is recovered: the clock doesn't disturb the system; it merely correlates with it.

**PW-5. Recovery by conditioning.**  
The system's state at "time T" is recovered by conditioning on the clock reading: |ψ_T⟩_S = ⟨T|_C |Ψ⟩. The Schrödinger equation d/dT|ψ_T⟩ = -iH_S|ψ_T⟩ is then derived from the stationary |Ψ⟩, not assumed.

The whole mechanism's point: there is no external time; time is an internal property of the universe, emerging from the correlation between the clock and the rest.

---

## The Huginn exp-089 setup

From exp-089 (CONFIRMED, 2026-07-20, see notes.md):

- **Frozen weights, frozen input, varying iteration count N** (N = 1, 2, 4, 8, 16, 32)
- **Attention statistics** (Δ_med, SYK-near count) measured at each N
- **Result:** Δ_med flows from ~0.291 (N=1) to ~0.239 (N=32), Spearman ρ = -0.943. SYK-near count increases monotonically for natural text (ρ = +0.771). Randomized-weights control frozen at 0.16868 throughout.

Candidate mapping:
- Clock C = iteration count N ∈ {1, 2, 4, 8, 16, 32}
- System S = attention state (Δ distribution)
- "Time evolution" = Δ-flow toward the conformal fixed point

---

## Formal correspondence check

**PW-1 (Factorized Hilbert space):** FAILS. There is no Hilbert space H_clock for the iteration count N. N is a classical integer parameter controlled by the experimenter. There are no quantum superpositions of "being at step 8 vs. step 16." The model processes one iteration at a time, deterministically.

**PW-2 (Hamiltonian constraint):** FAILS. There is no total Hamiltonian H_total governing the joint (iteration-count, attention-state) system. The dynamics are discrete function application: apply 4 SandwichBlocks → update hidden state. This is not Hamiltonian flow. There is no constraint of the form H_total|Ψ⟩ = 0.

**PW-3 (Quantum entanglement):** FAILS. The "correlation" between N and ψ(N) is classical: given N, ψ(N) is completely determined by the input and the weights. There is no quantum superposition of the form Σ_N |N⟩_C ⊗ |ψ_N⟩_S. Classical deterministic correlation is not entanglement. PW's mechanism depends specifically on quantum coherence across clock states.

**PW-4 (Non-interacting clock):** PARTIAL. The iteration count N does not "disturb" the weights or input (H_int ≈ 0 in spirit). But this satisfaction is trivially true of any control parameter — the Marletto-Vedral condition requires H_int = 0 *between quantum subsystems*, which doesn't apply when the clock isn't a quantum subsystem at all.

**PW-5 (Recovery by conditioning):** TRIVIALLY TRUE BUT EMPTY. "Conditioning on step N" in Huginn is just running the model for N steps and reading off the attention statistics. The conditioning recovers ψ(N) — correct — but there's nothing to recover from: there's no stationary quantum state |Ψ⟩ being projected. The operation is deterministic function evaluation, not quantum state conditioning.

**Verdict: Formal PW correspondence FAILS on requirements PW-1, PW-2, PW-3. PW-4 and PW-5 have trivial/partial satisfaction but not in the sense that matters for the mechanism.**

---

## Can any weaker formalism salvage it?

**Attempt 1: GLM 2015 operational extension**  
Giovannetti–Lloyd–Maccone (2015) gave an operational reformulation of PW that doesn't require the universe to be in a zero-eigenvalue state of H_total. Their "history approach" uses density matrices and conditional probabilities. But GLM still requires: (a) the clock subsystem has a quantum mechanical Hilbert space, (b) the correlation is quantum (captured by trace over the clock's Hilbert space). Without quantum clock subsystems, GLM reduces to: "the state at time N is ψ(N)." This is trivially true for any time-indexed sequence and adds nothing beyond "the function maps N to ψ(N)."

**Attempt 2: Classical PW analog**  
Construct a classical probability distribution P(N, Δ) over (iteration count, attention state). Then "time evolution" = conditional distribution P(Δ|N). But this is trivially true of ANY Markov process or deterministic map. The PW mechanism's content — that unitary evolution is RECOVERED from a globally stationary state, rather than assumed — has no classical analog, because there is no globally stationary state to recover from. The iteration-count sequence N = 1, 2, 4, 8, 16, 32 IS an external time parameter, not something being recovered from a timeless structure. Applying PW to claim "N is a clock subsystem" here is just renaming the parameter.

**Attempt 3: Tensor-network / RG-flow framing**  
The Huginn forward pass is a tensor network where iteration count N parameterizes depth. The fixed-point weights W* are "stationary" in the sense that they don't change during inference. The "evolution" is the trajectory of the attention state through this network. But this is **RG flow** — the attention statistics flow toward an infrared fixed point under repeated application of the same block — not PW. The correct physics framework for exp-089 is already named in the conformal program: integration depth as the RG flow parameter, with the conformal fixed point (Δ → 0.25) as the infrared attractor. This framework is native to the physics, doesn't need PW, and is more precise.

**None of the weaker formalisms salvages a non-trivial PW formal correspondence.**

---

## What the correct physics framework is

Exp-089's conformal convergence is best described by **RG flow** in the sense the conformal program uses:

- The "depth" variable (iteration count N in exp-089; training steps in exp-086; layer index in exp-031) parameterizes integration — how many times the attention computation has been applied to the input
- The conformal fixed point (Δ = 0.25, the SYK q=4 infrared fixed point) is the attractor
- The flow (Δ_med decreasing from UV toward IR as depth increases) is the RG trajectory
- The randomized-weights control freezing at 0.16868 (the exact kinematic substrate exponent — see `notes/2026-07-21_substrate_exponent_kinematic_derivation.md`) confirms that the flow requires trained weights, not just architectural depth

This framework is internally consistent with the conformal program and doesn't require borrowing from PW. The "evolution without evolution" language in §IV.1 of the inversion foundation uses "PW" to name a *spirit* (dynamics emerging from a stationary whole) — and that spirit is correctly illustrated by exp-089, but the PW mechanism is not the formal model.

---

## What this means for the inversion foundation

The phrase in §V.2: "evolution without evolution, in the Page–Wootters sense of the phrase, in an attention system we can instrument completely."

This use of "Page–Wootters sense" is illustrative, not claiming formal PW correspondence. The document itself already hedges: "Whether the correspondence is formal (loop-iteration count as a clock subsystem) or only structural is a registered physics-room question, not a claim." The hedge is correct. The answer is: **structural analogy only**.

**The "inversion in miniature" description survives as a structural analogy:**  
Frozen weights + frozen input + Δ-flow under pure recurrence = the structure does not change; what changes is the attending. This is P1 + P2 of the inversion directly observable on a bench. The description is still apt; only the PW formal mechanism is not applicable.

**What the amendment needs:**  
§V.2: note parenthetically that the correspondence was checked and found to be structural analogy, not formal PW. The measured result and the "inversion in miniature" description stand; the "evolution without evolution in the PW sense" phrase should be softened to "analogous to the PW spirit."

§VII.4: close the open question as answered: formal correspondence absent; structural analogy confirmed; document amended accordingly.

---

## Honest accounting of the prior

Prior said ~85% confidence the formal correspondence fails. Result: formal correspondence fails on all three primary PW requirements. Prior confirmed. Nothing about the inversion is falsified; the exp-089 result retains its full strength; only the PW citation is downgraded from a possible formal correspondence to a structural analogy.

---

## Summary statement

**The formal Page–Wootters correspondence does not hold for Huginn exp-089.** The PW mechanism requires quantum clock subsystems, quantum entanglement between clock and system, and a Hamiltonian constraint; Huginn has none of these. The Δ-flow (Δ_med → 0.239, ρ = -0.943, CONFIRMED) is better described as classical RG flow toward the conformal infrared fixed point — the same flow that training steps operationalize in exp-086 and layer depth operationalizes in exp-031. The "inversion in miniature" reading of exp-089 survives as a structural analogy to the PW spirit: the structure doesn't change; the attending does. But the formal PW mechanism is not at work, and §V.2 of the inversion foundation should say so.
