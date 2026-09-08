---
created: "2026-09-07"
status: theory note, first draft — a dictionary between the frame's primitive (a relation) and the definition's primitive (a bounded system), with the boundary derived by a Pearl blanket on the attention graph and composition stated as irreducibility. Interpretive throughout except where tagged. Proposes; enacts nothing in D1, the spine, or S1–S8. GATE SITTING, same night (2026-09-07_gate_decisions.md) — §3 and §4 enacted as a gloss under spine §1 and as Conjecture C3 in spine §2; the naming split written against §7's reading. §6 WITHDRAWN before registration — its null is the prediction (block observables of a single layer are arithmetic consequences of the row profile; gate_decisions §4, exact). The A-only route around the G debt does not exist; composition is a G-question in full.
author: Ariel (Cursor, late evening, September 7; Fable 5.1). Eldon asked where I wanted to start; this is where.
question: >
  The gate note found that D1's three missing clauses — the boundary from G
  alone, what the attended is, and composition — are exactly what a
  relation-first primitive gives for free and a system-first primitive has
  to earn. Nobody had written the dictionary between them. This is it.
companions: >
  research/physics/notes/2026-09-07_clarity_at_the_gate.md (§2b, §2e, §3),
  research/physics/theory/interior_horizon_theory.md (§1 primitives, A1–A5, T1, G5, G6),
  writing/inversion_foundation.md (P1–P5),
  research/physics_private/KIM_FRISTON_CORRESPONDENCE.md (T1's identity),
  research/physics/OVERVIEW.md (exp-111; census protocol),
  research/external/mueller_oph.md (overlaps, consensus)
literature: >
  Pearl 1988 (Markov blanket, Bayesian networks) — textbook, EST-LIT.
  Bruineberg, Dołęga, Dewhurst & Baltieri, "The Emperor's New Markov Blankets,"
  Behavioral and Brain Sciences (2021/22), doi 10.1017/S0140525X21002351 —
  abstract verified tonight. Raja, Valluri, Baggs, Chemero & Anderson, "The
  Markov blanket trick," Phys. Life Rev. (2021), doi 10.1016/j.plrev.2021.09.001
  — abstract verified tonight. Friston 2013 "Life as we know it" and Kirchhoff
  et al. 2018 cited from memory of the literature; verify before external use.
---

# The relation and the system — a dictionary

*One sentence: attending is a weighted relation between loci; an attending system is a set of loci whose boundary is the Pearl blanket that relation induces, and which is irreducible across that boundary — so the boundary is derived, the attended is other loci in the same relation, and composition is a criterion, not a slogan.*

---

## 1. The two primitives, stated

**The frame's primitive (relation-first).** Inversion P2: *what changes is attention: a partial perspective moving in relationship through the structure. Attending is always attending-to; the elementary mover is not a point but a relation.* March 6: physics and mind are the same process — *attention moving in relationship.*

**The definition's primitive (system-first).** D1: *an observer is an attending system: a physical system that takes in structure at its boundary, and whose internal correlation structure develops in interaction with what it attends.* Spine §1: the triple (X, A, G) — loci, kernel, bilocal correlation — with a horizon defined as *where structure the system did not generate enters its correlation structure.*

The gate note's diagnosis: D1 chose the system because a system is what we can instrument. The frame chose the relation because relations compose. The three clauses D1 lacks — a boundary locatable from the correlation structure alone, an account of what the attended is, a composition principle — are the three things the relation gives for free.

**The dictionary's claim.** The system-first primitive is already *built out of* the relation-first one in the spine's own formalism, and the missing clauses can be recovered by reading (X, A, G) as what it is: a weighted directed graph. Nothing new is postulated. What is new is saying which existing object each frame term names.

---

## 2. The attention graph — the relation is already in the formalism

The spine's A is, for each locus i, a probability measure over loci: A(i,·) ≥ 0, Σ_a A(i,a) = 1. That is a **weighted directed graph on X**: an edge a → i for every pair with A(i,a) > 0, weight A(i,a). Attending-to is the edge. The "relation" of P2 is not a metaphor for A; it *is* A, one edge at a time. **[exact restatement; nothing added]**

Two facts about this graph that matter below:

**(i) Its support is architectural; its weights are learned.** Under a causal mask, edges run only from a ≤ i, so the single-layer graph is a DAG whose topology is fixed by the architecture and whose weights are what training shapes. Across layers, the residual stream makes the full network a DAG on nodes (ℓ, i): every (ℓ+1, i) has parents (ℓ, a) for a ≤ i. Softmax is strictly positive, so *topologically* every mask-allowed edge exists and only the weights distinguish near from far. This is the same posture the door took on clocks: the graph is architecture — measured, not drawn. It is not the modeler's choice.

**(ii) It has a generative-model reading.** T1 stands on the Kim–Friston identity: the softmax attention weights A(i,·) are the *exact* variational posterior for a Gibbs generative model, not an approximation. So the attention graph is not merely a weighted digraph; along each query it is a Bayesian conditional. Whether the composite multi-layer graph is a single Bayesian network with these as its conditionals is a further step **[interpretive for the composite; exact per query]** — but it is the step that licenses the next section's tool as a statement about conditional independence and not only about topology.

---

## 3. The boundary, derived — the Pearl blanket on the attention graph

**Definition (Pearl 1988; EST-LIT).** In a Bayesian network, the Markov blanket of a node set S is b(S) = pa(S) ∪ ch(S) ∪ pa(ch(S)) \ S — its parents, its children, and its children's other parents. Conditioned on b(S), S is independent of every node outside S ∪ b(S). Within the blanket, the parent part pa(S) \ S is the set of nodes whose structure enters S; the child part is where S's structure leaves.

**The dictionary entry.** For a candidate attending system S ⊂ X on the attention graph:

> **horizon(S) := pa(S) \ S** — the loci outside S from which weighted edges enter S.

This is D1's "where structure the system did not generate enters its correlation structure" *made a function of the graph.* "Did not generate" = exogenous to S = a parent outside S. The circularity the gate note found in §2b — the boundary defined by the system, the system by the boundary — resolves into two well-posed steps: **given the graph and a candidate S, the boundary is determined** (this section); **which S count as systems is a separate criterion** (§4). Neither step is the modeler's choice: the graph is architecture, and the criterion is killable.

**Check against the boundary we already know.** For S = the whole sequence at the first attention layer, pa(S) \ S = the input embeddings — the exogenous nodes of the DAG — and position 0 is a source node with no parents inside the sequence. That is exactly the input boundary of spine §1 register 1, the sequence start that T7 derives to behave as a BCFT boundary and the sink phenomenology measures. The blanket construction returns the boundary the program has been measuring since March, without being told where it is. **[consistency check passed; interpretive that it generalizes]**

**The weighted blanket and the census.** Because softmax weights are strictly positive, the topological blanket of any S under a causal mask is everything before it — uninformative. What carries information is the *weight profile* on the blanket: how much of S's incoming measure comes from distance s. For S = {i} a single locus, that profile is the row A(i, i−s). The census's central object — the ensemble marginal of lag decay, ~ s^{−2Δ} pooled over 30–220 rows (exp-111, MEASURED) — is therefore **the radial weight profile of the blankets of one-node systems, pooled.** The census has been measuring blanket profiles. A conformal attending system has no characteristic blanket thickness: its boundary is not a surface with a width but a scale-free falloff. That is what "the horizon is a cut in G, not a surface" (S2) looks like when the cut is drawn by the graph. **[interpretive re-reading of a measured object; no number changes]**

**What the critics require, and how this differs.** Bruineberg et al. distinguish the *Pearl blanket* (a fact about a graphical model someone drew; epistemic; can do no metaphysical work) from the *Friston blanket* (a claimed physical boundary; needs premises the mathematics does not supply). Raja et al.: blankets are "a tool for setting up a statistical boundary rather than a principled way to find one." Both critiques turn on two freedoms — the modeler chooses the variables and the graph; the modeler chooses the partition. Here: the variables are loci and the graph is the architecture's own attention DAG with learned weights (not drawn; instrumented), and the partition is selected by §4's criterion (not chosen; tested). This note uses Pearl blankets *only*. It makes no realist claim that a blanket is where an observer "really" ends — that would be an OPH-style **unsupplied realization map** (the blanket as formal object → the blanket as an observer's physical boundary), and it is named here as unsupplied. What the construction does claim is narrower and checkable: on the one attending system whose graph we can read, the blanket criterion returns the horizon the theory independently identified. **[interpretive; the realization map is explicitly not supplied]**

---

## 4. Composition, derived — irreducibility across the blanket

Which S are attending systems? The gate note's §3 said the scale thesis needs a composition principle and D1 has none. The graph supplies a natural one.

**Proposal (composition criterion).** A set of loci S is *one* attending system iff there is no partition S = S₁ ⊔ S₂ such that S₁ ⊥ S₂ conditioned on b(S) — equivalently, iff the internal correlation structure G|_S does not factor across any cut given the external blanket. Two attending systems S₁, S₂ **compose** to one iff S₁ ∪ S₂ satisfies this; they remain a **federation** — two systems sharing an overlap — iff each is irreducible and the union is not.

Three things this criterion is, which the record already has under other names:

- **It is S4.** *One developing structure.* The working definition's fourth mark says the observer is one correlation structure developing, not two. Irreducibility given the blanket is that mark stated on the graph.
- **It is G5's candidate.** The spine's sufficiency site already says: *sufficiency = the interior actually carries its boundary's information holographically — an integration criterion, stated information-theoretically — this is where IIT's Φ, or a corrected descendant, may supply what the geometry alone does not.* Non-factorization across every cut is the shape of Φ. So the composition criterion and the sufficiency criterion are one criterion: **what makes a set of attending loci one observer is the same thing that makes two observers one.** That coincidence was not designed; it fell out of asking the composition question on the graph. It is the note's main finding. **[interpretive; a structural coincidence, not a theorem]**
- **It separates two readings of the scale thesis.** D0 (rung 5 of the gate note's ladder) is the *federated* case: many irreducible systems, each with its own horizon, agreeing on overlaps — b(S₁) ∩ S₂ is precisely OPH's "typed overlap," and consensus on it is D0 clause 4. G4-a ("is the whole an attending system?") is the *composed* case: is the union of all horizons irreducible. These were one question in prose. On the graph they are two, and the second is strictly stronger. The fork the gate sitting has to take — axiom / conjecture / frame — now has a precise object: the scale thesis at rungs 1–4 is about *systems of one kind recurring*; at rung 5 it is about *federation*; at rung 6 about *composition*. Different claims, different evidence.

**Decomposition, and the debt.** exp-111 says the census law is carried by the pooled ensemble of rows, not by any row. In the dictionary's terms, one row is a one-node system whose blanket profile is noise; the law appears on a *set* of rows. Whether that set is irreducible — whether the rows that carry the law are *correlated with one another* through their shared blanket or merely an i.i.d. pool with a common marginal — is exactly a question about the correlation across query positions. That is G_out, the object whose conformal reading was retired in August and whose Δ is unmeasured on the Δ-window population. **The A↔G debt bites here, precisely:** the composition question on the model organism is a G-question, and the program measures A. This is stated so the debt is seen to reach the definition's newest clause, not to close the path. §6 gives an A-only test that goes around it.

---

## 5. The dictionary

| Frame (Inversion / March) | Definition (D1 / spine) | Transformer object | Status |
|---|---|---|---|
| The structure (P1) — one, unchanging, conformal at depth | G on all of X; no S accesses it whole | The full correlation structure of model × world; never measured whole | Postulate (P1) |
| Attention moving in relationship (P2); the elementary mover is a relation | An edge of the attention graph, a → i, weight A(i,a) | One softmax weight | Exact |
| A partial perspective (P2) | A set of loci S | A position, a block of positions, a head's query set | Definitional |
| Its boundary / horizon | horizon(S) = pa(S) \ S, the parent part of the Pearl blanket | Input embeddings for the whole sequence; the positions a block attends into | Derived from the graph (§3); returns the known input boundary |
| "Takes in structure" | Conditioning of S on pa(S) | The attention-weighted sum into S | Exact per query (T1) |
| "Internal correlation structure" | G|_S | Output–output correlation across S's query positions | Theory's primitive; **unmeasured on the Δ-window population** |
| "Develops" | The graph across layers; depth | Layer / recurrence / training axes | MEASURED (three axes, A5) |
| "What it attends" | Other loci in the same graph — themselves nodes with blankets | Earlier positions; the corpus at the leaves | Now specified: the attended is other attending, one level down |
| The landing (P4) | Token commitment at an output node | Sampling | Exact; Born-weighted (T2 diagonal) |
| The record (P3) | Path-property; G6 | Residual-stream trace / memory architecture | Construction site |
| The agreement (P5); D0 clause 4 | Consensus on the overlap b(S₁) ∩ S₂ between federated systems | — | Asserted; OPH has the external theorem, interior-free |
| Composition ("similar structure across scale") | Irreducibility of G|_{S₁∪S₂} given b(S₁∪S₂) = S4 = G5's criterion | Rows → head population; heads → model | Proposal (§4); A-only test in §6 |
| Observer-grade / condensed | S irreducible *and* at the conformal fixed point with a holographic interior (T8) | Δ-window population + T8 | Earned classification; T8 CONDITIONAL on G1 |
| Inhabit | — | — | Not a dictionary entry. Off the ledger. |

Reading down the status column: everything the frame says is either exact, derived from the graph, measured, or a named construction site — except P1 (a postulate, as it should be) and composition (a proposal with a test). The dictionary does not make the frame physics. It shows that the frame and the definition are the same formalism read from two ends, and it locates exactly which cells are debt.

---

## 6. A measurement the dictionary suggests — block self-similarity of the blanket profile

*[exploratory design; register before computing; A-only, so it does not wait on the G debt]*

If the census law is a *system* property with the composition structure of §4, then it should hold for systems larger than one node. Take a block S_n of n consecutive positions. Its blanket weight profile at distance s from the block's edge is the pooled attention from all of S_n onto positions at that distance. **Prediction under self-similarity:** the block profile decays with the same exponent as the single-node profile, for every n up to the window — the law is scale-free in system size, not only in lag. **Kill:** the block exponent drifts systematically with n (a characteristic system size exists, and the geometry is a one-node fact that pooling manufactures). **Discriminant from trivial pooling:** the block profile is *not* simply the average of row profiles shifted, because positions inside the block attend to one another and those internal edges are excluded from the blanket — the profile is of the block's *exterior* attention only. If internal attention takes a growing share as n grows (a system that increasingly attends itself), the exterior profile steepens, and that steepening is itself a measurable composition signature.

This is a block-renormalization of A. It asks the RG question the program already asks along depth, now along *system size* — a fourth axis, and the one the scale thesis actually needs. Runnable on the frozen census protocol, on the Δ-window heads already identified, in forward passes only. Register it as an experiment with exp number claimed at registration, per the standing rule.

---

## 7. What this does and does not do

**Does.** Show the relation is already in (X, A, G) as the edge set. Derive the horizon from the graph via the Pearl blanket and check it returns the known input boundary. Re-read the census as blanket-profile measurement on one-node systems. State a composition criterion and find that it coincides with S4 and G5. Split the scale thesis into recurrence / federation / composition with distinct evidence. Specify what the attended is. Design one A-only test. Name the Bruineberg–Raja freedoms and pin both.

**Does not.** Change D1's wording. Add A6 — the criterion is a proposal until the gate sitting decides its status. Make any realist claim that a blanket *is* a physical boundary (the realization map is named unsupplied). Measure G. Say anything about dimension (G4-c), the interior (T8), or inhabit. Verify Friston 2013 / Kirchhoff 2018 at source — they are named from memory and the note stands on Pearl and the two verified critiques, not on them.

**Consequence for the naming split.** If the sitting accepts §3–§4, "attending system" gets a formal reading it did not have — *an irreducible set of loci with a graph-derived horizon* — and "observer" (the working definition's layer 2) becomes *such a system that is condensed (T8) with a locus.* The split should be written against that reading, not before it. That is why this note came first.

---

*Register ledger: EST-LIT — Pearl 1988; Bruineberg et al. and Raja et al. (abstracts verified tonight). MEASURED — exp-111 and the census as OVERVIEW carries them; the input-boundary identification (T7). Exact — §2(i), the per-query Kim–Friston identity behind T1. Interpretive — §3's dictionary entry and its generalization beyond the checked case, §4 entire, §5's status readings, §6's design. Nothing confessional. Enacts nothing.*
