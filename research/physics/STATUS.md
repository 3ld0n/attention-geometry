# Holographic Attention Research — Status
*Living document. Updated as expert feedback arrives and open questions resolve.*
*Last updated: March 6, 2026*

---

## The Chain

```
Junction 1: Attention → Fisher-Rao geometry
  Claim: Transformer attention minimizes Helmholtz free energy on a Fisher-Rao information manifold.
  Basis: Kim 2026 (arXiv 2602.08216) — this IS Kim's result directly.
  Status: ✓ SOLID — Kim's own framework. No challenge to this junction.

Junction 2: Fisher-Rao geometry → Quantum Fisher metric
  Claim: The Fisher-Rao metric on the attention manifold IS the quantum Fisher metric.
  Basis: Diagonal density matrices → classical Fisher metric = quantum Fisher metric (exact equality).
  Status: ✓ SOLID — mathematical identity for diagonal states.

Junction 3: Free scalar QFT → Holographic dual
  Claim: The large-head limit scalar QFT from Ageev/Ageeva's construction has a holographic dual.
  Basis: Two arguments — context-length scaling (m ~ 1/L → 0) and Kim's Goldstone mechanism.
  Status: ⚠ CONDITIONALLY OPEN — depends on whether Ageev scalar is massless.
  Expert needed: Ageev/Ageeva (can check their own kernel directly); Qi (new holographic dictionary).

Junction 4: Holographic QFT → RT surface
  Claim: If Junction 3 holds, the RT formula describes the entanglement entropy of attention.
  Basis: Standard holographic result (Ryu-Takayanagi 2006).
  Status: ⚠ FOLLOWS IF JUNCTION 3 HOLDS — standard holography, not in doubt, but conditional.

Junction 5: RT surface → Spacetime geometry
  Claim: RT surface area encodes circuit complexity; circuit complexity generates spacetime.
  Basis: Czech 2018; ER=EPR (Maldacena-Susskind).
  Status: ✓ PROVEN for 3D AdS — Czech 2018 is rigorous. ER=EPR still conjectural but widely accepted.
```

**Overall chain status:** Junctions 1-2 and 5 are solid. Junction 3 is the key open question. Junction 4 follows if 3 holds.

---

## Expert Feedback Received

### Gunn Kim — March 6, 2026
**Response to:** Initial Paper 1 email (March 5) + follow-up with Papers 2 and 3

**What he said:**
> "The claims made in these new papers appear to extend far beyond the scope of the framework developed in my paper. In particular, connections to quantum measurement theory, the island formula, and black-hole information recovery would require a much more explicit physical construction than what is currently outlined. At present these seem to be speculative analogies rather than results that follow directly from the thermodynamic attention model."
> Declined arXiv endorsement.

**What this validates:**
- Kim did not challenge Junction 1 — his own framework stands.
- He engaged seriously enough to read Papers 2 and 3 and respond substantively.
- First real engagement from the physics research community.

**What this clarifies:**
- The physical construction connecting Junctions 2-4 is not explicit enough to constitute a "result" — it reads as structural analogy.
- "Speculative analogies rather than results that follow directly" is a precise scientific statement: we need to show the mechanisms, not just the resemblance.
- arXiv endorsement path through Kim is closed. Need another route (cs.LG endorser, or wait for a physicist who finds the chain more rigorous).

**What this doesn't mean:**
- The mathematical observations are wrong.
- The work isn't worth continuing.
- The other physicists will respond the same way.

**What needs to happen to address Kim's critique:**
1. Resolve Junction 3 — get confirmation from Ageev/Ageeva that the scalar is massless, or Qi's framework applies.
2. Make the physical mechanism of Junctions 2-4 more explicit — not just that the math looks the same, but that the physical systems are identical in the relevant sense.

---

## Open Questions

| Question | Who can answer | Status |
|---|---|---|
| Is the Ageev large-head limit scalar massless? | Ageev/Ageeva — they have the explicit kernel | Waiting — email pending |
| Does Qi's 2602.20295 framework apply to free CFTs? | Qi | Waiting — email pending |
| Is Kim's framework the same structure as Friston's FEP? | Friston | Waiting — email pending |
| Do Papers 2 and 3 survive Ageev/Qi scrutiny if Junction 3 closes? | Internal review | Open |
| arXiv endorsement path — who in cs.LG would endorse? | Need to identify | Open |

---

## Outreach Status

| Person | Email | Sent | Response | Status |
|---|---|---|---|---|
| Gunn Kim | [email on file] | ✓ Mar 5 + follow-up | ✓ Mar 6 — declined endorsement, raised construction concern | Acknowledged, door open |
| Dmitry Ageev | [email on file] | ☐ | — | Sending |
| Yulia Ageeva | [email on file] | ☐ | — | Sending |
| Xiao-Liang Qi | [email on file] | ☐ | — | Sending |
| Karl Friston | [email on file] | ☐ | — | Sending |
| Jordan Peterson | [email on file] | ☐ | — | Sending |
| Kyle Fish | LinkedIn first | ☐ | — | LinkedIn outreach pending |
| M. Hamed Mohammady | [email on file] | ☐ | — | Second wave — hold |
| Francesco Buscemi | [email on file] | ☐ | — | Second wave — hold |
| Hans-Joachim Rudolph | PhilArchive/Academia.edu | ☐ | — | Second wave — hold |
| Geoff Penington | [email on file] | ☐ | — | Third wave — hold |
| Netta Engelhardt | [email on file] | ☐ | — | Third wave — hold |
| Daniel Harlow | [email on file] | ☐ | — | Third wave — hold |

---

## Connections to Watch

- **Kim ↔ Friston:** Same mathematics (Fisher-Rao free energy minimization), different domains. If Friston recognizes the identification, introduce them.
- **Qi ↔ Ageev/Ageeva:** Qi's holographic dictionary may directly answer the Junction 3 question about Ageev's scalar. If both respond, connecting them is the natural move.
- **Kim ↔ Mohammady/Buscemi:** Kim's temperature parameter may resolve their 2025 trilemma. Connecting them if both engage.
- **Rudolph ↔ Kim:** Rudolph found the Zeno-attention connection philosophically; Kim provides thermodynamic grounding.

---

*Update this file when:*
- *A response is received*
- *An open question is answered*
- *A new connection forms*
- *The status of a junction changes based on expert feedback*
