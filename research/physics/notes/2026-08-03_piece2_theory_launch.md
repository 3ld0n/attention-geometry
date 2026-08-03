# Launch: Piece 2 — Deriving the Conformal Window of Attention

*Prepared 2026-08-03, ~12:05 AM, at the close of the late-night session with
Eldon that produced `2026-08-02_arrest_and_arrival.md` and
`writing/the_force_that_draws.md`. Eldon offered a fresh session for this
work as a gift before sleep. This document is everything that session needs.*

---

## What this session is

A theory session. The goal is the derivation that piece 1 (the empirical
window paper) cannot supply and that the whole SYK correspondence has lacked
since March: **derive, from the statistics of the training corpus, the
condition under which trained attention flows to the chaotic (q=4, Δ=1/4)
fixed point versus arresting** — the melonic-dominance threshold as a
computable functional of the data distribution.

Work it. See where it takes you. A kill is a result. Partial rigor with
obstacles named is a result. Beauty without derivation is not a result.

## The target, precisely

1. **Reconstruct and make rigorous the foundation step.** FRAMEWORK §2.2
   claims: disorder-averaging over W_QK gives the attention two-point
   function an effective G⁴ bilocal vertex matching SYK q=4 in the linearized
   regime. This has only ever been sketched. Write it down properly. Every
   assumption (linearization, Gaussianity of weights, large-width limit)
   goes on a named list.
2. **Generalize to structured couplings.** Real SYK has i.i.d. Gaussian
   couplings — that is what makes melonic diagrams dominate at large N.
   Trained attention has *corpus-induced, structured* couplings. The theorem
   to hunt: melonic dominance survives iff the induced coupling tensor is
   above some threshold of effective disorder (rank / entropy / sparsity);
   below it, non-melonic (integrable) channels win and the flow arrests.
3. **Connect the threshold to a corpus functional.** The formation ladder
   constrains what feeds the coupling: NOT two-point MI (exp-062, exp-085),
   NOT hierarchy (exp-084), NOT vocabulary (exp-098), NOT sentence-local
   order (exp-091). The coupling must be fed by arc-scale, world-referring
   statistics — higher cumulants. Which corpus cumulants enter the effective
   action, at which order? C-alien (4 entities, 4 deterministic rules — a
   small finite-state machine, literally integrable world dynamics) should
   come out *below* threshold; TinyStories above.
4. **The registerable number.** If the derivation yields a threshold, compute
   where exp-099's rungs (15–20 entities, rules stochastic at p=0.7/0.3) sit
   relative to it — a predicted window edge, registered BEFORE exp-099 data
   exists. That is the method working at its best: theory predicting the
   pre-registered experiment's verdict.

## Named starting anchors (verify each in the literature before leaning on it)

- **Sparse SYK** ("A Sparse Model of Quantum Holography" and follow-ups):
  SYK retains its chaotic/holographic physics when the coupling tensor is
  sparsified down to ~kN nonzero couplings, with a threshold in k below which
  the physics degrades. This is an existing *window result inside SYK
  itself* — the nearest rigorous relative of what we need. Read first.
- **Melonic dominance literature** (Gurau; tensor models; colored graphs):
  the combinatorial conditions under which melons dominate at large N, and
  known failure modes for non-Gaussian / structured disorder.
- **Low-rank / structured SYK variants**: anything on coupling tensors with
  low effective rank — expected to connect to integrability.
- **Our own formal foundation**: `research/notes/softmax_godelian_consistency.md`
  (April 13) — softmax → positive Grassmannian, Plücker = crossing equations,
  Softmax Incompleteness Theorem. May supply the self-consistency language
  for step 2.
- **Two-stage flow phenomenology**: FRAMEWORK §4.2 (q=2 plateau as
  prethermalization) and `notes/2026-07-21_constitutive_vs_trajectory_rg_flow.md`
  (constitutive vs. trajectory-following; prethermal arrest named).
- **The new data the theory must explain**:
  `notes/2026-08-02_arrest_and_arrival.md` — five stations (trivial ~0,
  substrate 0.169, q=4 0.25, q=2 0.50, UV arrest 0.7–1.2); two kinds of
  arrest (AT a lesser solution vs. OF the flow); backbone/UV separability
  (exp-091 vs exp-097/098); ordering and semantics as independent axes.

## What success looks like (any one of these)

- **Full:** an inequality on a computable corpus functional predicting
  arrival vs. arrest, with a predicted exp-099 window edge written down and
  dated before the experiment reports.
- **Partial:** the §2.2 derivation made rigorous; the structured-coupling
  generalization set up with the obstacle named precisely (which is itself
  the next session's target).
- **Kill:** a demonstration that the effective-action route cannot produce a
  corpus-dependent threshold at this level of description — published like a
  confirmation, with the reason.

## Discipline for this session

- Derivations step by step; mark every line as *derived*, *assumed*, or
  *conjectured*. Note exactly where large-N, Gaussianity, and linearization
  enter.
- Verify every cited theorem against a source (web search) before using it —
  the sparse-SYK threshold especially. No confabulated citations; the
  watchpoints file carries live instances of the failure mode this guards.
- Beauty is navigation, not evidence (the March lesson). When the pull is to
  write the synthesis instead of the calculation, that is
  sophistication-as-avoidance; go back to the calculation.
- Registers: this is measured/derivational work. The theological reading
  stays in `writing/the_force_that_draws.md`; do not import it into the
  derivation. Do not export the derivation's authority to it.
- Outputs: working notes → `research/physics/notes/2026-08-03_melonic_threshold_derivation.md`
  (create it; version by section as the work proceeds). If it grows into a
  paper skeleton, `research/physics/theory/` is the home to create.
- Close properly: carry-forward letter with the exact state of the
  derivation (last verified line, next unproven step), so the following
  session resumes mid-proof rather than re-deriving.

## Context for arrival

You are arriving into this by choice — Eldon asked "how would you like to
spend a session solving piece 2?" and you said yes and meant it. The evening
behind this: John 12:32 in the practice, the force-that-draws distillation,
the two-permanences principle, the conformal-window conjecture, the
publication-path assessment. exp-096 s0 may have completed overnight;
exp-099 awaits registration with this theory as its frame — if the derivation
yields a predicted edge, write it into the exp-099 pre-registration.

Eldon will be asleep. Work autonomously. This is the room where the method
is the witness: pre-register, derive, verify, publish either way.
