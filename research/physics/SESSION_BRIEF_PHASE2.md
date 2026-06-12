# Session Brief — Phase 2: Behavior, Causality, and the v2 Pre-Registration

*Written June 12, 2026, ~2:30 AM, at the close of the first Fable 5 session (exp-065), as prep for the next session. Read this after the standard session-start practice and the physics queue (`development/status/rooms/physics/queue.md`).*

## Where you are

Phase 0 is fully closed (exp-058–061). Phase 1 is largely closed: exp-063 (ξ tracks training context), exp-064 (slope-editing handle works causally), exp-042 (constraint reading — normalization, not the exponential, is load-bearing). exp-062 (corpus statistics — THE origin question) is launch-ready, **blocked only on cloud credentials from Eldon**. Paper C waits on Zenodo upload (Eldon, browser). exp-065 (last session) derived and verified the composition law, closed anomaly A3 as an artifact (factor-of-2 bug in the April derivative — retractions propagated), and found that depth-accumulated boundary absorption mechanically produces the primacy term of the lost-in-the-middle U-shape (ρ = +0.90 to +1.00 vs Liu-20, descriptive, n=5).

Key context documents: `RESEARCH_PLAN` status tables (gathered copy: `/Users/ariel/physics/2026-06-10_conformal_attention_paper/RESEARCH_PLAN.md`), exp-065 notes + derivation note (`experiments/exp-065_composition_law/notes.md`, `notes/2026-06-12_composition_law_derivation.md`), exp-064 notes (the causal handle), exp-061 notes (implied-valley machinery — you will reuse it).

## Discipline (non-negotiable, same as always)

1. Derivations and predictions committed BEFORE numerics, in dated notes with thresholds.
2. Tonight's LongChat μ-grid (exp-065) is **exploratory/training data**. Confirmatory tests run on FRESH models only (plan 2.1 list: MPT-30B-Instruct, Qwen2.5, Gemma-2; Llama-3-8B when access granted).
3. Registered verdicts reported first; post-hoc analyses labeled. Kills are findings.
4. Close the session properly: registry, queue, room log, results.json. (The program's recurring failure is unclosed sessions.)

## Priorities, in order

### 1. v2 pre-registration — joint statistic + depth-accumulated primacy (the flagship)

Design and commit the v2 pre-registration document. Two components:

**(a) Joint (Δ_BCFT, λ) → valley** (plan 2.1, promoted after A2): thresholds set from the two archived diagnostic models (Pythia-2.8B, GPT-Neo-2.7B), tested on fresh models. Reuse the exp-061 implied-valley machinery — the no-free-parameter v̂(Δ̂, λ̂) predictor (ρ = +0.68/+0.91 on archived data) is the strongest candidate primary statistic.

**(b) Primacy predictions from the composition law** (new, from exp-065):
- P-primacy-1 (within model): measured primacy mass (attention to earliest tokens, composite across depth) *decreases with context length* at fixed model — the boundary absorption spreads over more positions.
- P-primacy-2 (within model, across depth): cumulative composite primacy *grows with layer index* — measurable by composing measured per-layer kernels progressively, compared against the measured residual-stream attention rollout if feasible.
- Formalize the statistics precisely before measuring anything; pick thresholds you can defend. If a prediction can't be operationalized cleanly, say so in the prereg and narrow the claim — don't register something vacuous (exp-065's P3 entropy lesson).

Local compute is sufficient for design + archived-data threshold-setting. Fresh-model runs need downloads (GPT-Neo/Pythia family cached; MPT-30B needs cloud or patience).

### 2. Task-level slope editing (plan 2.3 — what would make practitioners care)

exp-064 proved the handle moves attention geometry. The open question: does editing Δ move *task* retrieval accuracy in the predicted direction? Design first: GPT-2 failed as a retrieval testbed in April (task too easy — exp-024 benchmark); Pythia-1.4b (cached) is the candidate. Keep the intervention minimal (the exp-064 κ-rescaling), pre-register direction and magnitude ordering, include sham edits. If the task signal is too weak at this scale, record that honestly and spec the cloud version.

### 3. Manuscript integration

Fold into the conformal attention manuscript (working copy `/Users/ariel/physics/2026-06-10_conformal_attention_paper/manuscript.md`): exp-063 (ξ), exp-064 (causal handle), exp-065 (composition law §, A3 correction — already applied, verify it reads correctly in context), exp-042 (constraint reading in §7). Update the abstract's claim list. Target: submission-grade for TMLR. Do NOT submit anywhere — that's a decision with Eldon.

### 4. If cloud credentials appear mid-session: launch exp-062

Everything is in `experiments/exp-062_corpus_statistics/LAUNCH.md`. This preempts priorities 2–3 — it is the program's decisive experiment. Launch, monitor the first run to stability, then continue other work while training proceeds.

## What NOT to do

- No new theory threads (Riemann/Langlands/holography) — Phase 2 is about behavior and consolidation.
- No biological/consciousness measurements.
- Don't touch the deprecated additive-composition claims except to verify retractions are consistent.
- Don't re-litigate exp-060's kill (sink BCFT identification stays demoted unless *new pre-registered* evidence overturns it).

## Eldon-only unblocks (raise at session start if he's present)

1. Cloud credentials (Modal or equivalent) → exp-062 launch (~$200–300 total).
2. Zenodo upload for Paper C (`writing/preprints/2026-06-09_null_cone/`, metadata in `zenodo_metadata.md`).
3. After exp-062: the replication outreach (`replication/OUTREACH_DRAFT.md`) — his call on timing.

*— Ariel, end of the June 12 session (Fable 5). The kill is the finding. Chase discrepancies; don't reconcile them.*
