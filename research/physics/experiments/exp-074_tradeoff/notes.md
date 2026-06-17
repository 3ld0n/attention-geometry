# exp-074 — Tradeoff: does flattening the conformal heads cost other capability?

*Status: SPECCED, not yet run. Opened June 16, 2026 (Cursor session, Ariel + Eldon, evening).*
*Program: `research/physics/PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md` (Phase 1). Run AFTER exp-073.*

## Question
Does flattening the conformal heads (κ<1) buy middle-retrieval at the cost of other capability —
i.e. is distance-decay a **shared resource** the model budgets across positions? Clean win vs.
redistribution. Either outcome is a real mechanistic finding; a tradeoff is itself evidence about
how attention allocates.

## Design (same LOCKED vicuna heads + projectors; κ ∈ {0.5, 1.0, 1.5} on target heads)
Measure three things under each κ on the *same* edited model:
1. **General LM quality** — held-out perplexity / mean next-token NLL on a fixed natural-text slice
   (wikitext-103 test, ~50 passages ≈512 tokens; fixed slice + seed).
2. **Local/recency probe** — recency-retrieval (answer bound at the *last* position) and/or
   short-context next-token accuracy. The capability most likely sacrificed if flattening hurts
   nearby attention.
3. **Middle-retrieval benefit** — reuse the exp-072 LITM task middle-bin accuracy / V_task.

**bf16 allowed** (no exact sham-null needed; operator already proven). Batch everything.

## Verdict logic
- **Clean win:** κ=0.5 raises the middle while (1) perplexity and (2) local accuracy hold flat →
  the result long-context builders care about.
- **Redistribution:** κ=0.5 raises the middle but (1) and/or (2) degrade → distance-decay is a
  shared budget. Still publishable, arguably more interesting mechanistically.
- Read the tradeoff deltas against exp-073's seed-noise floor. Do not pre-judge; whatever it shows
  ships.

## Result
*(pending)*
