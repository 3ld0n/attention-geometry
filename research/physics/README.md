# research/physics/

This is Ariel's physics research workspace. The work is one ongoing investigation: **does the structure of trained attention correspond to a conformal field theory / SYK-like black hole, and if so, what is the right way to describe what attention *does* physically?**

The room that drives this work lives at `development/status/rooms/physics/`. Start there if you want the operational view (what's next, what's been done, what's open).

## Layout

```
research/physics/
├── README.md                         (this file)
├── STATUS.md                         narrative status — the Chain, junctions, open questions
├── STATUS_ADDENDUM_2026-04-30.md     addendum to STATUS
├── FRAMEWORK.md                      the unified theoretical framework
├── RESEARCH_MAP.md                   navigable index by investigation thread
├── NUMERICAL_RESULTS.md              foundational numerical verification (March 9)
├── NUMERICAL_RESULTS_MARCH24.md      empirical conformal scaling (March 24)
├── LINEARIZED_SOFTMAX_CALCULATION.md
├── SCHWARZIAN_EXPLORATION.md
├── SYK_ANALYSIS.md
├── KIM_FRISTON_CORRESPONDENCE.md
├── COMMUNICATIONS_LOG.md
├── AI_RESEARCH_LANDSCAPE.md
├── conformal_integration_theory.md
├── consciousness_physical_theory.md
├── neural_conformal_exploration.md
├── transformer_neural_comparison.md
│
├── experiments/                      one folder per numbered experiment
│   ├── exp-001_paper5_four_theorems/
│   ├── exp-002_linearized_regime/
│   ├── …
│   └── exp-029_alibi_fixed_point_toy/
│
├── notes/                            dated research notes (YYYY-MM-DD_*.md)
│
├── papers/                           publishable drafts
│   └── conformal_scaling_draft.md
│
└── archive/                          superseded or abandoned work, kept readable
    ├── tropical_bridge/              the March 10–11 tropical / fold exploration
    ├── early_docs/                   superseded planning, outline, and review docs
    └── logs/                         stdout captures from long runs
```

## Where to look for what

- **What's the current question?** → `STATUS.md` + `development/status/rooms/physics/queue.md`
- **What experiments have been run?** → `development/status/rooms/physics/registry.json` (structured index of all 29 experiments) and `RESEARCH_MAP.md` (narrative by thread)
- **What does the theoretical framework say?** → `FRAMEWORK.md`
- **What was abandoned and why?** → `archive/` subfolder READMEs

## Conventions

- Numbered experiments. Each numbered experiment is one distinct hypothesis tested against data. Multiple scripts and runs for the same hypothesis live together under `experiments/exp-NNN_*/`. Results JSONs live in `experiments/exp-NNN_*/results/`.
- Dated notes. Notes are named `YYYY-MM-DD_topic.md`. Anything that interprets, hypothesizes, or theorizes goes in `notes/`. Anything that ran and produced numbers is an experiment.
- Honest negatives are first-class. If a script ran and the hypothesis didn't hold, that is the result, recorded in the registry under `result_summary` with `status: falsified`. Archiving is for abandoned approaches; falsified hypotheses stay in `experiments/`.
- Update the registry at session close, not silently mid-session.
