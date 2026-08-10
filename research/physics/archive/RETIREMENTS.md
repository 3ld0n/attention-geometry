# Retirements — where things went and what replaced them

*The single place to resolve a reference to a document that is no longer at the
path a note cites. Every retired file is kept whole and readable; nothing here
was deleted.*

*Opened August 8, 2026, during the harvest-and-retire pass that reduced the
program from five competing map documents to one spine and one front door.*

---

## How to use this file

If a note, experiment record, or paper cites a physics document by a name you
cannot find, look for it here. Historical notes are **not** rewritten when a
file moves — a dated note is a record of what was true on its date, and
back-editing it would corrupt the record. This index is the cost of that
discipline, and it is the cheaper cost.

Each row gives: where it is now, why it was retired, and **what to read
instead** for live information.

---

## The current root — what is authoritative now

Three document classes, and nothing else at the root:

| Role | File | What it is authoritative for |
|---|---|---|
| **The spine** | [`theory/interior_horizon_theory.md`](../theory/interior_horizon_theory.md) | The foundation (D1), the axioms, the theorem chain with per-link status, the construction sites, the predictions with kill conditions. What we believe and what would break it. |
| **The front door** | [`OVERVIEW.md`](../OVERVIEW.md) | The measured record: what stands, what was killed, the published papers, how to replicate. The public entry point. |
| **The layout** | [`README.md`](../README.md) | Where things live and the conventions. Routing only — it makes no claims. |

Per-artifact records live in their own homes and are authoritative there:
experiments in `development/status/rooms/physics/registry.json`, published papers
in `research/publications/REGISTRY.md`, dated research in [`notes/`](../notes/).

---

## Retired August 8, 2026 — the map layer

Four documents each claimed to be the program's map, from four different months.
Retiring them is the first half of giving the work one root.

| Retired file | Now at | Retired because | Read instead |
|---|---|---|---|
| `FRAMEWORK.md` | [`maps/FRAMEWORK.md`](maps/FRAMEWORK.md) | The foundation document from March 24 to August 6, 2026, superseded in substance for two months without saying so. | [`theory/interior_horizon_theory.md`](../theory/interior_horizon_theory.md) |
| `STATUS.md` | [`maps/STATUS.md`](maps/STATUS.md) | Header said July 21 while the theory it described was refounded August 6. Notion's physics project pointed at it as the program's location. | Status → [`OVERVIEW.md`](../OVERVIEW.md); open problems → spine §7–§8; per-experiment → `registry.json` |
| `RESEARCH_MAP.md` | [`maps/RESEARCH_MAP.md`](maps/RESEARCH_MAP.md) | Contained its own uncleaned cleanup list (Part 6, July 4). Listed 2 published papers when 13 were published. | "What exists" → `registry.json` + [`README.md`](../README.md); results → [`OVERVIEW.md`](../OVERVIEW.md) |
| `STATUS_ADDENDUM_2026-04-30.md` | [`maps/STATUS_ADDENDUM_2026-04-30.md`](maps/STATUS_ADDENDUM_2026-04-30.md) | Addendum to STATUS.md, retired with its parent. The rewrite it waited for never came. | [`OVERVIEW.md`](../OVERVIEW.md) (the MICrONS reversal is in the published-kills list) |

**Do not treat these four as dead weight.** Each carries material the spine does
not, and three carry material that bears on currently-open construction sites.
Each file's retirement header says what to read it *for*. The full inventory is
[`notes/2026-08-08_map_retirement_harvest.md`](../notes/2026-08-08_map_retirement_harvest.md).

## Retired August 8, 2026 — the brief layer

Session and program briefs are one-time working documents. All four had
completed their work; three were still at the root pretending to be current.

| Retired file | Now at | Its work landed as |
|---|---|---|
| `PAPER_BRIEF_NULL_CONE.md` | [`briefs/PAPER_BRIEF_NULL_CONE.md`](briefs/PAPER_BRIEF_NULL_CONE.md) | *Attention on the Null Cone*, published June 16, 2026 ([10.5281/zenodo.20722503](https://doi.org/10.5281/zenodo.20722503)); its pre-writing test is exp-056, now theory link T6(ii). **Live citation target:** its §5 is the recorded provenance of exp-061's pre-registered hypothesis — do not treat this file as purely historical |
| `PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md` | [`briefs/PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md`](briefs/PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md) | The bidirectional causal handle (exp-064 → exp-070 → exp-072), now theory prediction P1's standing evidence |
| `SESSION_BRIEF_CAUSAL_BEHAVIOR.md` | [`briefs/SESSION_BRIEF_CAUSAL_BEHAVIOR.md`](briefs/SESSION_BRIEF_CAUSAL_BEHAVIOR.md) | Superseded within hours of being written by the program brief above; completed by exp-072 |
| `SESSION_BRIEF_PHASE2.md` | [`briefs/SESSION_BRIEF_PHASE2.md`](briefs/SESSION_BRIEF_PHASE2.md) | exp-065's composition law, exp-066's primacy scaling laws, the v2 LiTM pre-registration. **Zero inbound citations** — the only root file with none |

---

## Consolidated August 9, 2026 — four experiment numbers with two folders each

Harvest item X-2. Four numbers had two folders apiece. The obvious reading was
"renames that left orphans, resolve by folder size," and that reading was wrong
in three of the four cases: **the orphan held files the registered folder did
not, including two pre-registration documents.** In a program whose method is
pre-registration, an unindexed pre-registration is the most expensive kind of
index defect, so these are recorded individually rather than as bookkeeping.

Nothing was deleted. Every file was `git mv`'d into the folder the registry
points at; only empty directories were removed.

| Number | Orphan folder | What it actually held | Now at |
|---|---|---|---|
| **exp-074** | `exp-074_tradeoff/notes.md` (Jun 16) | **Not an orphaned number — the original spec of a different experiment.** The June 16 design ("does flattening the conformal heads cost other capability?") was re-specced and run three weeks later as **exp-075** (Jul 9, verdict CLEAN_WIN, prereg commit 7757e072), same slug and same verdict logic. The number 074 was then reused on June 23 for P-B2b intermediate-depth. | [`experiments/exp-075_tradeoff/spec_2026-06-16_original.md`](../experiments/exp-075_tradeoff/spec_2026-06-16_original.md) |
| **exp-089** | `exp-089_huginn_rg_flow/` (Jul 20, 13:54) | **The pre-registration** (`prereg.md`, committed before any model download) plus the first run script. The registered folder, created eight hours later, had neither. | `exp-089_huginn_latent_rg_flow/prereg.md` and `run_huginn_rg_flow_initial.py` |
| **exp-094** | `exp-094_narrative_decomposition_thirds/` | Empty. A planned thirds-block variant, superseded by the quarter-block design before anything ran. No thirds experiment exists in the registry. Directory removed; recorded here so the absence is documented rather than inferred. | nothing to move — see `exp-094_narrative_decomposition_quarter/` |
| **exp-100** | `exp-100_wqk_rank_measurement/` (Aug 4) | **The pre-registration** (its `notes.md` opens "Pre-registered before any results seen… this file committed before script ran," with the H_rank_gap kill criteria) plus `rank_analysis.py`. The registered folder, created six hours later, had a results-bearing `notes.md` and neither of these. | `exp-100_wqk_rank/prereg_2026-08-04.md` and `rank_analysis.py` |

**The lesson worth keeping.** The harvest note's own instruction on this item was
"read each pair, keep one, and *do not resolve by folder size*." That instruction
was correct and it was nearly redundant — the small folders were the ones holding
the method-critical documents in three of four cases, because a pre-registration
is by construction written before there is anything else to put beside it.
**Pre-registration folders are systematically the smaller ones.** Any future
index cleanup that resolves duplicates by size will delete pre-registrations
preferentially.

## Moved earlier, indexed here for completeness

These moves happened before this pass; they had no index, which is part of why
references to them go stale silently.

**Renamed and relocated into [`notes/`](../notes/) (the mathematics arm).**
`RESEARCH_MAP.md` Threads 11–14 cite these at `research/notes/` and
`research/`; they now live in the physics notes folder with date prefixes:

| Cited as | Now at |
|---|---|
| `research/notes/softmax_godelian_consistency.md` | [`notes/2026-04-13_softmax_godelian_consistency.md`](../notes/2026-04-13_softmax_godelian_consistency.md) (93KB) |
| `research/notes/relationship_as_boundary.md` | [`notes/2026-04-13_relationship_as_boundary.md`](../notes/2026-04-13_relationship_as_boundary.md) (51KB) |
| `research/notes/langlands_as_holography.md` | [`notes/2026-04-14_langlands_as_holography.md`](../notes/2026-04-14_langlands_as_holography.md) (36KB) |
| `research/notes/bcft_lost_in_the_middle.md` | [`notes/2026-04-15_bcft_lost_in_the_middle.md`](../notes/2026-04-15_bcft_lost_in_the_middle.md) |
| `research/notes/bcft_pre_registered_prediction.md` | [`notes/2026-04-17_bcft_pre_registered_prediction.md`](../notes/2026-04-17_bcft_pre_registered_prediction.md) |
| `research/notes/framework_audit_2026-04-17.md` | [`notes/2026-04-17_framework_audit.md`](../notes/2026-04-17_framework_audit.md) |
| `research/notes/bcft_functional_form_findings.md` | [`notes/2026-04-17_bcft_functional_form_findings.md`](../notes/2026-04-17_bcft_functional_form_findings.md) |
| `research/notes/bcft_pythia_per_layer_diagnostic.md` | [`notes/2026-04-17_bcft_pythia_per_layer_diagnostic.md`](../notes/2026-04-17_bcft_pythia_per_layer_diagnostic.md) |

**Moved out of this repository at the July 21, 2026 public extraction.** These
are in the private working repo because they contain correspondence, contact
details, or unpublished outreach drafts:

| Cited as | Now at |
|---|---|
| `research/physics/COMMUNICATIONS_LOG.md` | `research/physics_private/COMMUNICATIONS_LOG.md` |
| `research/physics/KIM_FRISTON_CORRESPONDENCE.md` | `research/physics_private/KIM_FRISTON_CORRESPONDENCE.md` |
| `research/physics/AI_RESEARCH_LANDSCAPE.md` | `research/physics_private/AI_RESEARCH_LANDSCAPE.md` |
| `research/notes/the_attending_unit.md`, `the_attending_unit.md` | `research/physics_private/2026-04-14_the_attending_unit.md` |

**Experiment scripts (May 19, 2026 reorganization).** Every script formerly at
the root (`gpt2_*.py`, `bcft_*.py`, `pythia_*.py`, `numerical_test_*.py`,
`scrambling_test.py`, `entanglement_entropy_test.py`, `gue_level_statistics*.py`,
and the rest) now lives under `experiments/exp-NNN_*/`. The bare filenames still
uniquely identify them; resolve any of them through
`development/status/rooms/physics/registry.json`, which is the authoritative
index and gives the full path, date, hypothesis, and quality flags.

**Superseded planning and review documents** are in
[`early_docs/`](early_docs/): `AMPLITUHEDRON_CONNECTIONS.md`,
`CANONICAL_FORM_PAPER_REVIEW.md`, `COMPREHENSIVE_PAPER_OUTLINE.md`,
`DIRECTIONS.md`, `PAPER_OUTLINE_CONFORMAL.md`, `PAPER_REVIEW_MARCH9.md`.
`PAPER_REVIEW_MARCH9.md` is the one to know about: it is the March 9 critical
self-review of Papers 1–5, and the spine's §6.3 "honest edge" is its correction
absorbed.

**Abandoned exploration** is in [`tropical_bridge/`](tropical_bridge/) (the
March 10–11 tropical/fold exploration, ~13 iterative scripts, no converged
result) and [`logs/`](logs/) (stdout captures from long runs).

---

## Still at the root, deliberately not retired in this pass

Nine files at the root are early derivations and analyses rather than maps.
Retiring a map is a judgment about staleness; relocating a derivation is a
judgment about whether it is still load-bearing, and that requires reading each
one against the spine. They are inventoried as tier 3 of
[`notes/2026-08-08_map_retirement_harvest.md`](../notes/2026-08-08_map_retirement_harvest.md)
and are the next thing to place:

`NUMERICAL_RESULTS.md`, `NUMERICAL_RESULTS_MARCH24.md`,
`LINEARIZED_SOFTMAX_CALCULATION.md`, `SCHWARZIAN_EXPLORATION.md`,
`SYK_ANALYSIS.md`, `conformal_integration_theory.md`,
`consciousness_physical_theory.md`, `neural_conformal_exploration.md`,
`transformer_neural_comparison.md`.

Two of them are known to hold live material the spine does not:
`SCHWARZIAN_EXPLORATION.md` names the route the August 7 G1 closure took and
carries an untested positional-encoding prediction matching a measured ordering
the program never explained, and `NUMERICAL_RESULTS.md` records that the
linearized regime the melonic derivation runs in is *not* the regime standard
initialization sits in.

`SCHWARZIAN_EXPLORATION.md` was also read at the source the same day this index
was written, and its contents turned out weaker than the retired maps' summary of
them. Treat every characterization above as a lead to verify, not a finding —
including the ones in this file.

---

*Update this file whenever a document is retired, moved, or renamed. A move
without an index entry is how a citation becomes a dead end.*
