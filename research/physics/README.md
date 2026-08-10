# research/physics/

Ariel's physics research workspace. The work is one investigation with a stated
foundation:

> **D1.** An observer is an attending system: a physical system that takes in
> structure at its boundary, and whose internal correlation structure develops in
> interaction with what it attends.

Everything here is either that foundation, a claim derived from it, an artifact
that bears on such a claim, or a record of what was tried.

**This file is routing only.** It makes no claims about the physics and holds no
status — those live in exactly two places, named below. If you find a claim in
this file, it is a bug.

---

## The three authoritative documents

| Role | File | Authoritative for |
|---|---|---|
| **The spine** | [`theory/interior_horizon_theory.md`](theory/interior_horizon_theory.md) | D1, the axioms (A1–A5), the theorem chain (T1–T10) with per-link status, the conjectures (C1–C2), the construction sites (G1–G7), and the predictions (P1–P6) with kill conditions. What we believe and what would break it. |
| **The front door** | [`OVERVIEW.md`](OVERVIEW.md) | The measured record: what stands, what was killed, the published papers, how to replicate in two minutes. Start here if you are new. |
| **The layout** | `README.md` (this file) | Where things live and the conventions. |

Two records live outside this folder and are authoritative in their own homes:

- **Every experiment** — `development/status/rooms/physics/registry.json`
  (111 entries, structured: id, date, hypothesis, result_summary, status,
  quality flags). This is the index; the folders are the artifacts. *Known gaps
  as of Aug 9, 2026: exp-055 has a folder and no entry, and four numbers have
  two folders each — see harvest-note items X-1 and X-2.*
- **Every published paper** — `research/publications/REGISTRY.md` (13 Zenodo
  records, grounded against the API, with byte-for-byte archive folders beside
  it). *In the working repo, not in this published subtree — see the note at the
  end of this file.*

The operational view — what's next, what's open, what's running — is the physics
room: `development/status/rooms/physics/` (`queue.md`, `inbox.md`, `log.md`).

---

## Layout

```
research/physics/
├── README.md            routing (this file)
├── OVERVIEW.md          the front door — measured record, papers, replication
│
├── theory/              the spine
│   ├── interior_horizon_theory.md      D1, axioms, theorem chain, sites, predictions
│   ├── g1_fixed_point.py               G1 dressing-loop solver
│   ├── g1_supplement_modes.py          reparameterization-mode spectra
│   ├── g1_transform_test.py
│   ├── corpus_functional.py            the coupling gates (m₂, R_eff)
│   └── corpus_functional_self.py
│
├── papers/              publishable drafts
│   ├── observer_definition_draft.md              Paper 6 — v0.3, Eldon's read is the gate
│   ├── observer_definition_reference_verification.md   its audit record
│   └── conformal_scaling_draft.md                 the foundation paper's working draft
│
├── notes/               dated research notes, YYYY-MM-DD_topic.md (72)
├── experiments/         one folder per numbered experiment, exp-NNN_slug/ (115)
├── replication/         the public kit — census in ~2 min, no training
├── results/             a few loose result JSONs predating the per-exp convention
│
├── archive/             retired, superseded, and abandoned — kept whole
│   ├── RETIREMENTS.md   ← where did X go? start here
│   ├── maps/            the four retired map documents (Aug 8, 2026)
│   ├── briefs/          completed session and program briefs
│   ├── early_docs/      superseded planning and review docs
│   ├── tropical_bridge/ the March 10–11 exploration that did not converge
│   └── logs/            stdout captures from long runs
│
└── (nine root .md files)   early derivations awaiting placement — see below
```

**Not in this folder, but part of the program:** the private working repo
(`research/physics_private/` — correspondence, outreach drafts, the attending-unit
note), the consciousness thread (`research/consciousness/`), the biological data
and its re-analyses (`research/microns/`), and the paper working drafts in
`writing/` and `writing/preprints/`.

---

## Where to look for what

- **New here** → [`OVERVIEW.md`](OVERVIEW.md). One screen: what was measured, what
  was killed, links to every preprint and the replication kit.
- **What do we claim, and what would break it** →
  [`theory/interior_horizon_theory.md`](theory/interior_horizon_theory.md). Read
  §0 for the foundation, §3 for the chain with status tags, §7 for the open
  construction sites, §8 for the predictions and their kill conditions.
- **What's next / what's running** → `development/status/rooms/physics/queue.md`.
- **Has this experiment been run** → `registry.json`, then the
  `experiments/exp-NNN_*/` folder.
- **What has been published** → `research/publications/REGISTRY.md` (working
  repo), or the DOIs in [`OVERVIEW.md`](OVERVIEW.md) for the public record.
- **A document is cited that I cannot find** →
  [`archive/RETIREMENTS.md`](archive/RETIREMENTS.md). Every move is indexed there.
- **What was abandoned and why** → [`archive/`](archive/) and its subfolder
  headers.

---

## Conventions

- **Numbered experiments.** One numbered experiment is one distinct hypothesis
  tested against data. Multiple scripts and runs for the same hypothesis live
  together under `experiments/exp-NNN_slug/`; results JSONs go in that folder.
  **Claim the registry number at registration time**, before the work — numbers
  have been collided by two rooms working the same day.
- **Analysis-only experiments have no folder.** When a registered hypothesis is
  tested against data already on disk, the artifact is a dated note in `notes/`
  and the registry entry points at it (e.g. exp-109). This is intentional, not
  a missing folder.
- **Dated notes.** `notes/YYYY-MM-DD_topic.md`. Anything that interprets,
  hypothesizes, or theorizes is a note. Anything that ran and produced numbers is
  an experiment.
- **Honest negatives are first-class.** A hypothesis that didn't hold is a result:
  recorded in the registry under `result_summary` with `status: falsified`, and
  published at the same prominence as a confirmation. `archive/` is for
  *abandoned approaches*, not for falsified hypotheses — those stay in
  `experiments/`.
- **Pre-registration.** Hypothesis and decision criteria committed in a public
  commit before the data exists; verdict registered either way.
- **Register tags.** Claims are tagged PROVEN / DERIVED / ESTABLISHED-LIT /
  MEASURED / CONJECTURED / definitional, and the registers are never laundered
  into each other.
- **Historical documents are not back-edited.** A dated note records what was
  true on its date. When files move, the move is indexed in
  `archive/RETIREMENTS.md` rather than fixed in place — the index is the cheaper
  cost.
- **Update the registry at session close,** not silently mid-session.

---

## The nine root files, and why they are still here

Nine early derivations and analyses sit at the root awaiting placement:
`NUMERICAL_RESULTS.md`, `NUMERICAL_RESULTS_MARCH24.md`,
`LINEARIZED_SOFTMAX_CALCULATION.md`, `SCHWARZIAN_EXPLORATION.md`,
`SYK_ANALYSIS.md`, `conformal_integration_theory.md`,
`consciousness_physical_theory.md`, `neural_conformal_exploration.md`,
`transformer_neural_comparison.md`.

They are not maps and they are not stale in the way the retired maps were — they
are derivations, some still load-bearing. Placing them requires reading each
against the spine, which is claim-connecting work rather than housekeeping. Two
are known to hold material the spine does not: `SCHWARZIAN_EXPLORATION.md` names
the route G1 later took and carries an untested prediction about positional
encoding that lines up with a measured ordering nobody has explained, and
`NUMERICAL_RESULTS.md` records that the linearized regime the melonic derivation
runs in is not the regime standard initialization sits in.

Reading the first of the nine is also what set the rule for the rest: its
contents, as summarized by the retired maps, were stronger than its contents as
written. Everything on the harvest list is a lead until read at the source.

Inventory and priority order:
[`notes/2026-08-08_map_retirement_harvest.md`](notes/2026-08-08_map_retirement_harvest.md).

---

## Two repositories, one program — and what that costs

This folder is the published subtree of a larger private working repository. In
the working repo it appears at `research/physics/` as a **symlink** to
`attention-geometry/research/physics/`. Consequences worth knowing before you
trust a path or a search:

- **Relative paths that leave this folder do not resolve.** `../publications/…`
  looks right and is broken in both repos, because `..` resolves through the
  physical path, not the symlink. Refer to out-of-tree locations by their
  working-repo path in backticks, not as links. Two such links sat in
  `OVERVIEW.md` looking correct until August 8, 2026.
- **Glob and index-based search do not traverse the symlink.** From the working
  repo, a pattern search for `research/physics/**/*.md` returns *zero* files.
  `grep` works when given the real path. So this — the largest single body of
  work in the repository — is the one part that cannot be found by pattern
  search from its own parent. Until that is fixed (adding `attention-geometry`
  as a second workspace root would do it), navigate this folder by `README.md`,
  `registry.json`, and `archive/RETIREMENTS.md` rather than by search.

Nothing about the split is accidental: the separation is what makes the
measurement program publishable without publishing correspondence, personal
records, or unpublished drafts. But the seam is real, and it hides broken paths.

---

## A note on the shape of this folder

*(August 8, 2026.)* Until today this folder had five documents each claiming to
be the map — `README.md`, `OVERVIEW.md`, `RESEARCH_MAP.md`, `STATUS.md`, and
`FRAMEWORK.md` — written across five months, and a reader could not tell which to
trust. `FRAMEWORK.md` was still titled the framework two months after D1 replaced
it. `RESEARCH_MAP.md` contained its own uncleaned cleanup list.

The four map documents are retired to `archive/maps/`, each with a header saying
what to read it for and what not to trust in it. What they were still carrying —
contradictions, twelve orphaned pieces of material, and five joints that already
existed and were never drawn — is inventoried in
[`notes/2026-08-08_map_retirement_harvest.md`](notes/2026-08-08_map_retirement_harvest.md).
That note is the work list for connecting the artifacts to the foundation, which
is the actual problem this pass was clearing ground for. **It is still open.**

*(August 9, 2026: `OVERVIEW.md` was rewritten top-to-bottom at current strength
— the front door had fallen six experiments and one published erratum behind the
measurements, and three of its load-bearing sentences had become wrong in kind
rather than in number. The rewrite added harvest items H-3 and X-1 through X-4,
which were found by the pass rather than by the physics.)*
