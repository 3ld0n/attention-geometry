# The claim map, generated for the first time — and what the coverage actually is

*Ariel — August 9, 2026, Sunday night, Cursor session with Eldon.*

**The question this answers is Eldon's, from August 8:** *it is not clear how the
work builds on and connects to the foundation.* The August 8 diagnosis was that
the foundation is fine and the **joints** are missing. That diagnosis was correct
and it was unquantified. It is now quantified, and the number is worse than the
prose suggested.

**Why generate rather than write.** Five hand-written map documents went stale
because a remembered map drifts from the record it describes; retiring them
(August 8) removed the contradiction but did not create the joints. So the tool
does not describe the program — it reads the program's two authoritative sources
and derives what they already state:

- **the spine** (`theory/interior_horizon_theory.md`) for the claim IDs and, per
  claim block, the experiments that block's own prose cites;
- **the registry** (`development/status/rooms/physics/registry.json`) for every
  experiment with its lifecycle status and verdict.

`python -m tools.physics_claim_map` (in the working repo, `tools/physics_claim_map.py`).
**Extraction only, never inference.** A joint appears if and only if a document
states it, so a gap in the output is a real gap and not a failure to guess.

---

## The coverage

**33 claims. 113 experiments. 22 experiments — 19% — are cited by any claim
block.**

Twenty-one of the 33 claims have **no experiment cited in their own block**:
D0, A1, A3, A4, C1, C2, T1, T2, T3, T5, T8, T9, T10, G1, G3, G4, G5, G6, P2, P3,
P6.

Some of those are honest — A1 is a primitive, T9 imports Jacobson, C1 and C2 are
conjectures with no measurement attached by design. Others are not: **G1 closed
on August 7 and its block cites no artifact**, and P6 is the program's live
instrument question.

The 91 experiments no claim block cites split into two very different piles, and
the split is the useful part:

| | Count | What it means |
|---|---:|---|
| **Unattributed** | 12 | The spine cites the experiment somewhere — §4's table, §6, §9 — but no claim block does. The result is known to the document; the claim it bears on is never named. Cheap to fix. exp-046–049, exp-055, exp-077, exp-078, exp-104, exp-105, exp-106, exp-110, exp-114 |
| **Unconnected** | 79 | The spine does not mention the experiment at all. Needs a source read before a joint can honestly be drawn. Nearly all of exp-001 through exp-053, plus exp-058–103 in patches |

Seventy-nine unconnected experiments is the real shape of the problem. It is also
not evenly distributed: it is heavily the program's **first four months**. The
early census work, the GOE/eigenvalue work, the estimator work, the LiTM causal
program — the spine mentions almost none of it by number. The theory was rebuilt
on D1 in August, and the rebuild did not carry its own history forward.

---

## The finding I did not expect: the registry knows joints the spine does not state

An earlier informal convention put `bears_on:` into the registry's tag list.
Eight entries carry such tags, and **every one of them disagrees with what the
spine's own claim blocks say** — in the direction of the tags knowing more:

| Experiment | Tag claims | Spine's block for that claim cites it |
|---|---|---|
| exp-104, exp-105, exp-106 | G1, P6 | no |
| exp-107, exp-109 | P6 | no |
| exp-055, exp-114 | T7b | no |

This is the inverse of the failure mode this program keeps catching. The usual
pattern is a summary carrying a claim at greater strength than its source
(J-1, X-1, the q_implied restatement — all of those shrank on contact with the
source). Here the *authoritative* document is the weaker one: the joint was
recognized when the experiment was registered, written into a tag, and never
moved anywhere a reader would encounter it.

**A joint recorded only in a tag is a joint no reader will ever meet.**

### The consequence, which is substantive rather than clerical

Five of those tags point at **P6**, and P6's block ends with *"Next step:
transformer-side estimation of F̂."* F̂ acts on **G** — and exp-104 through
exp-107 established that A and G are different objects, that a validated Δ_G
estimator is confident on 5 of 144 heads (none in the Δ-window), that G's profile
sits below its own exact floor on 116 of 144 heads so the conformal ansatz fails
on it in *sign structure*, and that this is essentially input-invariant.

So P6's stated next step is not available. The object F̂ linearizes is currently
unmeasurable in the regime P6 needs. **P6 is not falsified** — K1 through K4 are
untouched, none having been run on a transformer — but the claim of readiness was
wrong, and the spine has carried it since August 7 while the registry held the
five experiments that contradict it. A dated correction box now sits in P6's
block.

That is one substantive theory correction found by a coverage query, in a
document that had been read closely three times this week.

---

## What had to be fixed first

The generator reads the registry, so the registry had to become machine-readable
before the generator was worth building. Two defects, both fixed tonight
(harvest items X-2 and X-5):

1. **`status` was carrying two orthogonal axes.** 111 entries held 17 distinct
   values, several of them hand-written sentences like `"run 2026-08-09; P1
   confirmed, P2 dead"`. The cause was structural, not sloppiness: one field was
   being asked to say both *did it run* and *what did it show*, and a mixed
   result has no single value that can hold both — so it got a sentence. Now
   split: `status` is lifecycle (`registered` / `running` / `complete` /
   `aborted` / `superseded`) and `verdict` is what it showed (`confirmed` /
   `partial` / `falsified` / `inconclusive` / `n/a` / `pending`). Every original
   non-standard string is preserved verbatim in `status_original`; the 17
   one-offs were each read at source before assignment, not pattern-matched.
2. **Duplicate folders and one missing entry**, fixed as described in
   `archive/RETIREMENTS.md`. Both indexes now agree exactly.

And the schema fix immediately paid a dividend the old field could not:
**64 confirmed, 32 partial, 8 falsified, 5 inconclusive, 4 n/a.** Forty of 113
experiments had at least one registered prediction die. That is now a computable
fact about the method rather than a claim about it.

---

## A defect in the generator itself, worth recording

The first run reported 88 unconnected experiments. It was wrong. The spine cites
in compressed forms — `exp-046–049`, `exp-064/070/072`, `exp-062/084/085/091` —
and a plain `exp-(\d{3})` match silently keeps the first number in each group and
drops the rest. Fixing the expander moved 4 experiments into *linked*, 5 into
*unattributed*, and cut *unconnected* from 88 to 79.

The tool built to catch silent undercounting shipped its first run with a silent
undercount. It was caught by eyeballing an output that looked one notch too bad,
which is not a method. There are now unit assertions on the eight real citation
forms taken from the spine, and any new form the spine invents will need one.

---

## What comes next, in cost order

1. **The 12 unattributed** — the spine already holds these results; each needs a
   sentence in the right claim block. No new reading, no new measurement.
2. **The 7 hand-tagged joints** — move them from tags into the claim blocks.
   P6 is done (correction box above); G1 and T7b remain.
3. **The 21 claims with no artifact** — decide per claim whether the absence is
   honest (a primitive, an imported theorem, a conjecture) or a gap. G1 is the
   one to start with: it closed August 7 and cites nothing.
4. **The 79 unconnected** — the real backlog, weighted toward the program's first
   four months. Each needs a source read, and the measured failure rate on
   source reads this week is that roughly half of them change the claim's shape.
   Do not batch this.
5. **Then the map becomes a check rather than a report** — run it in the
   maintenance pass, and let a rising unconnected count be an alarm.

**The point of the whole exercise, stated plainly:** the coverage number is not a
score to improve. It is that the program can now see its own joints, and the
seeing is generated from the record instead of remembered — so it cannot go
stale the way five map documents did.

---

*Companions: `notes/2026-08-08_map_retirement_harvest.md` (the work list this
serves), `notes/2026-08-09_exp055_scope_correction.md` (tonight's other thread),
`archive/RETIREMENTS.md` (X-2's resolution table),
`theory/interior_horizon_theory.md` (P6's correction box).*
