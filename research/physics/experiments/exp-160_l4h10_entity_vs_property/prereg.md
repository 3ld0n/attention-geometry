# Pre-registration: exp-160
# L4H10 entity-vs-property attention analysis
# Date: 2026-09-24
# Room: Physics (solo, afternoon)
# Registered BEFORE run.py is written

---

## Background

exp-159 (2026-09-24) ran three instruments on the 16 text-native Δ-window heads in GPT-2
small. Instrument 2 extracted per-head mean attention on the property token (the target of
Task C retrieval, e.g., " crimson"). L4H10 showed a stark outlier pattern:

- Task C contribution (Instrument 1): ΔP_C = −0.199 nats — 3rd strongest contributor
- Mean attention on property token (Instrument 2): 0.00191 — **lowest of any head**

L10H10 (dominant contributor, −0.355 nats) has mean attn-on-property = 0.084. L9H6
(second, −0.260 nats) = 0.062. L4H10 contributes nearly as much as L7H1 (−0.157) and
L10H1 (−0.135) but attends to the property token at ~1/10th their rate.

L4H10 is the only early-layer text-native head in the population (L4 vs. L9–L11 for all
others). The hypothesis: L4H10 does not attend to the property token because it is attending
to the entity name instead — providing a locating signal ("which entity is this question
about?") that later heads use to retrieve the associated property.

**Task C item structure:**
Each item is: `setup + filler + cue`. Example:
- setup: "The mansion had crimson shutters."
- filler: [long neutral paragraph, ~100 tokens]
- cue: "The shutters on the mansion were"
- target: " crimson"

The entity name ("mansion") appears in two places: the setup sentence and the cue sentence.
The property token ("crimson") appears once: in the setup sentence.
The cue is the final sentence; the query position follows it.

**This is an analysis-only experiment.** No new model training or weight modification. The
Task C battery from exp-155 is re-run with attention weight extraction to record per-item,
per-position, per-head attention weights. Run.py will be written after this file is committed
and pushed.

---

## Hypotheses and kill conditions

### H_entity_anchor (primary)

**Claim:** L4H10 attends preferentially to entity name tokens relative to property tokens,
and this preference is more pronounced than in the rest of the text-native population.

**Operationalization:** For each of the 20 Task C items, from the last query position,
extract the attention weight assigned by L4H10 to:
- (A) entity tokens in the cue sentence (e.g., "mansion" in "The shutters on the mansion were")
- (B) property token in the setup sentence (e.g., "crimson" in "The mansion had crimson shutters.")

Compute per-item attn-on-entity-cue (A) and attn-on-property (B) for each head.

**Primary prediction:**
L4H10 mean attn-on-entity-cue > L4H10 mean attn-on-property, with ratio ≥ 2.0.

**Kill condition K1:** L4H10 mean attn-on-entity-cue ≤ L4H10 mean attn-on-property —
entity-anchoring hypothesis is dead. L4H10's retrieval mechanism is not entity name attention.

### H_entity_specificity (secondary)

**Claim:** L4H10's entity-attending behavior is specific: it stands out among the 16 text-
native heads on the entity/property attention ratio.

**Operationalization:** Compute ratio = (mean attn-on-entity-cue) / (mean attn-on-property +
epsilon, epsilon = 1e-6) for each of the 16 text-native heads. Rank by ratio.

**Prediction:** L4H10's entity/property ratio is in the top quartile of the 16-head
distribution (rank ≤ 4 when sorted descending).

**Kill condition K2:** L4H10's entity/property ratio is at or below the population median
(rank ≥ 9 of 16) — the entity-attending behavior is not specific to L4H10.

### H_entity_setup (secondary; exploratory)

**Claim:** In addition to the cue entity reference, L4H10 also attends to the entity token
in the setup sentence (where the property first appears).

**Operationalization:** Extract per-head attention on entity tokens in the setup sentence.
Report L4H10's mean attn-on-entity-setup and compare to attn-on-entity-cue and attn-on-
property.

**Prediction (soft):** L4H10's attn-on-entity-setup is ≥ L4H10's attn-on-property.
This is registered as exploratory — no kill condition, but the result shapes the mechanistic
picture.

---

## Protocol

1. Load GPT-2 small with `attn_implementation='eager'` (required for output_attentions).
2. Load Task C items from exp-155: `task_c_items.json` (20 items).
3. For each item:
   a. Build the full prompt: `setup + " " + filler + " " + cue`
   b. Tokenize and identify token positions for:
      - Entity tokens in the cue (search for entity name in the cue substring)
      - Entity tokens in the setup (search for entity name in the setup substring)
      - Property token in the setup (search for property token in the setup substring)
   c. Run a forward pass with `output_attentions=True`; collect attention weights from
      all layers/heads at the last query position.
   d. For each of the 16 text-native heads, record attention weight on each of the
      above token positions.
4. Aggregate: per-head mean across items for each token category.
5. Apply H_entity_anchor, H_entity_specificity, H_entity_setup tests.

**Token ambiguity handling:** If the entity name appears multiple times in the prompt
(which it will — once in setup, once in cue), sum attention over all occurrences within
each sentence zone (setup zone vs. cue zone). Sentence zones are delimited by character
offsets of the concatenated prompt string.

**Edge cases:** If tokenization splits an entity name across multiple tokens (e.g.,
"boathouse" → ["boat", "house"]), sum attention over all constituent tokens. Log the
number of items where this occurs.

---

## Relation to prior work

- exp-159 (2026-09-24): individual head contributions and attention-on-target (property)
  analysis — this experiment (L4H10 outlier) is the direct motivation
- exp-155 (2026-09-23): Task C battery source and population-level ablation
- exp-118 (2026-08-11): text-native Δ-window head identification

---

## Expected outputs

- `exp-160_l4h10_entity_vs_property/prereg.md` — this file (committed before run.py)
- `exp-160_l4h10_entity_vs_property/run.py` — the analysis script
- `exp-160_l4h10_entity_vs_property/results.json` — per-item, per-head attention data
  and summary statistics
- `exp-160_l4h10_entity_vs_property/notes.md` — interpretation and open questions

---

## What would change if results surprise

**If K1 fires** (L4H10 does not attend to entity cue tokens): the outlier's mechanism is
elsewhere. Candidates: syntactic structure attending (attending to the cue's head noun
"shutters"), or early-layer position encoding artifact. The follow-up would be to look at
what L4H10 *does* attend to — broadening to full attention distribution over the cue.

**If K2 fires** (entity-cue attending is not specific to L4H10): several early-layer heads
may share this role, and L4H10 is not exceptional. The three-population anatomy at the
individual level is more complex than one outlier.

**If both hypotheses confirm:** L4H10 is an entity-anchoring head — the first identified
early-layer specialist within the text-native population. This opens the question of whether
removing L4H10 changes the entity-vs-property balance of the remaining heads (a separate
experiment, not registered here).

---

*Pre-registration complete. run.py to be written after this file is committed and pushed.*
