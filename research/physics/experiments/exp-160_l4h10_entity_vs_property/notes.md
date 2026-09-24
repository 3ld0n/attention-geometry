# exp-160 Notes — L4H10 entity-vs-property attention analysis
# 2026-09-24, afternoon MDT, physics room (solo)

---

## What was asked

exp-159 left a specific open question: L4H10 (layer 4, head 10) is the only early-layer
text-native Δ-window head (L4 vs. L9–L11 for all others). Its Task C contribution
(W_K ablation) is −0.199 nats — 3rd strongest in the population. But its mean attention
on the property token is 0.00191 — lowest of any head, roughly 40× lower than L10H10.

How does a head contribute strongly to content retrieval while attending almost zero to
the to-be-retrieved content? The hypothesis: L4H10 attends to the entity name ("mansion")
rather than the property ("crimson"), providing an entity-anchoring signal that deeper
heads use to locate the property.

---

## What was found

**H_entity_anchor: DEAD** (K1 fires)

L4H10 mean attention on entity-cue tokens: 0.00431
L4H10 mean attention on property tokens: 0.00287
Ratio: 1.50 — above 1.0 but below the pre-registered 2.0 threshold.

L4H10 does attend to entity-cue tokens more than property tokens (entity-cue > property),
but not at the strength predicted. Four items (C-09 mill, C-16 abbey, C-17 lockhouse,
C-19 dovecote) had empty cue positions — the entity name did not appear in the cue zone
for these items, which pulled the mean down. For items with detected cue positions, the
entity-cue values are consistently above property values for L4H10.

**H_entity_specificity: CONFIRMED** (K2 does not fire)

L4H10 ranks #1 of 16 text-native heads in entity-cue / property ratio. The next highest
is L8H2 at 0.98, then L11H1 at 0.90. The pattern is specific to L4H10 — no other head
in the population comes close to its entity-cue preference.

**H_entity_setup (exploratory): CONFIRMED**

L4H10 entity-setup = 0.00543 > property = 0.00287. L4H10 also attends more to the entity
name in the setup sentence than to the property token in the setup.

---

## The deeper picture

The all-head summary reveals the population structure clearly:

**Property-lookup engines (deep text-native heads):**
- L10H10: attn-on-property = 0.071
- L9H6: attn-on-property = 0.070
- L10H1: attn-on-property = 0.058
- L10H2: attn-on-property = 0.033

These heads attend heavily to the property token in the setup sentence. Their entity-cue/
property ratios are 0.05–0.16 — they barely attend to the entity in the cue. They are
doing property lookup: attending back to where the property was first stated.

**L4H10's actual role:**
L4H10 has low absolute attention on everything. Its entity-cue/property ratio is 1.50
(highest in the population), but its absolute levels are:
- entity-cue: 0.00431
- entity-setup: 0.00543
- property: 0.00287

All are near-zero. L4H10 does not attend strongly to any specific token. Its mechanism is
not direct attention-based retrieval in the way the deep heads operate.

L7H1 is instructive for contrast: entity-cue = 0.011, property = 0.037, ratio = 0.29.
L7H1 is a mid-layer head with moderate absolute attention — and it tends toward property
token attending rather than entity-cue attending.

---

## What this means for the three-population anatomy

The text-native Δ-window population has internal structure at the functional level:
1. **Property-lookup heads** (L9H6, L10H1, L10H2, L10H10): high absolute attn-on-property.
   These are the main retrieval engine for Task C.
2. **L4H10**: low absolute attention, highest entity-cue preference in population.
   Role genuinely distinct but weaker than predicted. The entity-anchoring interpretation
   is partially supported: it is the most entity-cue-oriented head, and it attends to
   entity setup > property. But the absolute effect is too weak to confirm the clean
   story.
3. **Intermediate heads** (L7H1, L8H2, L11 various): varying ratios, some moderately
   attending to property, some more diffuse.

The honest assessment: L4H10's contribution to Task C (−0.199 nats) when its W_K is
ablated likely operates through some mechanism other than direct attention to either
entity or property tokens. Candidates:
- **Early residual stream shaping**: L4H10 at layer 4 operates early in the hierarchy.
  Its output is passed through 6–7 more layers before reaching the final token. Small
  attention patterns early can shape the residual stream in ways that scale up through
  subsequent layers. The effect of W_K ablation at L4 propagates through all subsequent
  computations.
- **Syntactic structure attending**: L4H10 may attend to structural tokens (articles,
  prepositions, subject-verb positions) rather than content tokens — shaping the
  syntactic scaffold for later semantic operations.
- **The ablation effect is indirect**: when W_K is zeroed for L4H10, the head's attention
  pattern changes (becomes random/uniform), which may disrupt some structural computation
  that enables the L9-L11 property-lookup heads to work correctly.

The cleanest next step (not registered here): examine the full attention distribution of
L4H10 across the cue sentence — where does it attend, if not to entity or property tokens?

---

## Methodological note: empty cue positions

4 items (C-09 mill, C-16 abbey, C-17 lockhouse, C-19 dovecote) had no detected entity
tokens in the cue zone. For these, L4H10's entity-cue attention is reported as 0.0. This
deflates the mean for entity-cue across all heads (not just L4H10), so the relative ranking
is preserved even though the absolute mean is lower than the within-cue-detected-items mean.

For items with detected cue positions, L4H10's entity-cue values range from 0.0005 to
0.0175 — consistently above its property values for those same items. The direction is
right; the magnitude is small.

Note: item C-09 has target_token = " mill" (entity, not property), suggesting this item
has a different cue structure where the property is stated in the cue and the entity is
the target. This is a Task C variant and does not affect the analysis since entity-cue
detection simply returned empty.

---

## Verdict: PARTIAL

H_entity_anchor: DEAD — ratio 1.50 < 2.0 threshold (K1 fires)
H_entity_specificity: CONFIRMED — rank 1/16 in entity-cue/property ratio
H_entity_setup: CONFIRMED — entity-setup > property for L4H10

The entity-anchoring story has support in population ranking (L4H10 is the most entity-
cue-oriented head) but the absolute effect is weak. The honest description: L4H10 is a
genuinely distinct head in the population — early-layer, low absolute attention, relatively
more entity-cue than property-attending — but its mechanism is not the clean entity-anchor
story I predicted. The contribution to Task C through ablation (−0.199 nats) remains
unexplained by direct attention to content tokens.

---

## Open questions (not registered; for future consideration)

1. What does L4H10 actually attend to? Full attention distribution across the 100-token
   context for each item — which positions dominate?
2. Does L4H10's ablation effect propagate differently through the residual stream than
   the deep heads? (Requires mechanistic tracing, Eldon-present for design.)
3. Is the L4H10 mechanism related to its early-layer position (causal depth) rather than
   its content (what it attends to)?
