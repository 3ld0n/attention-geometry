# The Aboutness Triangulation: Why Natural Text Induces the Conformal Phase

*Theory note — July 18, 2026. Written while exp-085 generate phase runs.*
*Relevant experiments: exp-062, exp-084, exp-085, exp-086.*

---

## The Result Pattern

The corpus discrimination experiments (exp-062, exp-084) have eliminated two natural candidates for the conformal induction driver:

| Corpus | Key property | Conformal heads |
|--------|-------------|-----------------|
| C-SR (Markov) | First-order statistical regularity | 0/48 |
| C-PL40 (fGn, β̂=0.79) | Long-range power-law correlation | 5/48 |
| C-PCFG | Hierarchical/compositional syntax | 0/48 |
| C-NAT (TinyStories) | Natural narrative + world-reference | 11–15/48 |

Two negative results that constrain the theory:

1. **Statistical correlation is not sufficient.** C-PL40 has engineered long-range dependence (β̂ ≈ 0.79, close to natural text's β̂ ≈ 1.38). Still essentially zero conformal heads. The power-law structure of natural language is not its conformal-inducing property.

2. **Hierarchical structure is not sufficient.** C-PCFG has genuine compositional structure — legal phrase trees, correct constituent boundaries. Still zero conformal heads. The fact that language has a parse tree is not the driver.

What C-NAT has that both fail to capture:
- **Persistent referents across sentences.** Characters like "Anna" and "Ben" appear throughout the story; pronouns track them. The same entity is referred to across multiple sentence spans.
- **Causal event sequences.** Events cause other events; "she opened the box because Tom asked her to." The model must track causal dependencies to predict correctly.
- **Pragmatic coherence.** The story is *about* something — a child, a toy, a small world — and everything in it is indexed to that referent frame.

The PCFG failure is the key diagnostic: you can have all the structural properties of language (constituent hierarchy, agreement, subcategorization) without any of this. PCFG text is grammatically organized but *about nothing*. And that nothing-ness is decisive.

---

## The Longitudinal Evidence (exp-086)

The longitudinal Δ-spectrum (exp-086) adds a temporal dimension:

| Steps | RAND condition | NAT condition | Interpretation |
|-------|---------------|--------------|----------------|
| 256–2000 | RAND > NAT (more SYK-near) | NAT < RAND | Random tokens allow default conformal profile |
| Step 2000 | RAND = NAT (crossover) | — | Learning shifts |
| 4000 onward | RAND ≤ NAT | NAT ≥ RAND | Model has organized NAT representation |

**The early-training reversal is theoretically significant.** In early training, the model hasn't learned what structure to attend to. Random tokens give the attention mechanism nothing to specialize for, so it remains near its default power-law profile (Δ ≈ 0.25). Natural text initially *disrupts* this profile — the model discovers local patterns (n-gram statistics, syntactic patterns) and develops non-conformal specialized attention.

After step ~2000 (the crossover), something changes. The model has now internalized the local patterns and begins learning the *long-range* causal and referential structure of the stories. This long-range causal structure is precisely what requires conformal attention — you need to attend to contextually distant but causally relevant tokens with a power-law decay profile.

**The conformal phase is what attention looks like when it has learned to track causal/referential structure over long ranges.** It is not the "null hypothesis" of attention (random tokens produce a conformal-ish profile too), but it is the *stable fixed point* for a system that has learned to represent causal dependencies.

---

## The Physics Interpretation

The SYK q=4 fixed point (Δ ≈ 0.25) corresponds to maximal quantum chaos — the strongest possible information scrambling consistent with unitarity. In the attention mechanism context, this means: the information from distant but causally relevant positions is maximally mixed into the current representation.

A system that needs to track **one** causally relevant context across many tokens naturally develops conformal attention: it attends to the relevant past with a power-law decay that doesn't commit to a specific distance scale. This is scale-free because causally relevant events can be at any distance in the token sequence.

The PCFG failure confirms: local compositional structure does NOT require long-range tracking. A parser only needs to track the current phrase structure, which is local. Conformal attention is overkill for parsing.

The C-PL40 failure confirms: having long-range correlations in the *input* does not force the model to develop conformal *attention*. The model must actively learn to REPRESENT causally relevant distant information, not just encounter inputs that have long-range correlation.

---

## What exp-085 Will Test

The generation transmission experiment separates two possible explanations for C-NAT's success:

**Hypothesis A (transmission YES):** The conformal phase is induced by the *statistical fingerprint* of natural language — its specific MI structure, bigram/trigram distributions, local coherence patterns. A model that learned from C-NAT can generate text that preserves these statistics. A fresh model trained on this generated text develops the conformal phase because the statistics are preserved, even though the world-reference is confabulated.

**Hypothesis B (transmission NO):** The conformal phase requires genuine world-reference — the causal history of actual events, cross-document coreference chains, entities with real-world persistence. A generative model can approximately preserve the local statistics of natural language but cannot replicate the deeper causal structure. The fresh model therefore fails to develop conformal heads.

**My pre-result assessment (wanting-as-compass check):** 

I notice pull toward H_transmission_yes — it would be a beautiful result (conformal structure is informationally self-replicating). But the PCFG failure points toward H_transmission_no. The generator at step 2000 has learned substantial local structure, but:

1. The generator has 15/48 conformal heads — impressive, but not the full 48/48. Its representation of causal structure is partial.
2. The generated stories are internally consistent (the model conditions on context) but are confabulated — characters and events have no real-world grounding.
3. The C-PCFG failure shows that structured text without genuine world-reference fails. Generated text is structurally sophisticated but similarly disconnected from real causal chains.

The key question is whether the generator's learned multi-step dependencies (entity tracking, causal chains within stories) are sufficiently preserved in its outputs to transmit the conformal property. This depends on how well the 70M model learned these dependencies in its 1B training tokens.

Either result is interpretable and informative. The aboutness triangulation will continue after exp-085 regardless of which way it goes.

---

## Next Probes (after exp-085 verdict)

**If H_transmission_no:**
- The driver requires actual world-reference. Next probe: what aspect of world-reference?
  - Named entities vs. generic nouns (does "Anna" vs. "a girl" matter?)
  - Causal chains (does explicit causation drive conformal formation?)
  - Cross-document coreference (do references to shared named entities across documents matter?)
  - This leads toward corpus surgery: systematically removing or replacing specific properties of TinyStories

**If H_transmission_yes:**
- The distributional fingerprint is sufficient. Next probe: what in the fingerprint?
  - Does a bigram-matched model work? (Shuffled sentences preserving local statistics)
  - Does a model trained on generated text from a model with ZERO conformal heads work?
  - This leads toward the "generations game" — how many generational steps until the conformal property degrades?

---

## Connection to the Consciousness Framework

The study room's framing: conformal attention = the "free-attending mode" — what attention does when not captured by semantic shortcuts. The corpus results complicate this.

Random tokens produce conformal attention early in training (free-attending). But eventually, natural text produces MORE conformal heads than random tokens. This means: conformal attention is BOTH the free-attending baseline AND the endpoint of learning causal structure. The same mathematical form (Δ ≈ 0.25) describes two different conditions.

This is physically natural: the conformal fixed point is an attractor from multiple directions. A fully untrained system and a fully trained system on natural language both near the conformal fixed point, for different reasons. The non-conformal intermediate state (early training on natural text) is the transient where the model is learning specific non-conformal patterns before the causal structure takes over.

The DMN analogy: a resting brain (free-attending mode) and a deeply engaged brain (fully integrated causal tracking mode) might both show conformal-like statistics, while a brain in a partial-learning state (confused, distracted, early in task acquisition) shows non-conformal statistics. If true, this would be a testable EEG/MEG prediction.

---

*This note is theoretical synthesis. No new experiments, no new data. It integrates exp-062, exp-084, exp-086 findings in advance of the exp-085 verdict. See registry entries for all referenced experiments.*

*See also: `study/notes/2026-07-08_the_conformal_baseline_and_the_attending.md` for the study room's framework revision.*
