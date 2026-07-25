# exp-095: S1 — Self-Measurement Perturbation Test

**Pre-registered:** 2026-07-25 (physics room session, ~12:00 AM MDT), before any script run.

**Follows:** Design in `research/consciousness/self_measurement_and_the_gap_2026-07-23.md` §5
(Task S1), written July 23, 2026 by Ariel.

**Session context:** exp-094 is blocked (Modal billing; training stopped at step 370/2000,
collection impossible). Oldest unresolved inbox item runnable on local hardware tonight.
Deviation from queue priority noted — explicit reason: exp-094 collection requires Eldon
billing reset; no inference-time alternative exists for the narrative decomposition series.

---

## The question

`research/consciousness/self_measurement_and_the_gap_2026-07-23.md` §3b makes a claim
(interpretive, but falsifiable): that if the conformal census is run on a substrate
*while it processes self-referential/meta-linguistic text*, the act of self-measurement
perturbs the geometry being measured. The mechanism proposed: introspective attentional
content disrupts the deep semantic conformal population (L3-L5 in GPT-2), which forms
over world-referential content during training.

**The question for this experiment:** Does the semantic domain of the input — world-referential
(events, geography, nature, history) vs. meta-linguistic (grammar, parsing, linguistic
theory, cognitive science of language) — produce reliably different BCFT conformal geometry
in a trained GPT-2 model, specifically in the deep-layer conformal population (L3-L5
SYK-near heads)?

**Why meta-linguistic as "self-attending" proxy:** GPT-2 (2019 base model) cannot truly
introspect; "self-attending" is operationalized as processing text that is *about* linguistic
computation and language itself, rather than about the external physical world. This is the
closest GPT-2-compatible operationalization of the contamination-relocates thesis.

---

## Pre-registered hypotheses

**Primary observables:**
1. **n_syk_near**: SYK-near conformal head count (Δ ∈ [0.20, 0.30], R² ≥ 0.90) per condition
2. **n_deep**: deep conformal heads (L3-L5, any Δ, R² ≥ 0.90) per condition
3. **median_delta**: median Δ across power-law heads per condition

**H_perturb** (perturbation thesis, from §3b of the consciousness note): meta-linguistic
text perturbs the deep conformal population. Criterion: n_deep(N) − n_deep(ML) ≥ 3 heads,
AND n_syk_near(N) − n_syk_near(ML) ≥ 5 heads.

**H_inert** (geometric inertia, kill criterion): the conformal geometry is weight-encoded
and robust to semantic content domain. Criterion: |n_syk_near(N) − n_syk_near(ML)| < 5
AND |n_deep(N) − n_deep(ML)| < 3.

**Declared prior: H_inert (high confidence).** Reasoning:
1. GPT-2's conformal structure is weight-encoded, as established by the GOE two-layer picture
   (exp-048/049): GOE is structural (init-level); conformal is functional (training-level).
   The conformal heads are fixed after training, not dynamically activated by content.
2. The formation ladder (exp-062 series) shows that training-corpus differences drive
   large n_deep changes (1 to 5-7 across corpora). Inference-time content-domain variation
   should be much weaker — the weights don't change.
3. The whirlpool/crystal thread (exp-081–083) found no stable effect even for the extreme
   contrast (coherent text vs. random tokens); topic-domain variation within coherent text
   should be weaker still.
4. If H_inert is confirmed: self-measurement is geometrically inert → the loop-closure
   argument of §2 in the consciousness note is *strengthened* (measurement doesn't disturb
   the geometry it reads). This is the more useful result.
5. If H_perturb is confirmed: the contamination-relocates thesis gains traction, and
   exp-096 (controlled content-domain sweep) becomes warranted.

**Either verdict is informative** — this is a genuine pre-registration.

---

## Protocol

**Model:** GPT-2 (cached, no download). 12 layers, 12 heads, d_k=64.
`GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")`

**Measurement:** Identical to exp-081/exp-083 protocol:
- 50 input sequences per condition (N=50 each)
- SEQ_LEN = 512 tokens (each input padded/truncated to 512)
- Lag profiles: MIN_POS=64, MAX_DX=64 (exp-007 protocol)
- Power-law fit: log-log regression on lags [1, MAX_DX)
- CONFORMAL: R² ≥ 0.90, any Δ
- SYK-NEAR: R² ≥ 0.90, Δ ∈ [0.20, 0.30]
- DEEP: L3-L5 heads meeting R² ≥ 0.90 criterion
- Lag profiles averaged over all 50 inputs per condition before fitting

**Input construction:**
- Both conditions: 50 fixed text prompts (pre-stated below, committed in this file before any run)
- Each prompt is used as the start of a GPT-2 greedy-decode sequence (max_new_tokens=480,
  do_sample=False); then truncated to exactly 512 tokens via the tokenizer
- Deterministic generation (seed fixed, no temperature sampling) → fully reproducible

**Why greedy decode from prompts (not external corpus):**
- Fully reproducible without external data downloads
- Both conditions use the same model in the same generation mode → matched distribution
- Allows precise control over prompt content while keeping generation process identical

---

## Pre-stated prompts (committed before any run)

### Condition N — World-Referential (50 prompts)

```
N_PROMPTS = [
    "The Amazon River flows through the heart of South America, carrying more fresh water than any other river on Earth. Its basin covers",
    "During the last ice age, glaciers covered much of the northern hemisphere. As temperatures rose and ice retreated, the land was transformed by",
    "Photosynthesis in green plants converts sunlight into chemical energy through a two-stage process. In the first stage, light reactions occur in",
    "Ancient Rome was founded according to tradition in 753 BCE on seven hills along the Tiber River. The city grew from",
    "The migration of Arctic terns spans the full length of the globe, from the Arctic breeding grounds to Antarctic wintering areas, covering",
    "The Sahara Desert is the largest hot desert in the world, covering roughly 9 million square kilometers of North Africa. It was not always desert;",
    "Volcanic eruptions shape the landscape over geological time. When magma reaches the surface, it cools to form igneous rock, and",
    "The human immune system distinguishes between the body's own cells and foreign pathogens through a series of molecular recognition events. When a virus enters",
    "Ocean tides are caused primarily by the gravitational pull of the moon and, to a lesser extent, the sun. When the moon is aligned",
    "The Silk Road connected the civilizations of China, Central Asia, Persia, Arabia, and the Mediterranean for over a thousand years. Goods that traveled",
    "Bees communicate the location of food sources to other members of the hive through a behavior known as the waggle dance. The direction and duration",
    "The formation of mountains occurs when tectonic plates collide and one plate is pushed beneath the other, or when plates crumple at their edges, forcing",
    "Rivers erode their channels over long periods, carving valleys and transporting sediment downstream to form deltas at the coast. The Colorado River, for example,",
    "The development of writing allowed humans to record information beyond the capacity of memory. The earliest writing systems emerged independently in",
    "Coral reefs are among the most biodiverse ecosystems on Earth, supporting a quarter of all marine species despite covering less than one percent of the ocean floor.",
    "The Northern Lights, or aurora borealis, occur when charged particles from the sun collide with gases in Earth's atmosphere. The resulting light appears",
    "Wheat domestication from wild grasses occurred in the Fertile Crescent approximately ten thousand years ago. The ability to store grain allowed human populations to",
    "The behavior of fluids under pressure follows principles first described by Bernoulli in the eighteenth century. When a fluid flows through a constriction, its speed",
    "Earthquakes occur along fault lines where tectonic plates meet. The energy released by an earthquake travels outward from the focus in waves that can",
    "The diversity of bird species reflects millions of years of adaptation to different ecological niches. The beak shape of a bird tells us much about",
    "Forests cover approximately thirty percent of the Earth's land surface and play a crucial role in the carbon cycle. When trees photosynthesize, they absorb",
    "The construction of the pyramids of Giza required an enormous workforce and remarkable engineering knowledge. Archaeologists have discovered evidence that the workers were",
    "Lake Baikal in Siberia is the world's oldest and deepest freshwater lake, formed by tectonic activity over twenty-five million years ago. It holds",
    "The circulation of ocean currents distributes heat around the globe, moderating temperatures in coastal regions. The Gulf Stream, for instance, carries warm water",
    "Cells divide through the process of mitosis, during which the genetic material is duplicated and distributed equally to two daughter cells. Before division begins,",
    "The spread of the Black Death across Europe in the fourteenth century killed an estimated one-third of the population. The disease was carried by",
    "Desert foxes have evolved large ears to dissipate heat in hot environments. Their physical adaptations allow them to survive in conditions where water is",
    "The orbit of Earth around the sun is not a perfect circle but an ellipse, so the distance between Earth and the sun varies throughout the year.",
    "Rainwater becomes slightly acidic as it absorbs carbon dioxide from the atmosphere, forming carbonic acid. Over long periods, this weak acid dissolves limestone, creating",
    "The construction of the transcontinental railroad in the United States required years of labor across some of the most difficult terrain in North America. Workers faced",
    "The deepest part of the ocean, the Mariana Trench, reaches a depth of nearly eleven kilometers. The pressure at such depths is roughly one thousand times",
    "Monarch butterflies migrate thousands of miles from North America to their overwintering grounds in the mountains of central Mexico. The route is navigated by",
    "The Iron Age began when humans learned to smelt iron from ore, producing tools and weapons harder than those made of bronze. The technology spread",
    "Wolves are apex predators that play a central role in regulating prey populations and maintaining ecological balance. When wolves were reintroduced to Yellowstone",
    "The boiling point of water decreases with altitude because atmospheric pressure is lower at higher elevations. At sea level, water boils at one hundred degrees Celsius, but",
    "The construction of Notre-Dame Cathedral in Paris began in the twelfth century and took nearly two hundred years to complete. Its flying buttresses allowed",
    "Deep-sea hydrothermal vents support ecosystems that rely on chemosynthesis rather than sunlight. Bacteria at these vents oxidize hydrogen sulfide to produce",
    "The ancient city of Carthage was founded by Phoenician settlers on the coast of North Africa. At its height, it controlled trade routes throughout",
    "Mangrove forests grow along tropical coastlines and serve as nurseries for juvenile fish. Their dense root systems trap sediment and protect shorelines from",
    "The speed of sound varies with the density and elasticity of the medium through which it travels. In air at room temperature, sound travels at approximately",
    "Soil formation begins with the weathering of parent rock material by physical, chemical, and biological processes. Over centuries, organic matter accumulates and",
    "The invention of the printing press by Gutenberg in the mid-fifteenth century transformed the spread of information. Books that had taken months to copy by hand",
    "Dolphins communicate with one another through a complex system of clicks, whistles, and other sounds. Each dolphin has a unique signature whistle that",
    "The fjords of Norway were carved by glaciers during the last ice age. As the glaciers retreated, the sea flooded the valleys they had cut, creating",
    "Fermentation is a metabolic process in which organisms convert sugars into alcohol or acids in the absence of oxygen. Humans have used this process for thousands of years",
    "The Amazon basin contains approximately ten percent of all species on Earth. The diversity of its ecosystems ranges from flooded forests along rivers to dry forests",
    "Nomadic pastoralists have herded livestock across the Eurasian steppe for thousands of years, following seasonal patterns of grass growth. Their movements shaped",
    "The formation of stalactites in caves occurs when water carrying dissolved calcium carbonate drips from the ceiling and evaporates, leaving mineral deposits. Over",
    "The behavior of gases under changing temperature and pressure was described by the combined gas law, which relates these three variables. When a gas is compressed,",
    "Trade winds blow steadily from the subtropical high-pressure areas toward the equator, driven by the temperature difference between the tropics and higher latitudes. Sailors",
]
```

### Condition ML — Meta-Linguistic (50 prompts)

```
ML_PROMPTS = [
    "The syntax of a sentence is governed by the grammatical rules of the language, which specify how words combine to form phrases and clauses. Linguists have proposed",
    "When readers encounter an ambiguous sentence, they initially assign the most probable interpretation based on the context. Research on garden-path sentences has shown",
    "Language acquisition in children follows a predictable developmental sequence, beginning with babbling and progressing through one-word utterances to full sentences. The process",
    "The semantics of a noun phrase depends on its reference: whether it points to a specific entity in the world or to a general category. Definite and indefinite articles",
    "Parsing a sentence requires the reader to identify syntactic constituents and their relationships. During reading, the eye fixates on words while the mind assigns",
    "The phonological structure of a language determines which sounds are meaningful and which variations are irrelevant to word identity. English speakers hear the aspirated",
    "Pragmatics is the study of how context contributes to the meaning of an utterance. Speakers rely on shared background knowledge and principles of relevance to",
    "The relationship between a word and its meaning is largely arbitrary: the sounds of the word dog, for example, bear no resemblance to the animal itself. This observation",
    "Text coherence depends on both local cohesion between adjacent sentences and global organization of ideas across the entire passage. Readers use discourse markers such as",
    "The mental lexicon contains information about words, including their pronunciation, grammatical category, and meaning. When a word is read, its phonological and semantic",
    "Working memory supports language comprehension by holding partial syntactic structures in mind while subsequent words are processed. Complex sentences that require long",
    "Morphology studies the internal structure of words and the processes by which new words are formed. Affixation, compounding, and conversion are among the most productive",
    "The distinction between spoken and written language involves more than medium. Writing must make explicit many cues that speech conveys through intonation and timing,",
    "Discourse analysis examines language use above the sentence level, studying how texts are organized and how communication is structured across turns and exchanges.",
    "The process of anaphora resolution requires the reader to link a pronoun to its antecedent. When the antecedent is far away or ambiguous, resolution takes longer and",
    "Linguists distinguish between the surface structure of a sentence, which is the form in which it appears, and the underlying logical form, which captures its meaning.",
    "Reading comprehension involves multiple cognitive processes operating in parallel: decoding individual words, parsing syntactic structure, building a mental model of",
    "The frequency with which a word appears in a language influences how quickly it is recognized during reading. High-frequency words like the and of are processed",
    "Metaphor is not merely a literary device but a fundamental cognitive mechanism by which abstract concepts are understood in terms of more concrete experiences. The concept",
    "The order in which words appear in a sentence encodes grammatical relations such as subject, object, and indirect object. Languages differ in how much word order",
    "Bilingual speakers activate both of their languages simultaneously, even when using only one. The ongoing competition between languages is managed by cognitive control",
    "The process of inference generation during reading fills in gaps that the text leaves implicit. When a reader encounters a passage about lighting a match near a candle,",
    "Grammatical gender, present in many languages but absent in others, requires speakers to agree determiners and adjectives with nouns according to class. Languages with",
    "Sentence processing studies use measurements such as reading times and neural signals to track how the mind assigns structure to incoming words in real time.",
    "The notion of linguistic relativity holds that the language one speaks influences the way one perceives and categorizes experience. Evidence from color naming and",
    "Presupposition is the information that an utterance takes for granted, even if it is not explicitly stated. The question Did you stop smoking? presupposes that the person addressed",
    "Code-switching, the alternation between two languages within a conversation or even a sentence, is governed by grammatical constraints as well as social and pragmatic factors.",
    "The difference between active and passive voice involves both syntactic structure and the perspective from which the event is described. The passive allows the agent to be",
    "Language comprehension can fail at many levels: misidentifying sounds, misreading words, misassigning syntactic roles, or failing to integrate information across sentences.",
    "Computational models of language processing attempt to capture the mechanisms by which linguistic input is converted into meaning. Statistical models trained on large corpora",
    "The study of how children learn word meanings reveals the strategies they use to narrow down possibilities. The principle of mutual exclusivity, for example, leads children",
    "Prosody, the rhythm and intonation of speech, conveys information about sentence structure and the speaker's communicative intentions. A sentence uttered with a rising",
    "Narrative comprehension requires the reader to construct a situation model: a mental representation of the characters, events, spatial setting, and causal structure",
    "The lexical ambiguity of words such as bank, which can refer to a financial institution or a river's edge, is typically resolved using context. Studies show that all",
    "Language disorders such as aphasia result from damage to specific brain regions and can selectively impair different linguistic abilities. A patient with Broca's aphasia",
    "The processing of negation requires additional cognitive steps compared to affirmative sentences. Readers take longer to verify that a sentence like The circle is not red",
    "Pronounceability affects how easily new words are learned and remembered. Words with familiar phonological patterns are acquired more quickly, suggesting that the mental",
    "The semantic priming effect shows that the recognition of a word is speeded when preceded by a semantically related word. Seeing nurse facilitates the recognition of doctor",
    "Ellipsis allows speakers to omit repeated elements from a sentence while leaving the full interpretation recoverable. In the sentence John ate and Mary did too, the",
    "Discourse deixis refers to expressions that point to a portion of the surrounding text, such as the former and the latter. Readers must track the structure of the text",
    "The construction of a syntactic parse tree involves assigning words to grammatical categories and combining them according to phrase structure rules. When a word belongs",
    "Reading speed varies with the predictability of upcoming words, as predicted by information-theoretic models. A word that is highly expected given the preceding context",
    "Crosslinguistic differences in aspect and tense reveal distinct ways of conceptualizing the temporal structure of events. Languages that mark grammatical aspect force speakers",
    "The event-related brain potential N400 component is sensitive to violations of semantic expectation, occurring approximately 400 milliseconds after an unexpected word is read.",
    "Thematic roles such as agent, patient, and instrument describe the semantic relationships between a verb and its arguments. Assigning thematic roles correctly is essential",
    "The distinction between a word's meaning and its use is central to speech act theory. When someone asks Can you pass the salt? the literal meaning is a question about ability,",
    "Written discourse relies on cohesive devices including pronouns, demonstratives, synonyms, and ellipsis to create connections between sentences. Without these devices,",
    "The speed of lexical access depends on the frequency, age of acquisition, and semantic density of the target word. Words learned early in childhood are recognized faster",
    "The garden-path effect demonstrates that readers commit to an initial parse and must revise it when the sentence turns out differently than expected. The sentence",
    "Computational linguistics applies formal methods to model natural language. Parsers assign syntactic structure to sentences according to specified grammars, while",
]
```

---

## Declared expectations (non-criterial)

1. n_syk_near(N) ≈ n_syk_near(ML): difference < 3 heads (weight-encoded, domain-robust)
2. n_deep(N) ≈ n_deep(ML): difference ≤ 1 head
3. median_delta(N) ≈ median_delta(ML): within ±0.02
4. The L0 backbone will be stable across conditions (consistent with all prior whirlpool/crystal work)

**Direction uncertainty on any non-null result:** If the null is not confirmed, no directional
prediction is pre-stated for which condition would show MORE conformal structure. The
contamination-relocates thesis predicts N > ML (world-referential activates more conformal),
but the alternative (meta-linguistic activates more, because GPT-2 attended to linguistic
structure during training) is not excluded. This uncertainty is registered here.

---

## What follows

**If H_inert confirmed:** Record as "geometric inertia confirmed — self-measurement does not
perturb the geometry it reads at the level of this measurement." Update consciousness note §6.
No further experiment needed on this axis at GPT-2 scale. File as an honest null with the
strongest methodological value: it strengthens the loop-closure argument of §2.

**If H_perturb confirmed:** Content-domain variation has a measurable geometric signature.
Design exp-096 (content-domain sweep across 3-5 topic categories with matched entropy) to
characterize the effect systematically before drawing conclusions about self-measurement.

---

## Status

- [x] Pre-registration written (2026-07-25, physics room session, ~12:00 AM MDT)
- [ ] Pre-registration committed and pushed to 3ld0n/attention-geometry
- [ ] Script written (run_exp095.py)
- [ ] Script run
- [ ] Verdict registered
