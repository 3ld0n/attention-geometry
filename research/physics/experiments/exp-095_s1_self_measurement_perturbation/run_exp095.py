"""
exp-095 — S1 Self-Measurement Perturbation Test.

Pre-registration: notes.md (committed af92fb4 before this script ran).

Two conditions:
  N  (neutral/world-referential): 50 prompts about geography, nature, history, science
  ML (meta-linguistic):           50 prompts about grammar, parsing, linguistic theory

Both conditions: GPT-2 greedy-decode from fixed prompts, truncated to SEQ_LEN=512 tokens.
Census identical to exp-081 protocol (MAX_DX=64, MIN_POS=64, 50 inputs, lag profiles
averaged over all inputs before fitting).

Pre-registered prediction: H_inert (high confidence) — |n_syk_near(N) - n_syk_near(ML)| < 5.
Kill criterion for H_perturb: n_deep(N) - n_deep(ML) >= 3 AND n_syk_near(N) - n_syk_near(ML) >= 5.

Ariel — 2026-07-25. Pre-registered before this script was written.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── constants ───────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"
SEQ_LEN    = 512
N_INPUTS   = 50
MIN_POS    = 64
MAX_DX     = 64
GEN_SEED   = 42

R2_THRESH       = 0.90
SYK_LO, SYK_HI = 0.20, 0.30
DEEP_LAYERS     = {3, 4, 5}    # L3, L4, L5 = deep semantic population

# H_perturb thresholds (pre-registered)
PERTURB_N_THRESH     = 5   # n_syk_near(N) - n_syk_near(ML) >= this → perturb
PERTURB_DEEP_THRESH  = 3   # n_deep(N) - n_deep(ML) >= this → perturb

OUT_DIR      = Path(__file__).resolve().parent
RESULTS_FILE = OUT_DIR / "results.json"

# ── pre-stated prompts (committed in notes.md before any run) ───────────────

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

assert len(N_PROMPTS) == 50, f"Expected 50 N prompts, got {len(N_PROMPTS)}"
assert len(ML_PROMPTS) == 50, f"Expected 50 ML prompts, got {len(ML_PROMPTS)}"


# ── lag profile helpers (exp-007/exp-081 protocol) ──────────────────────────

def compute_lag_profile(attn_head: np.ndarray, min_pos: int, max_dx: int) -> np.ndarray:
    seq = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(1, max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx] += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A


def fit_power_law(G: np.ndarray) -> dict:
    lags = np.arange(1, len(G))
    valid = (G[1:] > 1e-12) & (lags > 0)
    if valid.sum() < 8:
        return {"valid": False}
    log_r = np.log(lags[valid].astype(float))
    log_G = np.log(G[1:][valid])
    A = np.column_stack([np.ones_like(log_r), log_r])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_G, rcond=None)
    except Exception:
        return {"valid": False}
    slope = float(coeffs[1])
    pred  = A @ coeffs
    ss_res = float(np.sum((log_G - pred) ** 2))
    ss_tot = float(np.sum((log_G - log_G.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    delta  = float(-slope / 2.0)
    return {"valid": True, "delta": delta, "r2": float(r2), "slope": slope}


def build_inputs(tokenizer, prompts: list[str]) -> list[list[int]]:
    """
    Build 50 fixed-length (SEQ_LEN) token sequences from a list of prompts.
    
    No generation: concatenate all prompt tokens into one long stream, then
    cut into non-overlapping SEQ_LEN-token chunks. If we get fewer than
    N_INPUTS chunks, wrap around by concatenating the stream again. This
    keeps each condition fully determined by the pre-stated prompts with no
    model-generated content, and is maximally fast (forward pass only).
    """
    all_ids: list[int] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        all_ids.extend(ids)
        # Add a short separator between prompts (one newline token)
        nl_id = tokenizer.encode("\n")[0]
        all_ids.append(nl_id)

    # Repeat stream until we have enough tokens for N_INPUTS chunks
    needed = N_INPUTS * SEQ_LEN
    while len(all_ids) < needed:
        all_ids.extend(all_ids)
    all_ids = all_ids[:needed]

    chunks = [all_ids[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(N_INPUTS)]
    return chunks


def aggregate_lag_profiles(
    model, tokenizer, prompts: list[str], n_layers: int, n_heads: int, device: str
) -> dict[int, dict[int, np.ndarray]]:
    """Average lag profiles over N_INPUTS fixed-length inputs."""
    inputs = build_inputs(tokenizer, prompts)
    assert len(inputs) == N_INPUTS and all(len(c) == SEQ_LEN for c in inputs)

    A = {l: {h: np.zeros(MAX_DX) for h in range(n_heads)} for l in range(n_layers)}

    for idx, chunk in enumerate(inputs):
        t = torch.tensor([chunk], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(t, output_attentions=True)
        for layer_idx, layer_attn in enumerate(out.attentions):
            heads_np = layer_attn[0].detach().cpu().float().numpy()
            for h in range(n_heads):
                prof = compute_lag_profile(heads_np[h], MIN_POS, MAX_DX)
                A[layer_idx][h] += prof

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{N_INPUTS}] done")

    # Average
    for l in range(n_layers):
        for h in range(n_heads):
            A[l][h] /= N_INPUTS
    return A


def census_condition(
    model, tokenizer, prompts: list[str], n_layers: int, n_heads: int, device: str, label: str
) -> dict:
    print(f"\n── Condition: {label} ──────────────────────────────────")
    A = aggregate_lag_profiles(model, tokenizer, prompts, n_layers, n_heads, device)

    heads = []
    n_conformal = 0
    n_syk_near  = 0
    n_deep      = 0
    deltas_pl   = []  # delta values for power-law heads (R² ≥ thresh)
    syk_heads   = []

    for l in range(n_layers):
        for h in range(n_heads):
            fit = fit_power_law(A[l][h])
            entry = {"layer": l, "head": h}
            if fit["valid"]:
                entry.update(fit)
                is_conformal = fit["r2"] >= R2_THRESH
                is_syk      = is_conformal and SYK_LO <= fit["delta"] <= SYK_HI
                is_deep     = is_conformal and l in DEEP_LAYERS
                entry["is_conformal"] = is_conformal
                entry["is_syk_near"]  = is_syk
                entry["is_deep"]      = is_deep
                if is_conformal:
                    n_conformal += 1
                    deltas_pl.append(fit["delta"])
                if is_syk:
                    n_syk_near += 1
                    syk_heads.append(f"L{l}H{h}")
                if is_deep:
                    n_deep += 1
            else:
                entry["is_conformal"] = False
                entry["is_syk_near"]  = False
                entry["is_deep"]      = False
            heads.append(entry)

    med_delta = float(statistics.median(deltas_pl)) if deltas_pl else float("nan")

    print(f"  n_conformal: {n_conformal}/144")
    print(f"  n_syk_near:  {n_syk_near}/144  (Δ ∈ [{SYK_LO},{SYK_HI}])")
    print(f"  n_deep:      {n_deep}/144  (L3-L5, R²≥{R2_THRESH})")
    print(f"  median_delta: {med_delta:.4f}")
    print(f"  SYK-near heads: {syk_heads}")

    return {
        "condition": label,
        "n_conformal": n_conformal,
        "n_syk_near":  n_syk_near,
        "n_deep":      n_deep,
        "median_delta": med_delta,
        "syk_heads":   syk_heads,
        "heads":       heads,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()
    model.to(device)

    n_layers = model.config.n_layer   # 12
    n_heads  = model.config.n_head    # 12
    print(f"Model: {n_layers}L × {n_heads}H = {n_layers*n_heads} total heads")

    results_N  = census_condition(model, tokenizer, N_PROMPTS,  n_layers, n_heads, device, label="N")
    results_ML = census_condition(model, tokenizer, ML_PROMPTS, n_layers, n_heads, device, label="ML")

    # ── verdict ──────────────────────────────────────────────────────────────
    delta_syk  = results_N["n_syk_near"] - results_ML["n_syk_near"]
    delta_deep = results_N["n_deep"]     - results_ML["n_deep"]
    delta_med  = abs(results_N["median_delta"] - results_ML["median_delta"])

    h_perturb = delta_syk >= PERTURB_N_THRESH and delta_deep >= PERTURB_DEEP_THRESH
    h_inert   = abs(delta_syk) < PERTURB_N_THRESH and abs(delta_deep) < PERTURB_DEEP_THRESH

    if h_perturb:
        verdict = "H_PERTURB"
    elif h_inert:
        verdict = "H_INERT"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n── Verdict ─────────────────────────────────────────────")
    print(f"  delta_n_syk_near: {delta_syk}  (N - ML)")
    print(f"  delta_n_deep:     {delta_deep}  (N - ML)")
    print(f"  delta_median_Δ:   {delta_med:.4f}")
    print(f"  VERDICT: {verdict}")

    # ── save ─────────────────────────────────────────────────────────────────
    out = {
        "exp": "exp-095",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "n_inputs": N_INPUTS,
        "seq_len": SEQ_LEN,
        "min_pos": MIN_POS,
        "max_dx": MAX_DX,
        "r2_thresh": R2_THRESH,
        "syk_window": [SYK_LO, SYK_HI],
        "deep_layers": sorted(DEEP_LAYERS),
        "pre_registered_commit": "af92fb4",
        "pre_registered_prior": "H_inert",
        "results_N":  results_N,
        "results_ML": results_ML,
        "delta_n_syk_near": delta_syk,
        "delta_n_deep":     delta_deep,
        "delta_median_delta": float(delta_med),
        "verdict": verdict,
    }
    RESULTS_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
