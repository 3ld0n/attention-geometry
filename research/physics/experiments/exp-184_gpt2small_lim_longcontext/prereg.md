# Pre-registration: exp-184
# GPT-2 small text-native Δ-window heads — lost-in-the-middle at long contexts
# Date: 2026-11-19
# Room: Physics (solo)
# Registered BEFORE run.py is written

---

## Background

exp-159 Instrument 3 (2026-09-24) ran an initial LiM probe for GPT-2 small text-native
Δ-window heads. Result: K4 fired (no lost-in-the-middle effect), replaced by a recency
gradient (+0.155 nats for setup near end vs. start). The exp-159 notes identified the
reason: items were ~80-90 tokens in a 1024-token window — far shorter than the multi-
document contexts in which Liu et al. (2023) observed LiM. At short contexts, attention
distributions are not compressed enough for a U-shaped curve to form.

This experiment extends that probe to contexts approaching GPT-2 small's full window
(~900 tokens of context + ~100-token query). The question: does LiM emerge at longer
contexts, and do the text-native Δ-window heads contribute differentially by target
position?

This is the LiM instrument from the functional characterization task
(task:functionally-characterize-the-text-native-semantic-population-...). exp-159
Instrument 3 was a preliminary probe; this is the proper test at the intended scale.

---

## Population under study

The 16 text-native Δ-window heads in GPT-2 small identified in exp-118 (WikiText-native
census). For the ablation instrument, the top-4 contributors from exp-159 Instrument 1
are used: L10H10 (ΔP_C = −0.355), L9H6 (−0.260), L4H10 (−0.199), L7H1 (−0.157). These
carried >70% of the population-level Task C signal. L9H4 (anomalous positive head) is
excluded from ablation.

---

## Context construction protocol

Multi-document retrieval context:
- 10 passage blocks, each following a template:
  "About [NAME]: [NAME] holds the position of [ROLE] at [COMPANY], based in [CITY].
   [NAME] joined [COMPANY] in [YEAR] and reports to [MANAGER]. [COMPANY]'s main
   product is [PRODUCT]."
- Each block is approximately 50-55 tokens (GPT-2 tokenization).
- 10 blocks = ~500-550 tokens. A query block adds ~20 tokens. Total ~520-570 tokens.
  (GPT-2 small context = 1024 tokens; this targets roughly half the window for clean
  signal — if the model degrades completely at longer contexts, K1 fires.)
- One block is the TARGET (contains the queried information); the other 9 are distractors
  with distinct names, companies, and facts.

Target positions:
- **Early (position 1):** target is block 1, distractors fill blocks 2–10.
- **Middle (position 5):** target is block 5, distractors fill blocks 1–4, 6–10.
- **Late (position 10):** target is block 10, distractors fill blocks 1–9.

Query format (after all 10 blocks):
  "Question: What company does [NAME] work at? Answer: [NAME] works at"
  → model completes with [COMPANY] token.

30 target identities (distinct names and companies), each tested at all 3 positions,
giving 90 forward passes for the baseline instrument. If fewer than 20 identities
produce above-floor baseline log-probs (K1 screen), reduce to 20 identities.

---

## Instrument 1 — Position sensitivity of baseline model

**Protocol:** Run 90 forward passes (30 identities × 3 positions). Record log-probability
of the first token of [COMPANY] at each position.

**Primary hypothesis H_lim:**
U-shaped performance: log-prob(middle) < log-prob(early) AND log-prob(middle) < log-prob(late),
with difference ≥ 0.15 nats.
*Declared confidence: LOW (25%).* GPT-2 small was not trained for multi-document retrieval
at this scale. Recency gradient is the more likely pattern.

**Alternative hypothesis H_recency:**
Monotone recency gradient: log-prob(late) > log-prob(middle) > log-prob(early), or
log-prob(late) > log-prob(early) with no U-shape.
*Declared confidence: HIGH (65%).* Extends exp-159's K4 finding to longer contexts.

**Kill condition K1:** Mean log-prob across positions ≤ −10 nats AND variance across
positions < 0.10 nats — retrieval has completely collapsed; no position sensitivity
measurable. If K1 fires, neither H_lim nor H_recency can be assessed.

---

## Instrument 2 — Differential tn head contribution by position

**Protocol:** For each of the 30 identities and 3 positions, run with top-4 tn heads
ablated (W_K=0, same protocol as exp-159 Instrument 1). Record ΔP = log-prob(ablated)
− log-prob(baseline) for each position.

**Hypothesis H_tn_differential:**
The ablation effect differs by target position: mean |ΔP(late)| − mean |ΔP(early)| > 0.10
nats AND this difference is in the direction consistent with the dominant position effect
(i.e., heads contribute more in the retrieval condition where the model performs better).
*Declared confidence: MODERATE-LOW (35%).* If the tn heads' function is primarily
content-retrieval without position awareness (consistent with exp-159 Instrument 1
showing content-selectivity), their contribution may be position-uniform.

**Kill condition K2:** max position-wise difference in mean |ΔP| < 0.05 nats — the
ablation effect is position-uniform; tn heads are not position-sensitive retrievers.

---

## Stopping rules and allowed amendments

1. If K1 fires, the experiment is recorded as K1-failure and the conclusion is:
   GPT-2 small cannot support coherent multi-document retrieval at this context length;
   the LiM functional characterization of the semantic population is complete at exp-159's
   finding (K4: recency gradient at short contexts; K1: retrieval collapse at long contexts).

2. If fewer than 10 of 30 identities produce any retrieval signal (log-prob > −15 nats at
   any position), item count may be extended to 60 identities. This amendment is
   pre-registered here.

---

## Declared prior on the full finding

Expected outcome: H_recency fires (or K1); K2 fires (tn heads are position-uniform).
The functional characterization of GPT-2 small tn heads will be: content-selective
retrievers (exp-159 Instruments 1–2) with a recency preference (exp-159 Instrument 3,
this experiment), no position-sensitive contribution to LiM.

If H_lim fires against expectation: this would tie the geometric census to a specific
long-range integration failure mode; the census would be identifying heads involved
in LiM-sensitive retrieval. This would require surfacing to OVERVIEW.md and revising
the interpretation of the two-population anatomy.

---

## Relation to prior work

- exp-118 (Aug 11): identifies the 16 tn Δ-window heads in GPT-2 small
- exp-155 (Sep 23): population-level Task C double dissociation (content retrieval)
- exp-159 (Sep 24): individual-level characterization; Instrument 3 = short-context
  LiM probe (K4: recency gradient, contexts ~80-90 tokens)
- **exp-184 (this):** long-context LiM proper test (~500-570 tokens; near GPT-2 small's window)

---

*Pre-registration complete. run.py to be written after this file is committed and pushed.*
