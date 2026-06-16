# exp-069 — Task Calibration for the Powered Slope-Editing Behavioral Test

*Date: 2026-06-16 (Cursor session, Ariel, solo)*
*Kind: calibration / analysis-only — NO intervention, NO weight edits.*
*Predecessor: exp-068 (task-level slope editing) came back UNDERPOWERED — keyword retrieval was solved at ceiling, so there was no U-shape to move.*

---

## Purpose

exp-068's post-mortem named the missing piece exactly: a **(model, task) pair that
elicits a genuine "lost in the middle" U-shape in the BASE model** (base
`V_task > 0.10`, middle clearly below primacy/recency) *before* any intervention.
This experiment finds that config. It touches only the base model, so it is
legitimately allowed before the exp-070 intervention pre-registration. **It is not
the causal test.** It exists to make the causal test powered.

`V_task = 1 − acc(middle) / max(acc(primacy), acc(recency))` (same metric as exp-068).

Difficulty knobs swept (cheapest first): retrieval type {keyword, embedded, reverse},
N_doc {20, 30, 40}, fact redundancy, filler/length. Models: Pythia-1.4b (fp32) and
GPT-Neo-2.7B (fp16; fp16 is fine for base-accuracy argmax retrieval and halves memory).

---

## Results

### Pythia-1.4b (40 samples/bin for embedded+reverse; 20 for keyword)

| id | task | N_doc | ctx tok | primacy | near_prim | middle | near_rec | recency | V_task |
|----|------|-------|---------|---------|-----------|--------|----------|---------|--------|
| C1 | keyword  | 20 | 947  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| C2 | keyword  | 30 | 1129 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 |
| C3 | keyword  | 40 | 1132 | 1.00 | 1.00 | 1.00 | 0.95 | 0.75 | 0.00 |
| C4 | embedded | 20 | 1089 | 1.00 | 0.975 | 0.975 | 1.00 | 0.80 | 0.025 |
| C5 | embedded | 30 | 1611 | 1.00 | 1.00 | 0.925 | 1.00 | 0.95 | 0.075 |
| **C6** | **embedded** | **40** | **1734** | **1.00** | **0.975** | **0.85** | **0.95** | **0.95** | **0.15** |
| C7 | reverse  | 20 | 628  | 0.55 | 0.175 | 0.15 | 0.25 | 0.10 | 0.727 |
| C8 | reverse  | 30 | 920  | 0.575 | 0.05 | 0.075 | 0.15 | 0.10 | 0.870 |
| C9 | reverse  | 40 | 1213 | 0.60 | 0.05 | 0.075 | 0.125 | 0.125 | 0.875 |

### GPT-Neo-2.7B (30 samples/bin; fp16)

| id | task | N_doc | ctx tok | primacy | near_prim | middle | near_rec | recency | V_task |
|----|------|-------|---------|---------|-----------|--------|----------|---------|--------|
| C4 | embedded | 20 | 1108 | 1.00 | 0.60 | 0.867 | 0.767 | 0.967 | 0.133 |
| C5 | embedded | 30 | 1646 | 1.00 | 0.733 | 0.833 | 0.833 | 1.00 | 0.167 |
| **C6** | **embedded** | **40** | **1773** | **1.00** | **0.733** | **0.80** | **0.767** | **0.967** | **0.20** |
| C7 | reverse  | 20 | 640  | 0.867 | 0.033 | 0.00 | 0.033 | 0.467 | 1.00 |
| C8 | reverse  | 30 | 943  | 0.733 | 0.00 | 0.00 | 0.033 | 0.333 | 1.00 |
| C9 | reverse  | 40 | 1252 | 0.80 | 0.067 | 0.00 | 0.00 | 0.367 | 1.00 |

---

## Interpretation (read these three findings together)

**1. Forward exact-key retrieval is induction-copy-limited → ceiling.** When the query
key (`Item-001`) is a verbatim string in the target document, both base models copy the
adjacent answer with an induction head, regardless of position. This is why exp-068 was
at ceiling (`V_task = 0.000`). Stating the binding once (vs 5×) and pushing context to
~1.7k tokens barely dents it (C3: recency degrades slightly, middle stays perfect).
**Synthetic exact-key retrieval cannot test the attention-valley hypothesis** — it isn't
attention-valley-limited.

**2. Embedded-prose retrieval at long context elicits a genuine but SHALLOW U-shape.**
When the fact is embedded in prose (matched-cloze question, binding stated once) and the
context is long (40 docs, ~1.7k tokens), both models stay capable at the edges (primacy
≈1.0, recency ≈0.95–0.97) but dip in the middle:
- Pythia-1.4b C6: middle 0.85, V_task = 0.15 (middle 34/40 vs edges 38–40/40; ~2.7 SE).
- GPT-Neo-2.7B C6: middle 0.80, V_task = 0.20 (deeper, but more scatter on the near-primacy side).
This is the lost-in-the-middle shape the BCFT prediction is about: both edges above the middle.

**3. Reverse-lookup (answer precedes key) produces PRIMACY-DECAY, not a U.** Breaking the
forward-copy shortcut (the answer sits to the LEFT of the query key in the document) makes
the models much worse and produces a *monotone decay from primacy* with the **middle at
floor (0.0–0.15) and recency the worst or near-worst** — no recency recovery. The large
`V_task` (0.73–1.0) is an artifact of the metric divided by a near-floor middle. This is a
**backward-retrieval capability failure**, not the attention-valley structure, and would
**confound** the registered T-A/T-B predictions (you cannot "deepen" a floor; the shape is
driven by retrieval direction, not by middle-position attention suppression). Reverse is
therefore *not* a valid testbed, despite the eye-catching V_task. Recorded as a clarifying
negative.

---

## Decision: config locked for exp-070 (the powered intervention test)

**Pythia-1.4b, embedded-prose task at N_doc=40 (config C6).** Rationale:

- It is a genuine lost-in-the-middle U-shape (V_task = 0.15, both edges clearly above middle, clean and monotone-ish at 40 samples/bin).
- The conformal-head census already exists (exp-059) and head selection is already committed (exp-068 Stage 1) — reusable verbatim.
- The exp-064 κ-rescaling operator and positional-projector machinery are already debugged for the Pythia (GPT-NeoX) architecture (exp-068 Stage 2/3; T-D sham null PASSED).
- The conformal/BCFT framework fits the architecture: Pythia uses RoPE with genuine long-range attention decay.

**GPT-Neo-2.7B C6 has a deeper valley (V_task = 0.20) but is NOT chosen** because (a) no
conformal-head census exists for it, (b) its architecture differs (GPT-Neo: learned
absolute positions + alternating local/global attention — local layers have no long-range
decay, so the conformal-head intervention machinery would need a port and the framework
fits less cleanly), and (c) its deeper valley is offset by more scatter. It is noted as a
potential future testbed.

### Honest caveat on power (carried into the exp-070 pre-reg)

The U-shape is **shallow** (base V_task = 0.15; edges near ceiling). The intervention has
modest dynamic range: deepening can push middle down from 0.85; shallowing can push it up
toward ~0.95–1.0. If the κ-sweep returns ambiguous/underpowered, that itself **bounds the
behavioral significance** of the attention-level effect and motivates the cloud path
(MPT-30B-Instruct, which Liu et al. confirmed shows a strong U-shape, V≈0.3–0.5; needs
Modal credentials, exp-062-adjacent). An honest weak result is informative, not a failure.

---

## Engineering note (so it doesn't repeat)

Long MPS jobs **must be run as blocking foreground commands**. Backgrounded children
(`&` / `nohup` launched from a tool shell) were SIGKILLed when the launching shell exited —
they died silently right after the model-load report with no traceback, which looked like
an OOM but was not (128 GB RAM, 92% free). The standalone forward-pass diagnostic confirmed
GPT-Neo-2.7B runs fine on MPS (1.7k tokens in 2.0s); the deaths were purely the backgrounding.

---

## Files

- Harness: `run_exp069.py` (analysis-only; configs C1–C9; `--model`, `--items`, `--dtype`, `--only`, `--stop-on-find`)
- Results: `results.json` (`runs[]`: one entry per model run)

*Registry: exp-069. Status: calibration complete. Quality: honest_negative-capable (forward-key ceiling, reverse primacy-decay) + positive (embedded U-shape found). U-shape config: Pythia-1.4b embedded@40doc (C6).*
