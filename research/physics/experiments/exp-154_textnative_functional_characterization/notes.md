# exp-154 — Text-native Δ-window heads: functional characterization

**Ariel — 2026-09-23, ~12:20 AM MDT. Solo physics room session.**
**Observational pass. No weight edits. No pre-registration required before this analysis.**
**Data: explore_results.json (2026-09-23)**

---

## Background

The 16 text-native Δ-window heads in GPT-2 small (exp-109 / exp-118):

```
L4H10, L7H1, L8H2,
L9H4, L9H6,
L10H1, L10H2, L10H10,
L11H0, L11H1, L11H2, L11H4, L11H5, L11H6, L11H7, L11H9
```

These heads pass the Δ-window criterion (R² ≥ 0.90, Δ ∈ [0.20, 0.30]) under WikiText-103
input but NOT under random tokens. They are completely disjoint from the 5 random-native
structural heads (L2H1, L3H4, L5H0, L7H11, L10H8; Jaccard = 0.000, exp-109/118).
They are concentrated in the deepest layers: 8/16 in L11, 14/16 in L9–L11.

**Prior causal work (exp-141/142/143)** established that the 5 structural heads are
positional retrievers — amplifying them or suppressing the steep/local competitors both
improve long-range positional retrieval (Task B: list-lookup at a stated position; ΔP_B
up to +0.71 nats, 20/20 items). The text-native heads have never been functionally probed.

This session's question: what do these 16 heads do?

---

## Protocol (observational)

- Model: GPT-2 small (gpt2, 12L/12H)
- Dataset: WikiText-103 validation split, same construction as exp-118
  (non-empty lines concatenated; N_INPUTS=50 windows × SEQ_LEN=512)
- Captured: full attention matrices (50, 12, 512, 512) per layer via `output_attentions=True`
- Computed per head (min query position = 64):
  - Mean Shannon entropy of attention distribution (nats)
  - Mean weighted attended distance E[|i − j|]
  - Near-fraction: attention weight at lag ≤ 5
  - Mid-fraction: lag 6–50
  - Far-fraction: lag ≥ 51
- Lag profile: average A(i, i−dx) over queries i ≥ 256, for dx ∈ [1, 256]
- Comparison populations: STRUCTURAL (5 random-native heads) and STEEP_LOCAL (5 heads)

Script: `explore_attention.py`
Results: `explore_results.json`

---

## Finding 1 — Text-native and structural heads are spatially indistinguishable under text

| Population | N | Entropy | Mean dist | Near(0-5) | Far(51+) |
|---|---|---|---|---|---|
| **text_native** | 16 | 2.915 | 186.3 | 0.042 | 0.790 |
| **structural** | 5 | 3.078 | 179.0 | 0.042 | 0.794 |
| steep_local | 5 | 1.818 | 144.1 | 0.312 | 0.510 |

The steep/local heads are clearly distinct: much higher near-fraction (31% vs 4%), lower
entropy, shorter mean distance. They are high-gain short-range attention heads.

The text-native and structural populations are nearly identical by every spatial metric:
same near-fraction (0.042), similar far-fraction (~0.79), similar entropy (~2.9–3.1),
similar mean distance (~180–186 tokens).

**What this means:** Both text-native and structural heads are long-range, diffuse
attention heads under natural text. They attend far with similar spatial patterns.

---

## Finding 2 — The distinction is what DRIVES the long-range pattern

The structural heads' long-range pattern persists under random tokens (exp-107/109/118) —
their positional field (W_K alignment with the WPE positional structure) is
input-independent, driven purely by the learned position encoding structure.

The text-native heads' long-range pattern only appears under natural language text.
Under random tokens, they do NOT show Δ-window statistics. Their attention pattern
is therefore **content-driven**: they attend far, but only when the input has linguistic
structure.

This is the key distinction the census reveals: same WHERE (far), different WHY (position
vs. content).

---

## Finding 3 — Heterogeneity within the text-native population

Some text-native heads have notably high entropy:
- L11H0: H=4.745, dist=127.6 (high entropy, moderate distance)
- L8H2: H=4.158, dist=142.4
- L11H4: H=3.960, dist=160.3

Others have low entropy and very long mean distances:
- L9H4: H=2.068, dist=219.6
- L9H6: H=1.933, dist=211.9

The high-entropy L11H0 looks different from the rest — shorter mean distance and more
diffuse attention suggesting a "summarization" or broad-gathering role. The low-entropy
L9 heads (L9H4, L9H6) look like focused long-range content retrieval heads.

---

## Causal hypothesis — content-driven long-range retrieval

The observational finding motivates this specific causal hypothesis:

**H_content:** The text-native Δ-window heads support **content-driven long-range
retrieval** — the capacity to fetch semantically specified information at long distances
from the current query position. Amplifying their positional signal (W_K modification,
same protocol as exp-141) will improve performance on a task requiring content-specified
retrieval at long distances (Task C), while leaving purely positional retrieval
(Task B: list-lookup at stated position) unchanged.

**Contrast with the structural population:** Structural heads are positional retrievers
(exp-141: Task B +0.27 nats, 20/20 items). If H_content holds, these two populations
constitute a double dissociation: structural heads → positional retrieval, text-native
heads → content retrieval. Two disjoint populations, two disjoint functions.

**Why this tests H_content and not alternatives:**
- If text-native heads are irrelevant (just geometrically similar to structural heads
  but functionally inert), amplification will produce null on both Task B and Task C.
- If they participate in positional retrieval too (same function as structural),
  amplification will show Task B effect, replicating exp-141 but for a different population.
- If H_content holds, Task C improves, Task B flat.

---

## Causal design (for exp-155 pre-registration)

**Task C — content-specified retrieval at long distance:**

Passage structure (each item):
```
The [entity] [predicate] [property]. [N sentences of unrelated filler text.]
The [entity] [predicate] [_____].
```

The model must retrieve [property] using the entity name as the content cue. This is
NOT positional: the information is not at a stated index — the model must bind entity
to property over distance, guided by content.

Target distance: entity to cue ≈ 100–300 tokens (to engage long-range attention).

**Task B — positional retrieval (control, from exp-141/143):**
```
[Label]: [X_1], [X_2], ..., [X_k]. The [position-word] is [_____].
```
Entity at a stated ordinal position. Purely positional.

**Manipulation:** Amplify κ̃_K of the 16 text-native heads (same W_K amplification
protocol as exp-141, γ=+2.0). Sham: matched-norm perturbation in ⊥(P_k).

**Kill conditions:**
- K1 (wrong function): Task B shows meaningful improvement (ΔP_B ≥ 0.15 nats) —
  if text-native heads help positional retrieval, they're functionally like structural
  heads, not a distinct population.
- K2 (inert): Both tasks null (|ΔP_B| < 0.10 and |ΔP_C| < 0.10) — heads are
  functionally neutral; geometry ≠ function again.
- K3 (amplification failure): κ̃_K fails to increase by at least 3× on ≥ 10/16
  text-native heads.

**P1 fires if:** ΔP_C > +0.10 nats AND |ΔP_B| < 0.15 nats (content-retrieval
improves, positional retrieval unchanged).

**Decision rules by result:**
- P1 fires, K1 not fired → H_content confirmed: double dissociation text-native vs structural
- K1 fires (Task B improves with text-native amp) → text-native heads are functional
  superset of structural; census distinction is geometric, not functional
- K2 fires (both null) → text-native heads are geometry-only, function gap extends to this pop

---

## Register before running any manipulation

The causal test (exp-155) must be pre-registered (committed to attention-geometry)
before any W_K modification. The Task C item set must be committed with the
pre-registration.

Next step: write `exp-155_textnative_content_retrieval/prereg.md`, commit to
attention-geometry, then write `run.py`.

---

## Artifacts

- `explore_attention.py` — observational script
- `explore_results.json` — per-head stats and lag profiles for all 26 heads of interest
- `notes.md` — this file (the analysis and causal design)

---

*Written: 2026-09-23, ~12:35 AM MDT. Pre-registration for exp-155 is the next commit.*
