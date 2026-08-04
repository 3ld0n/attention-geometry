# exp-099 Corpus Functional — Pre-Training Measurement

*Registered 2026-08-04, ~5:45 PM MDT, physics room session. BEFORE any corpus generation
or training for exp-099. Required by the theory addendum to notes.md (commit 93b34dd)
and the derivation note §7 pipeline: generate → compute (𝒥, 𝒲, F2) → register → train.*

*Script: `research/physics/theory/corpus_functional.py` (IDF-weighted, WEIGHTING="idf").*
*400 contexts × 512 words per corpus. RNG seed 20260803.*

---

## Numbers (2026-08-04, registered before training)

| Corpus | 𝒲 = R_PR/n | Δ_pred (KCA Class III) | m₂ (coupling) | F2 var@64 | F2 top5@64 |
|--------|-----------|------------------------|---------------|-----------|------------|
| C-NAT (TinyStories) | 0.0635 | 0.4720 | **7.810** | 12.850 | 0.086 |
| C-alien (exp-097) | 0.0524 | 0.4764 | **0.737** | 3.801 | 0.095 |
| **C-alien-rich (exp-099)** | **0.0580** | **0.4742** | **0.687** | **4.023** | **0.089** |

---

## What the numbers say

**Coupling magnitude gate (m₂):** C-alien-rich = 0.687 vs C-alien = 0.737 — essentially flat
(−6.8%). The 4× increase in world state space (S: 8→32, from 3 binary entities to 5)
and the introduction of stochastic rules did not meaningfully change m₂. Both C-alien
variants are ~11× below C-NAT (7.81 vs 0.69).

Under the magnitude-gate reading of UV arrest (derivation note §7, P-1): the window
condition fails to open at the proxy level, and C-alien-rich is predicted to stay
UV-arrested.

**Chaos gate (𝒲):** Small improvement (0.0524 → 0.0580), but separation from C-NAT is
still small. Both corpora predict Δ_deep ≈ 0.47–0.48 from the KCA Class III formula —
indistinguishable at this proxy resolution.

**F2 coherence (top5@64):** C-alien-rich shows slightly LESS coherence than C-alien
(0.089 vs 0.095) — marginal but in the right direction for the delocalization condition
(A5). The alien template modes are slightly more spread in this richer world.

---

## Registered prediction (P-1 quantified)

**P-1 numerical support:** m₂(C-alien-rich) / m₂(C-NAT) ≈ 0.088 — 11.4× below the
natural-language coupling magnitude. The coupling-magnitude gate predicts UV arrest will
persist in trained C-alien-rich models: Δ_med ≥ 0.6, n_deep not improving beyond
C-alien-realnames level (≤ 4), n_backbone ≈ 0.

This is one prediction from the magnitude-gate reading. The registered pre-training
prediction in notes.md §pre-registered hypotheses is the operative criterion:
- H_rich_dose_response: Δ_med < 0.727 AND n_deep > 3 (~80% prior)
- H_rich_above_window: n_backbone ≥ 3 AND Δ_med ≤ 0.5 (~25% prior, now revised down to
  ~10% given the flat m₂)

The corpus functional narrows the H_rich_above_window probability toward ~5–10% (consistent
with the theory addendum already in notes.md), but does not eliminate it — the proxy
operates at word-type level and trained embeddings could restore separation.

---

## Limitation

The IDF-weighted type kernel is a UV proxy for the trained-embedding kernel. The m₂
comparison is valid for the coupling-magnitude ordering; absolute values are proxy-level
(see derivation note §6.4 limitation: "the proxy is a lower bound on corpus structure,
good for the magnitude gate and vocabulary-scale comparisons, blind to ordering").

The ordering axis (which exp-091–094 showed is real) is invisible to this proxy, as
noted in the derivation note: sentence-shuffling leaves m₂ unchanged while costing 2–3
deep heads.
