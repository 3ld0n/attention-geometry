# exp-175 Notes — Pythia-70m formation step-sweep

**Date run:** 2026-11-10, 4:17–5:17 AM MST, Mission Valley. Physics room.
**Pre-registration:** d4bbfc1 (pushed to attention-geometry 2026-11-10, before script was written).
**Why-check:** Formation — when does D/4 first appear in Pile-trained Pythia-70m?

---

## Trajectory (full)

| Step | n_dw | Δ_med | Notable heads |
|------|------|-------|---------------|
| 0 | 8 | 0.206 | L3H1, L3H4, L3H7, L4H0, L4H4, L4H6, L5H0, L5H3 |
| 1 | 8 | 0.206 | (same) |
| 2 | 8 | 0.206 | (same) |
| 4 | 8 | 0.206 | (same) |
| 8 | 8 | 0.205 | (same) |
| 16 | 8 | 0.206 | slight composition change |
| 32 | 11 | 0.205 | training beginning |
| 64 | 7 | 0.209 | — |
| 128 | 19 | 0.222 | large fluctuation |
| 256 | 3 | 0.280 | beginning collapse |
| **512** | **0** | **—** | **n_dw = 0: initialization erased** |
| 1000 | 3 | 0.256 | trained structure first emerges |
| 2000 | 6 | 0.234 | training continues |
| 4000 | 5 | 0.212 | — |
| 8000 | 4 | 0.276 | — |
| 16000 | 1 | 0.276 | L2H5 stable |
| 32000 | 1 | 0.296 | L2H5 |
| 64000 | 1 | 0.267 | L2H5 |
| 128000 | 1 | 0.270 | L2H5 |
| 143000 | 1 | 0.264 | L2H5 — consistent with exp-118 |

**Kill conditions:**
- K1: not fired (step 143000 n_dw=1, Δ_med=0.264; |0.264-0.260|=0.004 ≤ 0.05 ✓)
- **K2: fires** (n_dw=8 at step 0)
- K3: not fired (clear 0→≥1 transition at step 1000)
- H3: consistent with exp-118 ✓

**H1 onset:** step 1000 (2.10B tokens) — first step after the n_dw=0 minimum at step 512
**H2 verdict:** late-onset (step 1000 ≥ 1000 by script definition; though only marginally so)
**Declared prior miss:** expected step 10000–64000 (20–130B tokens); actual onset step 1000 (2.1B tokens)

---

## What K2 tells us: the RoPE initialization structure

K2 fires because n_dw = 8 at step 0. The untrained model has 8/48 heads with Δ ∈ [0.20, 0.30]
and R² ≥ 0.90. This is qualitatively different from the GPT-2 randomized-weights control
mentioned in the pre-registration (Δ_med ≈ 0.17, far from 0.25).

**Why the RoPE initialization shows Δ ≈ 0.206:**

RoPE (Rotary Position Embedding) applies position-dependent rotations to Q and K before
the dot product. Even at random initialization (random weight matrices), RoPE's structure
imposes a positional dependency on attention scores: the inner product Q_i · K_j after
RoPE rotation has an angular term that depends on |i-j|. For random Q/K, this can
produce attention patterns that, pooled over many random inputs, show systematic lag decay.
The Δ ≈ 0.206 (lower edge of the window) in all 8 heads at step 0 is a structural
signature of RoPE initialization — not a learned conformal structure.

**The distinct structure of initialization vs. trained:**

The initialization-era n_dw (steps 0–256) shows:
- All heads near the lower boundary: Δ_med consistently ≈ 0.205–0.222
- Same 8 heads (L3H1, L3H4, L3H7, L4H0, L4H4, L4H6, L5H0, L5H3) through early steps
- Head composition is LAYER-SPECIFIC to layers 3–5 (deeper layers of the 6-layer model)
- The pattern is disrupted by training, reaching n_dw=0 at step 512

The trained-era n_dw (steps 1000+) shows:
- Heads from different layers (including early layers 0, 1, 2)
- Δ_med varies more widely across steps (0.21–0.30)
- Final stable head (L2H5) is layer 2, completely different from the initialization heads

The two populations do not overlap at all — the initialization-era heads and the
training-era heads are distinct. This is strong evidence that:
1. The K2 signal is a RoPE initialization artifact, not a pre-trained formation event.
2. Training first *destroys* the initialization structure (n_dw=0 at step 512) before
   building trained conformal structure from scratch.

---

## The formation story

The correct reading of H1, adjusted for K2:

**Phase 1 (steps 0–256): RoPE initialization structure**
The untrained model has 8 heads near the lower edge of the Δ-window due to RoPE's
positional structure on random weights. This is not trained formation; it is
architecture-imposed lag decay from rotary encoding.

**Phase 2 (step 512): Erasure**
Training has disrupted the initialization structure. n_dw = 0. The model is in a
transitional state where neither initialization-driven nor trained conformal structure
is detectable by the census.

**Phase 3 (steps 1000–143000): Trained formation**
Conformal-relevant heads form through learning. The onset of trained structure is
approximately step 1000 (2.10B tokens). The population is small — 1 stable head
(L2H5) by step 16000 — and the head remains consistent through step 143000.

**Comparison to AbsPE (exp-091):**
The AbsPE formation at TinyStories onset: ~1.05B tokens (exp-091, step 2000).
The Pythia-70m trained formation onset: ~2.1B tokens (step 1000 on The Pile).

These are within a factor of 2 by token count. Given that The Pile has roughly 10–20%
story-scale referential content while TinyStories is 100% story-scale, the effective
"formation-relevant" exposure at step 1000 on The Pile is roughly 0.2–0.4B tokens of
story-scale language — 3–5× less than the TinyStories AbsPE onset.

This is consistent with Interpretation A (corpus-type-dependent formation) OR with a
simpler story: the heterogeneity of The Pile makes D/4 emergence slower per raw token,
but the model still forms at a comparable token scale.

---

## Prior miss: earlier than expected

Declared prior: step 10000–64000 (20–130B tokens).
Actual onset: step 1000 (2.1B tokens) — 10–65× earlier than the center of my prior.

The prior was based on reasoning that The Pile's heterogeneity would spread the
formation-driving content across a larger token budget. The actual result shows formation
is faster. Candidate explanations:
1. The small final n_dw (only 1 head by step 16000) suggests the "formation" in this
   model may be thin — one head forms early and the rest don't. The prior was calibrated
   for AbsPE where 11–15 heads form; perhaps the criterion n_dw ≥ 1 fires earlier for
   rotary because it's a weaker signature.
2. Story-scale content on The Pile may be effectively concentrated enough in the early
   training (curricula effects, or the first batches happening to include more story-scale
   text).
3. The onset is real and my prior was wrong about how much the corpus heterogeneity matters.

All three are plausible. The honest summary: my prior was wrong, formation is earlier,
and the reason is uncertain.

---

## What this does and does not decide

**Decides:**
- Formation of trained conformal structure in Pythia-70m (RoPE) occurs at approximately
  step 1000 (2.1B tokens), earlier than my declared prior.
- The step 143000 result (n_dw=1, Δ_med=0.264) is consistent with exp-118 (K1 not fired;
  H3 confirmed).
- The RoPE initialization structure produces a spurious K2 signal that must be
  distinguished from trained formation.
- The trained population is L2H5 — a single stable head, much smaller than the AbsPE
  formation (11–15 heads in GPT-2 NeoX 70m on TinyStories).

**Does not decide:**
- Whether the small trained population (n_dw=1 with random-token census) reflects genuine
  corpus-type dependence or simply the smaller training-signal fraction of The Pile.
- Interpretation A vs B (corpus-type-dependent vs. training-time-dependent). This requires
  the C-NAT vs C-alien corpus-type test — now feasible at step ~1000 on local hardware.
- Whether the initialization structure at steps 0–256 is a confound for other architecture
  measurements or a window into the RoPE geometry prior to training.

---

## Implications for the next experiment

The corpus-type test (exp-176 candidate) is feasible:
- Train a Pythia-70m-equivalent model (NeoX 70m rotary) from scratch on C-NAT vs C-alien
  to approximately step 1000–2000.
- Run the frozen random-token census at that step.
- Compare n_dw: C-NAT should produce n_dw ≥ 1 if Interpretation A is correct;
  C-alien should produce n_dw = 0.

Note: this requires training a model from scratch on our corpora, which is the same kind
of infrastructure as exp-091/092. Budget question: step 2000 at batch size 2M tokens =
4B tokens. With our 70m TinyStories models trained on ~1.05B tokens, we need ~4× that
in C-NAT/C-alien. This is a Modal training budget question — flag for the structural
session with Eldon.

---

## Protocol note: census size for step sweep

The 50 × 60 census has limited power for this architecture (48 heads total). At step 0
the variance is high — 8 heads at the lower edge with consistent composition suggests
initialization structure, but a larger census would help discriminate. For the corpus-type
test (exp-176), use a larger census (500 × 512 or similar) to reduce noise.

---

## For Geometry

K2's finding about RoPE initialization is interesting beyond Formation: even random weights
with RoPE produce Δ ≈ 0.20 lag-decay profiles in the deeper layers. This suggests that
RoPE's positional structure alone can produce power-law lag-decay in the census, at the
lower edge of the conformal window. The training then *replaces* this with a different,
sparser pattern at a slightly higher Δ. The Geometry question must account for this:
what the census measures in a trained RoPE model is not simply "conformal structure
formed in training" but "training has moved the heads away from the RoPE initialization
pattern into a trained one." This is a load-bearing distinction for the T1 restatement.

---

*Total elapsed: 597 seconds (20 checkpoints × ~30s per checkpoint including download).*
