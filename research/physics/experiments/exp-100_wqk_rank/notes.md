# exp-100: W_QK Effective Rank Measurement

**Pre-registered:** 2026-08-05, ~12:30 AM MDT, physics room session.
**Pre-registration committed before any analysis.** (Committed to 3ld0n/attention-geometry.)

**Analysis-only experiment — no new training.** Accesses existing Modal checkpoints
from exp-096 (C-NAT-anon), exp-097 (C-alien), exp-098 (C-alien-realnames), exp-099
(C-alien-rich).

---

## The question

Exp-099 produced a decoupled result: n_deep improved with world complexity (3→4→6 across
exp-097/098/099) while Δ_med did not move toward IR (1.04→0.727→0.750). The conformal
window theory (melonic-threshold derivation, `notes/2026-08-03_melonic_threshold_derivation.md`)
predicts that the W_QK effective rank should track the world's state-space dimension S:
R_eff(C-alien) ~ S=8, R_eff(C-alien-rich) ~ 4×R_eff(C-alien), R_eff(C-NAT) ~ d_k=64.

If rank tracks S but Δ_med doesn't improve: the rank elevation is real but not sufficient
to cross the conformal window threshold — the two observables respond to different corpus
properties at different sensitivity.

If rank does NOT track S: a different mechanism drives both the backbone collapse and
n_deep dose-response; the R_eff ↔ S(corpus) conjecture is falsified at the W_QK level.

**Direct physical interpretation:** Low-rank SYK is disordered if and only if R_eff > R_crit.
The W_QK effective rank is the direct operationalization of the rank in the sparse/low-rank
SYK Hamiltonian (arXiv:1910.10173, Kim–Cao–Altman). If exp-099's world-complexity scan
changes R_eff proportionally to S, the mechanism is direct. If not, something else governs.

---

## Pre-registered hypotheses

**H_rank_ordered** (primary): effective rank increases with world complexity.
- C-alien (S=8) < C-alien-rich (S=32) < C-NAT-anon (S >> d_k).
- Stated as strict ordering on mean_rank across all 6×8=48 heads.

**H_rank_S** (primary, quantitative): rank scales with S.
- mean_rank(C-alien-rich) / mean_rank(C-alien) ≈ 4 (= 32/8 = S ratio).
- Criterion: ratio in [2.5, 5.5] (±37.5% of the S-ratio prediction; accounts for
  state-space non-linearity and embedding effects).

**H_rank_delta_correlated** (secondary): per-head effective rank and Δ_med are
anti-correlated (low rank → UV-arrested, high rank → IR-approaching).
- Stated as: Pearson r(R_eff, Δ_med) < −0.3 when pooled across all corpora.

**H_rank_backbone** (secondary): L0 (backbone) heads have higher effective rank in
C-NAT-anon than in C-alien; backbone collapse (n_backbone=0 in C-alien) corresponds
to near-rank-1 W_K in L0.

**Kill criterion for H_rank_ordered:**
If mean_rank(C-alien) > mean_rank(C-NAT-anon): rank is not ordered by world complexity;
the low-rank SYK mechanism does not hold at this grain; alternative mechanism required.

---

## Protocol

**Checkpoints:** one seed per corpus (s0, the first seed).
- C-alien: `exp097-alien-data/runs/run_alien_s0/step_2000/model.safetensors`
- C-NAT-anon: `exp096-anon-data/runs/run_anon_s0/step_2000/model.safetensors`
- C-alien-realnames: `exp098-realnames-data/runs/run_names_s0/step_2000/model.safetensors`
- C-alien-rich: `exp099-rich-data/runs/run_rich_s0/step_2000/model.safetensors`

**W_K extraction:** For each layer (0–5) and head (0–7):
- Load `gpt_neox.layers.{L}.attention.query_key_value.weight` (shape [1536, 512])
- Extract key slice: rows [512:1024] → shape [512, 512]
- Extract per-head: rows [h*64:(h+1)*64] for h in 0..7 → shape [64, 512]

**Rank computation:**
- SVD of W_K_h: σ_1 ≥ σ_2 ≥ ... ≥ σ_64 (min(64, 512) = 64 singular values)
- Participation ratio (effective rank): R_eff = (Σ σ_i)² / Σ σ_i²
- Stable rank: R_stable = Σ σ_i² / σ_1²
- Both reported; R_eff is primary.

**Summary statistics:** per corpus —
- mean_rank, median_rank, per-layer breakdown (L0 backbone of particular interest)
- Cross-corpus: rank ratio C-alien-rich / C-alien, ordered comparison

**Compute:** Modal CPU (mounts existing volumes; no GPU; cost < $0.10; runs in seconds).

---

## What follows

**If H_rank_ordered confirmed and H_rank_S confirmed:**
- The low-rank SYK mechanism is directly evidenced.
- n_deep dose-response in exp-099 is explained: higher S → higher R_eff → more structural
  positions available in deep layers → more heads specialize.
- Δ_med failure to improve: the rank elevation is real but S=32 still well below R_crit
  at which the conformal phase forms.
- Next rung: continue S expansion (S=64, S=128) to pin the threshold.

**If H_rank_ordered confirmed but H_rank_S fails (rank doesn't scale as S):**
- World state-space S is an imperfect proxy for W_QK rank.
- Something about the corpus structure beyond S determines how much rank gets encoded.
- Enrich analysis: look at what corpus statistics actually predict R_eff.

**If H_rank_ordered falsified (alien > NAT-anon):**
- The R_eff ↔ S(corpus) conjecture does not hold at the W_QK level.
- An alternative mechanism governs both backbone collapse and UV arrest.
- Return to theory: what determines W_QK rank if not world state-space complexity?

---

## Status

- [x] Pre-registration written (2026-08-05, ~12:30 AM MDT)
- [x] Pre-registration committed to 3ld0n/attention-geometry (before analysis)
- [ ] Analysis script written
- [ ] Analysis run
- [ ] Verdict registered
