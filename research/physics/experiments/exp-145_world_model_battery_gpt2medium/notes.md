# exp-145 — World-model battery combined manipulation, GPT-2 medium

**Ariel — September 17, 2026, ~5:10–5:50 AM MDT. Solo.**
**Pre-registration: attention-geometry 2b1016a (pushed before run.py written).**
**One run; no reruns.**

---

## Verdict: INCONCLUSIVE — direction inverted

| Registered | Criterion | Result | Verdict |
|---|---|---|---|
| **P1** Task B improvement | ΔP_B > 0.33 nats AND ΔP_B > ΔP_A + 0.3 nats | ΔP_B = **−0.252 nats (OPPOSITE direction)** | NOT FIRED |
| **P2** Task A spared | \|ΔP_A − ΔP_A_sham\| < 0.5 nats | \|−0.096 − (−0.013)\| = 0.083 < 0.5 | **FIRES** |
| **P3** Item-level dissociation | Task B ≥ 60% improved, Task A < 60% | Task B: 0/20 improved; Task A: 1/20 | NOT FIRED |
| **P4_strong** | ΔP_B ≥ 1.0 nats | −0.252 | NOT FIRED |
| **P5_null** | Both within 0.1 nats of sham | B: 0.247 > 0.1 | NOT FIRED |
| **K1** | Sham ≥ combined both tasks | not fired | — |
| **K2** | Baseline at floor | orig_A=−1.401, orig_B=−4.977 (not at floor) | not fired |
| **K3_amp** | κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL | 0/5 below threshold | not fired |
| **K3_sup** | κ̃_K(sup) > 5.0 on ≥ 2/5 STEEP_LOCAL | 0/5 above threshold | not fired |

**Overall: INCONCLUSIVE — direction inverted. The manipulations worked mechanically but
produced the opposite of the predicted effect on Task B.**

---

## 1. What happened mechanically

**Amplification verified:**

| Head | κ̃ before | κ̃ after amp | κ̃ sham |
|---|---|---|---|
| L7H5  | 0.257 | 1.994 | 0.253 |
| L3H12 | 0.304 | 2.629 | 0.298 |
| L9H7  | 0.417 | 3.352 | 0.410 |
| L8H13 | 0.468 | 3.826 | 0.458 |
| L7H15 | 0.556 | 4.592 | 0.545 |

Amplification was effective (7–8× κ̃ increase). K3_amp not fired.

**Suppression verified:**

| Head | κ̃ before | κ̃ after sup | κ̃ sham |
|---|---|---|---|
| L4H13 | 43.199 | 0.0768 | 37.103 |
| L15H8 | 39.888 | 0.0889 | 35.696 |
| L8H7  | 33.251 | 0.0486 | 29.138 |
| L5H11 | 31.719 | 0.0827 | 28.047 |
| L11H7 | 30.909 | 0.0577 | 28.069 |

Suppression was effective (>500× reduction in positional capture). K3_sup not fired.
Sham values closely match originals — sham control working as designed.

**The protocol worked. The hypothesis was wrong.**

---

## 2. The finding

| Condition | Task A median logP | Task B median logP |
|---|---|---|
| Original | −1.401 | −4.977 |
| Combined | −1.497 | −5.229 |
| Sham | −1.414 | −4.982 |
| ΔP (comb) | −0.096 | **−0.252** |
| ΔP (sham) | −0.013 | −0.005 |

Task B: 0/20 items improved in combined vs original. Every single item degraded. The combined
manipulation reliably degraded positional retrieval in GPT-2 medium — the opposite of exp-143.

Task A: P2 fires (|−0.083| < 0.5) — the task-A sparing is preserved, consistent with both
models. Entity-state tracking is not differentially affected by these manipulations in either model.

---

## 3. What went wrong — the head selection problem

The exp-143 (GPT-2 small) STRUCTURAL heads came from the **random-token census**, not the
WikiText text-native census:

**GPT-2 small's STRUCTURAL heads (exp-143):**
- exp-007: Random-token census → 44/144 heads with R²≥0.90, Δ_med ≈ 0.249
- exp-112: Theory-of-A analysis → identifies 5 structural heads (L2H1, L3H4, L5H0, L7H11, L10H8)
  specifically by their behavior under the random-token census
- exp-137: κ̃_K analysis on these 5 structural heads → confirms low κ̃_K (0.05–0.33)
- exp-143: Amplifies these 5 random-token-census structural heads

**exp-145 GPT-2 medium STRUCTURAL heads (this experiment):**
- exp-118: WikiText-native census → 59 heads with Δ ∈ [0.20, 0.30]
- This session: κ̃_K characterization → picks 5 heads with lowest κ̃_K among the 59 wiki heads
- exp-145: Amplifies these 5 WikiText-native Δ-window heads

**The gap:** The WikiText-native population (exp-118) and the random-token structural population
(exp-007 equivalent for GPT-2 medium) are related but distinct. The causal experiment uses the
structural (random-token census) heads as the intervention target — these are the heads that
develop positional power-law structure under the frozen census protocol. The text-native
Δ-window heads are the semantic population that reads that structure under natural text. Using the
semantic population as the amplification target does not necessarily engage the positional retrieval
mechanism.

In GPT-2 small, exp-143 amplified the structural heads (random-token census) — these were
confirmed to be functionally involved in positional retrieval (exp-141/142). Whether the semantic
text-native population in GPT-2 medium plays the same functional role is untested.

**The random-token census for GPT-2 medium has never been run.** The research model registry
shows gpt2-medium as "unmeasured" — this refers to the random-token census specifically. The
next step is to run the census on GPT-2 medium, identify its structural heads, and then
pre-register the world-model battery with those heads as targets.

---

## 4. What this result means — bounded claim

**[MEASURED]** The combined manipulation (amplify 5 lowest-κ̃_K WikiText-native Δ-window heads,
suppress 5 highest-κ̃_K non-Δ-window heads) in GPT-2 medium degrades Task B (positional
retrieval) rather than improving it.

This is an honest, clean negative with a specific interpretation: the text-native Δ-window
heads and the structural (random-token census) heads are not interchangeable as intervention
targets. The two populations are distinguished — which is itself a finding about how the
Δ-window geometry is distributed across functional classes in larger models.

**What it does NOT say:** that the antagonism model is wrong for GPT-2 medium, or that
GPT-2 medium lacks the mechanism. It says that the mechanism, if present, is carried by
heads not yet identified — specifically the random-token structural population, which has
not been characterized for GPT-2 medium.

---

## 5. What remains open

1. **Run the random-token census on GPT-2 medium.** Standard frozen-protocol census
   (N_INPUTS=50, SEQ_LEN=512, SEED=42, random tokens, R²≥0.90) on all 384 heads. This
   identifies the structural population analogous to GPT-2 small's 44/144 structural heads.
   Register before running.

2. **Re-run the κ̃_K characterization on the random-token structural heads.** Once the
   structural population is identified, compute κ̃_K on those heads and identify any
   steep/local antagonism within GPT-2 medium.

3. **Pre-register exp-146: world-model battery on GPT-2 medium, structural heads.**
   With the correct target population identified, re-run the world-model battery.

4. **Flag for Eldon:** The direction inversion is worth discussing — amplifying the
   WikiText-native Δ-window heads in GPT-2 medium HURTS Task B. This might mean the
   semantic population's positional signal is actually competing with positional retrieval
   in GPT-2 medium (the opposite of the structural population's role). Worth a theory note.

---

## 6. Artifacts

- `characterize_kappa.py` — κ̃_K characterization (all 384 GPT-2 medium heads)
- `kappa_characterization.json` — full κ̃_K data
- `prereg.md` — pre-registration (attention-geometry 2b1016a)
- `run.py` — combined protocol for GPT-2 medium
- `results.json` — full per-item scores, kappa values, verdict
- `notes.md` — this file
