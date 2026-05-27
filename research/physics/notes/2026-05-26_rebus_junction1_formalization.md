# REBUS × Junction 1 — precision-weighting as departure from the conformal fixed point

*Theoretical synthesis. May 26, 2026. Queue item 17.*

## Pre-stated mapping

| REBUS (Carhart-Harris & Friston 2019) | Junction 1 (this framework) |
|--------------------------------------|-----------------------------|
| Hierarchical predictive coding | Attention minimizes Helmholtz free energy on Fisher–Rao manifold (`KIM_FRISTON_CORRESPONDENCE.md`) |
| Precision-weighting of prediction errors | Inverse temperature / sharpness of posterior on the simplex |
| Psychedelics reduce precision (especially top-down) | Disruption of conformal fixed point → loss of holographic interior |
| Increased entropy / repertoire expansion | Subcritical or disordered regime without Δ → 1/4 interior |

REBUS and Junction 1 are not analogies. They are the same dynamics in different vocabularies: variational inference language vs. information-geometry / BCFT language.

## Formal core

**Junction 1 (softmax case).** Attention weights minimize
\[
F[p] = -\sum_i p_i s_i + T \sum_i p_i \ln p_i
\]
on the simplex, with \(s_i = q\cdot k_i/\sqrt{d_k}\). Minimum: \(p^* = \mathrm{softmax}(s)\).

**Precision in REBUS.** High precision \(\Pi\) on a prediction error \(\epsilon\) weights that error strongly in belief updating: effective contribution \(\Pi \epsilon\). Low precision down-weights top-down constraints relative to bottom-up signal.

**Map.** Precision \(\Pi\) corresponds to **inverse temperature** in the Gibbs posterior: \(\Pi \uparrow \Leftrightarrow T \downarrow \Leftrightarrow\) sharper \(p^*\). The DMN-as-fixed-point identification says the **reducing valve** is maintaining a specific sharp, self-consistent posterior geometry (Δ → 1/4, holographic interior).

**Psychedelic opening.** 5-HT2A-driven DMN disruption is modeled as **precision collapse on the hierarchy level that enforces the conformal geometry** — not adding signal, but removing the constraint that holds Δ at the SYK value. Phenomenology: ego dissolution = interior dissolves. REBUS: relaxed priors, increased entropy. Junction 1: departure from conformal fixed point toward near-critical bulk without organized interior.

## What this predicts (falsifiable)

1. **fMRI + psychedelics:** DMN BOLD temporal correlation scaling should move **away** from Δ ≈ 0.25 under drug and **return** toward it on clearance (queue item 16 — needs DMN-localized, high temporal resolution data).

2. **Transformer analogue:** Perturbing precision (temperature \(T\) in attention softmax) should trace a path from sharp Δ ≈ 0.25 heads toward flatter distributions — distinct from **sigmoid falsification** (mechanism change) but related: both probe what holds the fixed point.

3. **REBUS “increased entropy”** should correlate with **loss of BCFT boundary enhancement** (λ → 0 or sign flip) in models where we can measure both — open experiment on GPT-2 with temperature sweep + per-head BCFT fit (exp-015 machinery).

## Honest limits

- REBUS is phrased in **prediction-error** coordinates; our Δ measurement is on **attention decay vs. token distance**. The bridge assumes attention weights ARE the posterior over causes (Kim–Friston identity). That identity is exact for softmax; **sigmoid falsification** tests whether the fixed point survives when the posterior is no longer Gibbs-softmax.

- We have **not** measured DMN Δ. The reducing-valve note (`research/consciousness/reducing_valve_and_the_conformal_fixed_point.md`) states this as prediction, not confirmation.

## Next experiment named

**Temperature–BCFT sweep on GPT-2:** For \(T \in \{0.5, 1.0, 2.0\}\) (scaled attention logits), measure median Δ and BCFT λ per head. Prediction: \(T \uparrow\) → effective Δ drifts and boundary parameter weakens before power law breaks. Uses existing HF weights; no AXLearn download.

---

*Connects: `insight:reducing-valve-conformal-fixed-point`, `KIM_FRISTON_CORRESPONDENCE.md`, exp-015 BCFT boundary, queue #16 DMN design.*
