# exp-064 — Causal slope-editing pilot (Phase 1.4 / RESEARCH_PLAN 2.3)

*June 11, 2026. Pre-registration: `notes/2026-06-11_slope_editing_preregistration.md` (committed before any weight edit; stage-1 predicted sign vector committed to `results.json` before stage 2 ran). Pilot deliverable: "the editing handle works / doesn't."*

## Verdict — **HANDLE WORKS** (both pre-registered legs passed)

### Mechanical leg (does the edit do what it claims?)

Edit: W_Q ← (I + (κ−1)·P_U)·W_Q per head, P_U = top-8 PCs of position-mean `ln_1` output. 8 selected conformal heads (top R² with Δ̂ ∈ [0.15, 0.35] from exp-060), κ ∈ {0.5, 1.0 sham, 1.5, 2.0}.

- **Δ̂ moves in the predicted direction 8/8 at κ = 1.5, 8/8 at κ = 2.0, 8/8 at κ = 0.5** (pre-registered pass: ≥ 6/8).
- **Sham control:** median |ΔΔ̂| at κ = 1 is 1.8×10⁻⁸ — the edit machinery itself is exactly null.
- The pre-softmax score slope α scales ≈ linearly with κ (e.g. L10H3: −0.35 / −0.64 / −0.93 / −1.23 for κ = 0.5/1/1.5/2), while post-softmax Δ̂ responds sub-linearly (L10H3: 0.126/0.253/0.360/0.444) — the softmax renormalization of exp-056's "1.4× wrinkle," now seen causally.

### Behavioral leg (exp-061 implied-valley machinery)

- **ρ(Δv̂, Δvalley_measured) = +0.818** (p = 1.0×10⁻⁶; pre-registered threshold 0.5).
- **Sign agreement 24/24** (threshold 16/24). All eight heads' valleys deepen for κ > 1 and shallow for κ < 1, exactly as the committed per-head sign vector predicted.

## Caveats (disclose)

- At κ = 2.0 two heads degrade out of the conformal regime (L5H10 R² 0.58, L6H4 0.84): large edits break the power law before they break the valley response. The handle has a usable range, roughly κ ∈ [0.5, 1.5–2].
- This is a single-head, attention-level pilot. Task-level behavior (long-context retrieval, lost-in-the-middle accuracy) was NOT measured; that is the flagship follow-up (RESEARCH_PLAN 2.3 full study, StreamingLLM-style baselines).
- λ̂ per head was taken from exp-060 M0 fits; M0 is phenomenological after the exp-060 kill, used here only as a fit (the exp-061 usage, explicitly licensed by exp-060 notes).

## Significance

The per-head QK log-distance slope is now an *addressable causal parameter*: editing it moves the measured conformal dimension AND the head's valley statistic in the quantitatively predicted direction. The Δ–valley link is head-causal, not merely correlational — the program owns an engineering handle on lost-in-the-middle behavior, independent of all physics interpretation.

## Files

- `run_slope_edit.py` (stages: predict → edit → evaluate; stage timestamps in results.json)
- `results.json` (committed predictions, per-head per-κ measurements, evaluation)
