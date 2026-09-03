# exp-133 — MLP block-0 gate pre-activations mechanism

**Pre-registered:** 2026-09-03 (commit before any script written or forward pass run)

**Question:** exp-132 showed the MLP in block 0 amplifies position-correlated structure from
σ(h^(0.5)) = 0.144 (MLP input) to σ(mlp_out) = 0.313 (MLP output), ratio 2.168. The amplification
mechanism is unknown. This experiment measures the intermediate states to identify it.

**GPT-2 MLP block architecture:**
1. pre_act = h^(0.5) @ W_fc    (W_fc = mlp.c_fc, d_model × 4·d_model = 768 × 3072)
2. h_gelu   = GeLU(pre_act)     (element-wise; changes magnitude distribution but not position structure directly)
3. mlp_out  = h_gelu @ W_proj  (W_proj = mlp.c_proj, 3072 × 768)

---

## Hypothesis

The amplification is gate-driven: W_fc's linear projection already carries the position-correlated
structure at σ ≥ 0.313 before GeLU sees it. GeLU then passes it through (or preserves it), and
W_proj projects back to d_model without destroying it.

Alternative: W_fc delivers σ < 0.313, and either (a) GeLU nonlinearly selects/amplifies the
position-correlated dimensions (σ post-GeLU > σ pre-GeLU), or (b) W_proj creates the correlation
in the projection back to d_model.

---

## Protocol

- Model: GPT-2 small (gpt2)
- Census: random-token, 50 sequences, seed=42, mean-first aggregation (same as exp-131/exp-132)
- Hook locations: h^(0.5) [MLP input], pre_act [after W_fc, before GeLU], h_gelu [after GeLU, before W_proj], mlp_out [after W_proj]
- Metric: position-correlation σ = OLS slope of mean cosine similarity vs log(|i−j|), R² recorded

---

## Predictions

**P1 (gate-driven, pre-registered):**
σ(pre_act, mean-first) ≥ 0.313
Interpretation: W_fc's projection already contains the full amplified signal; GeLU and W_proj are
not the source of amplification — they preserve or marginally transform a signal that was already
present at the gate input.

**P2 (GeLU passes through, contingent on P1):**
If P1 confirmed: σ(h_gelu) ≈ σ(pre_act) (within 0.05)
Interpretation: GeLU does not substantially amplify or attenuate the position-correlation that W_fc
created; the nonlinearity is mechanism-neutral at this grain.

---

## Kill conditions

**K1:** σ(pre_act) < 0.313
P1 falsified. The amplification does not originate in the W_fc linear projection alone.
Two sub-cases to distinguish post-hoc:
  - K1a: σ(h_gelu) ≥ 0.313 → GeLU is the nonlinear selector (position-correlated dimensions are
    preferentially activated by GeLU from a lower-σ pre_act)
  - K1b: σ(h_gelu) < 0.313 → neither W_fc nor GeLU creates the correlation; it must arise in
    W_proj's projection back to d_model (position-correlated directions in the 4d_model space
    happen to align with W_proj's columns)

---

## Verdict map

| σ(pre_act) | σ(h_gelu) | Verdict |
|---|---|---|
| ≥ 0.313 | ≈ pre_act | P1 CONFIRMED: gate-driven; W_fc linear projection is the amplifier |
| ≥ 0.313 | > pre_act | P1 CONFIRMED (partial): W_fc initiates, GeLU amplifies further |
| < 0.313 | ≥ 0.313 | K1a: GeLU nonlinear selection |
| < 0.313 | < 0.313 | K1b: W_proj as amplifier |

Registered threshold for K1: 0.313 (exp-132's measured σ_mlp_out).
All comparisons mean-first, random-token census, same as exp-131/exp-132.

---

## Context

- **exp-131** identified the correct protocol (mean-first × random-token census): σ_mlp0 = 0.313
- **exp-132** confirmed: σ(h^(0.5)) = 0.144, σ(mlp_out) = 0.313, ratio 2.168; pass-through falsified
- This is the next rung of the G7 multi-layer mechanism investigation

*Pre-registration commit establishes the ordering; results will be appended below.*
