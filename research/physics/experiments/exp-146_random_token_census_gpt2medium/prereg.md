# exp-146 — Pre-registration: Random-Token Census on GPT-2 Medium

**Pre-registration commit: 8aa31dd (attention-geometry, pushed before run.py written)**
**Registered before run.py is written.**
**Ariel — September 18, 2026, ~12:30 AM MDT. Solo.**

---

## 1. Context and motivation

exp-145 (September 17, 2026) ran the world-model battery combined manipulation on
GPT-2 medium, using WikiText-native Δ-window heads (exp-118, 59 heads) as intervention
targets. Result: INCONCLUSIVE — direction inverted. ΔP_B = −0.25 nats; Task B
DEGRADED on 0/20 items. The opposite of exp-143 (GPT-2 small, confirmed).

The root cause diagnosis from exp-145: **WikiText-native Δ-window heads (exp-118) and
random-token structural heads are distinct populations as causal intervention targets.**
The GPT-2 small structural heads used in exp-141/142/143 were identified via the
random-token census (the same frozen protocol used in exp-110/112). That population
has never been run on GPT-2 medium.

The WikiText-native census (exp-118) uses text-native input distributions and identifies
heads with conformal scaling on natural language. The random-token census uses a uniform
random-token input (N_INPUTS=50, SEQ_LEN=512, SEED=42) and identifies the structural
positional-carrier population — heads whose Δ reflects the pure positional geometry
independent of semantic content. These are the heads the κ̃_K intervention target
logic was built around in GPT-2 small.

**This experiment:** run the frozen random-token census protocol on all 384 heads of
GPT-2 medium, identify the structural population, and prepare the correct targets for
the world-model battery extension (exp-147 candidate).

---

## 2. Protocol (frozen)

**Identical to the published replication census**
(`research/physics/replication/measure_conformal_heads.py`) with model changed to
GPT-2 medium.

Parameters:
```
MODEL = "openai-community/gpt2-medium"   # 24 layers, 16 heads, D_model=1024, D_head=64
N_INPUTS = 50
SEQ_LEN = 512
SEED = 42
R2_MIN = 0.90                            # OLS R² threshold for a conformal head
DELTA_MIN = 0.05                         # minimum Δ to report
FIT_LO, FIT_HI = 8, 256                 # lag window for OLS fit
DEEP_LO = 256                            # minimum query position (q_i ≥ max(DEEP_LO, dx))
```

Input: 50 sequences of 512 random tokens drawn uniformly from the full vocabulary
(torch.randint with SEED=42).

Measurement: per-head attention weight matrix (fp32, output_attentions=True), lag-averaged
profile A(dx) for dx ∈ [0, SEQ_LEN−1], OLS fit of log A vs log dx over FIT_LO ≤ dx ≤ FIT_HI.

Output: per-head (Δ, R²) across all 384 heads.

**Structural population:** heads with R² ≥ 0.90 AND Δ ∈ [0.20, 0.30]. (This is the
Δ-window criterion applied under the random-token protocol — the "structural" subset of
the conformal population.)

**Full conformal population:** heads with R² ≥ 0.90 AND Δ ≥ 0.05. (Standard census
output; the structural subset is the primary target for exp-147.)

---

## 3. Hypothesis

GPT-2 medium has a random-token structural population (R² ≥ 0.90, Δ ∈ [0.20, 0.30])
that is **distinct** from the WikiText-native Δ-window population identified in exp-118
(59 heads). The exp-145 inversion was caused by using the wrong target population for
the world-model battery.

The structural population will be identifiable, interpretable as positional-carrier heads
(analogous to GPT-2 small's 5 structural heads: (2,1), (3,4), (5,0), (7,11), (10,8)),
and will provide the correct target set for a corrected world-model battery experiment.

---

## 4. Predictions

| Label | Criterion | Interpretation if fired |
|---|---|---|
| **P1** (population exists) | n_structural ≥ 1 head with R²≥0.90, Δ∈[0.20,0.30] | A structural population is measurable in GPT-2 medium |
| **P2** (populations distinct) | Jaccard(structural, WikiText-native) < 0.8 | The two census protocols identify substantially different head sets |
| **P3** (Δ in window) | Median Δ of structural population ∈ [0.20, 0.30] | Structural heads are in the SYK-q4 conformal window |
| **P4** (scale comparison) | n_structural > n_structural_gpt2small = 5 | Larger model has larger structural population (scale hypothesis) |

P1 is required for the experiment to be useful (the battery extension requires targets).
P2 is the direct test of the exp-145 diagnosis.
P3 follows from the protocol definitions but is stated explicitly because failing it
would indicate a protocol discrepancy worth diagnosing.
P4 is secondary — scale comparison against GPT-2 small's 5 structural heads.

**Primary outcome:** the structural head list itself. This is the artifact exp-147 will use.
The predictions above characterize its properties.

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | n_conformal = 0 (no heads with R²≥0.90, Δ≥0.05) | Census fails — model not behaving conformal under random-token protocol |
| **K2** | Jaccard(structural, WikiText-native) = 1.0 | The two populations are identical — exp-145 diagnosis was wrong; the inversion requires a different explanation |
| **K3** | n_structural = 0, but n_conformal > 0 with Δ > 0.30 | Structural population shifted out of window — investigate before battery |

K1 would be a severe protocol failure; GPT-2 medium trained on WebText should show
conformal heads under any reasonable input regime given the prior evidence chain.
K2 would require revising the exp-145 diagnosis.

---

## 6. Analysis plan

1. Run the frozen census on all 384 heads of GPT-2 medium (fp32, output_attentions=True)
2. Report per-head (Δ, R², layer, head) sorted by Δ
3. Identify structural population: R² ≥ 0.90, Δ ∈ [0.20, 0.30]
4. Identify full conformal population: R² ≥ 0.90, Δ ≥ 0.05
5. Compute Jaccard overlap between structural and WikiText-native (exp-118's 59 heads)
6. Compare structural population size to GPT-2 small (5 heads)
7. Report: structural head list (layer, head, Δ, R²) — this is the primary artifact
8. Evaluate P1–P4, K1–K3
9. Archive results.json; note heads for exp-147 candidate pre-registration

---

## 7. Connection to the record

- **Follows from:** exp-145 (world-model battery GPT-2 medium — direction inverted,
  root cause: wrong target population); exp-118 (WikiText-native census, 59 heads);
  exp-112 (GPT-2 small structural population via random-token census)
- **Bears on:** P1 (functional characterization of structural population at medium scale)
- **Enables:** exp-147 (corrected world-model battery in GPT-2 medium, with
  random-token structural heads as intervention targets)
- **Analysis-only:** false (new forward passes; GPT-2 medium locally cached at
  ~/.cache/huggingface/hub/models--openai-community--gpt2-medium)
- **Pre-registration required before run.py:** yes
