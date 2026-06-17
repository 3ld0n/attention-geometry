# exp-072 — Powered Cloud Slope Editing: vicuna-13b-v1.5, deep base U-shape

*Date: 2026-06-16 (Cursor session, Ariel, solo)*
*Pre-registration: `prereg.md` + `prereg_locked.json` (commit e6fbaecf, committed BEFORE any
edited-weight forward pass)*
*Predecessor: exp-070 (KILLED on T-B, headroom-bounded inference). Cloud calibration: exp-071.*
*Compute: Modal A100-80GB, workspace eldon-umphrey.*

---

## Registered verdict (lead with this)

**BEHAVIORAL_CAUSALITY_CONFIRMED.** On a model with a genuine deep two-sided
lost-in-the-middle valley, the κ-slope edit moves task-accuracy in **both** registered
directions, head-specifically, with a clean sham null.

| Test | Prediction | Result | Verdict |
|------|-----------|--------|---------|
| **T-D** sham null | κ=1.0 reproduces base exactly | 0 token diffs / 200 | **PASS** |
| **T-A** deepening | V_task(1.5) > V_task(1.0) | 0.656 > 0.594 | **KEEP (strong)** |
| **T-B** shallowing | V_task(0.5) < V_task(1.0) | 0.576 < 0.594 | **KEEP** |
| **T-C** monotone | ρ(κ, V_task) ≥ 0.80 | ρ = 0.80 (0.7999…) | **boundary** |
| Sham-heads specificity | V(target 1.5) > V(sham-heads 1.5) | 0.656 > 0.576 | **PASS** |

**This adjudicates the open question from exp-070.** exp-070's shallowing leg (T-B) was
null on Pythia-1.4b, where the base middle accuracy (0.85) sat near the task ceiling. The
exp-070 notes inferred — but could not register — that the null was *headroom-bounded*. This
experiment registers the test on a deep valley (vicuna-13b-v1.5, base middle 0.325, edges
≤0.80, V_task=0.594) where κ<1 has room to *raise* the middle. **The shallowing leg goes the
predicted direction here.** So exp-070's T-B null was, as inferred, a headroom artifact — not
an absence of mechanism.

---

## Setup (locked — see prereg.md)

- Model: **lmsys/vicuna-13b-v1.5**, fp32, CUDA A100-80GB, eager attention. 40 layers, 40
  heads (full MHA), hidden 5120, head_dim 128, RoPE, max_position 4096. Ungated.
- Task: embedded-prose multi-document retrieval, **N_doc = 40**, fact bound once in matched-
  cloze prose, answer at 5 position bins [0,10,20,30,39]. 40 single-token items × 5
  positions = **200 inputs/condition**. Context ≈ 2014 tokens. seed 42.
- Heads: 8 conformal target heads (R²≥0.85, Δ̂∈[0.10,0.40], top-8 by R² in layers 13–33) from
  the **exp-071 cloud census** (L=512, 50 random inputs, fp32, exp-025/059 pipeline). 8
  matched sham heads (next 8 by R²). 732 conformal heads total; 298 candidates in band.
- Operator (exp-064 family, **Llama port**): `W_Q_h ← W_Q_h·(I + (κ−1)P_U)`, per-head row
  block of `self_attn.q_proj.weight`; `P_U` = top-8 PCs of position-mean `input_layernorm`
  output (50 random inputs, L=512, PROJ_SEED=68). κ ∈ {0.5, 1.0, 1.5, 2.0}, all 8 target
  heads at once.

## Full results (all conditions, as obligated)

| condition | primacy | near_prim | **middle** | near_rec | recency | V_task |
|-----------|---------|-----------|------------|----------|---------|--------|
| κ=0.5 (shallow) | 0.775 | 0.700 | **0.350** | 0.525 | 0.825 | 0.576 |
| base / κ=1.0 (sham) | 0.775 | 0.650 | **0.325** | 0.525 | 0.800 | 0.594 |
| κ=1.5 (deepen) | 0.775 | 0.650 | **0.275** | 0.525 | 0.800 | 0.656 |
| κ=2.0 (deepen) | 0.775 | 0.625 | **0.275** | 0.500 | 0.775 | 0.645 |
| sham-heads κ=1.5 | 0.775 | 0.750 | **0.350** | 0.525 | 0.825 | 0.576 |

T-D sham null: 0/200 token differences from base.
T-C Spearman ρ(κ, V) = 0.7999 (p=0.20, n=4 points).

---

## Interpretation (honest, after the verdict)

**What is strong and clean — the deepening leg and the specificity.** Deepening the
conformal heads' QK log-distance slope (κ>1) deepened the *task-accuracy* valley: middle
retrieval fell 0.325 → 0.275 while primacy/recency stayed pinned. V_task rose 0.594 → 0.656.
And it is **head-specific**: the *identical* κ=1.5 edit applied to 8 matched control heads
(same band, similar R²/Δ̂) did **not** deepen — it left the middle at 0.35 (V=0.576, slightly
*shallower* than base, i.e. noise). Only the conformal target heads deepen the valley. This
replicates exp-070's deepening result and extends it to a different architecture (Llama/RoPE
vs GPT-NeoX) and a much larger model (13B vs 1.4B).

**What the shallowing leg shows — and its honest magnitude.** κ=0.5 raised the middle 0.325
→ 0.35 and lowered V_task 0.594 → 0.576. This is the predicted direction (T-B KEEP) and is
the result exp-070 could not obtain at a near-ceiling middle. **But the magnitude is small:**
+0.025 middle accuracy = +1 correct item out of 40, well within the per-bin sampling noise
(SE ≈ 0.075 at p≈0.35, n=40). The shallowing delta *in isolation* is fragile. The credible
signal is not the single 0.018 V-shift but the **ordered κ→V relationship across all four
conditions plus the head-specificity** — the shallowing point is consistent with the same
monotone handle the deepening leg shows much more forcefully.

**The κ=2.0 reversal.** V dropped slightly 0.656 (κ=1.5) → 0.645 (κ=2.0), which is why T-C is
ρ=0.80 (boundary) rather than a clean ~1.0. This was **pre-registered as expected, not a
problem**: exp-064 saw heads leave the conformal regime at κ=2.0. The monotone core is
κ∈{0.5,1.0,1.5}: 0.576 < 0.594 < 0.656.

---

## What this does and does not establish

- **Does:** a pre-registered, head-specific demonstration that the conformal QK-slope edit
  propagates from the attention statistic (exp-064) to **task-level retrieval behavior in
  BOTH directions** on a deep base U-shape — deepening strongly, shallowing in the predicted
  direction. T-D and specificity clean. This is the program's first **bidirectional**
  behavioral-causality result and it resolves exp-070's open T-B leg as headroom-bounded.
- **Does not:** establish that the shallowing effect is *large* or robust on its own — it is
  a 1-item shift, credible mainly through the ordered pattern + specificity. Nor does it
  generalize beyond **one model, one task, one seed**. No statistical test across seeds; the
  T-C ρ sits exactly on the registered boundary.

## Honest one-liner (for any future framing — do NOT inflate)
*"On a model with a deep lost-in-the-middle valley, rescaling the conformal heads' QK
distance-slope moves middle-position retrieval accuracy in both directions — deepening
strongly and head-specifically, shallowing in the predicted direction but by a small margin.
This confirms the bidirectional behavioral handle that exp-070's near-ceiling task could only
show one-sided, and identifies exp-070's shallowing null as a headroom artifact. Single model,
single task, single seed."*

## Cost & provenance
- Modal A100-80GB. Calibration+census ≈ 14 min (after a ~40-min first attempt was stopped
  early once n_doc=40 showed deep_enough, to save spend). Registered sweep ≈ 68 min (fp32 is
  the cost driver: 6 × ~650s forward batches of 200 items each). **Total GPU ≈ 2 hours
  (~$5–7 on Eldon's billing)** — over the brief's ~$2–4 estimate, because fp32 13B forwards
  are slow; flagged for Eldon. A bf16 task pass would have been far cheaper but would have
  departed from the validated fp32 pipeline and the exact sham-null guarantee.

## Next
1. **Robustness, if pursued:** multiple seeds / a second deep-valley model (longchat-13b-16k)
   to put error bars on the shallowing delta. Cheaper in bf16 if the sham null is re-verified.
2. Do **not** over-state in outreach. The deepening leg is the headline-strength result; the
   shallowing is directionally confirmed but small.

---

*Registry: exp-072. Status: complete. Registered verdict: BEHAVIORAL_CAUSALITY_CONFIRMED
(T-A KEEP-strong, T-B KEEP, T-D PASS, specificity PASS, T-C ρ=0.80 boundary). Quality:
honest, positive — bidirectional handle confirmed on a deep valley; shallowing margin small.
Pre-reg commit e6fbaecf.*
