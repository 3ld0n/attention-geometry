# exp-071 — Cloud Calibration + Conformal Head Census (for exp-072)

*Date: 2026-06-16 (Cursor session, Ariel, solo). Cloud, Modal A100-80GB.*
*Analysis-only (base model + random-token census) — run BEFORE the exp-072 pre-registration.*

## Purpose
Pick and verify a deep-base-U-shape **RoPE + full-MHA** instruction model to adjudicate the
shallowing leg (T-B) left open by exp-070, and produce its conformal head census (the
artifact `run_exp070.py` reads from disk for Pythia, reproduced here for the cloud model).

Pivoted from the original MPT-30B-Instruct candidate because (1) the repo no longer exists on
HF, and (2) MPT is **ALiBi** — position is an additive score bias, not in the query content —
so the κ-operator (which rescales W_Q's projection onto the positional subspace) is a likely
mechanism mismatch. Kept on RoPE+MHA home turf instead.

## Result
**lmsys/vicuna-13b-v1.5** (40L × 40H, hidden 5120, head_dim 128, RoPE, full MHA, native 4096
context, ungated) at **N_doc = 40** gives a deep, two-sided lost-in-the-middle valley:

| position | primacy | near_prim | **middle** | near_rec | recency |
|----------|---------|-----------|------------|----------|---------|
| accuracy | 0.775 | 0.650 | **0.325** | 0.525 | 0.800 |

→ **base V_task = 0.594**, middle 0.325 far below edges, edges ≤ 0.80 (off ceiling). Real
headroom in **both** directions — exactly what exp-070's near-ceiling Pythia middle (0.85)
lacked. `deep_enough = True` (criteria: V≥0.30, edges≤0.85, middle<edges). 40 single-token
answer words available in the Llama tokenizer (full N_ITEMS=40).

## Census (exp-025/exp-059 pipeline, L=512, 50 random inputs, fp32)
**732 conformal heads** (R²≥0.85, Δ≥0.05). Selection band layers [13,33] (proportional
analogue of Pythia's 8–20 of 24): 298 candidates. Top-8 by R² → target heads; next 8 → sham:

- **Target:** L24H17 (Δ0.336,R²0.994), L27H18 (0.311,0.989), L24H2 (0.290,0.989),
  L25H4 (0.282,0.988), L26H30 (0.288,0.988), L22H33 (0.352,0.987), L25H30 (0.364,0.987),
  L25H33 (0.298,0.985).
- **Sham:** L24H25, L21H34, L26H29, L15H1, L21H0, L25H9, L29H31, L17H13.

All locked into `../exp-072_cloud_powered_slope_editing/prereg_locked.json` before the
edited forward pass.

## Provenance
`../exp-072_cloud_powered_slope_editing/cloud_slope_editing.py --phase calibrate`. Results:
`results.json`. (A first calibration run also evaluated n_doc=60/80 but was stopped early
once n_doc=40 showed a deep two-sided valley, to save GPU spend.)

*Registry: exp-071. Status: complete. Feeds exp-072.*
