---
date: 2026-05-19
topic: rope-delta-bcft-verification
status: confirmed-negative
experiment: exp-030
---

# RoPE Δ Verification: Cron Numbers vs Raw BCFT JSON

## Question

Do Pythia-410m, 1.4b, and 6.9b have the median Δ values asserted in `notes/2026-05-18_rope-perturbation-extraction.md` and `notes/2026-05-19_rope-perturbation-extraction.md`? Does same depth (L=24) imply different Δ for 410m vs 1.4b?

## Method

Loaded raw head records from exp-025 BCFT pre-registered runs (`bcft_pre_registered_run_2026-04-17T092056Z.json`, `092239Z.json`). Computed median Δ under six filters (all heads; R2≥0.85; R2>0.90; BCFT conformal; SYK-near with |Δ−0.25|≤0.05). Compared to cron assertions and to March exp-011 published medians (70m/160m/410m, different protocol: seq_len=256, R2>0.90 only).

Script: `research/physics/experiments/exp-030_rope_delta_bcft_verification/verify_pythia_delta_bcft.py`.

## Results

### Pythia-6.9b — not in data

No `pythia-6.9b` entry in any BCFT pre-registered JSON. Cron Δ_med ≈ **0.60** matches **Pythia-70m** from March exp-011 (6 layers, R2>0.90 median Δ = 0.60), not a 32-layer model.

### Pythia-410m — cron ≈ March, not BCFT

| Source | median Δ | Notes |
|--------|----------|-------|
| Cron notes | 0.281–0.284 | asserted from "BCFT data" |
| March exp-011 | 0.28 | R2>0.90, seq_len=256, full 24 layers |
| **This extraction (BCFT JSON)** | **0.361** | R2>0.90, seq_len=512, **15/24 layers in file** |
| SYK-near (R2≥0.85, \|Δ−0.25\|≤0.05) | 0.253 | heads actually at q=4 |

### Pythia-1.4b — cron ≈ March 160m, not BCFT 1.4b

| Source | median Δ | Layers |
|--------|----------|--------|
| Cron notes | 0.375–0.383 | claimed L=24 |
| March **pythia-160m** | **0.38** | **L=12** |
| **This extraction (BCFT JSON)** | **0.327** | R2>0.90, 21/24 layers in file |
| SYK-near | **0.248** | near SYK Δ |

### Same depth, different Δ?

Only if you compare **incommensurate numbers** (410m March depth test 0.28 vs 1.4b mislabeled 160m 0.38). On the **same BCFT files** with SYK-near filter: 410m **0.253**, 1.4b **0.248**, difference **0.004**. The May 19 "parameter density scaling" narrative built on the cron table is **not supported**.

### Data quality

BCFT 410m run stored **239 heads** (layers 0–14) vs **384** expected. Any model-level median from that file is partial. Re-run or complete extraction before using 410m BCFT medians in scaling laws.

## Interpretation

The May 18–19 cron placeholder did real work flagging depth-convergence complexity, then **confabulated a number table** by mixing March depth-test results with wrong model labels. The honest negative from May 18 ("data access limitation") was accurate; May 19's asserted extractions were worse than the floor — specific numbers without valid provenance.

**What remains open (real):** RoPE models deviate from Δ=0.25 on the BCFT conformal subset (410m ~0.36, 1.4b ~0.34 at R2≥0.85, δ≥0.05). A capacity-dependent perturbation may still exist but needs **one protocol**, **complete layer coverage**, and models actually measured (not 70m labeled 6.9b).

## Next experiment

**GPT-2 Medium + Large** depth convergence (queue Tier 1 item 3) — learned-embedding control, ~1.5–3 GB downloads. Do not fit Δ(N/L) from the falsified three-point table.
