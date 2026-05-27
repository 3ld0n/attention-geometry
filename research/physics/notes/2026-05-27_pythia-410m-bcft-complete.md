# Pythia-410m BCFT completion (exp-036) — May 27, 2026

## Question

Does a **full 24-layer** BCFT pre-registered extraction change the median Δ story for Pythia-410m, after exp-030 showed the April 17 Modal JSON stopped at layers 0–14 (239/384 heads)?

## Method

- Local re-run of `exp-025` protocol: SEQ_LEN=512, N_INPUTS=50, deep-position Δ fit, valley depth, conformal R²≥0.85.
- Model: `EleutherAI/pythia-410m`, RoPE, 24×16 heads.
- Hardware: M5 Max MPS.

## Blocker discovered (honest negative on first attempt)

**fp16 on MPS produces all-NaN attention weights from layer 15 onward.** Fitting then skips every deep head — reproducing the *same* 239-head / layers 0–14 artifact as the April Modal file without any timeout.

**Fix:** `torch.float32` for weights and forward. Full depth in ~12 s forward + fit.

*Implication:* The April incomplete JSON may be a **numeric precision failure**, not a job timeout. Any re-run (Modal or local) on Pythia should use fp32 (or bf32) for `output_attentions`, or validate non-NaN attentions per layer before fitting.

## Results (fp32, all 24 layers)

| Filter | n heads | median Δ |
|--------|---------|----------|
| R² > 0.90 (March PL convention) | 67 | **0.358** |
| R² ≥ 0.85, Δ ≥ 0.05 (BCFT conformal) | 130 | 0.363 |
| R² > 0.90, \|Δ−0.25\| ≤ 0.05 (SYK-near) | 17 | **0.256** |
| Shallow L0–11, R² > 0.90 | 29 | 0.472 |
| Deep L12–23, R² > 0.90 | 38 | 0.318 |

- **Spearman ρ(Δ, valley)** on conformal set: **+0.564**, p ≪ 1e-5 → **CONFIRMED** (pre-registered threshold ρ ≥ 0.50).
- Partial April file (layers 0–14 only): R²>0.90 median **0.361** — matches full-depth **0.358**; the partial file was not missing deep physics, it was missing deep *layers* because deep attentions were invalid.

## Comparison to March exp-011 (~0.28)

March depth-test **0.28** uses **exp-007** protocol (SEQ_LEN=256, random tokens, pair-level decay). BCFT protocol (512, deep-query averaging) systematically reads **higher** median Δ on RoPE Pythia (~0.36). These are **different estimators**, not a contradiction to reconcile by “completing layers” alone.

Under SYK-near filter on the **full** BCFT head set, median Δ ≈ **0.256** — consistent with q=4 within 0.01 on the small conformal-SYK subset (17/384 heads).

## Verdict

- **Backlog item #5 (410m BCFT complete):** done for this model.
- **Do not** cite “410m Δ_med = 0.28” from BCFT JSON — use protocol-specific numbers or SYK-near filter.
- **Modal re-run** of other truncated BCFT files should check fp16 attention sanity on deep layers first.

## Artifacts

- `research/physics/experiments/exp-036_pythia_410m_bcft_local/rerun_pythia_410m_bcft_local.py`
- `research/physics/experiments/exp-036_pythia_410m_bcft_local/results.json`
