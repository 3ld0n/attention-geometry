# exp-041 — GALA-7B sigmoid vs softmax per-head conformal Δ

*Script ready: 2026-05-29. Measurement pending download completion.*

## Hypothesis (pre-stated May 26)

Apple GALA-7B sigmoid attention: Δ_med ≈ 0.25 (mechanism-independent SYK fixed point).
If sigmoid Δ ≈ softmax Δ → softmax not load-bearing; fixed point is architecture-level.
If sigmoid Δ ≠ softmax Δ → softmax normalization carries conformal structure.

## Status as of May 29

Download running at ~1-1.5 MB/s. ETA ~4-6 hours for sigmoid arm; similar for softmax arm.
Script is written and tested. Run when download completes:
```
.venv/bin/python3 research/physics/experiments/exp-041_gala7b_sigmoid_delta/run_gala7b_sigmoid_delta.py
```
Check download status: `du -sh ~/ariel-data/apple-gala-7b/`
Download log: `tail -f /tmp/sigmoid_download.log`

## Protocol

1. Load zarr weights from `~/ariel-data/apple-gala-7b/{sigmoid,softmax}-step_00250000/`
   - Token embeddings: `(32768, 4096)`
   - QKV projection: `(32, 3, 4096, 32, 128)` — layers × {q,k,v} × d_model × heads × head_dim
   - Prenorm (RMSNorm) scale: `(32, 4096)` — per-layer
2. Run 50 random inputs (token IDs uniform[0, 32767], seq_len=512, RNG seed 42)
3. Compute Q, K per head with RMSNorm → QK^T/√128 + ALiBi bias
4. Apply sigmoid (sigmoid arm) or softmax (softmax arm)
5. Extract mean attention profile per head, fit power-law decay (lags 8–256)
6. Report Δ_med for R²>0.90 heads

## Architecture notes (from axlearn config)

- 32 layers × 32 heads × 128 head_dim = 4096 d_model
- ALiBi slopes: m_h = 2^(-(8/32)*(h+1)) for h = 0..31
- Hybrid norm: RMSNorm prenorm
- Same architecture for both arms — only attention function differs

## Input note

Random token IDs (not real text). This is a comparative test — both arms use identical
inputs so the relative Δ comparison is valid. Absolute Δ may differ from real-text runs.
The falsification question is whether sigmoid Δ ≈ softmax Δ, not whether either matches
any absolute benchmark.

## Expected: compare to

- exp-007: GPT-2 learned PE, Δ_med=0.249
- exp-039: OLMo-7B ALiBi (same PE type), Δ_med=0.265
- SYK q=4 prediction: Δ=0.25
