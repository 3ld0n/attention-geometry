# exp-033 — Apple sigmoid vs softmax conformal falsification

**Status (May 26):** Infrastructure probe only — **no Δ numbers**.

## Hypothesis

Trained Apple 7B **sigmoid** attention (ALiBi, hybrid norm) shows median per-head Δ ≈ 0.25 (R²>0.90), matching softmax baseline on the same stack and GPT-2 (exp-007). If not → softmax normalization is load-bearing for the SYK fixed point.

## May 26 session

- Unblocked GCS (`gsutil` works; exp-032 was wrong about missing SDK).
- axlearn partial install on Py3.12: `tensorflow-io==0.37.1`, `--no-deps`, manual JAX stack; full trainer load still needs more deps (`tokamax`, etc.).
- Checkpoint sizes: ~72.7 GB full; ~24.5 GB model weights per arm (notebook partial copy).

See `notes/2026-05-26_sigmoid-infrastructure-unblock.md`.

## Next

1. Finish axlearn deps OR follow notebook in clean env.
2. Download sigmoid + softmax model weights (~50 GB total).
3. `InferenceRunner` + exp-007 protocol on both arms.
