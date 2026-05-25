# Apple sigmoid-attention — access blockers (exp-032)

*2026-05-25. Infrastructure probe; measurement not run.*

## Pre-stated falsification

If trained **sigmoid** attention yields median Δ ≈ 0.25 (same as GPT-2 softmax, exp-007/031), the SYK fixed point is **attention-mechanism-independent** (universality strengthened). If Δ ≠ 0.25 or no clean power law, **softmax normalization is load-bearing** (Category B in `the_attending_unit.md`).

## What exists

Apple’s paired checkpoints (ICLR 2025, `apple/ml-sigmoid-attention`):

| Model | GCS path (under `gs://axlearn-public/experiments/`) | Attention | Pos enc |
|-------|-----------------------------------------------------|-----------|---------|
| 7b-sigmoid | `gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/.../step_00250000` | Sigmoid | ALiBi + hybrid norm |
| 7b-softmax | `gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/.../step_00250000` | Softmax | ALiBi + hybrid norm |

**Not on HuggingFace.** No `apple/*sigmoid*` weights in HF search (May 25, 2026).

OpenELM (queue warm-up) is **RoPE + standard softmax**, not sigmoid.

## What was tried today

1. `gsutil` — **not installed** on this machine.
2. `pip install axlearn[core,apple-silicon] @ git+apple/axlearn` — **failed**: `tensorflow-io==0.37.3` not available for **Python 3.12** (only 0.37.1 listed).
3. `apple/OpenELM-270M` via transformers 5.8.1 — **failed**: `OpenELMConfig.__post_init__()` rejects `use_cache` from config.json.

## Verdict

**exp-032 status: aborted** (honest negative). No Δ number for sigmoid — infrastructure gap, not a physics null.

## Unblock checklist (for Eldon or next session)

1. **Python 3.11 venv** (or axlearn env that satisfies tensorflow-io pin) + install per notebook:
   `pip install --ignore-installed "axlearn[core,apple-silicon,gcp] @ git+https://github.com/apple/axlearn.git"`
2. **Google Cloud SDK** (`gcloud` + `gsutil`), auth, download ~7B checkpoints locally (128GB RAM sufficient for bfloat16 inference).
3. **Attention extraction** via AXLearn `InferenceRunner` — adapt exp-007 decay protocol; note ALiBi on both arms (compare sigmoid vs softmax **mechanism**, not vs GPT-2 learned embeddings).
4. Optional: fix OpenELM transformers compat for Apple RoPE baseline (separate from sigmoid).

Artifacts: `experiments/exp-032_sigmoid_infrastructure/probe_sigmoid_access.py`, `results.json`.
