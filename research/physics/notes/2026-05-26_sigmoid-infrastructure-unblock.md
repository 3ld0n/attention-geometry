# Apple sigmoid falsification — infrastructure unblock (May 26, 2026)

*Follows exp-032 (aborted). exp-033 infrastructure probe.*

## Question

Can we run the Tier 1 falsification test (paired 7B sigmoid vs softmax, exp-007 Δ protocol) on this machine today?

## What changed since May 25

| Blocker (exp-032) | May 26 status |
|-------------------|---------------|
| `gsutil` not installed | **Resolved** — `/opt/homebrew/bin/gsutil`, GCS list OK |
| axlearn pip fails on Py3.12 (tensorflow-io 0.37.3) | **Workaround** — `tensorflow-io==0.37.1`, `pip install --no-deps axlearn`, then manual `jax`, `flax`, `aqtp`, `seqio`, `nltk` |
| OpenELM warm-up | Still broken (transformers `use_cache`); irrelevant to sigmoid |
| Checkpoint size unknown | **Measured** — full checkpoint ~72.7 GB; **model weights only** (`gda/model` + `tf_*` per notebook) ~24.5 GB per arm |

## Verdict

**Infrastructure partially ready. No Δ measured.**

- GCS access confirmed for both `gala-7B-sigmoid-*` and `gala-7B-hybridnorm-*` (softmax baseline).
- `SigmoidAttention` imports from axlearn after dependency stack install.
- Trainer configs load after full axlearn deps (see `probe_infrastructure.py` results).
- **Remaining:** download ~25 GB × 2 checkpoints; implement attention extraction via `InferenceRunner` (notebook: `apple/ml-sigmoid-attention` `pretrained/axlearn_load_pretrained.ipynb`); adapt exp-007 decay + R²>0.90 median Δ.

## Install recipe (Python 3.12, this repo)

```bash
.venv/bin/pip install tensorflow-io==0.37.1
.venv/bin/pip install --no-deps "axlearn @ git+https://github.com/apple/axlearn.git"
.venv/bin/pip install jax==0.8.3 jaxlib==0.8.3 flax chex optax aqtp seqio==0.0.20 nltk==3.9.2
```

Note: `pip install axlearn[core,...]` still fails on pinned `tensorflow-io==0.37.3`. axlearn upstream now requires `==3.12.*` (May 26: Py3.11 venv is wrong direction).

## Falsification (unchanged)

- Sigmoid median Δ ≈ 0.25 → universality (mechanism-independent fixed point).
- Δ ≠ 0.25 or no clean power law → softmax normalization load-bearing.
- Compare **sigmoid vs softmax on same ALiBi stack**, not vs GPT-2 learned embeddings.

## For Eldon

Downloading both model-weight trees is ~50 GB and long-running. Worth confirming before `gsutil -m cp`. Script path: `research/physics/experiments/exp-033_sigmoid_measurement/`.

**Do not publish** "What the Stones Listen To" as final synthesis until this test completes or the essay is updated with the result.
