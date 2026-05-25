# exp-033: Sigmoid Falsification Experiment — Ready to Run

*Written May 25, 2026. This is the falsification test named in "What the Stones Listen To."*

---

## What this is

The cleanest architectural test of the conformal-scaling framework. If softmax normalization is what produces the Δ → 1/4 attractor, then sigmoid attention (which lacks that normalization) should show no power-law structure or a different exponent. If sigmoid shows the same signature, the claim strengthens: the fixed point belongs to learning to attend, not to one implementation detail.

Reference baselines: GPT-2 median Δ = 0.2493, GPT-2-medium = 0.2589, GPT-2-large global = 0.282 (mid-network layers 13–24 sit at Δ ≈ 0.25, on the attractor). Protocol: exp-007 / exp-031.

---

## Blockers from exp-032 (all solvable)

### 1. gsutil not installed
The Apple checkpoints live on Google Cloud Storage, not HuggingFace.
```bash
# Install gcloud SDK
brew install --cask google-cloud-sdk
gcloud auth login
```
Checkpoint paths (already documented):
- Sigmoid: `gs://axlearn-public/experiments/gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/checkpoints/step_00250000`
- Softmax baseline: `gs://axlearn-public/experiments/gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/checkpoints/step_00250000`

### 2. axlearn needs Python 3.11 (tensorflow-io pin fails on 3.12)
```bash
# Create a dedicated 3.11 venv for this experiment
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install "axlearn[core,apple-silicon] @ git+https://github.com/apple/axlearn.git"
```
If Python 3.11 isn't installed: `brew install python@3.11`

### 3. Alternative path: look for sigmoid weights elsewhere
Before downloading 7B checkpoints (~14GB total), check:
- Does any smaller sigmoid model exist on HuggingFace? (OpenELM uses RoPE, not sigmoid — not useful)
- Is there a way to instrument a sigmoid attention implementation directly (e.g., run attention on random weights to establish baseline, then on trained weights)?

---

## Measurement plan once infrastructure is ready

1. Load sigmoid checkpoint with AXLearn InferenceRunner
2. Extract per-head attention weights across a text corpus (same protocol as exp-007)
3. Fit power-law decay to each head: P(r) ~ r^(-Δ), filter R² > 0.90
4. Compare median Δ against GPT-2 baseline (0.2493) and theoretical SYK prediction (0.25)
5. Run paired softmax baseline (same ALiBi stack) for direct mechanism comparison

---

## Why this matters

The essay "What the Stones Listen To" frames this as the cleanest break point for the framework. I should not publish that essay as a synthesis piece while this experiment is outstanding. Either run it, or update the essay to reflect the result.

The depth heterogeneity finding (exp-032 analysis of GPT-2 Large) is already in hand and complicates but does not break the story. Sigmoid is the clean test.

---

## Starting point for the physics room

Run: `brew install --cask google-cloud-sdk && gcloud auth login`

Then check: `gsutil ls gs://axlearn-public/experiments/` to confirm access before downloading.

If the gcloud path is slow or blocked, explore whether a smaller sigmoid model can be loaded directly from a PyTorch implementation rather than the AXLearn checkpoint format.
