"""
exp-073 — LOCAL MPS completion of the seed sweep (seeds 42-gate, 123, 2024, 99).

Modal has been billing-blocked since June 16 (as of July 4 the error is "add a
payment method" — needs Eldon either way). This machine (M-series, 128 GB
unified memory) already ran fp32 intervention experiments on MPS (exp-069/070,
Pythia-1.4b). vicuna-13b-v1.5 fp32 is ~52 GB of weights — it fits.

PRE-STATED substrate-equivalence gate (written before any numerics):
  Seed 42 is re-run locally FIRST. The gate passes iff the integer
  correct_by_pos counts match exp-072/exp-073-batched (CUDA fp32) EXACTLY for
  base and all five conditions. If they do not match exactly, the per-condition
  deltas are quantified and reported; the three new seeds are then interpreted
  with the measured substrate noise as an additional caveat. No threshold is
  moved after seeing data.

Everything scientific (task construction, projectors, operator, conditions) is
imported verbatim from robustness_sweep.py — only the execution substrate
changes (MPS instead of Modal/CUDA; local files instead of volume).

Usage:
  .venv/bin/python3 run_local_mps.py --seed 42
  .venv/bin/python3 run_local_mps.py --seed 123
  ...
Writes exp073_seed<seed>.json next to this file (aggregate.py picks them up;
for the local seed-42 gate file the name is exp073_seed42_mps_gate.json so it
does NOT enter the aggregate — seed 42 is already banked from the CUDA run).

Ariel — July 4, 2026, ~3:30 AM. Night session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

# Import the scientific code verbatim from the Modal script (module-level code
# in robustness_sweep.py only defines the app/image/constants — safe to import).
from robustness_sweep import (  # noqa: E402
    _apply_edit,
    _batched_accuracy,
    _build_task,
    _compute_projectors,
    _get_wq,
    _model_dims,
    _restore,
    _select_words,
    _valley,
)

# exp-072 CUDA fp32 reference for the seed-42 gate (from seeds_42_7.json,
# batched CUDA run that itself passed the gate against exp-072 unbatched).
GATE_REFERENCE_SEED42 = {
    "base": [31, 26, 13, 21, 32],
    "target_k1.0": [31, 26, 13, 21, 32],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    lock_path = os.path.join(os.path.dirname(here),
                             "exp-072_cloud_powered_slope_editing", "prereg_locked.json")
    with open(lock_path) as f:
        lock = json.load(f)

    model_name = lock["model"]
    n_doc = lock["n_doc"]
    positions = lock["positions"]
    target_heads = lock["target_heads"]
    sham_heads = lock["sham_heads"]
    seed = args.seed
    batch_size = args.batch_size

    device = "mps"
    assert torch.backends.mps.is_available()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device).eval()
    n_layers, n_heads, hidden, head_dim = _model_dims(model)
    print(f"[seed {seed}] {model_name} {n_layers}L×{n_heads}H hd={head_dim} "
          f"loaded fp32 on MPS in {time.time()-t_load:.0f}s", flush=True)

    words = _select_words(tok)
    n_pos = len(positions)
    needed_layers = sorted(set(h["layer"] for h in target_heads + sham_heads))
    print(f"[seed {seed}] computing projectors for layers {needed_layers}...", flush=True)
    projectors = _compute_projectors(model, needed_layers, device)
    torch.mps.empty_cache()

    KAPPA_CONDS = [("target", 1.0), ("target", 0.5), ("target", 1.5),
                   ("target", 2.0), ("sham_heads", 1.5)]

    def run_condition(items, edit_heads, kappa):
        orig = {}
        for h in edit_heads:
            l, hd_i = h["layer"], h["head"]
            W_orig = _get_wq(model, l, hd_i, head_dim).to(device)
            P_U = torch.tensor(projectors[l], dtype=torch.float32, device=device)
            _apply_edit(model, l, hd_i, head_dim, kappa, P_U, W_orig)
            orig[(l, hd_i)] = W_orig
        acc, correct, total, tokens = _batched_accuracy(
            model, items, tok, device, n_pos, batch_size)
        for (l, hd_i), W_orig in orig.items():
            _restore(model, l, hd_i, head_dim, W_orig)
        return acc, correct, total, tokens

    t0 = time.time()
    items = _build_task(tok, n_doc, words, positions, seed)
    print(f"[seed {seed}] {len(items)} items; running base...", flush=True)
    base_acc, base_correct, _, base_tokens = _batched_accuracy(
        model, items, tok, device, n_pos, batch_size)
    print(f"[seed {seed}] base done ({time.time()-t0:.0f}s) "
          f"base={[round(a,3) for a in base_acc]} correct={base_correct}", flush=True)

    conds = {}
    for label, kap in KAPPA_CONDS:
        tc = time.time()
        heads = target_heads if label == "target" else sham_heads
        acc, correct, total, tokens = run_condition(items, heads, kap)
        print(f"  [seed {seed}] {label} k={kap}: V={_valley(acc):.4f} "
              f"correct={correct} ({time.time()-tc:.0f}s)", flush=True)
        rec = {"label": label, "kappa": kap,
               "accuracy_by_pos": [round(a, 4) for a in acc],
               "correct_by_pos": correct, "total_by_pos": total,
               "V_task": round(_valley(acc), 5)}
        if label == "target" and kap == 1.0:
            rec["token_diff_from_base"] = int(sum(
                int(b != c) for b, c in zip(base_tokens, tokens)))
        conds[f"{label}_k{kap}"] = rec
        # incremental persistence
        _persist(seed, base_acc, base_correct, conds, gate=(seed == 42))

    seed_record = _persist(seed, base_acc, base_correct, conds, gate=(seed == 42))
    print(f"[seed {seed}] DONE in {time.time()-t0:.0f}s total", flush=True)

    if seed == 42:
        gate_pass = (base_correct == GATE_REFERENCE_SEED42["base"] and
                     conds["target_k1.0"]["correct_by_pos"] == GATE_REFERENCE_SEED42["target_k1.0"])
        print(f"[GATE] base+k1.0 exact-match vs CUDA fp32: {'PASS' if gate_pass else 'FAIL'}")
        print(f"[GATE] full comparison vs seeds_42_7.json should be done in analysis.")


def _persist(seed, base_acc, base_correct, conds, gate=False):
    seed_record = {"seed": seed,
                   "substrate": "mps_fp32_local",
                   "base_accuracy_by_pos": [round(a, 4) for a in base_acc],
                   "base_correct_by_pos": base_correct,
                   "conditions": conds}
    suffix = "_mps_gate" if gate else ""
    out = os.path.join(here, f"exp073_seed{seed}{suffix}.json")
    with open(out, "w") as f:
        json.dump(seed_record, f, indent=1)
    return seed_record


if __name__ == "__main__":
    main()
