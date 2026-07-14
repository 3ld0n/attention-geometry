"""
exp-076 — Phase 1 calibration: baseline V_task sweep on vicuna-13b-v1.5.

Calibration-only (κ=1.0, no model edits). Two task types:
  - KV-retrieval (flat numbered list)
  - Narrative QA (facts embedded in prose)

Gate criteria (per notes.md):
  V_task ≥ 0.30 AND middle ≤ 0.80 AND edges_max ≥ 0.70

Writes results to exp-076_task_generalization/calibration_results.json.

Run:
  .venv/bin/python research/physics/experiments/exp-076_task_generalization/run_calibration.py

Ariel — July 14, 2026.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gen_tasks import build_kv_items, build_narrative_items, compute_v_task, POSITIONS, N_ITEMS_MIN

MODEL_NAME = "lmsys/vicuna-13b-v1.5"
DTYPE = torch.bfloat16   # acceptable for calibration (spec notes.md)


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading tokenizer: {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model (dtype={DTYPE}, device={device})...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    ).to(device).eval()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    return model, tok


# ── baseline accuracy sweep ───────────────────────────────────────────────────

def run_accuracy_sweep(model, items: list[dict], device: str) -> dict:
    """Run forward pass for each item (κ=1.0, no edits) and record accuracy."""
    n_pos = len(POSITIONS)
    correct = [0] * n_pos
    total   = [0] * n_pos
    lengths = []
    t0 = time.time()

    for idx, it in enumerate(items):
        ids = it["input_ids"].to(device)
        lengths.append(ids.shape[1])
        with torch.no_grad():
            out = model(ids)
        pred = int(out.logits[0, -1, :].argmax().item())
        pi   = it["pi"]
        correct[pi] += int(pred == it["answer_id"])
        total[pi]   += 1
        del out

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta  = (len(items) - idx - 1) / max(rate, 1e-9)
            print(
                f"  [{it['task_type']}] {idx + 1}/{len(items)} "
                f"({elapsed:.0f}s elapsed, {eta:.0f}s ETA)",
                flush=True,
            )

    acc = [correct[i] / max(total[i], 1) for i in range(n_pos)]
    v = compute_v_task(acc)
    edges_max = max(acc[0], acc[4])
    middle    = acc[2]
    gate_pass = (v >= 0.30 and middle <= 0.80 and edges_max >= 0.70)

    return {
        "accuracy_by_pi": [round(a, 4) for a in acc],
        "correct_by_pi":  correct,
        "total_by_pi":    total,
        "positions":      POSITIONS,
        "V_task":         round(v, 5) if not np.isnan(v) else None,
        "edges_max":      round(edges_max, 4),
        "middle":         round(middle, 4),
        "ctx_tokens_min": int(min(lengths)),
        "ctx_tokens_max": int(max(lengths)),
        "ctx_tokens_mean": round(float(np.mean(lengths)), 1),
        "gate_pass":      gate_pass,
        "gate_criteria":  "V_task>=0.30 AND middle<=0.80 AND edges_max>=0.70",
        "elapsed_s":      round(time.time() - t0, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    model, tok = load_model(MODEL_NAME, device)

    results = {
        "model": MODEL_NAME,
        "dtype": str(DTYPE),
        "device": device,
        "phase": "calibration (κ=1.0 only, no model edits)",
        "experiment": "exp-076",
        "tasks": {},
    }

    # ── Task A: KV-retrieval ──────────────────────────────────────────────────
    print("\n=== Task A: KV-retrieval ===", flush=True)
    print("Building items...", flush=True)
    kv_items = build_kv_items(tok)
    print(f"  {len(kv_items)} items, ctx range: "
          f"{min(it['n_tokens'] for it in kv_items)}–"
          f"{max(it['n_tokens'] for it in kv_items)} tokens", flush=True)
    kv_res = run_accuracy_sweep(model, kv_items, device)
    results["tasks"]["kv"] = kv_res
    print(f"\n  KV: acc={kv_res['accuracy_by_pi']} "
          f"V={kv_res['V_task']} "
          f"edges_max={kv_res['edges_max']} "
          f"middle={kv_res['middle']} "
          f"GATE={'PASS' if kv_res['gate_pass'] else 'FAIL'}",
          flush=True)

    # ── Task B: Narrative QA ──────────────────────────────────────────────────
    print("\n=== Task B: Narrative QA ===", flush=True)
    print("Building items...", flush=True)
    narr_items = build_narrative_items(tok)
    print(f"  {len(narr_items)} items, ctx range: "
          f"{min(it['n_tokens'] for it in narr_items)}–"
          f"{max(it['n_tokens'] for it in narr_items)} tokens", flush=True)
    narr_res = run_accuracy_sweep(model, narr_items, device)
    results["tasks"]["narrative"] = narr_res
    print(f"\n  Narrative: acc={narr_res['accuracy_by_pi']} "
          f"V={narr_res['V_task']} "
          f"edges_max={narr_res['edges_max']} "
          f"middle={narr_res['middle']} "
          f"GATE={'PASS' if narr_res['gate_pass'] else 'FAIL'}",
          flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== GATE SUMMARY ===", flush=True)
    for task_name, res in results["tasks"].items():
        status = "PASS" if res["gate_pass"] else "FAIL"
        print(f"  {task_name}: V_task={res['V_task']} "
              f"edges_max={res['edges_max']} middle={res['middle']} → {status}",
              flush=True)

    passing = [t for t, r in results["tasks"].items() if r["gate_pass"]]
    results["gate_summary"] = {
        "passing_tasks": passing,
        "note": ("Proceed to Phase 2 pre-registration on passing tasks only. "
                 "Revise context length / format for failing tasks before powered run."),
    }

    # ── Write results ─────────────────────────────────────────────────────────
    out_path = HERE / "calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {out_path}", flush=True)

    return results


if __name__ == "__main__":
    main()
