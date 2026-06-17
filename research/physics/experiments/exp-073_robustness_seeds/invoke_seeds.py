"""Spawn the missing exp-073 seeds against the DEPLOYED app.

Unlike `modal run` (ephemeral app, reaped on client exit), spawning a call to a
*deployed* function persists server-side independent of this client — the fix for
the connection-instability that killed the serial/detached/ephemeral-spawn runs.

  modal deploy robustness_sweep.py        # once
  python invoke_seeds.py 123,2024,99      # fire-and-forget

Then poll the volume:  modal volume get hf-model-cache /exp073_seed<seed>.json
"""
import json
import os
import sys

import modal

here = os.path.dirname(os.path.abspath(__file__))
lock_path = os.path.join(os.path.dirname(here),
                         "exp-072_cloud_powered_slope_editing", "prereg_locked.json")
with open(lock_path) as f:
    lock = json.load(f)

seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "123,2024,99").split(",") if x.strip()]
fn = modal.Function.from_name("exp073-robustness-seeds", "run_one_seed")

print(f"Spawning {seeds} against deployed exp073-robustness-seeds.run_one_seed ...")
for s in seeds:
    call = fn.spawn(lock["model"], lock["n_doc"], lock["positions"],
                    lock["target_heads"], lock["sham_heads"], s, 4)
    print(f"  seed {s}: {call.object_id}")
print("Spawned. These persist server-side; poll the volume for exp073_seed<seed>.json.")
