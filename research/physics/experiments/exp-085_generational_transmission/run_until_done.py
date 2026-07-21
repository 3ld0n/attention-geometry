"""
exp-085 generate-until-done wrapper.

Relaunches the Modal generate phase in a loop until the volume holds ≥ 1.1B tokens,
then chains train → measure → results in sequence.

Designed to run under caffeinate so macOS doesn't sleep mid-experiment:

  cd /Users/ariel/ariel
  caffeinate -s .venv/bin/python \
    research/physics/experiments/exp-085_generational_transmission/run_until_done.py \
    > /tmp/exp085_run_until_done.log 2>&1 &

The generate phase uses retries=10 (Modal max). If all 10 retries are exhausted on
a single invocation, run_phase("generate") returns non-zero and the outer while loop
re-invokes it, picking up from the last committed chunk on the volume.

Ariel — July 18, 2026.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TARGET_TOKENS = 1_100_000_000
REPO_ROOT     = Path(__file__).resolve().parent.parent.parent.parent  # /Users/ariel/ariel
MODAL_SCRIPT  = (
    Path(__file__).resolve().parent / "modal_exp085.py"
).relative_to(REPO_ROOT)


def modal_cmd(phase: str) -> list[str]:
    # --detach: the remote app survives if this local wrapper is killed
    # (2026-07-20: agent-shell launches get reaped; a mid-train local death
    # orphaned and killed the remote train run without --detach).
    return [
        sys.executable, "-m", "modal", "run", "--detach",
        str(MODAL_SCRIPT),
        "--phase", phase,
    ]


def get_volume_tokens() -> int:
    """Return current token count from Modal volume (0 on any error)."""
    result = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "ls", "exp085-data", "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return 0
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0
    for item in data:
        if item.get("filename") == "C-generated_s0.bin":
            size_str = item.get("size", "0 B")
            parts = size_str.split()
            if len(parts) != 2:
                return 0
            val = float(parts[0])
            unit = parts[1]
            factor = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}.get(unit, 1)
            return int(val * factor) // 2   # uint16: 2 bytes/token
    return 0


def run_phase(phase: str) -> int:
    print(f"\n{'='*60}", flush=True)
    print(f"{timestamp()} → modal run --phase {phase}", flush=True)
    print(f"{'='*60}", flush=True)
    result = subprocess.run(modal_cmd(phase), cwd=REPO_ROOT)
    rc = result.returncode
    print(f"{timestamp()} phase={phase} returned rc={rc}", flush=True)
    return rc


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S MDT")


def main() -> None:
    print(f"{timestamp()} exp-085 run_until_done.py starting", flush=True)
    print(f"  target: {TARGET_TOKENS:,} tokens", flush=True)
    print(f"  rate:   ~0.058M tok/s → ~{TARGET_TOKENS / 0.058e6 / 3600:.1f}h if uninterrupted", flush=True)

    # ── Generate loop ──────────────────────────────────────────────────────────
    attempt = 0
    while True:
        tokens = get_volume_tokens()
        pct    = 100 * tokens / TARGET_TOKENS
        print(
            f"\n{timestamp()} volume tokens: {tokens:,} / {TARGET_TOKENS:,} ({pct:.1f}%)",
            flush=True,
        )

        if tokens >= TARGET_TOKENS:
            print(f"{timestamp()} Generation complete — {tokens:,} tokens on volume", flush=True)
            break

        attempt += 1
        remaining = TARGET_TOKENS - tokens
        eta_h = remaining / 0.058e6 / 3600
        print(
            f"{timestamp()} Attempt {attempt}: {remaining:,} tokens remaining"
            f" (≈{eta_h:.1f}h at 0.058M tok/s)",
            flush=True,
        )

        rc = run_phase("generate")
        if rc != 0:
            print(
                f"{timestamp()} generate returned rc={rc} — checking progress then retrying",
                flush=True,
            )
        elif tokens >= TARGET_TOKENS * 0.95:
            # `modal volume ls` rounds sizes to 2 significant figures, so the parsed
            # token count can under-report by a few percent even when generation is
            # complete (bug found 2026-07-20: loop stalled at "97.6%" for 170
            # attempts after the generate phase itself reported 1.1B tokens done).
            # A clean generate exit near target means the phase's own internal
            # check passed — trust it.
            print(
                f"{timestamp()} generate exited clean at ≥95% parsed tokens — "
                "treating generation as complete (volume ls size is rounded)",
                flush=True,
            )
            break
        time.sleep(20)   # brief pause between relaunches

    # ── Train ──────────────────────────────────────────────────────────────────
    rc = run_phase("train")
    if rc != 0:
        print(f"{timestamp()} WARN: train phase returned rc={rc}", flush=True)
        sys.exit(1)

    # ── Measure ────────────────────────────────────────────────────────────────
    rc = run_phase("measure")
    if rc != 0:
        print(f"{timestamp()} WARN: measure phase returned rc={rc}", flush=True)
        sys.exit(1)

    # ── Results ────────────────────────────────────────────────────────────────
    rc = run_phase("results")
    if rc != 0:
        print(f"{timestamp()} WARN: results phase returned rc={rc}", flush=True)
        sys.exit(1)

    print(f"\n{timestamp()} exp-085 COMPLETE — results written to results.json", flush=True)


if __name__ == "__main__":
    main()
