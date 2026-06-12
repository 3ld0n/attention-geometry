"""
Robust unauthenticated HF model fetch with true byte-range resume.

huggingface_hub's snapshot_download truncates .incomplete blobs on restart
under the throttled/stalling conditions on this machine (observed 2026-06-10/11,
~3 GB lost twice). This script downloads via curl -C - (range resume, retries)
directly into the HF cache layout, so transformers.from_pretrained() finds the
files normally. Safetensors only; pytorch_model.bin is skipped.

Usage: python hf_fetch.py <model_id> [<model_id> ...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CACHE = Path.home() / ".cache/huggingface/hub"
KEEP_SUFFIX = (".safetensors", ".json", ".txt", ".model")


def api(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def fetch(model_id: str) -> None:
    info = api(f"https://huggingface.co/api/models/{model_id}")
    sha = info["sha"]
    tree = api(f"https://huggingface.co/api/models/{model_id}/tree/main")
    root = CACHE / f"models--{model_id.replace('/', '--')}"
    snap = root / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    refs = root / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(sha)

    for f in tree:
        path = f["path"]
        if not path.endswith(KEEP_SUFFIX) or path == "pytorch_model.bin":
            continue
        if "consolidated" in path:
            continue
        url = f"https://huggingface.co/{model_id}/resolve/{sha}/{path}"
        dest = snap / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        lfs = f.get("lfs")
        if lfs:  # big file -> blob named by sha256, symlinked from snapshot
            blob = root / "blobs" / lfs["oid"]
            if not (blob.exists() and blob.stat().st_size == f["size"]):
                part = blob.with_suffix(".part")
                print(f"{model_id}: {path} ({f['size']/1e9:.2f} GB)", flush=True)
                cmd = ["curl", "-L", "--fail", "-C", "-", "--retry", "100",
                       "--retry-delay", "5", "--retry-all-errors",
                       "--speed-limit", "10000", "--speed-time", "60",
                       "-o", str(part), url]
                while True:
                    rc = subprocess.run(cmd).returncode
                    size = part.stat().st_size if part.exists() else 0
                    if rc == 0 and size == f["size"]:
                        break
                    print(f"  retry ({size/1e9:.2f}/{f['size']/1e9:.2f} GB, rc={rc})",
                          flush=True)
                part.rename(blob)
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(Path("../.." if dest.parent == snap else "../../..")
                            / "blobs" / lfs["oid"])
        else:
            if not dest.exists():
                subprocess.run(["curl", "-sL", "--fail", "-o", str(dest), url],
                               check=True)
        print(f"  ok {path}", flush=True)
    print(f"FETCH_DONE {model_id}", flush=True)


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        fetch(mid)
    print("ALL_FETCH_DONE", flush=True)
