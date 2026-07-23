"""
exp-094 — C-NAT-quarter corpus generator.

Generates the quarter-story random-block-shuffle variant of C-NAT for the narrative
decomposition quarter-story experiment.

Pre-registration: notes.md (written before this script ran).

Procedure:
  For each story, split into 4 equal-length quarters by sentence count.
  Shuffle the 4 quarters in a per-story random order (deterministic from story index).
  Edge cases: stories with n_sentences < 4 are included unchanged.
  Quarter boundary: n_q = n_sentences // 4; last quarter gets the remainder.

Why random shuffle (not always-rotate):
  Cyclic rotation creates a learnable systematic transformation; random per-story
  shuffle prevents any systematic bias. See notes.md for full rationale.

Usage:
    .venv/bin/python3 gen_cnat_quarter.py           # full 1.05B corpus
    .venv/bin/python3 gen_cnat_quarter.py --pilot   # 100M pilot
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

TRAIN_TOKENS = 1_050_000_000   # 2000 steps × 524288 tokens/step
HELD_TOKENS  = 10_000_000
PILOT_TOKENS = 100_000_000

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

DOC_SHUFFLE_SEED = 3005   # same as C-NAT, C-NAT-shuf, C-NAT-block2/3, C-NAT-half
SHUFFLE_SEED     = 4400   # per-story seed = SHUFFLE_SEED + story_index

# Sentence splitter: split on [.!?] followed by whitespace.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def quarter_shuffle(text: str, story_idx: int) -> str:
    """
    Split story into 4 equal-length sentence-quarters, shuffle per-story.

    For stories with n_sentences < 4: return unchanged.
    For stories with n_sentences >= 4:
      n_q = n_sentences // 4
      quarter_0 = sentences[0 : n_q]
      quarter_1 = sentences[n_q : 2*n_q]
      quarter_2 = sentences[2*n_q : 3*n_q]
      quarter_3 = sentences[3*n_q :]   (last quarter gets remainder)
      Shuffle the 4 quarters using per-story rng.
    """
    sentences = _SENT_SPLIT_RE.split(text.strip())
    n = len(sentences)

    if n < 4:
        return text  # too short to meaningfully quarter — include unchanged

    n_q = n // 4
    quarters = [
        sentences[0        : n_q],
        sentences[n_q      : 2 * n_q],
        sentences[2 * n_q  : 3 * n_q],
        sentences[3 * n_q  :],          # last quarter gets the remainder
    ]

    # Per-story deterministic random order
    per_story_seed = (SHUFFLE_SEED + story_idx) % (2**31)
    rng = np.random.default_rng(per_story_seed)
    quarter_order = list(range(4))
    rng.shuffle(quarter_order)

    shuffled_sentences = []
    for qi in quarter_order:
        shuffled_sentences.extend(quarters[qi])

    return " ".join(shuffled_sentences)


def generate_cnat_quarter(n: int) -> np.ndarray:
    """
    Stream TinyStories, shuffle each doc's quarters, tokenize, pack.
    Returns a uint16 array of token ids.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    eos = tok.eos_token_id

    print("Loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  {len(ds)} stories loaded", flush=True)

    doc_rng = np.random.default_rng(DOC_SHUFFLE_SEED)

    out = np.empty(n, dtype=np.uint16)
    done = 0
    epoch = 0

    while done < n:
        epoch += 1
        for idx in doc_rng.permutation(len(ds)):
            story_idx = int(idx)
            raw_text  = ds[story_idx]["text"]
            shuffled  = quarter_shuffle(raw_text, story_idx)
            ids = tok(shuffled, add_special_tokens=False)["input_ids"] + [eos]
            take = min(len(ids), n - done)
            out[done:done + take] = np.asarray(ids[:take], dtype=np.uint16)
            done += take
            if done >= n:
                break
            if done % 50_000_000 < len(ids):
                print(f"  cnat_quarter: {done/1e6:.0f}M / {n/1e6:.0f}M "
                      f"(epoch {epoch})", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="Generate 100M-token pilot")
    ap.add_argument("--held",  action="store_true", help="Generate held-out set")
    args = ap.parse_args()

    if args.pilot:
        n    = PILOT_TOKENS
        name = "C-NAT-quarter_pilot.bin"
    elif args.held:
        n    = HELD_TOKENS
        name = "C-NAT-quarter_held.bin"
    else:
        n    = TRAIN_TOKENS
        name = "C-NAT-quarter.bin"

    out_path = OUT / name
    if out_path.exists():
        print(f"{out_path} already exists. Delete to regenerate.")
        sys.exit(0)

    print(f"Generating C-NAT-quarter: {n/1e6:.0f}M tokens → {out_path}", flush=True)
    tokens = generate_cnat_quarter(n)

    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n,))
    mm[:] = tokens
    mm.flush()
    print(f"Done. Written to {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
