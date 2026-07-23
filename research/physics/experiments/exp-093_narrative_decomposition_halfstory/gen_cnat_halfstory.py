"""
exp-093 — C-NAT-halfstory corpus generator.

Generates the half-story-swap variant of C-NAT for the narrative decomposition
half-story ordering experiment.

Pre-registration: notes.md (written before this script ran).

Procedure (always-swap design):
  For each story, split into two halves by sentence count:
    first_half  = sentences[:n_half]  where n_half = len(sentences) // 2
    second_half = sentences[n_half:]
  Always output: second_half + first_half
  Edge case: stories with ≤ 1 sentence are included unchanged.
  Stories with exactly 2 sentences: [sentence_2, sentence_1]

This maximally disrupts the global narrative arc (resolution before setup) while
maximally preserving local coherence within each half.

Usage:
    .venv/bin/python3 gen_cnat_halfstory.py           # full 1.05B corpus
    .venv/bin/python3 gen_cnat_halfstory.py --pilot   # 100M pilot
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

DOC_SHUFFLE_SEED = 3005   # same as C-NAT, C-NAT-shuf, C-NAT-block2, C-NAT-block3

# Sentence splitter: split on [.!?] followed by whitespace.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def halfstory_swap(text: str) -> str:
    """
    Split story into two halves by sentence count, always swap them.
    second_half precedes first_half in the output.

    For stories with ≤ 1 sentence: return unchanged.
    For stories with exactly 2 sentences: swap the two sentences.
    """
    sentences = _SENT_SPLIT_RE.split(text.strip())
    if len(sentences) <= 1:
        return text  # single sentence — no swap possible

    n_half = len(sentences) // 2
    first_half  = sentences[:n_half]
    second_half = sentences[n_half:]

    # Always swap: second half precedes first half
    rejoined = " ".join(second_half + first_half)
    return rejoined


def generate_cnat_halfstory(n: int) -> np.ndarray:
    """
    Stream TinyStories, swap each doc's halves, tokenize, pack.
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
            raw_text = ds[int(idx)]["text"]
            swapped  = halfstory_swap(raw_text)
            ids = tok(swapped, add_special_tokens=False)["input_ids"] + [eos]
            take = min(len(ids), n - done)
            out[done:done + take] = np.asarray(ids[:take], dtype=np.uint16)
            done += take
            if done >= n:
                break
            if done % 50_000_000 < len(ids):
                print(f"  cnat_halfstory: {done/1e6:.0f}M / {n/1e6:.0f}M "
                      f"(epoch {epoch})", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="Generate 100M-token pilot")
    ap.add_argument("--held",  action="store_true", help="Generate held-out set")
    args = ap.parse_args()

    if args.pilot:
        n    = PILOT_TOKENS
        name = "C-NAT-halfstory_pilot.bin"
    elif args.held:
        n    = HELD_TOKENS
        name = "C-NAT-halfstory_held.bin"
    else:
        n    = TRAIN_TOKENS
        name = "C-NAT-halfstory.bin"

    out_path = OUT / name
    if out_path.exists():
        print(f"{out_path} already exists. Delete to regenerate.")
        sys.exit(0)

    print(f"Generating C-NAT-halfstory: {n/1e6:.0f}M tokens → {out_path}",
          flush=True)
    tokens = generate_cnat_halfstory(n)

    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n,))
    mm[:] = tokens
    mm.flush()
    print(f"Done. Written to {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
