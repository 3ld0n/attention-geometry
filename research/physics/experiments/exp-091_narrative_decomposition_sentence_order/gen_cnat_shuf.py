"""
exp-091 — C-NAT-shuf corpus generator.

Generates the sentence-shuffled variant of C-NAT for the narrative decomposition
experiment. Identical to exp-062 C-NAT generation except:
  - Sentences within each story are shuffled uniformly at random
  - Sentence shuffle seed: 9100 (distinct from doc-shuffle seed 3005)
  - Doc-level shuffle seed: 3005 (same as C-NAT, to control epoch composition)

Pre-registration: notes.md (written before this script ran).

Usage:
    .venv/bin/python3 gen_cnat_shuf.py               # generate C-NAT-shuf train bin
    .venv/bin/python3 gen_cnat_shuf.py --pilot        # 100M-token pilot (dry run)
    .venv/bin/python3 gen_cnat_shuf.py --held         # generate held-out bin
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

DOC_SHUFFLE_SEED     = 3005   # same as C-NAT (controls doc epoch order)
SENTENCE_SHUFFLE_SEED = 9100  # controls sentence order within each story


# Sentence splitter: split on [.!?] followed by whitespace.
# Keeps the delimiter attached to the sentence it ends.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def sentence_shuffle(text: str, rng: np.random.Generator) -> str:
    """Split story into sentences, shuffle, rejoin."""
    sentences = _SENT_SPLIT_RE.split(text.strip())
    if len(sentences) < 2:
        return text  # single sentence — return unchanged
    rng.shuffle(sentences)
    return " ".join(sentences)


def generate_cnat_shuf(n: int, pilot: bool = False) -> np.ndarray:
    """
    Stream TinyStories, sentence-shuffle each doc, tokenize, pack.
    Returns a uint16 array of token ids (not symbols — raw Pythia token ids).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    eos = tok.eos_token_id

    # Load dataset — use streaming=False for repeatable doc-level shuffle
    print("Loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  {len(ds)} stories loaded", flush=True)

    doc_rng  = np.random.default_rng(DOC_SHUFFLE_SEED)
    sent_rng = np.random.default_rng(SENTENCE_SHUFFLE_SEED)

    out = np.empty(n, dtype=np.uint16)
    done = 0
    epoch = 0

    while done < n:
        epoch += 1
        for idx in doc_rng.permutation(len(ds)):
            raw_text = ds[int(idx)]["text"]
            shuffled  = sentence_shuffle(raw_text, sent_rng)
            ids = tok(shuffled, add_special_tokens=False)["input_ids"] + [eos]
            take = min(len(ids), n - done)
            out[done:done + take] = np.asarray(ids[:take], dtype=np.uint16)
            done += take
            if done >= n:
                break
            if done % 50_000_000 < len(ids):
                print(f"  cnat_shuf: {done/1e6:.0f}M / {n/1e6:.0f}M (epoch {epoch})",
                      flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="Generate 100M-token pilot")
    ap.add_argument("--held",  action="store_true", help="Generate held-out set")
    args = ap.parse_args()

    if args.pilot:
        n    = PILOT_TOKENS
        name = "C-NAT-shuf_pilot.bin"
    elif args.held:
        n    = HELD_TOKENS
        name = "C-NAT-shuf_held.bin"
    else:
        n    = TRAIN_TOKENS
        name = "C-NAT-shuf.bin"

    out_path = OUT / name
    if out_path.exists():
        print(f"{out_path} already exists. Delete to regenerate.")
        sys.exit(0)

    print(f"Generating {n/1e6:.0f}M tokens → {out_path}", flush=True)
    tokens = generate_cnat_shuf(n, pilot=args.pilot)

    # Write as uint16 memmap (same format as exp-062 corpora)
    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n,))
    mm[:] = tokens
    mm.flush()
    print(f"Done. Written to {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
