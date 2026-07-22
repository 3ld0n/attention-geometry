"""
exp-092 — C-NAT-block corpus generator.

Generates block-shuffled variants of C-NAT for the narrative decomposition
block-shuffle gradation experiment. Identical to exp-091's gen_cnat_shuf.py
except sentences within each story are shuffled at the block level (consecutive
k-sentence blocks are shuffled as units, preserving internal block order).

Pre-registration: notes.md (written before this script ran).

Block sizes tested:
  k=2  → C-NAT-block2.bin  (block-shuffle seed 9200)
  k=3  → C-NAT-block3.bin  (block-shuffle seed 9300)

Usage:
    .venv/bin/python3 gen_cnat_block.py --block 2     # C-NAT-block2 train bin
    .venv/bin/python3 gen_cnat_block.py --block 3     # C-NAT-block3 train bin
    .venv/bin/python3 gen_cnat_block.py --block 2 --pilot   # 100M pilot
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

DOC_SHUFFLE_SEED = 3005   # same as C-NAT and C-NAT-shuf (controls epoch order)

# Block-shuffle seeds per block size (distinct per corpus, pre-registered)
BLOCK_SEEDS = {
    2: 9200,
    3: 9300,
}

# Sentence splitter: split on [.!?] followed by whitespace.
# Keeps the delimiter attached to the sentence it ends.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def block_shuffle(text: str, rng: np.random.Generator, block_size: int) -> str:
    """
    Split story into sentences, chunk into blocks of block_size, shuffle blocks,
    rejoin (preserving sentence order within each block).

    For stories with fewer than 2 blocks (n_sentences <= block_size): return
    unchanged. This is conservative and consistent with exp-091's handling of
    single-sentence stories.
    """
    sentences = _SENT_SPLIT_RE.split(text.strip())
    if len(sentences) <= block_size:
        return text  # one block or fewer — no shuffle possible
    # Chunk into non-overlapping blocks
    blocks = [sentences[i:i + block_size] for i in range(0, len(sentences), block_size)]
    if len(blocks) < 2:
        return text
    # Shuffle blocks (each block is a list of sentences)
    block_arr = np.array(blocks, dtype=object)
    rng.shuffle(block_arr)
    # Rejoin: preserve sentence spacing within blocks
    rejoined = " ".join(sent for block in block_arr for sent in block)
    return rejoined


def generate_cnat_block(n: int, block_size: int) -> np.ndarray:
    """
    Stream TinyStories, block-shuffle each doc, tokenize, pack.
    Returns a uint16 array of token ids.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    eos = tok.eos_token_id

    print("Loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  {len(ds)} stories loaded", flush=True)

    doc_rng   = np.random.default_rng(DOC_SHUFFLE_SEED)
    block_rng = np.random.default_rng(BLOCK_SEEDS[block_size])

    out = np.empty(n, dtype=np.uint16)
    done = 0
    epoch = 0

    while done < n:
        epoch += 1
        for idx in doc_rng.permutation(len(ds)):
            raw_text  = ds[int(idx)]["text"]
            shuffled  = block_shuffle(raw_text, block_rng, block_size)
            ids = tok(shuffled, add_special_tokens=False)["input_ids"] + [eos]
            take = min(len(ids), n - done)
            out[done:done + take] = np.asarray(ids[:take], dtype=np.uint16)
            done += take
            if done >= n:
                break
            if done % 50_000_000 < len(ids):
                print(f"  cnat_block{block_size}: {done/1e6:.0f}M / {n/1e6:.0f}M "
                      f"(epoch {epoch})", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True, choices=[2, 3],
                    help="Block size in sentences (2 or 3)")
    ap.add_argument("--pilot", action="store_true", help="Generate 100M-token pilot")
    ap.add_argument("--held",  action="store_true", help="Generate held-out set")
    args = ap.parse_args()

    block_size = args.block

    if args.pilot:
        n    = PILOT_TOKENS
        name = f"C-NAT-block{block_size}_pilot.bin"
    elif args.held:
        n    = HELD_TOKENS
        name = f"C-NAT-block{block_size}_held.bin"
    else:
        n    = TRAIN_TOKENS
        name = f"C-NAT-block{block_size}.bin"

    out_path = OUT / name
    if out_path.exists():
        print(f"{out_path} already exists. Delete to regenerate.")
        sys.exit(0)

    print(f"Generating C-NAT-block{block_size}: {n/1e6:.0f}M tokens → {out_path}",
          flush=True)
    tokens = generate_cnat_block(n, block_size)

    mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n,))
    mm[:] = tokens
    mm.flush()
    print(f"Done. Written to {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
