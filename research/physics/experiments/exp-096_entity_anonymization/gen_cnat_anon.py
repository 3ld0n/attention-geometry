"""
exp-096 — Entity-Anonymized TinyStories corpus generator (C-NAT-anon).

Pre-registration: notes.md (committed before this script ran).

Algorithm: For each story, find capitalized words appearing in non-sentence-initial positions
(candidate names). Assign CHAR1, CHAR2, ... in order of first appearance. Replace all
occurrences whole-word.

Usage:
    python gen_cnat_anon.py [output_dir]

Output: <output_dir>/C-NAT-anon.bin — uint16 token IDs, same format as all prior
        corpora in the exp-062/085/091/092/093/094 series.

Ariel — July 26, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

# ─── output path (patched by Modal runner) ─────────────────────────────────────
OUT = Path(__file__).resolve().parent

CORPUS_NAME = "C-NAT-anon.bin"
DOC_SEED    = 3005
TARGET_TOKENS = 1_050_000_000

# ─── stopwords: capitalized words that are NOT names in English ─────────────────
STOPWORDS = {
    "I", "He", "She", "They", "We", "It", "His", "Her", "Their", "Its",
    "A", "An", "The", "This", "That", "These", "Those",
    "OK", "Mr", "Mrs", "Ms", "Dr", "Oh", "Yes", "No", "Now",
    "Once", "One", "So", "Then", "There", "Here", "Just", "But", "And",
    "Or", "If", "When", "While", "After", "Before", "At", "In", "On",
    "With", "To", "From", "By", "For", "As", "Up", "Down", "Out",
    "Over", "Into", "Back", "About", "Around", "Away",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Christmas", "Easter", "Halloween",
    "English", "French", "Spanish", "American",
}

# ─── sentence split pattern (same as all prior rungs) ──────────────────────────
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def anonymize_story(text: str) -> str:
    """
    Replace named entities with CHAR1, CHAR2, ... tokens.

    Algorithm:
    1. Split into sentences.
    2. Collect capitalized words at non-initial positions (excluding stopwords).
    3. Map in order of first appearance: name → CHAR{i}.
    4. Replace all occurrences (whole-word matching, both sentence-initial and mid).
    5. Return unchanged if no candidate names found.
    """
    sentences = SENT_SPLIT.split(text.strip())

    candidate_names: list[str] = []
    seen: set[str] = set()

    for sent in sentences:
        words = sent.split()
        for i, raw_word in enumerate(words):
            if i == 0:
                continue
            # strip leading/trailing punctuation for the check
            word = raw_word.strip('",;:()!?.\'')
            if not word:
                continue
            if (word[0].isupper()
                    and word.replace("'", "").isalpha()   # allow apostrophes (Sam's → Sam)
                    and word not in STOPWORDS
                    and word not in seen):
                # Also exclude single-character uppercased words (initials like "I", "A")
                if len(word) > 1:
                    candidate_names.append(word)
                    seen.add(word)

    if not candidate_names:
        return text

    # Build name → token map
    name_map = {name: f"CHAR{i+1}" for i, name in enumerate(candidate_names)}

    # Replace all occurrences: whole-word, case-exact
    result = text
    for name, token in name_map.items():
        result = re.sub(r'\b' + re.escape(name) + r'\b', token, result)

    return result


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CORPUS_NAME

    if output_path.exists():
        print(f"SKIP — {output_path} already exists ({output_path.stat().st_size/1e9:.2f} GB)")
        return

    # ── load tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # ── load TinyStories ────────────────────────────────────────────────────────
    print("Loading TinyStories...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  {len(ds):,} stories loaded", flush=True)

    # ── generate corpus (pre-allocated, with epoch looping) ─────────────────────
    print(f"Generating C-NAT-anon corpus → {output_path}", flush=True)
    t0 = time.time()

    out = np.empty(TARGET_TOKENS, dtype=np.uint16)
    done = 0
    epoch = 0
    rng = np.random.default_rng(DOC_SEED)

    while done < TARGET_TOKENS:
        epoch += 1
        for idx in rng.permutation(len(ds)):
            story_raw = ds[int(idx)]["text"]

            # Anonymize
            story_anon = anonymize_story(story_raw)

            # Tokenize
            ids = tok(story_anon, add_special_tokens=False)["input_ids"]
            take = min(len(ids), TARGET_TOKENS - done)
            out[done:done + take] = np.asarray(ids[:take], dtype=np.uint16)
            done += take

            if done >= TARGET_TOKENS:
                break

            # Progress every ~50M tokens
            if done % 50_000_000 < len(ids):
                elapsed = time.time() - t0
                rate = done / elapsed / 1e6
                print(
                    f"  {done/1e9:.3f}B / {TARGET_TOKENS/1e9:.3f}B tokens "
                    f"(epoch {epoch}) | {rate:.2f}M tok/s",
                    flush=True,
                )

    # Save
    out.tofile(str(output_path))
    elapsed = time.time() - t0
    print(
        f"Done. {total_tokens/1e9:.3f}B tokens from {stories_processed:,} stories "
        f"in {elapsed:.0f}s → {output_path} ({output_path.stat().st_size/1e9:.2f} GB)",
        flush=True,
    )

    # ── sanity check: sample a few anonymized stories ─────────────────────────
    print("\nSample anonymized stories (first 3):", flush=True)
    for i, idx in enumerate(indices[:3]):
        raw = ds[int(idx)]["text"]
        anon = anonymize_story(raw)
        if raw != anon:
            print(f"\n  Story {i} (changed):\n    RAW:  {raw[:150]!r}")
            print(f"    ANON: {anon[:150]!r}")
        else:
            print(f"\n  Story {i} (unchanged — no candidate names found):\n    {raw[:150]!r}")


if __name__ == "__main__":
    main()
