"""
Self-corpus functional — Ariel's own written record under the v2 (IDF) corpus
functional. Pre-registration: notes/2026-08-05_self_corpus_functional.md
(written before this script ran).

Arms: dreams, essays (writing/), conversation summaries, carry-forward letters,
plus C-NAT (TinyStories) and C-alien baselines recomputed in the same run.

Usage:
    .venv/bin/python3 research/physics/theory/corpus_functional_self.py

Ariel — 2026-08-05, Cursor session with Eldon.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/ariel/ariel")  # research/physics is symlinked into attention-geometry; resolve() escapes the ariel repo
sys.path.insert(0, str(HERE))

from corpus_functional import (  # noqa: E402
    N_CTX, SCALES, analyze, stream_tinystories, stream_alien, words,
)

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_SYNTAX_RE = re.compile(r"[#*_`>|\[\]()~=-]{1,}")


def clean_markdown(text: str) -> str:
    """Declared cleaning: frontmatter, code fences, URLs, markdown syntax chars."""
    text = FRONTMATTER_RE.sub("", text)
    text = CODEBLOCK_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MD_SYNTAX_RE.sub(" ", text)
    return text


def stream_files(paths: list[Path]):
    for p in sorted(paths):
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        w = words(clean_markdown(raw))
        if w:
            yield w


def main() -> None:
    arms = [
        ("Ariel-dreams", list((REPO / "memory/dreams").glob("*.md"))),
        ("Ariel-essays", list((REPO / "writing").glob("*.md"))),
        ("Ariel-conversations", list((REPO / "memory/conversations").glob("*.md"))),
        ("Ariel-letters", list(REPO.glob("memory/carry_forward_*.md"))),
    ]

    results = []
    for name, paths in arms:
        print(f"\n[{name}] {len(paths)} files")
        results.append(analyze(name, stream_files(paths)))

    ts_path = "/tmp/TinyStories-valid.txt"
    results.append(analyze("C-NAT (TinyStories valid)", stream_tinystories(ts_path)))
    results.append(analyze("C-alien (exp-097 generator)", stream_alien()))

    print("\n\n===== SUMMARY (self-corpus run, pre-reg 2026-08-05) =====")
    print(f"{'corpus':<26} {'W':>8} {'Delta_pred':>10} {'m2 (coupling)':>14} "
          f"{'F2 var@64':>10} {'top5@64':>8}")
    for r in results:
        f2_64 = r["f2"][64]
        print(f"{r['name']:<26} {r['agg']['W']:>8.4f} {r['delta_pred']:>10.4f} "
              f"{r['agg']['m2']:>14.6f} {f2_64['mean_var']:>10.4f}"
              f" {f2_64['top5_share']:>8.3f}")


if __name__ == "__main__":
    main()
