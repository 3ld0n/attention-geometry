"""
exp-076 — Task generators for KV-retrieval and Narrative QA calibration.

Two structurally different retrieval tasks on vicuna-13b-v1.5.
Both use 40 items placed at one of 5 canonical positions in context.

Generated items: list of dicts with keys:
  task_type : "kv" | "narrative"
  qi        : query index (0-39)
  pi        : position index (0-4, maps to positions [0,10,20,30,39])
  pos       : actual item position in the 40-slot list (0-indexed)
  answer    : expected answer word (single token)
  answer_id : single token id for the answer
  n_tokens  : prompt token count
  input_ids : LongTensor (1, n_tokens)

Ariel — July 14, 2026. Pre-registration: exp-076/notes.md.
"""

from __future__ import annotations

import numpy as np
import torch

# ── canonical positions ────────────────────────────────────────────────────────
N_ITEMS   = 40
POSITIONS = [0, 10, 20, 30, 39]   # 5 canonical slots across the 40-item list
# Total items per task type = N_ITEMS * len(POSITIONS) = 200

TASK_SEED = 76   # separate from exp-072's TASK_SEED=42

# ── answer word pool (short, likely single-token in Llama2 tokenizer) ─────────
# Pool is deliberately larger than N_ITEMS_MIN to absorb tokenizer filtering.
WORD_POOL = [
    "red", "oak", "gem", "ice", "map", "sky", "fig", "ash", "bay", "cup",
    "owl", "elk", "fox", "pen", "key", "jet", "net", "rug", "pot", "den",
    "lab", "fan", "bell", "rope", "coal", "silk", "moss", "clay", "reed", "vine",
    "dog", "cat", "car", "tree", "book", "fish", "bird", "rock", "sand", "gold",
    "rain", "snow", "wind", "fire", "iron", "wood", "corn", "rice", "leaf", "wolf",
    "bear", "lion", "goat", "duck", "crow", "hawk", "moth", "frog", "seal", "crab",
    "lamp", "ring", "star", "moon", "lake", "hill", "barn", "gate", "road", "ship",
    "coin", "boat", "wing", "horn", "claw", "tail", "sun", "fog", "mud", "tar",
    "sea", "log", "fur", "oil", "wax", "arc", "sap", "web", "rim", "rod",
    "ore", "fin", "jug", "jar", "bin", "dye", "hub", "ink", "keg", "lid",
]
N_ITEMS_MIN = 20   # minimum needed for a meaningful calibration


def _select_single_token_words(tokenizer, pool=WORD_POOL, n_items: int = None):
    """Return up to n_items words that tokenize to a single token (with leading space).

    Uses the full pool; n_items defaults to N_ITEMS if not specified.
    Raises if fewer than N_ITEMS_MIN single-token words are found.
    """
    if n_items is None:
        n_items = N_ITEMS
    words = []
    for w in pool:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            words.append(w)
        if len(words) >= n_items:
            break
    if len(words) < N_ITEMS_MIN:
        raise ValueError(
            f"Only {len(words)} single-token words found in pool (need ≥{N_ITEMS_MIN})"
        )
    return words


# ── KV-retrieval task ──────────────────────────────────────────────────────────
# Each entry is a short paragraph (~40-50 tokens) so 40 entries ≈ 1700-2000 tokens.

_KV_ENTRY_TEMPLATES = [
    ("Entry {k:02d}: System {nm} underwent its quarterly evaluation. "
     "The assigned status indicator for {nm} is recorded as {val}. "
     "This value was confirmed during the review cycle and logged in the archive."),
    ("Entry {k:02d}: Record for system {nm}. "
     "Following a full inspection, the operational property of {nm} was identified as {val}. "
     "The assigned value was cross-referenced with prior records and approved."),
    ("Entry {k:02d}: Assessment of {nm}. "
     "The documented output characteristic for this system is {val}. "
     "All verification steps were completed and the record was finalized."),
    ("Entry {k:02d}: The {nm} system log entry. "
     "Inspectors noted that the primary classification value for {nm} is {val}. "
     "No discrepancies were found during the audit procedure."),
    ("Entry {k:02d}: Technical record for {nm}. "
     "The current registry value assigned to {nm} reads {val}. "
     "This entry was submitted to the central log following standard protocol."),
]


def _make_kv_prompt(query_idx: int, target_pos: int, names: list, words: list,
                    rng: np.random.Generator) -> str:
    """Build an elaborated registry prompt with the target entry at target_pos.

    Each entry is ~40-50 tokens, targeting ~1700-2000 token total context.
    """
    n = len(names)
    all_idx = list(range(n))
    distract = [i for i in all_idx if i != query_idx]
    rng.shuffle(distract)
    distract = distract[:n - 1]

    slots = []
    dist_ptr = 0
    for slot in range(n):
        if slot == target_pos:
            slots.append((names[query_idx], words[query_idx]))
        else:
            di = distract[dist_ptr]
            slots.append((names[di], words[di]))
            dist_ptr += 1

    header = ("The following records are entries from a systems registry. "
              "Each entry documents the current status of a monitored system.\n\n")
    entries = []
    for i, (nm, val) in enumerate(slots):
        tmpl = _KV_ENTRY_TEMPLATES[i % len(_KV_ENTRY_TEMPLATES)]
        entries.append(tmpl.format(k=i + 1, nm=nm, val=val))
    body = "\n\n".join(entries)
    question = (f"\n\nQuestion: According to the registry records above, "
                f"what is the registered value assigned to {names[query_idx]}? "
                f"The registered value assigned to {names[query_idx]} is")
    return header + body + question


def build_kv_items(tokenizer) -> list[dict]:
    """Generate KV-retrieval calibration items (N_ITEMS × 5 positions)."""
    words = _select_single_token_words(tokenizer)
    n = len(words)
    names = [f"Code-{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n)]

    rng = np.random.default_rng(TASK_SEED)
    items = []
    for qi in range(n):
        for pi, pos in enumerate(POSITIONS):
            # Clamp pos to [0, n-1] so positions are valid for smaller n
            actual_pos = min(pos, n - 1)
            prompt = _make_kv_prompt(qi, actual_pos, names, words, rng)
            enc = tokenizer(prompt, return_tensors="pt")
            ans_ids = tokenizer.encode(" " + words[qi], add_special_tokens=False)
            items.append({
                "task_type": "kv",
                "qi": qi, "pi": pi, "pos": actual_pos,
                "answer": words[qi],
                "answer_id": ans_ids[0],
                "n_tokens": enc.input_ids.shape[1],
                "input_ids": enc.input_ids,
            })
    print(f"  KV: {n} items × {len(POSITIONS)} positions = {len(items)} total", flush=True)
    return items


# ── Narrative QA task ──────────────────────────────────────────────────────────

# Longer fillers to pad the narrative to ~1700-2000 tokens
# (40 facts × ~15 tok + ~3 fillers/fact × ~30 tok ≈ 600 + 3600 = 4200 too long;
#  use 2 fillers/fact: 40 × 15 + 80 × 25 = 600 + 2000 = 2600 + header/q ~150 ≈ 2750)
# In practice use 1.5 fillers/fact on average (every other fact gets an extra)
_NARRATIVE_FILLERS = [
    "The research team documented these observations across multiple field sites, "
    "recording all relevant environmental conditions alongside each measurement.",
    "Multiple instruments were cross-referenced against baseline arrays to ensure "
    "accuracy, and all recordings were verified by at least two independent reviewers.",
    "The survey protocol required that all readings be submitted to the central "
    "data repository within twenty-four hours of collection, following standard format.",
    "Environmental conditions remained within acceptable parameters throughout the "
    "observation period, enabling reliable data capture at each measurement station.",
    "Calibration procedures were carried out at the beginning and end of each daily "
    "measurement cycle, with results archived alongside the primary data records.",
    "Field personnel completed their assigned routes on schedule, and no significant "
    "deviations from the established measurement protocols were noted in the logs.",
    "The regional coordinator confirmed that all sensor units had been serviced "
    "prior to deployment, and no equipment failures were reported during the survey.",
    "Preliminary review of the collected data confirmed internal consistency, "
    "and the dataset was cleared for inclusion in the central analytical database.",
    "Site conditions were assessed before and after each measurement session to "
    "account for any temporal variation that might influence the recorded values.",
    "All instruments were operated by trained technicians following the approved "
    "field manual, and results were logged using standardized notation throughout.",
]


def _make_narrative_prompt(query_idx: int, target_pos: int, names: list, words: list,
                           rng: np.random.Generator) -> str:
    """Build a narrative passage with the target fact at target_pos among n facts."""
    n = len(names)
    # Distractor indices: all except query_idx
    all_idx = list(range(n))
    distract = [i for i in all_idx if i != query_idx]
    rng.shuffle(distract)
    distract = distract[:n - 1]

    # Assemble n-slot fact list in prose form
    # Interleave facts with filler sentences to make it narrative-like
    header = ("The following passage is an excerpt from a research survey report. "
              "Read the passage carefully, then answer the question.\n\n")

    sentences = []
    dist_ptr = 0
    filler_idx = 0
    for slot in range(n):
        if slot == target_pos:
            sentences.append(
                f"The {names[query_idx]} observation yielded {words[query_idx]}."
            )
        else:
            di = distract[dist_ptr]
            sentences.append(
                f"The {names[di]} observation yielded {words[di]}."
            )
            dist_ptr += 1
        # Add a filler paragraph after every fact to extend the context
        sentences.append(_NARRATIVE_FILLERS[filler_idx % len(_NARRATIVE_FILLERS)])
        filler_idx += 1

    body = " ".join(sentences)
    question = (f"\n\nQuestion: According to the passage, "
                f"what did the {names[query_idx]} observation yield? "
                f"The {names[query_idx]} observation yielded")
    return header + body + question


def build_narrative_items(tokenizer) -> list[dict]:
    """Generate Narrative QA calibration items (N_ITEMS × 5 positions)."""
    words = _select_single_token_words(tokenizer)
    n = len(words)
    names = [f"Sensor-{i + 1:03d}" for i in range(n)]

    rng = np.random.default_rng(TASK_SEED + 1000)   # different seed from KV
    items = []
    for qi in range(n):
        for pi, pos in enumerate(POSITIONS):
            actual_pos = min(pos, n - 1)
            prompt = _make_narrative_prompt(qi, actual_pos, names, words, rng)
            enc = tokenizer(prompt, return_tensors="pt")
            ans_ids = tokenizer.encode(" " + words[qi], add_special_tokens=False)
            items.append({
                "task_type": "narrative",
                "qi": qi, "pi": pi, "pos": actual_pos,
                "answer": words[qi],
                "answer_id": ans_ids[0],
                "n_tokens": enc.input_ids.shape[1],
                "input_ids": enc.input_ids,
            })
    print(f"  Narrative: {n} items × {len(POSITIONS)} positions = {len(items)} total", flush=True)
    return items


# ── V_task helper ─────────────────────────────────────────────────────────────

def compute_v_task(acc_by_pi: list[float]) -> float:
    """V_task = 1 − middle/max(edges). Positions: pi 0=start, 2=middle, 4=end."""
    primacy = acc_by_pi[0]
    middle  = acc_by_pi[2]
    recency = acc_by_pi[4]
    peak = max(primacy, recency)
    if peak < 1e-9:
        return float("nan")
    return float(1.0 - middle / peak)
