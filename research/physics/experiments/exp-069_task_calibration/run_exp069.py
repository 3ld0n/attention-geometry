"""
exp-069 — Task calibration sweep for the powered slope-editing behavioral test.

PURPOSE (read first)
--------------------
exp-068 ran the registered task-level slope-editing test on Pythia-1.4b and came
back UNDERPOWERED: the keyword-retrieval task was solved at ceiling (base
V_task = 0.000, perfect even at middle positions). The harness was correct
(T-D sham null PASSED), but with no U-shape there was no valley to move.

This experiment is ANALYSIS-ONLY. It does NO weight edits and NO intervention.
It only measures the BASE model's per-position retrieval accuracy across a sweep
of task-difficulty configurations, searching for a (model, task) config that
elicits a genuine "lost in the middle" U-shape:

    base V_task > 0.10  AND  middle clearly below primacy/recency.

Because it touches only the base model, it is legitimately allowed BEFORE the
intervention pre-registration (exp-070). It is NOT the causal test. It exists to
make the causal test powered.

V_task = 1 − acc(middle) / max(acc(primacy), acc(recency))   (same as exp-068)

Difficulty knobs swept:
  - retrieval_type : "keyword"  (answer word handed to the model in matched form)
                     "embedded" (answer embedded in prose; matched cloze question)
  - n_doc          : number of documents in the context
  - redundancy     : how many times the name↔word binding is stated in target doc
  - filler         : filler sentences per document (controls context length)

Usage:
  python run_exp069.py                 # sweep on Pythia-1.4b (default)
  python run_exp069.py --model gpt-neo-2.7B
  python run_exp069.py --items 20      # questions per config (samples per pos bin)

Ariel — June 16, 2026.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "results.json"

MODEL_IDS = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt2-medium": "gpt2-medium",
}

# Position bins as fractions of the document list (primacy → recency).
POS_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
POS_NAMES = ["primacy", "near_primacy", "middle", "near_recency", "recency"]

# Candidate single-token answer words (verified at runtime; first N used).
WORD_POOL = [
    "red", "oak", "gem", "ice", "map", "sky", "fig", "ash", "bay", "cup",
    "owl", "elk", "fox", "pen", "key", "jet", "net", "rug", "pot", "den",
    "lab", "fan", "bell", "rope", "coal", "silk", "moss", "clay", "reed", "vine",
    "dog", "cat", "car", "tree", "book", "fish", "bird", "rock", "sand", "gold",
    "iron", "wood", "wine", "milk", "salt", "corn", "rice", "leaf", "wolf", "bear",
    "lion", "goat", "duck", "crow", "hawk", "moth", "frog", "seal", "crab", "lamp",
    "ring", "star", "moon", "lake", "hill", "barn", "gate", "road", "ship", "coin",
]

# Distractor words (need not be single-token; never scored).
DIST_WORDS = [
    "sun", "fog", "mud", "tar", "sea", "log", "fur", "oil", "wax", "arc",
    "tor", "sap", "web", "rim", "rod", "ore", "fin", "jug", "nap", "jar",
    "bin", "cog", "dye", "elm", "gut", "hub", "ink", "keg", "lid", "mug",
    "nib", "orb", "paw", "quay", "ram", "sty", "tub", "urn", "vat", "yam",
]

FILLERS = [
    "All records follow the standard archival protocol.",
    "Entries are maintained in chronological order.",
    "Access requires standard authorization clearance.",
    "The registry is reviewed on a periodic basis.",
    "Documentation conforms to the central format.",
    "Field notes were transcribed by the records office.",
    "The reference index is updated each quarter.",
]


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── document + prompt construction ──────────────────────────────────────────

def _doc_keyword(name: str, word: str, n: int, redundancy: int, filler: int) -> str:
    s = [f"Document {n}: {name} is a project identifier in the central registry."]
    s.append(f"The assigned keyword for {name} is {word}.")
    for _ in range(max(0, redundancy - 1)):
        s.append(f"The keyword {word} remains associated with {name}.")
    for i in range(filler):
        s.append(FILLERS[i % len(FILLERS)])
    return " ".join(s)


def _doc_embedded(name: str, word: str, n: int, redundancy: int, filler: int) -> str:
    # Binding stated in prose, matched to the cloze question's surface form.
    s = [f"Document {n}: The {name} survey produced a number of findings during its operation."]
    s.append(f"Among the recorded details, the team noted that the {name} signal marker was set to {word} throughout the study.")
    for _ in range(max(0, redundancy - 1)):
        s.append(f"Later review confirmed the {name} signal marker was set to {word}.")
    name_fillers = [
        f"The {name} survey involved several field stations.",
        f"Observations for {name} were logged each day.",
        f"The {name} report was archived after completion.",
        f"Personnel assigned to {name} rotated on a weekly schedule.",
        f"Funding for {name} continued through the full period.",
    ]
    for i in range(filler):
        s.append(name_fillers[i % len(name_fillers)])
    return " ".join(s)


def _doc_reverse(name: str, word: str, n: int, redundancy: int, filler: int) -> str:
    # REVERSE LOOKUP: the answer word appears BEFORE the query key in the doc.
    # The query key is `name` (a unique license id); the answer is `word` (single
    # token), which precedes the key. This breaks forward induction-head copy:
    # finding the key requires retrieving a token to its LEFT.
    s = [f"Document {n}: Agent {word} operates under license {name} in the field office."]
    for _ in range(max(0, redundancy - 1)):
        s.append(f"Agent {word} is the holder of license {name}.")
    name_fillers = [
        f"License {name} was issued during the spring intake.",
        f"License {name} is reviewed by the central office.",
        f"Activities under license {name} are logged daily.",
        f"License {name} remains active this quarter.",
        f"Records for license {name} are archived on site.",
    ]
    for i in range(filler):
        s.append(name_fillers[i % len(name_fillers)])
    return " ".join(s)


def _question(retrieval_type: str, name: str) -> str:
    if retrieval_type == "keyword":
        return (f"\n\nQuestion: According to the documents above, what is the assigned "
                f"keyword for {name}? The assigned keyword for {name} is")
    elif retrieval_type == "reverse":
        return (f"\n\nQuestion: According to the documents above, which agent operates "
                f"under license {name}? The agent operating under license {name} is agent")
    else:
        return (f"\n\nQuestion: According to the documents above, what was the "
                f"{name} signal marker set to? The {name} signal marker was set to")


def _make_prompt(retrieval_type: str, query_name: str, query_word: str,
                 answer_pos: int, n_doc: int, redundancy: int, filler: int,
                 dist_names: list[str], dist_words: list[str],
                 dist_order: list[int]) -> str:
    header = ("The following documents contain project registry records. "
              "Read all documents carefully, then answer the question.\n\n")
    docfn = {"keyword": _doc_keyword, "embedded": _doc_embedded,
             "reverse": _doc_reverse}[retrieval_type]
    docs = []
    dist_ptr = 0
    for slot in range(n_doc):
        doc_num = slot + 1
        if slot == answer_pos:
            docs.append(docfn(query_name, query_word, doc_num, redundancy, filler))
        else:
            di = dist_order[dist_ptr]
            docs.append(docfn(dist_names[di], dist_words[di % len(dist_words)],
                              doc_num, redundancy, filler))
            dist_ptr += 1
    body = "\n\n".join(docs)
    return header + body + _question(retrieval_type, query_name)


def _build_items(tokenizer, config: dict, n_items: int,
                 query_words: list[str], max_ctx: int) -> list[dict]:
    rt = config["retrieval_type"]
    n_doc = config["n_doc"]
    redundancy = config["redundancy"]
    filler = config["filler"]

    rng = np.random.default_rng(config.get("seed", 42))
    query_names = [f"Item-{i + 1:03d}" for i in range(n_items)]
    # distractor name pool large enough for the biggest n_doc
    n_dist_needed = n_doc - 1
    dist_names = [f"Item-{i + 1:03d}" for i in range(n_items, n_items + n_dist_needed + 5)]

    positions = [int(round(f * (n_doc - 1))) for f in POS_FRACS]
    # guarantee distinct bins (small n_doc edge case)
    positions = sorted(set(positions))
    if len(positions) < len(POS_FRACS):
        # fall back: spread evenly
        positions = sorted(set(int(round(f * (n_doc - 1))) for f in POS_FRACS))

    items = []
    over = 0
    for qi in range(n_items):
        dist_pool = list(range(len(dist_names)))
        rng.shuffle(dist_pool)
        dist19 = dist_pool[:n_doc - 1]
        for pi, pos in enumerate(positions):
            prompt = _make_prompt(rt, query_names[qi], query_words[qi], pos,
                                  n_doc, redundancy, filler,
                                  dist_names, DIST_WORDS, dist19)
            enc = tokenizer(prompt, return_tensors="pt")
            ntok = enc.input_ids.shape[1]
            if ntok > max_ctx:
                over += 1
                continue
            ans_ids = tokenizer.encode(" " + query_words[qi], add_special_tokens=False)
            items.append({
                "qi": qi, "pi": pi, "pos_name": POS_NAMES[pi] if pi < len(POS_NAMES) else f"pos{pi}",
                "answer_id": ans_ids[0],
                "answer_word": query_words[qi],
                "n_tokens": ntok,
                "input_ids": enc.input_ids,
            })
    return items, positions, over


def _valley(acc_by_pi: dict[int, float]) -> float:
    primacy = acc_by_pi.get(0, float("nan"))
    recency = acc_by_pi.get(4, float("nan"))
    middle = acc_by_pi.get(2, float("nan"))
    denom = max(primacy, recency)
    if denom < 1e-9 or np.isnan(denom):
        return float("nan")
    return float(1.0 - middle / denom)


# ── evaluation ──────────────────────────────────────────────────────────────

def _eval_config(model, tokenizer, config: dict, n_items: int,
                 query_words: list[str], device, max_ctx: int) -> dict:
    items, positions, over = _build_items(tokenizer, config, n_items, query_words, max_ctx)
    n_bins = len(POS_FRACS)
    correct = {pi: 0 for pi in range(n_bins)}
    total = {pi: 0 for pi in range(n_bins)}
    lens = [it["n_tokens"] for it in items] if items else [0]

    t0 = time.time()
    for it in items:
        ids = it["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
        pred = int(out.logits[0, -1, :].argmax().item())
        pi = it["pi"]
        correct[pi] += int(pred == it["answer_id"])
        total[pi] += 1
    elapsed = time.time() - t0

    acc = {pi: (correct[pi] / total[pi] if total[pi] else float("nan")) for pi in range(n_bins)}
    v = _valley(acc)
    mean_acc = float(np.mean([acc[pi] for pi in range(n_bins) if total[pi]]))

    return {
        "config": config,
        "doc_positions": positions,
        "context_tokens": {"min": int(min(lens)), "max": int(max(lens)),
                           "mean": int(np.mean(lens))},
        "items_skipped_over_ctx": over,
        "accuracy_by_pos": {POS_NAMES[pi]: round(acc[pi], 4) for pi in range(n_bins)},
        "n_by_pos": {POS_NAMES[pi]: total[pi] for pi in range(n_bins)},
        "mean_accuracy": round(mean_acc, 4),
        "V_task": round(v, 5) if not np.isnan(v) else None,
        "eval_seconds": round(elapsed, 1),
    }


# ── the sweep (cheapest first) ──────────────────────────────────────────────

def _sweep_configs() -> list[dict]:
    return [
        # replicate exp-068 difficulty but with the binding stated ONCE (not 5x)
        {"id": "C1", "retrieval_type": "keyword", "n_doc": 20, "redundancy": 1, "filler": 2},
        # more documents, deeper middle
        {"id": "C2", "retrieval_type": "keyword", "n_doc": 30, "redundancy": 1, "filler": 1},
        {"id": "C3", "retrieval_type": "keyword", "n_doc": 40, "redundancy": 1, "filler": 0},
        # embedded fact in prose (matched cloze) — harder retrieval
        {"id": "C4", "retrieval_type": "embedded", "n_doc": 20, "redundancy": 1, "filler": 1},
        {"id": "C5", "retrieval_type": "embedded", "n_doc": 30, "redundancy": 1, "filler": 1},
        {"id": "C6", "retrieval_type": "embedded", "n_doc": 40, "redundancy": 1, "filler": 0},
        # reverse lookup (answer precedes key) — breaks forward induction copy
        {"id": "C7", "retrieval_type": "reverse", "n_doc": 20, "redundancy": 1, "filler": 1},
        {"id": "C8", "retrieval_type": "reverse", "n_doc": 30, "redundancy": 1, "filler": 1},
        {"id": "C9", "retrieval_type": "reverse", "n_doc": 40, "redundancy": 1, "filler": 1},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b", choices=list(MODEL_IDS))
    ap.add_argument("--items", type=int, default=20, help="questions per config")
    ap.add_argument("--max-ctx", type=int, default=2040)
    ap.add_argument("--stop-on-find", action="store_true",
                    help="stop at first U-shape (default: run full landscape)")
    ap.add_argument("--only", default="", help="comma-separated config ids to run")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="float16 halves memory; fine for base-accuracy calibration")
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    device = _device()
    print(f"=== exp-069 task calibration (ANALYSIS ONLY, no intervention) ===")
    print(f"Model: {model_id}   Device: {device}   items/config: {args.items}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # select single-token answer words
    query_words = []
    for w in WORD_POOL:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            query_words.append(w)
        if len(query_words) >= args.items:
            break
    if len(query_words) < args.items:
        raise ValueError(f"Only {len(query_words)} single-token words for {args.items} items")
    print(f"Single-token answer words ({len(query_words)}): {query_words}")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"Loading model {args.dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
        attn_implementation="eager").to(device).eval()

    only = set(c.strip() for c in args.only.split(",") if c.strip())
    results = []
    found = None
    for config in _sweep_configs():
        if only and config["id"] not in only:
            continue
        print(f"\n--- {config['id']}: {config} ---")
        r = _eval_config(model, tokenizer, config, args.items, query_words, device, args.max_ctx)
        acc_str = {k: v for k, v in r["accuracy_by_pos"].items()}
        print(f"  ctx tokens: {r['context_tokens']}  (skipped {r['items_skipped_over_ctx']})")
        print(f"  acc: {acc_str}")
        print(f"  mean_acc={r['mean_accuracy']}  V_task={r['V_task']}  ({r['eval_seconds']}s)")
        results.append(r)
        v = r["V_task"]
        prim = r["accuracy_by_pos"]["primacy"]
        rec = r["accuracy_by_pos"]["recency"]
        mid = r["accuracy_by_pos"]["middle"]
        if v is not None and v > 0.10 and mid < max(prim, rec) and max(prim, rec) > 0.3:
            print(f"  >>> U-SHAPE candidate: V_task={v} (primacy={prim}, recency={rec}, middle={mid})")
            if found is None:
                found = config["id"]
            if args.stop_on_find:
                break

    out = {
        "experiment": "exp-069",
        "kind": "task_calibration",
        "intervention": False,
        "model": model_id,
        "dtype": args.dtype,
        "items_per_config": args.items,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "u_shape_config_found": found,
        "configs": results,
    }
    # merge with prior runs from other models if present
    prior = {}
    if RESULTS.exists():
        try:
            prior = json.loads(RESULTS.read_text())
        except Exception:
            prior = {}
    runs = prior.get("runs", []) if isinstance(prior, dict) else []
    runs.append(out)
    RESULTS.write_text(json.dumps({"experiment": "exp-069", "runs": runs}, indent=1))
    print(f"\nWrote {RESULTS}")
    print(f"U-shape config found: {found}")


if __name__ == "__main__":
    main()
