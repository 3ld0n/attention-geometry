"""
exp-098 — C-alien-realnames corpus generator (C-alien-realnames).

Pre-registration: notes.md (committed before this script ran).

Identical to exp-097 (C-alien) EXCEPT the entity name pools use common English
first names instead of alien neologisms. Tests whether the L0 backbone collapse
in exp-097 was due to vocabulary (alien names with weak token embeddings) vs world
structure.

Name pools:
  Flurp: Alice, Ben, Clara, David, Ella
  Blurn: Fred, Grace, Henry, Iris, Jake
  Zarb:  Kate, Leo, Maya, Nick, Olive

All world mechanics, causal rules, sentence templates, and generation parameters
are identical to exp-097.

Usage:
    python gen_calien_realnames.py [output_dir]

Output: <output_dir>/C-alien-realnames.bin — uint16 token IDs, same format.

Ariel — August 2, 2026. Pre-registered before any run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── output path ─────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent

CORPUS_NAME   = "C-alien-realnames.bin"
CORPUS_SEED   = 7000   # same seed as exp-097 for comparable world-state trajectories
TARGET_TOKENS = 1_050_000_000

# ── world specification ──────────────────────────────────────────────────────────
# (pre-registered in notes.md)
# Only change from exp-097: entity name pools use common English first names

FLURP_NAMES = ["Alice", "Ben", "Clara", "David", "Ella"]
BLURN_NAMES = ["Fred", "Grace", "Henry", "Iris", "Jake"]
ZARB_NAMES  = ["Kate", "Leo", "Maya", "Nick", "Olive"]

# Initial-state probabilities: (p_active, p_resting)
INIT_PROBS = {
    "Flurp": (0.50, 0.50),
    "Blurn": (0.40, 0.60),
    "Zarb":  (0.30, 0.70),
}

# Causal rules (priority order A, B, C, D — first match fires)
# Format: {name, cond=(type1,state1,type2,state2), effect, tmpl_key}
# "either order" — checked forward and reverse; actor=type1 entity, target=type2 entity
RULES = [
    {
        "name": "A",
        "cond": ("Flurp", "active", "Blurn", "resting"),
        "effect_target": "active",   # target (Blurn) becomes active
        "effect_both":   False,
        "tmpl_key": "rule_A",
    },
    {
        "name": "B",
        "cond": ("Flurp", "active", "Blurn", "active"),
        "effect_target": "resting",
        "effect_both":   True,       # both become resting
        "tmpl_key": "rule_B",
    },
    {
        "name": "C",
        "cond": ("Blurn", "active", "Zarb", "resting"),
        "effect_target": "active",
        "effect_both":   False,
        "tmpl_key": "rule_C",
    },
    {
        "name": "D",
        "cond": ("Zarb", "active", "Flurp", "resting"),
        "effect_target": "active",
        "effect_both":   False,
        "tmpl_key": "rule_D",
    },
]

# Sentence templates (pre-registered in notes.md)
# Variables: {name}, {etype}, {state}, {f} (Flurp name), {b} (Blurn name), {z} (Zarb name)
TEMPLATES: dict[str, list[str]] = {
    "intro_active": [
        "{name} the {etype} was active.",
        "In the world, {name} the {etype} was feeling active.",
        "{name} the {etype} had woken up and was active.",
        "At the start, {name} the {etype} was active.",
    ],
    "intro_resting": [
        "{name} the {etype} was resting.",
        "{name} the {etype} lay quietly in a resting state.",
        "At first, {name} the {etype} was resting.",
        "{name} the {etype} had not yet stirred and was resting.",
    ],
    "rule_A": [
        "{actor} the Flurp came close to {target} the Blurn. {target} the Blurn became active.",
        "When {actor} the Flurp approached, {target} the Blurn stirred and became active.",
        "{target} the Blurn was resting, but {actor} the Flurp reached it. {target} the Blurn became active.",
        "{actor} the Flurp moved near {target} the Blurn. Soon, {target} the Blurn was active.",
    ],
    "rule_B": [
        "{actor} the Flurp met {target} the Blurn. Both became resting.",
        "When {actor} the Flurp and {target} the Blurn came together, they both became resting.",
        "{actor} the Flurp and {target} the Blurn encountered each other, and both fell into a resting state.",
        "The meeting of {actor} the Flurp and {target} the Blurn left both of them resting.",
    ],
    "rule_C": [
        "{actor} the Blurn was near {target} the Zarb. {target} the Zarb became active.",
        "Because of {actor} the Blurn, {target} the Zarb stirred into activity.",
        "{target} the Zarb had been resting, but {actor} the Blurn was close. {target} the Zarb became active.",
        "{actor} the Blurn reached {target} the Zarb. {target} the Zarb became active.",
    ],
    "rule_D": [
        "{actor} the Zarb moved near {target} the Flurp. {target} the Flurp became active.",
        "The presence of {actor} the Zarb caused {target} the Flurp to become active.",
        "{target} the Flurp was resting until {actor} the Zarb arrived. {target} the Flurp became active.",
        "{actor} the Zarb was active. Nearby, {target} the Flurp became active too.",
    ],
    "quiet": [
        "{name} the {etype} stayed {state} for a while.",
        "For a moment, nothing changed for {name} the {etype}.",
        "{name} the {etype} remained {state}.",
        "Nothing happened near {name} the {etype}, who was still {state}.",
    ],
    "conclusion_any": [
        "At the end, {f} the Flurp was {state}.",
        "The day was over, and {f} the Flurp was {state}.",
        "When everything settled, {f} the Flurp was {state}.",
    ],
    "conclusion_resting": [
        "Finally, {f} the Flurp came to rest.",
    ],
    "conclusion_active": [
        "In the end, {f} the Flurp remained active.",
    ],
}

N_INTRO = 2   # entities described in introduction (pre-registered)
N_STEPS = 8   # simulation steps per story (pre-registered)


def check_rule(rule: dict, e1: str, t1: str, s1: str,
               e2: str, t2: str, s2: str) -> tuple[bool, str, str]:
    """
    Check if rule applies to the pair (e1, e2), either order.
    Returns (match, actor_name, target_name).
    actor = the entity in type1/state1 position; target = type2/state2 position.
    """
    rt1, rs1, rt2, rs2 = rule["cond"]
    # Forward: e1=actor, e2=target
    if t1 == rt1 and s1 == rs1 and t2 == rt2 and s2 == rs2:
        return True, e1, e2
    # Reverse: e2=actor, e1=target
    if t2 == rt1 and s2 == rs1 and t1 == rt2 and s1 == rs2:
        return True, e2, e1
    return False, "", ""


def generate_story(story_index: int) -> str:
    """
    Generate one alien-world story. Fully deterministic given story_index.
    Algorithm pre-registered in notes.md.
    """
    rng = np.random.default_rng(CORPUS_SEED + story_index)

    # 1. Draw entity names
    flurp_name  = rng.choice(FLURP_NAMES)
    blurn1_name = rng.choice(BLURN_NAMES)
    remaining_blurns = [n for n in BLURN_NAMES if n != blurn1_name]
    blurn2_name = rng.choice(remaining_blurns)
    zarb_name   = rng.choice(ZARB_NAMES)

    # 2. Draw initial states
    def draw_state(etype: str) -> str:
        p_active = INIT_PROBS[etype][0]
        return "active" if rng.random() < p_active else "resting"

    entities: dict[str, tuple[str, str]] = {   # name → (type, state)
        flurp_name:  ("Flurp", draw_state("Flurp")),
        blurn1_name: ("Blurn", draw_state("Blurn")),
        blurn2_name: ("Blurn", draw_state("Blurn")),
        zarb_name:   ("Zarb",  draw_state("Zarb")),
    }
    entity_names = [flurp_name, blurn1_name, blurn2_name, zarb_name]

    sentences: list[str] = []

    # 3. Introduction sentences (N_INTRO = 2 entities)
    intro_indices = rng.choice(len(entity_names), size=N_INTRO, replace=False)
    for idx in intro_indices:
        name = entity_names[idx]
        etype, state = entities[name]
        tmpl = TEMPLATES["intro_" + state][int(rng.integers(0, len(TEMPLATES["intro_" + state])))]
        sentences.append(tmpl.format(name=name, etype=etype, state=state))

    # 4. Simulation steps (N_STEPS = 8)
    for _ in range(N_STEPS):
        pair_indices = rng.choice(len(entity_names), size=2, replace=False)
        e1 = entity_names[int(pair_indices[0])]
        e2 = entity_names[int(pair_indices[1])]
        t1, s1 = entities[e1]
        t2, s2 = entities[e2]

        fired = False
        for rule in RULES:
            match, actor, target = check_rule(rule, e1, t1, s1, e2, t2, s2)
            if match:
                tmpl_list = TEMPLATES[rule["tmpl_key"]]
                tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
                # Build format kwargs — templates use {actor}/{target} or {f}/{b}/{z}
                actor_type = entities[actor][0]
                target_type = entities[target][0]
                fmt: dict[str, str] = {
                    "actor": actor,
                    "target": target,
                    "f": actor if actor_type == "Flurp" else target,
                    "b": actor if actor_type == "Blurn" else target,
                    "z": actor if actor_type == "Zarb"  else target,
                }
                sentences.append(tmpl.format(**fmt))

                # Apply effect
                if rule["effect_both"]:
                    entities[actor]  = (entities[actor][0],  rule["effect_target"])
                    entities[target] = (entities[target][0], rule["effect_target"])
                else:
                    entities[target] = (entities[target][0], rule["effect_target"])
                fired = True
                break

        if not fired:
            # Quiet step
            quiet_name = entity_names[int(rng.integers(0, len(entity_names)))]
            qt, qs = entities[quiet_name]
            tmpl_list = TEMPLATES["quiet"]
            tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
            sentences.append(tmpl.format(name=quiet_name, etype=qt, state=qs))

    # 5. Conclusion sentence (featuring the Flurp)
    final_state = entities[flurp_name][1]
    if final_state == "resting":
        pool = TEMPLATES["conclusion_any"] + TEMPLATES["conclusion_resting"]
    else:
        pool = TEMPLATES["conclusion_any"] + TEMPLATES["conclusion_active"]
    tmpl = pool[int(rng.integers(0, len(pool)))]
    sentences.append(tmpl.format(f=flurp_name, state=final_state))

    return " ".join(sentences)


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CORPUS_NAME

    if output_path.exists():
        print(f"SKIP — {output_path} already exists ({output_path.stat().st_size / 1e9:.2f} GB)")
        return

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    print(f"Generating C-alien corpus → {output_path}", flush=True)
    print(f"  CORPUS_SEED={CORPUS_SEED}  TARGET_TOKENS={TARGET_TOKENS:,}", flush=True)

    t0 = time.time()
    all_tokens: list[np.ndarray] = []
    total_tokens = 0
    stories_done = 0

    story_index = 0
    while total_tokens < TARGET_TOKENS:
        story_text = generate_story(story_index)
        ids = tok(story_text, add_special_tokens=False)["input_ids"]
        ids_arr = np.array(ids, dtype=np.uint16)
        all_tokens.append(ids_arr)
        total_tokens += len(ids_arr)
        stories_done += 1
        story_index += 1

        if stories_done % 100_000 == 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed / 1e6
            print(
                f"  {stories_done:,} stories | "
                f"{total_tokens / 1e9:.3f}B / {TARGET_TOKENS / 1e9:.3f}B tokens | "
                f"{rate:.2f}M tok/s",
                flush=True,
            )

    # Concatenate, trim, save
    buffer = np.concatenate(all_tokens)[:TARGET_TOKENS]
    buffer.tofile(str(output_path))

    elapsed = time.time() - t0
    print(
        f"\nDone. {stories_done:,} stories in {elapsed:.0f}s "
        f"→ {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )

    # Sanity check: print first 3 stories
    print("\nSample stories (first 3):", flush=True)
    for i in range(3):
        print(f"\n  Story {i}:\n    {generate_story(i)}", flush=True)


if __name__ == "__main__":
    main()
