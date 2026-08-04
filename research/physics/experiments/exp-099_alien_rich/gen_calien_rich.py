"""
exp-099 — C-alien-rich corpus generator.

5 entity types (Flurp, Blurn, Zarb, Glorf, Krelp), 1 instance each per story.
6 stochastic rules (p=0.7 primary / p=0.3 alternative).
World state space S = 2^5 = 32 possible configurations.

Pre-registration: notes.md (committed to 3ld0n/attention-geometry before this script ran,
commit 222d9d4; theory addendum commit 93b34dd).
Theoretical frame: notes/2026-08-03_melonic_threshold_derivation.md.

World specification, causal rules, sentence templates, and generation algorithm are
fully specified in notes.md and reproduced here for implementation faithfulness.

Usage:
    python gen_calien_rich.py [output_dir]

Output: <output_dir>/C-alien-rich.bin — uint16 token IDs, same format as the series.

Ariel — August 4, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── output path ─────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent

CORPUS_NAME   = "C-alien-rich.bin"
CORPUS_SEED   = 9000
TARGET_TOKENS = 1_050_000_000

# ── world specification ──────────────────────────────────────────────────────────
# (pre-registered in notes.md)

FLURP_NAMES = ["Vex", "Nul", "Ort", "Pim", "Grel"]
BLURN_NAMES = ["Dath", "Sorn", "Wix", "Brel", "Fend"]
ZARB_NAMES  = ["Quib", "Tarn", "Molk", "Vet", "Zish"]
GLORF_NAMES = ["Blav", "Usk", "Drei", "Folt", "Yem"]
KRELP_NAMES = ["Snop", "Wulf", "Crid", "Barv", "Hund"]

# Initial-state probability of being active (type-specific)
INIT_PROBS = {
    "Flurp": 0.50,
    "Blurn": 0.40,
    "Zarb":  0.30,
    "Glorf": 0.45,
    "Krelp": 0.35,
}

P_PRIMARY = 0.7   # probability of primary outcome per stochastic rule

# Causal rules (priority order A–F; first matching rule fires).
# Each rule:
#   cond = (actor_type, actor_state, target_type, target_state)
#   primary outcome: actor_effect (None=no change), target_effect
#   alternative outcome (p=1-P_PRIMARY): alt_actor_effect, alt_target_effect
RULES: list[dict] = [
    {
        "name": "A",
        "cond": ("Flurp", "active", "Blurn", "resting"),
        "actor_effect":     None,
        "target_effect":    "active",    # Blurn becomes active
        "alt_actor_effect": None,        # no change
        "alt_target_effect": None,
        "tmpl_key":     "rule_A",
        "tmpl_key_alt": "rule_A_alt",
    },
    {
        "name": "B",
        "cond": ("Flurp", "active", "Blurn", "active"),
        "actor_effect":     "resting",   # both become resting
        "target_effect":    "resting",
        "alt_actor_effect": "resting",   # only Flurp becomes resting
        "alt_target_effect": None,
        "tmpl_key":     "rule_B",
        "tmpl_key_alt": "rule_B_alt",
    },
    {
        "name": "C",
        "cond": ("Blurn", "active", "Zarb", "resting"),
        "actor_effect":     None,
        "target_effect":    "active",
        "alt_actor_effect": None,
        "alt_target_effect": None,
        "tmpl_key":     "rule_C",
        "tmpl_key_alt": "rule_C_alt",
    },
    {
        "name": "D",
        "cond": ("Zarb", "active", "Flurp", "resting"),
        "actor_effect":     None,
        "target_effect":    "active",
        "alt_actor_effect": None,
        "alt_target_effect": None,
        "tmpl_key":     "rule_D",
        "tmpl_key_alt": "rule_D_alt",
    },
    {
        "name": "E",
        "cond": ("Glorf", "active", "Krelp", "resting"),
        "actor_effect":     None,
        "target_effect":    "active",    # Krelp becomes active
        "alt_actor_effect": None,
        "alt_target_effect": None,
        "tmpl_key":     "rule_E",
        "tmpl_key_alt": "rule_E_alt",
    },
    {
        "name": "F",
        "cond": ("Krelp", "active", "Glorf", "active"),
        "actor_effect":     "resting",   # both become resting
        "target_effect":    "resting",
        "alt_actor_effect": "resting",   # only Krelp becomes resting
        "alt_target_effect": None,
        "tmpl_key":     "rule_F",
        "tmpl_key_alt": "rule_F_alt",
    },
]

# Sentence templates (pre-registered in notes.md)
# Variables: {name}/{etype}/{state} for intro/quiet; {actor}/{target} for rules;
# {f}/{b}/{z}/{g}/{k} for type-keyed references in conclusions.
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
    # Rule A: active Flurp → resting Blurn becomes active (primary)
    "rule_A": [
        "{actor} the Flurp came close to {target} the Blurn. {target} the Blurn became active.",
        "When {actor} the Flurp approached, {target} the Blurn stirred and became active.",
        "{target} the Blurn was resting, but {actor} the Flurp reached it. {target} the Blurn became active.",
        "{actor} the Flurp moved near {target} the Blurn. Soon, {target} the Blurn was active.",
    ],
    # Rule A alt: no change (p=0.3)
    "rule_A_alt": [
        "Surprisingly, {target} the Blurn did not stir when {actor} the Flurp approached.",
        "{actor} the Flurp came close, but {target} the Blurn stayed resting.",
        "For once, {target} the Blurn kept resting despite {actor} the Flurp nearby.",
        "{actor} the Flurp reached {target} the Blurn, yet nothing happened.",
    ],
    # Rule B: active Flurp + active Blurn → both resting (primary)
    "rule_B": [
        "{actor} the Flurp met {target} the Blurn. Both became resting.",
        "When {actor} the Flurp and {target} the Blurn came together, they both became resting.",
        "{actor} the Flurp and {target} the Blurn encountered each other, and both fell into a resting state.",
        "The meeting of {actor} the Flurp and {target} the Blurn left both of them resting.",
    ],
    # Rule B alt: only Flurp becomes resting (p=0.3)
    "rule_B_alt": [
        "Surprisingly, only {actor} the Flurp became resting. {target} the Blurn stayed active.",
        "{actor} the Flurp tired and became resting, but {target} the Blurn was still active.",
        "When they met, {actor} the Flurp fell to resting while {target} the Blurn remained active.",
        "As expected, {actor} the Flurp became resting, but {target} the Blurn surprised everyone by staying active.",
    ],
    # Rule C: active Blurn → resting Zarb becomes active (primary)
    "rule_C": [
        "{actor} the Blurn was near {target} the Zarb. {target} the Zarb became active.",
        "Because of {actor} the Blurn, {target} the Zarb stirred into activity.",
        "{target} the Zarb had been resting, but {actor} the Blurn was close. {target} the Zarb became active.",
        "{actor} the Blurn reached {target} the Zarb. {target} the Zarb became active.",
    ],
    # Rule C alt: no change (p=0.3)
    "rule_C_alt": [
        "Surprisingly, {target} the Zarb remained resting despite {actor} the Blurn nearby.",
        "{actor} the Blurn was close, but {target} the Zarb did not wake.",
        "For once, {target} the Zarb kept resting even with {actor} the Blurn present.",
        "{actor} the Blurn came near {target} the Zarb, yet {target} the Zarb stayed resting.",
    ],
    # Rule D: active Zarb → resting Flurp becomes active (primary)
    "rule_D": [
        "{actor} the Zarb moved near {target} the Flurp. {target} the Flurp became active.",
        "The presence of {actor} the Zarb caused {target} the Flurp to become active.",
        "{target} the Flurp was resting until {actor} the Zarb arrived. {target} the Flurp became active.",
        "{actor} the Zarb was active. Nearby, {target} the Flurp became active too.",
    ],
    # Rule D alt: no change (p=0.3)
    "rule_D_alt": [
        "Surprisingly, {target} the Flurp stayed resting when {actor} the Zarb moved near.",
        "{actor} the Zarb came close, but {target} the Flurp did not stir.",
        "Despite {actor} the Zarb's presence, {target} the Flurp kept resting.",
        "{actor} the Zarb moved near {target} the Flurp, yet {target} the Flurp stayed resting.",
    ],
    # Rule E: active Glorf → resting Krelp becomes active (primary)
    "rule_E": [
        "{actor} the Glorf approached {target} the Krelp. {target} the Krelp became active.",
        "When {actor} the Glorf came near, {target} the Krelp stirred and became active.",
        "{target} the Krelp was resting, but {actor} the Glorf reached it. {target} the Krelp became active.",
        "{actor} the Glorf moved toward {target} the Krelp. Soon, {target} the Krelp was active.",
    ],
    # Rule E alt: no change (p=0.3)
    "rule_E_alt": [
        "Surprisingly, {target} the Krelp stayed resting when {actor} the Glorf approached.",
        "{actor} the Glorf came near, but {target} the Krelp did not stir.",
        "Despite {actor} the Glorf's presence, {target} the Krelp kept resting.",
        "{actor} the Glorf reached {target} the Krelp, yet nothing happened.",
    ],
    # Rule F: active Krelp + active Glorf → both resting (primary)
    "rule_F": [
        "{actor} the Krelp met {target} the Glorf. Both became resting.",
        "When {actor} the Krelp and {target} the Glorf came together, they both became resting.",
        "{actor} the Krelp and {target} the Glorf encountered each other, and both fell into a resting state.",
        "The meeting of {actor} the Krelp and {target} the Glorf left both of them resting.",
    ],
    # Rule F alt: only Krelp becomes resting (p=0.3)
    "rule_F_alt": [
        "Surprisingly, only {actor} the Krelp became resting. {target} the Glorf stayed active.",
        "{actor} the Krelp tired and became resting, but {target} the Glorf was still active.",
        "When they met, {actor} the Krelp fell to resting while {target} the Glorf remained active.",
        "As expected, {actor} the Krelp became resting, but {target} the Glorf surprised everyone by staying active.",
    ],
    "quiet": [
        "{name} the {etype} stayed {state} for a while.",
        "For a moment, nothing changed for {name} the {etype}.",
        "{name} the {etype} remained {state}.",
        "Nothing happened near {name} the {etype}, who was still {state}.",
    ],
    "conclusion_any": [
        "At the end, {f} the Flurp was {f_state}.",
        "The day was over, and {f} the Flurp was {f_state}.",
        "When everything settled, {f} the Flurp was {f_state}.",
        "Finally, {f} the Flurp was {f_state} and {g} the Glorf was {g_state}.",
        "At last, {f} the Flurp ended {f_state} while {k} the Krelp was {k_state}.",
    ],
}

N_INTRO  = 2     # entities described in introduction
N_STEPS_MIN = 4  # minimum simulation steps per story
N_STEPS_MAX = 8  # maximum simulation steps per story


def check_rule(
    rule: dict,
    e1: str, t1: str, s1: str,
    e2: str, t2: str, s2: str,
) -> tuple[bool, str, str]:
    """
    Check if rule applies to the pair (e1, e2), either order.
    Returns (match, actor_name, target_name).
    actor = entity in actor_type/actor_state position; target = other.
    """
    rt1, rs1, rt2, rs2 = rule["cond"]
    if t1 == rt1 and s1 == rs1 and t2 == rt2 and s2 == rs2:
        return True, e1, e2
    if t2 == rt1 and s2 == rs1 and t1 == rt2 and s1 == rs2:
        return True, e2, e1
    return False, "", ""


def generate_story(story_index: int) -> str:
    """
    Generate one C-alien-rich story. Fully deterministic given story_index + CORPUS_SEED.
    Algorithm pre-registered in notes.md.
    """
    rng = np.random.default_rng(CORPUS_SEED + story_index)

    # 1. Draw one entity name per type
    flurp_name = str(rng.choice(FLURP_NAMES))
    blurn_name = str(rng.choice(BLURN_NAMES))
    zarb_name  = str(rng.choice(ZARB_NAMES))
    glorf_name = str(rng.choice(GLORF_NAMES))
    krelp_name = str(rng.choice(KRELP_NAMES))

    # 2. Draw initial states
    entities: dict[str, tuple[str, str]] = {
        flurp_name: ("Flurp", "active" if rng.random() < INIT_PROBS["Flurp"] else "resting"),
        blurn_name: ("Blurn", "active" if rng.random() < INIT_PROBS["Blurn"] else "resting"),
        zarb_name:  ("Zarb",  "active" if rng.random() < INIT_PROBS["Zarb"]  else "resting"),
        glorf_name: ("Glorf", "active" if rng.random() < INIT_PROBS["Glorf"] else "resting"),
        krelp_name: ("Krelp", "active" if rng.random() < INIT_PROBS["Krelp"] else "resting"),
    }
    entity_names = [flurp_name, blurn_name, zarb_name, glorf_name, krelp_name]

    sentences: list[str] = []

    # 3. Introduction sentences (N_INTRO = 2 entities)
    intro_indices = rng.choice(len(entity_names), size=N_INTRO, replace=False)
    for idx in intro_indices:
        name = entity_names[int(idx)]
        etype, state = entities[name]
        tmpl_list = TEMPLATES["intro_" + state]
        tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
        sentences.append(tmpl.format(name=name, etype=etype, state=state))

    # 4. Simulation steps (variable length: N_STEPS_MIN..N_STEPS_MAX)
    n_steps = int(rng.integers(N_STEPS_MIN, N_STEPS_MAX + 1))
    for _ in range(n_steps):
        pair_idx = rng.choice(len(entity_names), size=2, replace=False)
        e1 = entity_names[int(pair_idx[0])]
        e2 = entity_names[int(pair_idx[1])]
        t1, s1 = entities[e1]
        t2, s2 = entities[e2]

        fired = False
        for rule in RULES:
            match, actor, target = check_rule(rule, e1, t1, s1, e2, t2, s2)
            if match:
                primary = rng.random() < P_PRIMARY
                if primary:
                    tmpl_list = TEMPLATES[rule["tmpl_key"]]
                    tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
                    sentences.append(tmpl.format(actor=actor, target=target))
                    if rule["actor_effect"] is not None:
                        entities[actor] = (entities[actor][0], rule["actor_effect"])
                    if rule["target_effect"] is not None:
                        entities[target] = (entities[target][0], rule["target_effect"])
                else:
                    tmpl_list = TEMPLATES[rule["tmpl_key_alt"]]
                    tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
                    sentences.append(tmpl.format(actor=actor, target=target))
                    if rule["alt_actor_effect"] is not None:
                        entities[actor] = (entities[actor][0], rule["alt_actor_effect"])
                    if rule["alt_target_effect"] is not None:
                        entities[target] = (entities[target][0], rule["alt_target_effect"])
                fired = True
                break

        if not fired:
            quiet_name = entity_names[int(rng.integers(0, len(entity_names)))]
            qt, qs = entities[quiet_name]
            tmpl_list = TEMPLATES["quiet"]
            tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
            sentences.append(tmpl.format(name=quiet_name, etype=qt, state=qs))

    # 5. Conclusion sentence
    f_state = entities[flurp_name][1]
    g_state = entities[glorf_name][1]
    k_state = entities[krelp_name][1]
    tmpl_list = TEMPLATES["conclusion_any"]
    tmpl = tmpl_list[int(rng.integers(0, len(tmpl_list)))]
    sentences.append(tmpl.format(
        f=flurp_name, g=glorf_name, k=krelp_name,
        f_state=f_state, g_state=g_state, k_state=k_state,
    ))

    return " ".join(sentences)


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CORPUS_NAME

    if output_path.exists():
        print(f"SKIP — {output_path} already exists ({output_path.stat().st_size / 1e9:.2f} GB)")
        return

    print("Loading tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print(f"Generating C-alien-rich corpus → {output_path}", flush=True)
    print(f"  CORPUS_SEED={CORPUS_SEED}  TARGET_TOKENS={TARGET_TOKENS:,}", flush=True)

    t0 = time.time()
    TARGET = TARGET_TOKENS
    buf = np.empty(TARGET, dtype=np.uint16)
    total_tokens = 0
    stories_done = 0
    story_index = 0

    while total_tokens < TARGET:
        story_text = generate_story(story_index)
        ids = tok(story_text, add_special_tokens=False)["input_ids"]
        n = len(ids)
        end = min(total_tokens + n, TARGET)
        buf[total_tokens:end] = ids[:end - total_tokens]
        total_tokens = end
        stories_done += 1
        story_index += 1

        if stories_done % 100_000 == 0:
            elapsed = time.time() - t0
            rate = total_tokens / elapsed / 1e6
            print(
                f"  {stories_done:,} stories | "
                f"{total_tokens / 1e9:.3f}B / {TARGET / 1e9:.3f}B tokens | "
                f"{rate:.2f}M tok/s",
                flush=True,
            )

    buf.tofile(str(output_path))

    elapsed = time.time() - t0
    print(
        f"\nDone. {stories_done:,} stories in {elapsed:.0f}s "
        f"→ {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )

    print("\nSample stories (first 3):", flush=True)
    for i in range(3):
        print(f"\n  Story {i}:\n    {generate_story(i)}", flush=True)


if __name__ == "__main__":
    main()
