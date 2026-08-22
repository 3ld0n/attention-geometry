"""
exp-125: Polysemy interference test — length-equated redesign of exp-124

Pre-registration commit: attention-geometry 86313f2
Protocol: notes.md in this directory

Key change from exp-124: C_AB is now "He was thinking about the [word]" (6 words)
instead of "The [word]" (2 tokens). C_A and C_B are unchanged from exp-124.

Same K-statistic and rho formulas, same P1/P2 thresholds.
"""

import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- word list and context templates (registered in notes.md) ---
# C_A and C_B: unchanged from exp-124
# C_AB: NEW — "He was thinking about the [word]" (6 words) for all words

WORDS = [
    {
        "word": "bank",
        "c_a": "She checked her balance at the bank",
        "c_b": "The salmon leaped from the river bank",
        "c_ab": "He was thinking about the bank",
    },
    {
        "word": "bat",
        "c_a": "The cave was home to the little bat",
        "c_b": "He swung the wooden baseball bat",
        "c_ab": "He was thinking about the bat",
    },
    {
        "word": "crane",
        "c_a": "The tall wading whooping crane",
        "c_b": "The steel construction crane",
        "c_ab": "He was thinking about the crane",
    },
    {
        "word": "palm",
        "c_a": "The fortune teller read his palm",
        "c_b": "She climbed the tropical palm",
        "c_ab": "He was thinking about the palm",
    },
    {
        "word": "bark",
        "c_a": "The watchdog let out a loud bark",
        "c_b": "He peeled the rough pine bark",
        "c_ab": "He was thinking about the bark",
    },
    {
        "word": "spring",
        "c_a": "The worn mattress had a broken spring",
        "c_b": "Wildflowers bloomed in the spring",
        "c_ab": "He was thinking about the spring",
    },
    {
        "word": "pitcher",
        "c_a": "She poured lemonade from the glass pitcher",
        "c_b": "The left-handed starting pitcher",
        "c_ab": "He was thinking about the pitcher",
    },
    {
        "word": "club",
        "c_a": "She was president of the local book club",
        "c_b": "The warrior raised the heavy wooden club",
        "c_ab": "He was thinking about the club",
    },
    {
        "word": "match",
        "c_a": "She lit the candle with a wooden match",
        "c_b": "The World Cup qualifying match",
        "c_ab": "He was thinking about the match",
    },
    {
        "word": "board",
        "c_a": "He nailed down the long wooden board",
        "c_b": "She was appointed to the corporate board",
        "c_ab": "He was thinking about the board",
    },
    {
        "word": "light",
        "c_a": "She turned on the ceiling light",
        "c_b": "The feather was extremely light",
        "c_ab": "He was thinking about the light",
    },
    {
        "word": "date",
        "c_a": "She ate the sweet Medjool date",
        "c_b": "He was nervous before his first date",
        "c_ab": "He was thinking about the date",
    },
    {
        "word": "pool",
        "c_a": "The children splashed in the swimming pool",
        "c_b": "He lined up his shot at the pool",
        "c_ab": "He was thinking about the pool",
    },
    {
        "word": "sage",
        "c_a": "She seasoned the stuffing with fresh sage",
        "c_b": "The old hermit was a wise sage",
        "c_ab": "He was thinking about the sage",
    },
    {
        "word": "mole",
        "c_a": "The garden was riddled by the mole",
        "c_b": "She had a dark brown beauty mole",
        "c_ab": "He was thinking about the mole",
    },
    {
        "word": "plane",
        "c_a": "She boarded the commercial passenger plane",
        "c_b": "He leveled the wood with a carpenter's plane",
        "c_ab": "He was thinking about the plane",
    },
    {
        "word": "scale",
        "c_a": "The trout's shiny silver scale",
        "c_b": "He stepped onto the bathroom scale",
        "c_ab": "He was thinking about the scale",
    },
    {
        "word": "seal",
        "c_a": "The playful harbor seal",
        "c_b": "She broke the wax seal",
        "c_ab": "He was thinking about the seal",
    },
    {
        "word": "iron",
        "c_a": "She pressed the wrinkled shirt with the iron",
        "c_b": "The blacksmith worked with molten iron",
        "c_ab": "He was thinking about the iron",
    },
    {
        "word": "file",
        "c_a": "She found the documents in the manila file",
        "c_b": "He smoothed the rough edge with the file",
        "c_ab": "He was thinking about the file",
    },
]

CONTROL = {
    "word": "elephant",
    "c_a": "She watched the large gray elephant",
    "c_b": "The African bush elephant",
    "c_ab": "He was thinking about the elephant",
}


def get_next_token_dist(model, tokenizer, prompt: str) -> np.ndarray:
    """Return probability distribution over next token after prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def compute_k_and_rho(p_a: np.ndarray, p_b: np.ndarray, p_ab: np.ndarray):
    """
    K_W = TV(P_AB, best-fit mixture of P_A, P_B)
    rho_W = Pearson(delta_W, normalized sqrt(P_A * P_B))
    """
    diff = p_a - p_b
    denom = np.dot(diff, diff)
    if denom < 1e-10:
        lam = 0.5
    else:
        lam = np.clip(np.dot(p_ab - p_b, diff) / denom, 0.0, 1.0)

    mixture = lam * p_a + (1.0 - lam) * p_b
    K_w = 0.5 * np.sum(np.abs(p_ab - mixture))

    delta = p_ab - mixture
    geom = np.sqrt(p_a * p_b)
    geom_sum = geom.sum()
    geom_n = geom / geom_sum if geom_sum > 1e-10 else geom

    corr_matrix = np.corrcoef(delta, geom_n)
    rho_w = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0

    return float(lam), float(K_w), float(rho_w), mixture, delta


def sense_separation(p_a: np.ndarray, p_b: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p_a - p_b)))


def top_tokens_by_delta(delta: np.ndarray, tokenizer, n: int = 20):
    idx = np.argsort(np.abs(delta))[::-1][:n]
    return [
        {
            "token": tokenizer.decode([int(i)]),
            "token_id": int(i),
            "delta": float(delta[i]),
        }
        for i in idx
    ]


def context_length_in_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text)["input_ids"])


def run_experiment():
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.eval()
    print(f"Model loaded. Vocabulary size: {tokenizer.vocab_size}")

    results_per_word = {}
    context_lengths = {}

    print("\nRunning main word list (20 words × 3 contexts = 60 forward passes)...")

    for entry in WORDS:
        w = entry["word"]
        print(f"  {w}...", end=" ", flush=True)

        # Record context lengths in tokens for transparency
        len_a = context_length_in_tokens(tokenizer, entry["c_a"])
        len_b = context_length_in_tokens(tokenizer, entry["c_b"])
        len_ab = context_length_in_tokens(tokenizer, entry["c_ab"])
        context_lengths[w] = {"c_a_tokens": len_a, "c_b_tokens": len_b, "c_ab_tokens": len_ab}

        p_a = get_next_token_dist(model, tokenizer, entry["c_a"])
        p_b = get_next_token_dist(model, tokenizer, entry["c_b"])
        p_ab = get_next_token_dist(model, tokenizer, entry["c_ab"])

        D_w = sense_separation(p_a, p_b)
        lam, K_w, rho_w, mixture, delta = compute_k_and_rho(p_a, p_b, p_ab)
        top20 = top_tokens_by_delta(delta, tokenizer, n=20)

        results_per_word[w] = {
            "sense_separation": D_w,
            "lambda_star": lam,
            "K_w": K_w,
            "rho_w": rho_w,
            "included": D_w >= 0.10,
            "context_tokens": context_lengths[w],
            "top_delta_tokens": top20,
        }
        print(f"D={D_w:.3f}  K={K_w:.4f}  rho={rho_w:.3f}  tokens=[{len_a},{len_b},{len_ab}]")

    print(f"\n  {CONTROL['word']} (control)...", end=" ", flush=True)
    len_a_ctrl = context_length_in_tokens(tokenizer, CONTROL["c_a"])
    len_b_ctrl = context_length_in_tokens(tokenizer, CONTROL["c_b"])
    len_ab_ctrl = context_length_in_tokens(tokenizer, CONTROL["c_ab"])
    p_a_ctrl = get_next_token_dist(model, tokenizer, CONTROL["c_a"])
    p_b_ctrl = get_next_token_dist(model, tokenizer, CONTROL["c_b"])
    p_ab_ctrl = get_next_token_dist(model, tokenizer, CONTROL["c_ab"])
    D_ctrl = sense_separation(p_a_ctrl, p_b_ctrl)
    lam_ctrl, K_ctrl, rho_ctrl, _, _ = compute_k_and_rho(p_a_ctrl, p_b_ctrl, p_ab_ctrl)
    control_result = {
        "sense_separation": D_ctrl,
        "lambda_star": lam_ctrl,
        "K_control": K_ctrl,
        "rho_control": rho_ctrl,
        "context_tokens": {"c_a_tokens": len_a_ctrl, "c_b_tokens": len_b_ctrl, "c_ab_tokens": len_ab_ctrl},
    }
    print(f"D={D_ctrl:.3f}  K={K_ctrl:.4f}  rho={rho_ctrl:.3f}  tokens=[{len_a_ctrl},{len_b_ctrl},{len_ab_ctrl}]")

    qualifying = [w for w, r in results_per_word.items() if r["included"]]
    excluded = [w for w, r in results_per_word.items() if not r["included"]]

    K_values = [results_per_word[w]["K_w"] for w in qualifying]
    rho_values = [results_per_word[w]["rho_w"] for w in qualifying]

    K_mean = float(np.mean(K_values)) if K_values else None
    K_std = float(np.std(K_values)) if K_values else None
    K_max = float(np.max(K_values)) if K_values else None
    K_max_word = qualifying[int(np.argmax(K_values))] if K_values else None
    rho_mean = float(np.mean(rho_values)) if rho_values else None

    if K_mean is not None:
        p1_verdict = "CONFIRMED" if K_mean < 0.10 else "FALSIFIED"
    else:
        p1_verdict = "INCONCLUSIVE (no qualifying words)"

    if p1_verdict == "FALSIFIED" and rho_mean is not None:
        p2_verdict = "CONFIRMED" if rho_mean > 0.30 else "FALSIFIED"
    else:
        p2_verdict = "NOT TESTED"

    # Comparison with exp-124
    exp124_K_mean = 0.679
    exp124_K_control = 0.556
    delta_K_mean = K_mean - exp124_K_mean if K_mean is not None else None
    delta_K_control = K_ctrl - exp124_K_control

    print(f"\n=== RESULTS ===")
    print(f"Qualifying words: {len(qualifying)}/{len(WORDS)}")
    print(f"K_mean = {K_mean:.4f} (std={K_std:.4f}, max={K_max:.4f} [{K_max_word}])")
    print(f"rho_mean = {rho_mean:.4f}")
    print(f"K_control = {K_ctrl:.4f}")
    print(f"P1 verdict: {p1_verdict}")
    print(f"P2 verdict: {p2_verdict}")
    print(f"\nComparison with exp-124:")
    print(f"  K_mean:    {exp124_K_mean:.3f} → {K_mean:.3f} (Δ = {delta_K_mean:+.3f})")
    print(f"  K_control: {exp124_K_control:.3f} → {K_ctrl:.3f} (Δ = {delta_K_control:+.3f})")

    headline = (
        f"exp-125 (length-equated): K_mean={K_mean:.4f} (vs exp-124 0.679), "
        f"K_control={K_ctrl:.4f} (vs exp-124 0.556), "
        f"rho_mean={rho_mean:.4f}. "
        f"P1 {p1_verdict}."
    )

    results = {
        "experiment": "exp-125",
        "date": "2026-08-22",
        "model": "openai-community/gpt2",
        "prereg_commit": "attention-geometry 86313f2",
        "change_from_exp124": "C_AB changed from 'The [word]' (2 tokens) to 'He was thinking about the [word]' (6 words). C_A and C_B unchanged.",
        "n_words": len(WORDS),
        "n_qualifying": len(qualifying),
        "excluded_words": excluded,
        "qualifying_words": qualifying,
        "per_word": results_per_word,
        "control": control_result,
        "aggregate": {
            "K_mean": K_mean,
            "K_std": K_std,
            "K_max": K_max,
            "K_max_word": K_max_word,
            "rho_mean": rho_mean,
        },
        "exp124_comparison": {
            "exp124_K_mean": exp124_K_mean,
            "exp124_K_control": exp124_K_control,
            "delta_K_mean": delta_K_mean,
            "delta_K_control": delta_K_control,
        },
        "predictions": {
            "P1_threshold": 0.10,
            "P1_verdict": p1_verdict,
            "P2_threshold": 0.30,
            "P2_verdict": p2_verdict,
        },
        "headline": headline,
    }

    output_path = "research/physics/experiments/exp-125_polysemy_length_equated/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Headline: {headline}")

    return results


if __name__ == "__main__":
    run_experiment()
