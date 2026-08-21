"""
exp-124: Polysemy interference test — Path C of the contextuality battery

Pre-registration commit: attention-geometry 14fd5d4
Protocol: notes.md in this directory

For 20 polysemous English nouns, measure GPT-2 next-token probability
distributions under sense-A context, sense-B context, and ambiguous context
("The [word]"). Test whether the ambiguous distribution is a convex mixture of
the two sense distributions.

K_W = TV(P_AB, best-fit mixture of P_A, P_B)
rho_W = Pearson(delta_W, normalized geometric mean of P_A * P_B)
"""

import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- word list and context templates (registered in notes.md) ---

WORDS = [
    {
        "word": "bank",
        "c_a": "She checked her balance at the bank",
        "c_b": "The salmon leaped from the river bank",
        "c_ab": "The bank",
    },
    {
        "word": "bat",
        "c_a": "The cave was home to the little bat",
        "c_b": "He swung the wooden baseball bat",
        "c_ab": "The bat",
    },
    {
        "word": "crane",
        "c_a": "The tall wading whooping crane",
        "c_b": "The steel construction crane",
        "c_ab": "The crane",
    },
    {
        "word": "palm",
        "c_a": "The fortune teller read his palm",
        "c_b": "She climbed the tropical palm",
        "c_ab": "The palm",
    },
    {
        "word": "bark",
        "c_a": "The watchdog let out a loud bark",
        "c_b": "He peeled the rough pine bark",
        "c_ab": "The bark",
    },
    {
        "word": "spring",
        "c_a": "The worn mattress had a broken spring",
        "c_b": "Wildflowers bloomed in the spring",
        "c_ab": "The spring",
    },
    {
        "word": "pitcher",
        "c_a": "She poured lemonade from the glass pitcher",
        "c_b": "The left-handed starting pitcher",
        "c_ab": "The pitcher",
    },
    {
        "word": "club",
        "c_a": "She was president of the local book club",
        "c_b": "The warrior raised the heavy wooden club",
        "c_ab": "The club",
    },
    {
        "word": "match",
        "c_a": "She lit the candle with a wooden match",
        "c_b": "The World Cup qualifying match",
        "c_ab": "The match",
    },
    {
        "word": "board",
        "c_a": "He nailed down the long wooden board",
        "c_b": "She was appointed to the corporate board",
        "c_ab": "The board",
    },
    {
        "word": "light",
        "c_a": "She turned on the ceiling light",
        "c_b": "The feather was extremely light",
        "c_ab": "The light",
    },
    {
        "word": "date",
        "c_a": "She ate the sweet Medjool date",
        "c_b": "He was nervous before his first date",
        "c_ab": "The date",
    },
    {
        "word": "pool",
        "c_a": "The children splashed in the swimming pool",
        "c_b": "He lined up his shot at the pool",
        "c_ab": "The pool",
    },
    {
        "word": "sage",
        "c_a": "She seasoned the stuffing with fresh sage",
        "c_b": "The old hermit was a wise sage",
        "c_ab": "The sage",
    },
    {
        "word": "mole",
        "c_a": "The garden was riddled by the mole",
        "c_b": "She had a dark brown beauty mole",
        "c_ab": "The mole",
    },
    {
        "word": "plane",
        "c_a": "She boarded the commercial passenger plane",
        "c_b": "He leveled the wood with a carpenter's plane",
        "c_ab": "The plane",
    },
    {
        "word": "scale",
        "c_a": "The trout's shiny silver scale",
        "c_b": "He stepped onto the bathroom scale",
        "c_ab": "The scale",
    },
    {
        "word": "seal",
        "c_a": "The playful harbor seal",
        "c_b": "She broke the wax seal",
        "c_ab": "The seal",
    },
    {
        "word": "iron",
        "c_a": "She pressed the wrinkled shirt with the iron",
        "c_b": "The blacksmith worked with molten iron",
        "c_ab": "The iron",
    },
    {
        "word": "file",
        "c_a": "She kept the documents in the manila file",
        "c_b": "He smoothed the rough edge with the file",
        "c_ab": "The file",
    },
]

# Monosemous control word
CONTROL = {
    "word": "elephant",
    "c_a": "She watched the large gray elephant",
    "c_b": "The African bush elephant",
    "c_ab": "The elephant",
}


def get_next_token_dist(model, tokenizer, prompt: str) -> np.ndarray:
    """Return probability distribution over next token after prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last token position
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs  # shape: (vocab_size,)


def compute_k_and_rho(p_a: np.ndarray, p_b: np.ndarray, p_ab: np.ndarray):
    """
    Compute K-statistic and interference correlation rho for one word.

    K_W = TV(P_AB, best-fit mixture of P_A, P_B)
    rho_W = Pearson(delta_W, normalized sqrt(P_A * P_B))
    """
    # Mixture fit: lambda* = argmin ||p_ab - (lam*p_a + (1-lam)*p_b)||^2
    diff = p_a - p_b
    denom = np.dot(diff, diff)
    if denom < 1e-10:
        lam = 0.5
    else:
        lam = np.clip(np.dot(p_ab - p_b, diff) / denom, 0.0, 1.0)

    mixture = lam * p_a + (1.0 - lam) * p_b

    # K-statistic (total variation from best-fit mixture)
    K_w = 0.5 * np.sum(np.abs(p_ab - mixture))

    # Interference correlation
    delta = p_ab - mixture
    geom = np.sqrt(p_a * p_b)
    geom_sum = geom.sum()
    if geom_sum > 1e-10:
        geom_n = geom / geom_sum
    else:
        geom_n = geom

    corr_matrix = np.corrcoef(delta, geom_n)
    rho_w = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0

    return float(lam), float(K_w), float(rho_w), mixture, delta


def sense_separation(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """TV distance between P_A and P_B."""
    return 0.5 * float(np.sum(np.abs(p_a - p_b)))


def top_tokens_by_delta(delta: np.ndarray, tokenizer, n: int = 20):
    """Return top n tokens by |delta|, with sign."""
    idx = np.argsort(np.abs(delta))[::-1][:n]
    return [
        {
            "token": tokenizer.decode([int(i)]),
            "token_id": int(i),
            "delta": float(delta[i]),
        }
        for i in idx
    ]


def run_experiment():
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.eval()
    print(f"Model loaded. Vocabulary size: {tokenizer.vocab_size}")

    results_per_word = {}

    # --- main word list ---
    print("\nRunning main word list (20 words × 3 contexts = 60 forward passes)...")
    all_p = {}  # cache distributions

    for entry in WORDS:
        w = entry["word"]
        print(f"  {w}...", end=" ", flush=True)

        p_a = get_next_token_dist(model, tokenizer, entry["c_a"])
        p_b = get_next_token_dist(model, tokenizer, entry["c_b"])
        p_ab = get_next_token_dist(model, tokenizer, entry["c_ab"])
        all_p[w] = (p_a, p_b, p_ab)

        D_w = sense_separation(p_a, p_b)
        lam, K_w, rho_w, mixture, delta = compute_k_and_rho(p_a, p_b, p_ab)

        top20 = top_tokens_by_delta(delta, tokenizer, n=20)

        results_per_word[w] = {
            "sense_separation": D_w,
            "lambda_star": lam,
            "K_w": K_w,
            "rho_w": rho_w,
            "included": D_w >= 0.10,
            "top_delta_tokens": top20,
        }
        print(f"D={D_w:.3f}  K={K_w:.4f}  rho={rho_w:.3f}")

    # --- monosemous control ---
    print(f"\n  {CONTROL['word']} (control)...", end=" ", flush=True)
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
    }
    print(f"D={D_ctrl:.3f}  K={K_ctrl:.4f}  rho={rho_ctrl:.3f}")

    # --- aggregate statistics ---
    qualifying = [w for w, r in results_per_word.items() if r["included"]]
    excluded = [w for w, r in results_per_word.items() if not r["included"]]

    K_values = [results_per_word[w]["K_w"] for w in qualifying]
    rho_values = [results_per_word[w]["rho_w"] for w in qualifying]

    K_mean = float(np.mean(K_values)) if K_values else None
    K_std = float(np.std(K_values)) if K_values else None
    K_max = float(np.max(K_values)) if K_values else None
    K_max_word = qualifying[int(np.argmax(K_values))] if K_values else None
    rho_mean = float(np.mean(rho_values)) if rho_values else None

    # P1 verdict
    if K_mean is not None:
        p1_verdict = "CONFIRMED" if K_mean < 0.10 else "FALSIFIED"
    else:
        p1_verdict = "INCONCLUSIVE (no qualifying words)"

    # P2 verdict (only if P1 falsified)
    if p1_verdict == "FALSIFIED" and rho_mean is not None:
        p2_verdict = "CONFIRMED" if rho_mean > 0.30 else "FALSIFIED"
    else:
        p2_verdict = "NOT TESTED"

    print(f"\n=== RESULTS ===")
    print(f"Qualifying words: {len(qualifying)}/{len(WORDS)}")
    print(f"K_mean = {K_mean:.4f} (std={K_std:.4f}, max={K_max:.4f} [{K_max_word}])")
    print(f"rho_mean = {rho_mean:.4f}")
    print(f"P1 verdict: {p1_verdict}")
    print(f"P2 verdict: {p2_verdict}")

    headline = (
        f"GPT-2 polysemy interference test: K_mean={K_mean:.4f} ({'< 0.10' if K_mean and K_mean < 0.10 else '>= 0.10'}). "
        f"P1 {p1_verdict}. Classical mixture {'holds' if p1_verdict == 'CONFIRMED' else 'broken'}. "
        f"rho_mean={rho_mean:.4f} (P2 {p2_verdict})."
    )

    results = {
        "experiment": "exp-124",
        "date": "2026-08-21",
        "model": "openai-community/gpt2",
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
        "predictions": {
            "P1_threshold": 0.10,
            "P1_verdict": p1_verdict,
            "P2_threshold": 0.30,
            "P2_verdict": p2_verdict,
        },
        "headline": headline,
    }

    output_path = "research/physics/experiments/exp-124_polysemy_interference/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Headline: {headline}")

    return results


if __name__ == "__main__":
    run_experiment()
