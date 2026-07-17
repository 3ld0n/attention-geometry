"""
exp-085 — Generate corpus using vLLM (high-throughput replacement for gen_gen.py).

Pre-registration: notes.md (committed 2026-07-14, pre-reg bcd4bb20).
Protocol: temperature=1.0, seq_len=512, seed=85.

Background: model.generate() (transformers) gave 0.05M tok/s on A100-40GB,
exceeding Modal's 7200s timeout for 1.1B tokens (~21000s). vLLM's offline
engine + PagedAttention + CUDA graphs eliminate Python-level step overhead and
KV-cache reallocation bottlenecks. Expected: 1-5M tok/s (~220-1100s).

Sampling equivalence to pre-registration:
  - Context token RNG: np.default_rng(seed) as in gen_gen.py (same protocol).
  - Internal sampling: vLLM's per-request CUDA sampling (different RNG than
    torch.Generator, but pre-registration already accepts protocol-equivalence:
    "model.generate uses its own internal sampling; samp_rng is not directly
    applicable here (documented as protocol-equivalent)").

Usage:
  python gen_gen_vllm.py <checkpoint_dir> <output.bin> \\
      [--n_tokens 1100000000] [--tokenizer /neox_tokenizer]

Ariel — July 17, 2026.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# vLLM — installed in the Modal image; not in local .venv
from vllm import LLM, SamplingParams

# Sequences per vLLM generate() call. vLLM schedules these in its own
# internal batches (PagedAttention). 8192 seqs × 512 tok = 4.2M tokens/chunk.
# For a 70m model on A100-40GB, each chunk should take < 5 seconds.
CHUNK_SEQS = 8192


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", help="path to step_2000/ directory on volume")
    ap.add_argument("output_bin", help="output .bin path (uint16)")
    ap.add_argument("--n_tokens", type=int, default=1_100_000_000,
                    help="total tokens to generate (default 1.1B)")
    ap.add_argument("--seq_len", type=int, default=512,
                    help="tokens per generated sequence (default 512)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=85,
                    help="RNG seed for context-token draws (default 85)")
    ap.add_argument("--tokenizer", default="/neox_tokenizer",
                    help="tokenizer directory (pre-baked in Modal image)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint_dir)
    out_path = Path(args.output_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prompt is 1 token; generate seq_len-1 new tokens → seq_len total.
    n_new_tokens = args.seq_len - 1

    # Total sequences needed (round up; clip on last write).
    n_seqs_total = (args.n_tokens + args.seq_len - 1) // args.seq_len

    print("=== exp-085 generate (vLLM) ===", flush=True)
    print(f"  checkpoint : {ckpt_path}", flush=True)
    print(f"  output     : {out_path}", flush=True)
    print(f"  tokenizer  : {args.tokenizer}", flush=True)
    print(f"  n_tokens   : {args.n_tokens:,}", flush=True)
    print(f"  seq_len    : {args.seq_len}", flush=True)
    print(f"  n_seqs     : {n_seqs_total:,}", flush=True)
    print(f"  temperature: {args.temperature}", flush=True)
    print(f"  seed       : {args.seed}", flush=True)
    print(f"  chunk_seqs : {CHUNK_SEQS}", flush=True)

    # Read vocab_size from checkpoint config before loading vLLM.
    # (Avoids importing transformers inside the vLLM-loaded process where possible.)
    from transformers import GPTNeoXConfig  # noqa: PLC0415
    cfg = GPTNeoXConfig.from_pretrained(str(ckpt_path))
    vocab_size = cfg.vocab_size
    print(f"  vocab_size : {vocab_size}", flush=True)

    # ── load model ─────────────────────────────────────────────────────────────
    print("\nLoading model via vLLM ...", flush=True)
    llm = LLM(
        model=str(ckpt_path),
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=args.seq_len,   # 512: exact training context length
        trust_remote_code=False,      # standard GPTNeoXForCausalLM architecture
        enforce_eager=False,          # use CUDA graphs for max throughput
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=n_new_tokens,
        ignore_eos=True,    # always generate the full seq_len regardless of EOS
        seed=args.seed + 1000,  # separate from context-token seed (protocol-equiv)
    )

    # ── pre-registered RNG for context token draws ───────────────────────────
    ctx_rng = np.random.default_rng(args.seed)

    # ── generation loop ────────────────────────────────────────────────────────
    n_chunks = (n_seqs_total + CHUNK_SEQS - 1) // CHUNK_SEQS
    print(f"\nGenerating in {n_chunks} chunks of ≤{CHUNK_SEQS} seqs ...", flush=True)

    t0 = time.time()
    total_written = 0
    chunk_idx = 0

    with open(out_path, "wb") as fout:
        while total_written < args.n_tokens:
            remaining_seqs = n_seqs_total - (total_written // args.seq_len)
            batch_seqs = min(CHUNK_SEQS, remaining_seqs)
            if batch_seqs <= 0:
                break

            # Draw random start tokens (pre-registration: "1 token drawn
            # uniformly from vocabulary (separate generator)", seed=85).
            start_tokens = ctx_rng.integers(0, vocab_size, size=batch_seqs)

            # vLLM token-ID prompts (no text tokenization).
            prompts = [{"prompt_token_ids": [int(t)]} for t in start_tokens]

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            # Collect and write: prompt token + generated tokens → seq_len total.
            for output in outputs:
                if total_written >= args.n_tokens:
                    break
                prompt_tok = list(output.prompt_token_ids)   # [1 token]
                gen_toks   = list(output.outputs[0].token_ids)  # [seq_len-1 tokens]
                seq = prompt_tok + gen_toks  # length = seq_len

                remaining = args.n_tokens - total_written
                to_write = min(len(seq), remaining)
                np.array(seq[:to_write], dtype=np.uint16).tofile(fout)
                total_written += to_write

            chunk_idx += 1
            if chunk_idx % 5 == 0 or total_written >= args.n_tokens:
                elapsed = time.time() - t0
                rate = total_written / elapsed / 1e6
                eta = (args.n_tokens - total_written) / max(total_written, 1) * elapsed
                print(
                    f"  chunk {chunk_idx}/{n_chunks}: {total_written:,} tok "
                    f"({rate:.2f}M tok/s, ETA {eta:.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    print(
        f"\nDone: {total_written:,} tokens → {out_path} "
        f"({size_gb:.2f} GB) in {elapsed:.0f}s "
        f"({total_written / elapsed / 1e6:.2f}M tok/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
