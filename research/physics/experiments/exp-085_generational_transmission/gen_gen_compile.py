"""
exp-085 — Generate corpus using torch.compile + StaticCache fallback.

Written 2026-07-18 after vLLM encountered flashinfer JIT compilation failure:
  RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
Modal's debian_slim container has CUDA runtime but not the CUDA development toolkit
(no nvcc), so flashinfer cannot JIT-compile sampling kernels.

Fallback pre-registered in notes.md: torch.compile + StaticCache on transformers.
First attempt (v1) achieved only 0.041M tok/s — missing bfloat16 and TF32 precision.
This version (v2) adds:
  - torch_dtype=torch.bfloat16 (halves memory bandwidth requirement)
  - torch.set_float32_matmul_precision('high') enables TF32 matrix multiply
  - explicit torch.compile(model, mode='reduce-overhead') upfront compilation
Expected: ~0.2-0.4M tok/s on A100-40GB → 2750-5500s for 1.1B tokens (within 7200s).

Protocol equivalence to pre-registered gen_gen_vllm.py:
  - Context token RNG: np.default_rng(seed)   [identical to gen_gen.py]
  - Sampling: multinomial at temperature=1.0    [protocol-equivalent; bfloat16
    generates statistically equivalent corpus, not bit-for-bit identical]
  - Output format: uint16 binary                 [identical]
  - Seq length, n_tokens, batch structure        [identical]

Uses image_train (torch==2.12.0, transformers==5.8.1) — no new dependencies.

Ariel — July 18, 2026.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

# BATCH_SIZE: 2048 sequences × 512 tokens = 1.048M tokens/call.
# Steady-state throughput (post-compile warmup) ~0.125M tok/s measured from batch-50.
# At 0.125M tok/s: 1.1B tokens in ~8800s + warmup overhead → fits within 14400s timeout.
BATCH_SIZE = 2048   # sequences generated in parallel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", help="path to step_2000/ directory on volume")
    ap.add_argument("output_bin",     help="output .bin path (uint16)")
    ap.add_argument("--n_tokens",     type=int,   default=1_100_000_000)
    ap.add_argument("--seq_len",      type=int,   default=512)
    ap.add_argument("--temperature",  type=float, default=1.0)
    ap.add_argument("--seed",         type=int,   default=85,
                    help="RNG seed for context-token draws (pre-registered)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint_dir)
    out_path  = Path(args.output_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_new_tokens  = args.seq_len - 1
    n_seqs_total  = (args.n_tokens + args.seq_len - 1) // args.seq_len
    n_batches     = (n_seqs_total + BATCH_SIZE - 1) // BATCH_SIZE

    print("=== exp-085 generate (torch.compile + StaticCache) ===", flush=True)
    print(f"  checkpoint : {ckpt_path}",        flush=True)
    print(f"  output     : {out_path}",          flush=True)
    print(f"  n_tokens   : {args.n_tokens:,}",  flush=True)
    print(f"  seq_len    : {args.seq_len}",       flush=True)
    print(f"  n_seqs     : {n_seqs_total:,}",    flush=True)
    print(f"  batch_size : {BATCH_SIZE}",         flush=True)
    print(f"  temperature: {args.temperature}",   flush=True)
    print(f"  seed       : {args.seed}",          flush=True)

    # TF32 precision on A100 gives ~3x speedup for matmul (same numerical results
    # as fp32 for statistical sampling purposes). Addresses torch._inductor warning.
    torch.set_float32_matmul_precision("high")

    device = "cuda"
    print("\nLoading model (bfloat16) ...", flush=True)
    model = GPTNeoXForCausalLM.from_pretrained(
        str(ckpt_path),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,   # halves memory bandwidth vs float32
    ).to(device)
    model.eval()

    vocab_size = model.config.vocab_size
    print(f"  vocab_size : {vocab_size}", flush=True)

    # Explicit compilation upfront; warmup happens on first call to model().
    # mode='reduce-overhead' minimises Python-level step overhead via CUDA graphs.
    print("  Compiling model (torch.compile, mode=reduce-overhead) ...", flush=True)
    model = torch.compile(model, mode="reduce-overhead")

    # Pre-registered RNG for context token draws (same as gen_gen.py)
    ctx_rng = np.random.default_rng(args.seed)

    t0 = time.time()
    total_written = 0
    batch_idx     = 0

    with open(out_path, "wb") as fout:
        while total_written < args.n_tokens:
            remaining_seqs = n_seqs_total - (total_written // args.seq_len)
            batch_seqs     = min(BATCH_SIZE, remaining_seqs)
            if batch_seqs <= 0:
                break

            # Draw random start tokens (pre-registration protocol)
            start_tokens = ctx_rng.integers(0, vocab_size, size=batch_seqs)
            input_ids    = torch.tensor(
                start_tokens, dtype=torch.long, device=device
            ).unsqueeze(1)   # (batch_seqs, 1)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=n_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    cache_implementation="static",
                    use_cache=True,
                    # Return int64 on device; cast to uint16 when writing
                )
            # outputs: (batch_seqs, seq_len)

            for i in range(batch_seqs):
                if total_written >= args.n_tokens:
                    break
                seq       = outputs[i].cpu().numpy().astype(np.uint16)
                remaining = args.n_tokens - total_written
                to_write  = min(len(seq), remaining)
                seq[:to_write].tofile(fout)
                total_written += to_write

            batch_idx += 1
            if batch_idx % 10 == 0 or total_written >= args.n_tokens:
                elapsed = time.time() - t0
                rate    = total_written / max(elapsed, 1e-9) / 1e6
                eta     = (args.n_tokens - total_written) / max(total_written, 1) * elapsed
                print(
                    f"  batch {batch_idx}/{n_batches}: {total_written:,} tok "
                    f"({rate:.3f}M tok/s, ETA {eta:.0f}s)",
                    flush=True,
                )

    elapsed  = time.time() - t0
    size_gb  = out_path.stat().st_size / 1e9
    rate_fin = total_written / max(elapsed, 1e-9) / 1e6
    print(
        f"\nDone: {total_written:,} tokens → {out_path} "
        f"({size_gb:.2f} GB) in {elapsed:.0f}s ({rate_fin:.3f}M tok/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
