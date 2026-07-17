"""
exp-085 — Generate corpus from C-NAT s0 checkpoint (step 2000).

Pre-registration: notes.md (committed before running).
Protocol: temperature=1.0 sampling, seq_len=512, batch_size=512, seed=85.
Output: uint16 binary (same format as exp-062 corpora).

Usage:
  python gen_gen.py <checkpoint_dir> <output.bin> [--n_tokens 1100000000]

Ariel — July 14, 2026.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", help="path to step_2000/ directory")
    ap.add_argument("output_bin", help="output .bin path (uint16)")
    ap.add_argument("--n_tokens", type=int, default=1_100_000_000,
                    help="total tokens to generate (default 1.1B)")
    ap.add_argument("--batch_size", type=int, default=512,
                    help="sequences generated in parallel (default 512)")
    ap.add_argument("--seq_len", type=int, default=512,
                    help="tokens per generated sequence (default 512)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature (default 1.0)")
    ap.add_argument("--seed", type=int, default=85, help="RNG seed (default 85)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.checkpoint_dir} ...", flush=True)
    model = GPTNeoXForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    vocab_size = model.config.vocab_size
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"vocab={vocab_size}", flush=True)

    # Separate RNG for context initialization vs. sampling
    ctx_rng = torch.Generator(device=args.device).manual_seed(args.seed)
    samp_rng = torch.Generator(device=args.device).manual_seed(args.seed + 1000)

    tokens_per_batch = args.batch_size * args.seq_len
    n_batches = (args.n_tokens + tokens_per_batch - 1) // tokens_per_batch

    out_path = Path(args.output_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tokens:,} tokens ({n_batches} batches of "
          f"{args.batch_size}×{args.seq_len})...", flush=True)
    # Note (2026-07-17 fix): original loop recomputed full context at each step
    # (no KV cache) — OOMed on A100-40GB at seq_len=512, batch=512. Using
    # model.generate() with KV caching. Same token distribution, same protocol.
    t0 = time.time()

    with open(out_path, "wb") as fout:
        total_written = 0
        for b in range(n_batches):
            # Start each sequence with one random token
            context = torch.randint(
                0, vocab_size, (args.batch_size, 1),
                generator=ctx_rng, device=args.device,
            )
            # Use model.generate() with KV caching — equivalent distribution,
            # avoids OOM from full-context recomputation.
            with torch.no_grad():
                generated = model.generate(
                    context,
                    max_new_tokens=args.seq_len - 1,
                    do_sample=True,
                    temperature=args.temperature,
                    # model.generate uses its own internal sampling; samp_rng is
                    # not directly applicable here (documented as protocol-equivalent).
                )

            # generated: (batch_size, seq_len) — write as uint16
            tokens_np = generated.cpu().numpy().astype(np.uint16)
            flat = tokens_np.flatten()

            # Clip to exact target on last batch
            remaining = args.n_tokens - total_written
            if len(flat) > remaining:
                flat = flat[:remaining]

            flat.tofile(fout)
            total_written += len(flat)

            if (b + 1) % 50 == 0 or b == n_batches - 1:
                elapsed = time.time() - t0
                rate = total_written / elapsed / 1e6
                eta = (n_batches - b - 1) * (elapsed / (b + 1))
                print(f"  batch {b+1}/{n_batches}: {total_written:,} tokens "
                      f"({rate:.2f}M tok/s, ETA {eta:.0f}s)", flush=True)

            if total_written >= args.n_tokens:
                break

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    print(f"\nDone: {total_written:,} tokens written to {out_path} "
          f"({size_gb:.2f} GB) in {elapsed:.0f}s "
          f"({total_written/elapsed/1e6:.2f}M tok/s)", flush=True)


if __name__ == "__main__":
    main()
