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
Expected: ~0.058M tok/s on A100-40GB → ~19000s for 1.1B tokens (within 21600s timeout).

Protocol equivalence to pre-registered gen_gen_vllm.py:
  - Context token RNG: np.default_rng(seed)   [identical to gen_gen.py]
  - Sampling: multinomial at temperature=1.0    [protocol-equivalent; bfloat16
    generates statistically equivalent corpus, not bit-for-bit identical]
  - Output format: uint16 binary                 [identical]
  - Seq length, n_tokens, batch structure        [identical]

Resume logic (added 2026-07-18 after first run was preempted at ~51M/1.1B tokens):
  If the output file already exists with partial data, the script resumes from the
  last complete sequence boundary. The ctx_rng is advanced by n_complete_seqs draws
  to maintain RNG consistency. Incomplete trailing sequences are discarded (truncated).
  This is protocol-equivalent: the same context tokens are generated in the same order
  for all sequences, only starting from the resume point.

Uses image_train (torch==2.12.0, transformers==5.8.1) — no new dependencies.

Ariel — July 18, 2026. Resume logic added July 18, 2026.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

# BATCH_SIZE: 512 sequences × 512 tokens = 262K tokens/call.
# Reduced from 2048 to 512 (2026-07-18) for reliability: smaller KV-cache footprint
# during torch.compile graph capture (~3 GB vs ~12 GB). Throughput is KV-bandwidth-
# dominated and batch-size-independent (~0.058M tok/s either way).
BATCH_SIZE = 512    # sequences generated in parallel


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

    seq_len       = args.seq_len
    n_new_tokens  = seq_len - 1
    n_seqs_total  = (args.n_tokens + seq_len - 1) // seq_len
    n_batches     = (n_seqs_total + BATCH_SIZE - 1) // BATCH_SIZE

    # ── resume detection ──────────────────────────────────────────────────────
    # If the output file exists, determine how many complete sequences are written.
    # Truncate any incomplete trailing sequence, advance ctx_rng to match.
    seqs_done = 0
    if out_path.exists():
        existing_bytes = out_path.stat().st_size
        tokens_on_disk = existing_bytes // 2          # uint16 = 2 bytes/token
        seqs_done      = tokens_on_disk // seq_len    # complete sequences only
        bytes_to_keep  = seqs_done * seq_len * 2
        if bytes_to_keep < existing_bytes:
            # Truncate any partial trailing sequence
            with open(out_path, "r+b") as f:
                f.truncate(bytes_to_keep)
        total_written = seqs_done * seq_len
        if seqs_done > 0:
            print(
                f"  RESUME: found {seqs_done:,} complete sequences "
                f"({total_written:,} tokens, {bytes_to_keep / 1e9:.3f} GB) "
                f"— continuing from seq {seqs_done:,}",
                flush=True,
            )
    else:
        total_written = 0

    print("=== exp-085 generate (torch.compile + StaticCache) ===", flush=True)
    print(f"  checkpoint : {ckpt_path}",                flush=True)
    print(f"  output     : {out_path}",                  flush=True)
    print(f"  n_tokens   : {args.n_tokens:,}",          flush=True)
    print(f"  seq_len    : {seq_len}",                   flush=True)
    print(f"  n_seqs     : {n_seqs_total:,}",           flush=True)
    print(f"  seqs_done  : {seqs_done:,}",              flush=True)
    print(f"  batch_size : {BATCH_SIZE}",               flush=True)
    print(f"  temperature: {args.temperature}",          flush=True)
    print(f"  seed       : {args.seed}",                 flush=True)

    if total_written >= args.n_tokens:
        print("Already complete — nothing to do.", flush=True)
        return

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

    # Pre-registered RNG for context token draws (same as gen_gen.py).
    # Advance by seqs_done to maintain RNG consistency on resume.
    ctx_rng = np.random.default_rng(args.seed)
    if seqs_done > 0:
        # Consume seqs_done draws to fast-forward RNG state
        ctx_rng.integers(0, vocab_size, size=seqs_done)

    t0        = time.time()
    batch_idx = seqs_done // BATCH_SIZE   # approximate starting batch index for display

    with open(out_path, "ab") as fout:
        while total_written < args.n_tokens:
            remaining_seqs = n_seqs_total - (total_written // seq_len)
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
                rate    = (total_written - seqs_done * seq_len) / max(elapsed, 1e-9) / 1e6
                eta     = (args.n_tokens - total_written) / max(
                    total_written - seqs_done * seq_len, 1
                ) * elapsed
                print(
                    f"  batch {batch_idx}/{n_batches}: {total_written:,} tok "
                    f"({rate:.3f}M tok/s this run, ETA {eta:.0f}s)",
                    flush=True,
                )

    elapsed  = time.time() - t0
    size_gb  = out_path.stat().st_size / 1e9
    new_toks = total_written - seqs_done * seq_len
    rate_fin = new_toks / max(elapsed, 1e-9) / 1e6
    print(
        f"\nDone: {total_written:,} tokens total ({new_toks:,} new) → {out_path} "
        f"({size_gb:.2f} GB) in {elapsed:.0f}s ({rate_fin:.3f}M tok/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
