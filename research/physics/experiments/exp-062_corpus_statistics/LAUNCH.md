# exp-062 launch runbook (cloud)

*Everything below is fixed by the pre-registration (`notes/2026-06-11_corpus_statistics_preregistration.md`). The calibration loop is closed (notes.md log); H values are frozen in `gen_corpora.py` SPECS. Do not re-tune anything after training starts.*

## What exists locally (M5 Max, June 11)

- `alphabet.json` — frozen 256-id synthetic alphabet.
- `corpora/C-SR_full.bin`, `C-PL15_full.bin`, `C-PL25_full.bin`, `C-PL40_full.bin` (+ `.mi.json` β̂ of record) — 1.06B tokens each, uint16. Regenerable bit-identically from seeds with `gen_corpora.py full <name>` (same numpy/scipy versions).
- C-NAT is NOT yet generated (needs the full TinyStories download): run `gen_corpora.py full C-NAT` then `gen_corpora.py mi corpora/C-NAT_full.bin` on the cloud box (or locally once bandwidth is free).

## Cloud steps (one A100/H100 box, ~2–3 GPU-h per run, 7 runs)

```bash
pip install torch transformers datasets numba scipy
# 1) corpora: either rsync corpora/ up, or regenerate from seeds (identical):
python gen_corpora.py alphabet && for c in C-SR C-PL15 C-PL25 C-PL40 C-NAT; do
  python gen_corpora.py full $c && python gen_corpora.py mi corpora/${c}_full.bin
done

# 2) the 7 pre-registered runs (seeds fixed in prereg §3):
python train.py corpora/C-SR_full.bin   run_CSR    --init-seed 1000 --data-seed 2000 --micro-batch 128
python train.py corpora/C-PL15_full.bin run_CPL15  --init-seed 1000 --data-seed 2000 --micro-batch 128
python train.py corpora/C-PL25_full.bin run_CPL25  --init-seed 1000 --data-seed 2000 --micro-batch 128
python train.py corpora/C-PL40_full.bin run_CPL40  --init-seed 1000 --data-seed 2000 --micro-batch 128
python train.py corpora/C-NAT_full.bin  run_CNAT_s0 --init-seed 1000 --data-seed 2000 --micro-batch 128
python train.py corpora/C-NAT_full.bin  run_CNAT_s1 --init-seed 1001 --data-seed 2001 --micro-batch 128
python train.py corpora/C-NAT_full.bin  run_CNAT_s2 --init-seed 1002 --data-seed 2002 --micro-batch 128

# 3) measurement (fp32, frozen protocol; final checkpoint primary, trajectory descriptive):
for r in run_CSR run_CPL15 run_CPL25 run_CPL40; do
  python measure.py runs/$r/step_2000 ${r}_final
done
for r in run_CNAT_s0 run_CNAT_s1 run_CNAT_s2; do
  python measure.py runs/$r/step_2000 ${r}_final --full-vocab
done
# emergence trajectories (optional but cheap):
# for s in 64 128 256 384 512 768 1024 1536; do python measure.py runs/$r/step_$s ${r}_s$s [...]; done
```

## After measurement

Decision statistics exactly per prereg §6 (slope of Δ̂_med on β̂/2 over formed power-law corpora, 1,000-draw head bootstrap; formation criterion ≥ 10/48 conformal) and §7 (multi-seed thresholds). Write the analysis script only after measurements exist, implementing only the pre-registered formulas. Verdict → notes.md, registry, manuscript §11, SESSION_BRIEF §6.
