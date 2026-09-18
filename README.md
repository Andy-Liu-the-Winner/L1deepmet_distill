# L1 DeepMET Distillation

L1 trigger MET regression for CMS Phase-2. A network assigns a per-particle weight `w_i`
to each L1 PUPPI candidate; MET is the weighted vector sum `MET = s · Σ w_i · p⃗_i`. The
deployed model must stay EdgeConv / FlowGNN-FPGA-compatible, so a large teacher defines
the quality ceiling and a small student carries the hardware constraints.

> **Start here: [`DISTILLATION.md`](DISTILLATION.md)** — the handoff document. It covers
> the architectures (including the ParT transformer teacher), the distillation
> implementation, all results, reproduction commands, open threads and known dead ends.
> This README is only an entry point.

## Current state (2026-09-18)

- **Teachers done.** ParT transformer teacher (138k params) beats PUPPI, the v1 EdgeConv
  teacher and a GraphNet control on every scaled-resolution metric. GraphNet is kept as
  an architecture control.
- **Distillation done, and it is a null result.** The distilled 6.4k student is
  metric-identical to an identically-trained from-scratch student (all deltas ≤ 0.6 %).
  Weight mimicry succeeded; no physics metric moved. See `DISTILLATION.md` §7.
- **Recommended artifact: the from-scratch student**, `student_ckpts_scratch_r1/`. Same
  metrics, no teacher dependency in the pipeline.
- **Next move:** a cheap global feature for the student, not more distillation
  (`DISTILLATION.md` §8).

Validation results (scaled σ in GeV for qT 50–150 / 150–300, plateau R over 50–300):

| model | params | scaled σ(u⊥) | scaled σ(u∥) | plateau R |
|---|---|---|---|---|
| PUPPI baseline | — | 29.7 / 29.2 | 49.9 / 68.3 | 0.816 |
| v1 teacher | 6.4k | 25.5 / 27.5 | 43.5 / 62.5 | 0.782 |
| student, scratch | 6.4k | 25.2 / 25.6 | 43.1 / 57.8 | 0.855 |
| student, distilled | 6.4k | 25.1 / 25.6 | 42.8 / 57.9 | 0.846 |
| GraphNet teacher | 129k | 24.3 / 24.5 | 39.5 / 54.6 | 0.866 |
| ParT teacher (r2) | 138k | 24.9 / 24.2 | 31.8 / 53.8 | 0.874 |

## Data

L1 PUPPI candidates, 8 features per particle. **Verified column format:**

| col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----|---|---|---|---|---|---|---|---|
| | pt | px | py | eta | phi | puppiWeight | pdgid | charge |

Columns 4/5 are **phi and puppiWeight** — the pre-2026-08 README claimed d0/dz, which was
wrong and broke every loss function that indexed them (`CHANGES_L1_FIX.md`). Truth labels
are `(genMETx, genMETy)`; sign convention `MET = Σ w·p ≈ −genMET`.

Raw data: `data/data4L1/data_ttbar/`. The current pipeline trains on repacked shards in
`data/data4L1/data_ttbar/shards_v2/` — 995,000 events, seed-42 75/25 split.

## Environment

Cluster conda env `/home/export/xuantinl/envs/env` — **torch 1.12.1**, numpy 1.26,
torch_geometric / torch_scatter / torch_cluster, matplotlib, tqdm.

GPU use is capped at ~20 % of the shared A100 (`--mem_fraction 0.18`, `--gres=mps:20`).

## Quick start

```bash
# 0. one-time: repack the 1M single-event .pt files into shards (~30 min)
cd teacher_deepmet && sbatch repack_L1_job.slurm

# 1. teacher (ParT); --arch graph trains the GraphNet control instead
sbatch train_ParT_job.slurm --ckpts ../teacher_ckpts_ParT_r2 --lam 15

# 2. student — distilled, and the from-scratch A/B baseline
sbatch train_student_distill_job.slurm --ckpts ../student_ckpts_distill_ParT_r1
sbatch train_student_distill_job.slurm --ckpts ../student_ckpts_scratch_r1 --teacher_ckpt ''

# 3. plots
python compare_all_teachers.py --output ../teacher_comparison_new
python compare_students.py     --output ../student_comparison_results

# 4. teacher unit tests (rotation invariance, padding, step-0 PUPPI gate)
python test_ParT.py
```

Checkpoints are gitignored and live on the cluster only. New-style checkpoints hold EMA
weights in `state_dict` and raw weights in `raw_state_dict` — **load `state_dict`**.

## Layout

```
teacher_deepmet/     current pipeline: ParT + GraphNet teachers, shards, distillation, comparisons
  teacher_ParT_DESIGN.md      design spec for the transformer teacher (read before modifying it)
  teacher_GraphNet_DESIGN.md  design spec for the GNN control
  trainL1_student_distill.py  the distillation script (and its scratch baseline)
student_deepmet/     the deployed EdgeConv student architecture + superseded v1 pipeline
vae_teacher/         dead end, kept for the record (DISTILLATION.md §9)
*_ckpts*/            checkpoints (gitignored)
teacher_comparison_new/      4-way teacher plots
student_comparison_results/  student plots, pre-registration, measured results
plots_L1_20260128/, plots_L1_fixed/, teacher_comparison_results/   older plot sets
CHANGES_L1_FIX.md    the 2026-02 column-index fix
```

The v1 pipeline (`trainL1.py`, `evaluate.py`, `model/data_loader.py` in both trees) is
superseded but kept, since the v1 teacher is still a comparison baseline.
