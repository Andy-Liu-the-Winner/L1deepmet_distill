# L1 DeepMET Distillation

Baseline knowledge distillation for the L1 trigger DeepMET model. The student learns to match the teacher's per-particle weight outputs (output-level distillation only — no intermediate layer matching).

Both teacher and student are graph neural networks (EdgeConv-based) that predict per-particle weights used to compute MET:

```
METx = sum(weights * px)
METy = sum(weights * py)
```

Graphs are built in eta-phi space using a radius graph with deltaR = 0.4.

## Data

L1 trigger data with 8 features per particle:

| Column | Feature |
|--------|---------|
| 0 | pt |
| 1 | px |
| 2 | py |
| 3 | eta |
| 4 | d0 |
| 5 | dz |
| 6 | pdgid (categorical) |
| 7 | charge (categorical) |

6 continuous features, 2 categorical. Truth labels are `(genMETx, genMETy)`.

Expected data location: `data/data4L1/data_ttbar/` (PyTorch Geometric format, `.pt` files).

## Setup

Requires Python 3.9+ with:

```
torch
torch_geometric
torch_scatter
torch_cluster
numpy
matplotlib
tqdm
```

## Training

### Teacher

```bash
cd teacher_deepmet
python trainL1.py --data ../data/data4L1/data_ttbar --ckpts ../teacher_ckpts_L1
```

Uses `loss_fn_response_tune` which combines a resolution term (MSE on METx/METy) with a response correction term that penalizes deviations from response = 1 for events with qT > 50 GeV.

To resume from a checkpoint:
```bash
python trainL1.py --data ../data/data4L1/data_ttbar --ckpts ../teacher_ckpts_L1 --restore_file last
```

For GPU training via SLURM, see `train_L1_job.slurm`.

### Student (distillation)

```bash
cd student_deepmet
python trainL1.py --data ../data/data4L1/data_ttbar --ckpts ../student_ckpts_L1 --teacher_ckpt ../teacher_ckpts_L1/best.pth.tar
```

The student loss is a weighted combination:

```
loss = alpha * task_loss + (1 - alpha) * MSE(student_weights, teacher_weights)
```

where `alpha = 0.5`. The task loss is the same `loss_fn_response_tune` used by the teacher. The distillation term is just MSE between the student and teacher per-particle weight predictions — no temperature scaling, no intermediate feature matching.

The teacher is frozen during student training.

### Architecture

Both models use the same hidden_dim (32) and conv_depth (2). The student (`StudentGraphMETNetwork`) drops the PV association embedding compared to the full-data teacher, but for L1 data this feature isn't available anyway, so the architectures are effectively the same size (~6.4k params). The point of this setup is to validate that output-level distillation works before experimenting with smaller student architectures.

## Evaluation and Plots

After training both models, generate comparison plots:

```bash
cd plots_L1_20260128
python plot_L1.py --teacher_ckpts ../teacher_ckpts_L1 --student_ckpts ../student_ckpts_L1 --output ../plots_L1
```

This produces:
- `loss_curves.png` — training/validation loss over epochs
- `response.png` — MET response vs qT
- `resolution_perp.png` — perpendicular resolution vs qT
- `resolution_parallel.png` — parallel resolution vs qT

Teacher and student curves are overlaid on the same plots.

## Directory Structure

```
teacher_deepmet/       # teacher model, training, evaluation
student_deepmet/       # student model with distillation training
  teacher_model/       # copy of teacher's model definition (for loading weights)
plots_L1_20260128/     # plotting script + pre-fix plots
plots_L1_fixed/        # plots after bug fixes
CHANGES_L1_FIX.md      # log of bug fixes applied
```
