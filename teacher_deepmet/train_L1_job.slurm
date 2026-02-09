#!/bin/bash
#SBATCH --job-name=teacher_L1
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

# Navigate to training directory
cd /home/export/xuantinl/L1deepmet_distill/teacher_deepmet

# Resume training from epoch 2, run to epoch 10
python trainL1.py --data ../data/data4L1/data_ttbar --ckpts ../teacher_ckpts_L1_fixed --restore_file last

echo "Training completed!"
