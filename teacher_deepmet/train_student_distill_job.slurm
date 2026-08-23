#!/bin/bash
#SBATCH --job-name=student_distill
#SBATCH --partition=work
#SBATCH --gres=mps:20
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_student_distill_%j.out
#SBATCH --error=slurm_student_distill_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

cd /home/export/xuantinl/L1deepmet_distill/teacher_deepmet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
python trainL1_student_distill.py "$@"

echo "Training completed!"
