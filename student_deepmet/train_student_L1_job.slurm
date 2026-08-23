#!/bin/bash
#SBATCH --job-name=student_L1
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_student_%j.out
#SBATCH --error=slurm_student_%j.err

# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

# Navigate to student training directory
cd /home/export/xuantinl/L1deepmet_distill/student_deepmet

# Run student training with fixed L1 teacher
python trainL1.py \
    --data ../data/data4L1/data_ttbar \
    --ckpts ../student_ckpts_L1_fixed \
    --teacher_ckpt ../teacher_ckpts_L1_fixed/best.pth.tar

echo "Student training completed!"
