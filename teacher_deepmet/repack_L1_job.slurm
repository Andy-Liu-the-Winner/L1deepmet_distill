#!/bin/bash
#SBATCH --job-name=repack_L1
#SBATCH --partition=work
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_repack_%j.out
#SBATCH --error=slurm_repack_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

cd /home/export/xuantinl/L1deepmet_distill/teacher_deepmet
python repack_L1_data.py --data ../data/data4L1/data_ttbar

echo "Repacking completed!"
