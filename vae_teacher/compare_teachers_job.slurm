#!/bin/bash
#SBATCH --job-name=compare_TT
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_compare_TT_%j.out
#SBATCH --error=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_compare_TT_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env
cd /home/export/xuantinl/L1deepmet_distill/vae_teacher

python compare_teachers.py \
    --data         ../data/data4L1/data_ttbar \
    --teacher_ckpt ../teacher_ckpts_L1_fixed/best.pth.tar \
    --vae_ckpt     ../teacher_prime_ckpts/best_vae.pth.tar \
    --out_dir      ../teacher_comparison_results

EXIT_CODE=$?
echo "T vs T' comparison exit code: $EXIT_CODE"
exit $EXIT_CODE
