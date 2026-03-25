#!/bin/bash
#SBATCH --job-name=vae_tprime
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_vae_%j.out
#SBATCH --error=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_vae_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

cd /home/export/xuantinl/L1deepmet_distill/vae_teacher

python trainVAE.py \
    --data  ../data/data4L1/data_ttbar \
    --teacher_ckpt ../teacher_ckpts_L1_fixed/best.pth.tar \
    --ckpts ../teacher_prime_ckpts
EXIT_CODE=$?
echo "Phase 2 exit code: $EXIT_CODE"
exit $EXIT_CODE
