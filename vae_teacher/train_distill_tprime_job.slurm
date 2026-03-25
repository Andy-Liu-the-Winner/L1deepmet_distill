#!/bin/bash
#SBATCH --job-name=distill_tprime
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_distill_%j.out
#SBATCH --error=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_distill_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

cd /home/export/xuantinl/L1deepmet_distill/vae_teacher

python trainDistillTPrime.py \
    --data  ../data/data4L1/data_ttbar \
    --teacher_ckpt ../teacher_ckpts_L1_fixed/best.pth.tar \
    --vae_ckpt     ../teacher_prime_ckpts/best_vae.pth.tar \
    --ckpts        ../student_ckpts_tprime
EXIT_CODE=$?
echo "Phase 4 exit code: $EXIT_CODE"
exit $EXIT_CODE
