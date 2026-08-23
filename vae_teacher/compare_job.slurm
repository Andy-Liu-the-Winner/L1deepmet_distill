#!/bin/bash
#SBATCH --job-name=compare_students
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_compare_%j.out
#SBATCH --error=/home/export/xuantinl/L1deepmet_distill/vae_teacher/slurm_compare_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env
cd /home/export/xuantinl/L1deepmet_distill/vae_teacher

python compare_students.py \
    --data ../data/data4L1/data_ttbar \
    --ckpt_orig   ../student_ckpts_L1_fixed/best.pth.tar \
    --ckpt_tprime ../student_ckpts_tprime/best.pth.tar \
    --out_dir     ../student_comparison_results

EXIT_CODE=$?
echo "Comparison exit code: $EXIT_CODE"
exit $EXIT_CODE
