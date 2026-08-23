#!/bin/bash
#SBATCH --job-name=teacher_GraphNet
#SBATCH --partition=work
#SBATCH --gres=mps:20
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_GraphNet_%j.out
#SBATCH --error=slurm_GraphNet_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

cd /home/export/xuantinl/L1deepmet_distill/teacher_deepmet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# --lam 15: identical loss config to the ParT r2 run (qT-binned response term)
# so the architecture comparison is controlled.
python trainL1_ParT.py --data ../data/data4L1/data_ttbar/shards_v2 \
                     --ckpts ../teacher_ckpts_GraphNet_r1 --arch graph --lam 15 "$@"

echo "Training completed!"
