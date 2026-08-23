#!/bin/bash
# Monitor jobs and run plotting when complete

echo "Monitoring jobs 6 (teacher) and 7 (student)..."
echo "Started at: $(date)"

while true; do
    # Check if any jobs still running
    RUNNING=$(squeue -u xuantinl -h | wc -l)

    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "All jobs completed at: $(date)"
        break
    fi

    # Show current status
    echo -n "."
    sleep 60
done

echo ""
echo "Running plotting script..."

# Activate environment and run plotting
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /home/export/xuantinl/envs/env

cd /home/export/xuantinl/L1deepmet_distill/plots_L1_20260128
python plot_L1.py \
    --teacher_ckpts ../teacher_ckpts_L1_20260128 \
    --student_ckpts ../student_ckpts_L1_20260128 \
    --output .

echo ""
echo "Done! Check plots in: /home/export/xuantinl/L1deepmet_distill/plots_L1_20260128/"
ls -la *.png 2>/dev/null || echo "No plots generated yet"
