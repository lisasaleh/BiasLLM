#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ftAya
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=eval_aya_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load CUDA/12.6.0

cd $HOME/atcs/bias
source activate atics_p2

# Allocate more memory for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS_DIR="./aya_saved_models/"
RESULTS_DIR="./aya_eval_results/"
# Create output directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Evaluate all saved models
for checkpoint in $MODELS_DIR/aya-expanse-8b_*; do
    echo "Evaluating checkpoint: $checkpoint"
    python eval_ckpt.py \
        --checkpoint_path "$checkpoint" \
        --bs 1 \
        --quantize_4bit \
        --output_dir $RESULTS_DIR
done
