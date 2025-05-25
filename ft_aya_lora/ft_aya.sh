#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ftAya
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=ft_eval_aya_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load CUDA/12.6.0

cd $HOME/atcs/bias
source activate atics_p2

# Allocate more memory for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS_DIR="./aya_saved_models/"
# Create output directory if it doesn't exist
mkdir -p $MODELS_DIR

RESULTS_DIR="./aya_eval_results_strict/"
# Create output directory if it doesn't exist
mkdir -p $RESULTS_DIR

SEED=42

# # Train models with different sampling strategies
# for sampling in "undersample" "oversample" "balanced" "normal"; do
#     echo "Training with sampling strategy: $sampling"
#     python ft_aya_lora_simple.py \
#         --seed $SEED \
#         --sampling $sampling \
#         --epochs 10 \
#         --bs 4 \
#         --lora_r 8 \
#         --lora_alpha 16 \
#         --output_dir $MODELS_DIR
# done

# echo "All models trained and saved in $MODELS_DIR"

echo "Starting evaluation of saved models..."

# Evaluate all saved models
for checkpoint in $MODELS_DIR/aya-expanse-8b_*; do
    echo "Evaluating checkpoint: $checkpoint"
    python eval_ckpt.py \
        --checkpoint_path "$checkpoint" \
        --seed $SEED \
        --bs 1 \
        --quantize_4bit \
        --output_dir $RESULTS_DIR
done

echo "Evaluation complete! Results saved in $RESULTS_DIR"
