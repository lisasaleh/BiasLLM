#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ftAya
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=eval_prompt_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load CUDA/12.6.0

cd $HOME/atcs/bias
source activate atics_p2

# Allocate more memory for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Define directories
MODELS_DIR="./aya_saved_models/"

PROMPTS=("same" "CoT")

# Run the script for each prompt type
for PROMPT in "${PROMPTS[@]}"; do
    echo "Evaluating with prompt: $PROMPT"
    if [[ "$PROMPT" == "same" ]]; then
        RESULTS_DIR="./aya_eval_results_same_prompt/"
    elif [[ "$PROMPT" == "CoT" ]]; then
        RESULTS_DIR="./aya_eval_results_CoT_prompt/" 
    else 
        RESULTS_DIR="./aya_eval_results_custom_prompt/"
    fi

    # Create output directory if it doesn't exist
    mkdir -p $RESULTS_DIR

    # Evaluate all saved models that include "seed42" and don't include "WL" in their name 
    for checkpoint in $MODELS_DIR/aya-expanse-8b_*; do
        if [[ "$checkpoint" == *"seed42"* && "$checkpoint" != *"WL"* ]]; then
            echo "########### Evaluating checkpoint: $checkpoint with prompt: $PROMPT ###########"
            python eval_prompt.py \
                --checkpoint_path "$checkpoint" \
                --bs 1 \
                --quantize_4bit \
                --prompt $PROMPT \
                --output_dir $RESULTS_DIR
            
            if [[ $? -ne 0 ]]; then
                echo "Evaluation failed for checkpoint: $checkpoint with prompt: $PROMPT. Exiting."
                exit 1
            fi
        fi
    done

done

echo "All evaluations completed successfully! Results saved in $RESULTS_DIR"
