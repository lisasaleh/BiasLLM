# BiasLLM: Bias Detection in Dutch Government Documents

This repository contains code for detecting bias in Dutch government documents using various language models, including mT5 and Aya.

## Project Structure

```
BiasLLM/
├── mt5/                          # mT5 model training and evaluation
│   ├── finetuning_mt5/          # Fine-tuning scripts for mT5
│   └── mt5_classification/      # Classification tasks with mT5
├── aya/                         # Aya model implementation
│   ├── utils/                   # Utility functions for Aya
│   ├── ft_aya_lora/            # Fine-tuning Aya with LoRA
│   └── prompting/               # Prompt engineering experiments
├── results/                      # Training results and model checkpoints
│   ├── results_aya/     
│   ├── results_mt5/                
│   └── checkpoints/                    
└── environment.yml              # Conda environment configuration
```

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate atics_p2
```

2. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Models

### mT5
- Fine-tuned mT5 model for bias detection
- Supports multiple sampling strategies (undersample, oversample, balanced, normal)
- Includes focal loss implementation for handling class imbalance

### Aya
- Implementation using the Aya model for bias detection
- Includes various prompting strategies
- Supports fine-tuning with LoRA for efficient adaptation

## Usage

### Training mT5
```bash
python mt5/finetuning_mt5/pretraining_mt5_classification.py --sampling balanced --epochs 12 --lr 5e-5 --bs 64
```

Arguments:
- `--sampling`: Data sampling strategy (undersample, oversample, balanced, normal)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--bs`: Batch size
- `--focal_loss`: Use focal loss for handling class imbalance

### Running Aya
```bash
python aya/promting/aya_prompt.py
```

## Dataset

The project uses the Dutch Government Data for Bias Detection dataset, which contains:
- Training set: 1,811 examples
- Validation set: 713 examples
- Test set: 782 examples

## Results

Model checkpoints and evaluation results are stored in the `results/` directory.

## Environment

The project uses Python 3.10.12 with PyTorch 2.0.1 and Transformers 4.51.3. See `environment.yml` for full dependencies.

## Contributors

Ina Klaric, Adriana Haralambieva, Raya Mezeklieva, Weronika Orska, Lisa Saleh
