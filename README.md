# BiasLLM: Bias Detection in Dutch Government Documents with Multilingual LLMs

This repository contains code and results from a research project exploring how open-source large language models (LLMs) like Aya Expanse 8B and mT5 can be used to detect bias in Dutch government documents. We apply prompt-based methods and LoRA-based fine-tuning, and evaluate multiple sampling strategies and prompt styles.

---

## Models Used

- **Aya Expanse 8B**: A multilingual causal decoder model optimized for generation tasks, adjusted for classification problem in this project. Evaluated in zero-shot and fine-tuned settings using LoRA.
  - [Aya on HuggingFace](https://huggingface.co/CohereLabs/aya-expanse-8b)
- **mT5**: A multilingual encoder-decoder model used with instruction tuning and chain-of-thought prompting.
  - [mT5 Base](https://huggingface.co/google/mt5-base)
- **BERTje / RobBERT**: BERT-based Dutch language models used for baseline comparison (results from de Swart et al., 2025).

---

## Project Structure

```
BiasLLM/
├── mt5/                         # mT5 model training and evaluation
│   ├── finetuning_mt5/          # Fine-tuning scripts for mT5
│   └── mt5_classification/      # Classification tasks with mT5
├── aya/                         # Aya model implementation
│   ├── utils/                   # Utility functions for Aya
│   ├── ft_aya_lora/             # Fine-tuning Aya with LoRA
│   └── prompting/               # Prompt engineering experiments
├── results/                     # Training results and model checkpoints
│   ├── results_aya/     
│   ├── results_mt5/                
│   └── checkpoints/                    
└── environment.yml              # Conda environment configuration
```

## Dataset

The project uses the Dutch Government Data for Bias Detection dataset, which contains:
- Training set: 1,811 examples
- Validation set: 713 examples
- Test set: 782 examples
  
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

## Models Details

### mT5
- Fine-tuned mT5 model for bias detection
- Supports multiple sampling strategies (undersample, oversample, balanced, normal)
- Includes focal loss implementation for handling class imbalance

### Aya
- Implementation using the Aya model for bias detection
- Includes various prompting strategies
- Supports fine-tuning with LoRA for efficient adaptation

## Usage

### Running mT5

#### Training

```bash
python mt5/finetuning_mt5/pretraining_mt5_classification.py --sampling balanced --epochs 12 --lr 5e-5 --bs 64
```

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--sampling`     | Data sampling strategy (`undersample`, `oversample`, `balanced`, `normal`)  |
| `--epochs`       | Number of training epochs                                                   |
| `--lr`           | Learning rate                                                               |
| `--bs`           | Batch size                                                                  |
| `--wd`           | Weight decay                                                                |
| `--patience`     | Early stopping patience                                                     |
| `--output_dir`   | Output directory for model checkpoints and logs                             |
| `--seed`         | Random seed for reproducibility                                             |
| `--focal_loss`   | Use focal loss to handle class imbalance                                    |
| `--focal_gamma`  | Gamma value for focal loss (default: 2.0)                                   |

#### Focal Loss Option

To enable focal loss for handling class imbalance, add the `--focal_loss` flag:

```bash
python mt5/finetuning_mt5/pretraining_mt5_classification.py \
    --sampling balanced \
    --epochs 12 \
    --lr 5e-5 \
    --bs 64 \
    --focal_loss
```

This modifies the loss function to better handle underrepresented classes.

### Running Aya

#### Training with LoRA

To fine-tune Aya with LoRA using different sampling strategies, run the following script:

```bash
python aya/ft_aya_lora/ft_aya.sh
```

This script:
- Loads the `CohereLabs/aya-expanse-8b` model with 4-bit quantization
- Applies LoRA adapters
- Fine-tunes on the [Dutch Government Data for Bias Detection](https://huggingface.co/datasets/milenamileentje/Dutch-Government-Data-for-Bias-detection)
- Saves adapter

#### SLURM Training Arguments

Inside `ft_aya.sh`, these key arguments are passed to `ft_aya_lora_simple.py`:

| Argument       | Description                                   | Example                               |
|----------------|-----------------------------------------------|---------------------------------------|
| `--seed`       | Random seed for reproducibility               | `42`                                  |
| `--sampling`   | Data strategy:                                |`undersample`, `oversample`, `balanced`|
|                                                                        `normal`, `balanced`            |
| `--epochs`     | Number of training epochs                     | `10`                                  |
| `--bs`         | Batch size per device                         | `4`                                   |
| `--lora_r`     | LoRA rank                                     | `8`                                   |
| `--lora_alpha` | LoRA alpha                                    | `16`                                  |
| `--output_dir` | Directory to save trained adapters            | `./aya_saved_models/`                 |

### Evaluation

After training, the script automatically evaluates all saved models using `eval_ckpt.py`.

#### Evaluation Arguments

| Argument             | Description                                              |
|----------------------|----------------------------------------------------------|
| `--checkpoint_path`  | Path to LoRA adapter to evaluate                         |
| `--bs`               | Evaluation batch size                                    |
| `--quantize_4bit`    | Evaluate under 4-bit quantization                        |
| `--output_dir`       | Directory to store evaluation metrics (CSV + JSON)       |

## Results

Model checkpoints and evaluation results are stored in the `results/` directory.

## Environment

The project uses Python 3.10.12 with PyTorch 2.0.1 and Transformers 4.51.3. See `environment.yml` for full dependencies.

## Contributors

Ina Klaric, Adriana Haralambieva, Raya Mezeklieva, Weronika Orska, Lisa Saleh

## More Information
  
- Supervised by Vera Neplenbroek, Univeristy of Amsterdam
