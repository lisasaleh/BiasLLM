"""Finetune Aya Expanse 8B on teh DGDB dataset. 

Data imbalance in the dataset is handled using data sampling strategies 
(random oversampling, undersampling, balanced sampling), 
or using a weighted cross-entropy loss function.

NOTE: The last implementation of the weighted cross-entropy loss 
was not tested due to time limitations. Previous versions yielded faulty results.
-> The models were trained using the `ft_aya_lora_simple.py` script, 
which does not include the weighted loss function but only the data sampling strategies. 
"""

import os
import argparse
import random

# For larger CUDA memory allocation (NOTE: has to be set before importing PyTorch)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

from utils.data_utils import STRAT_ABBREV, sample_data
from utils.model_utils import preprocess, compute_metrics, TRAIN_MAX_SEQ_LENGTH

# ---------- Constants & Defaults ----------
TRAIN_MAX_SEQ_LENGTH = TRAIN_MAX_SEQ_LENGTH  # 256 by default ()
DEFAULT_QUANTIZE_4BIT = True
DEFAULT_FLASH_ATTENTION = True 
DEFAULT_GRAD_ACC_STEPS = 4  # increased to reduce memory usage
USE_GRAD_CHECKPOINTING = True  # enable gradient checkpointing to save memory


# ---------- Argument Parser ----------
def parse_args():
    p = argparse.ArgumentParser("Fine-tune Aya for bias detection using LoRA")
    # data & optimization
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling", choices=list(STRAT_ABBREV.keys()), default="normal")
    p.add_argument("--loss_type", type=str, 
                    choices=["standard", "weighted"], 
                    default="standard", 
                    help="Loss function type: standard CE, weighted CE")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=4, help="per-device train batch size")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay")
    p.add_argument("--patience", type=int, default=5, help="early stop patience")
    p.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    p.add_argument("--output_dir", type=str, default="./results_aya_lora")

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=8)             # rank: nb of params in adaptation layers 
                                                                # (more -> remembers better -> can learn more complex things)
    p.add_argument("--lora_alpha", type=int, default=16)        # scaling factor of adaptation layers weights 
                                                                # (higher -> LoRA layers have more influence on base model)
                                                                # usually set to double the rank
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--lora_bias",
        choices=["none", "all", "lora_only"],
        default="none",
        help="LoRA bias setting",
    )
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help="Comma-sep list or 'all-linear'",
    )

    # hardware / performance
    p.add_argument(
        "--quantize_4bit",
        action="store_true",
        default=DEFAULT_QUANTIZE_4BIT,
        help="Use bitsandbytes 4-bit quantization",
    )
    p.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=DEFAULT_FLASH_ATTENTION,
        help="Use FlashAttention2 (requires flash-attn installed)",
    )
    p.add_argument(
        "--use_grad_checkpointing", 
        action="store_true", 
        default=USE_GRAD_CHECKPOINTING
    )
    p.add_argument(
        "--grad_acc_steps",
        type=int,
        default=DEFAULT_GRAD_ACC_STEPS,
        help="Gradient accumulation steps",
    )

    return p.parse_args()

# ---------- Token-Level Weighted Loss Implementation ----------
class WeightedCrossEntropyLoss(nn.Module):
    """
    Vocabulary-level weighted cross-entropy loss.
    Creates a weight tensor for the entire vocabulary where only target tokens
    ("ja" and "nee") get class weights, all other tokens get weight 1.0.
    """
    def __init__(self, class_weights, ignore_index=-100, tokenizer=None):
        super().__init__()
        self.class_weights = class_weights  # [weight_nee, weight_ja]
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        
        # Get token IDs for "ja" and "nee"
        self.ja_token_ids = self.tokenizer("ja", add_special_tokens=False).input_ids
        self.nee_token_ids = self.tokenizer("nee", add_special_tokens=False).input_ids

        # Create vocabulary-level weight tensor
        vocab_size = self.tokenizer.vocab_size
        self.weight_tensor = torch.ones(vocab_size, dtype=torch.float)
        
        # Set weights for target tokens
        for ja_token_id in self.ja_token_ids:
            if 0 <= ja_token_id < vocab_size:
                self.weight_tensor[ja_token_id] = class_weights[1]  # weight for "ja"
                
        for nee_token_id in self.nee_token_ids:
            if 0 <= nee_token_id < vocab_size:
                self.weight_tensor[nee_token_id] = class_weights[0]  # weight for "nee"
        
        print(f"WeightedCrossEntropyLoss initialized:")
        print(f"  ja_token_ids: {self.ja_token_ids} (weight: {class_weights[1]})")
        print(f"  nee_token_ids: {self.nee_token_ids} (weight: {class_weights[0]})")
        print(f"  vocab_size: {vocab_size}")
        print(f"  Non-unit weights in vocabulary: {(self.weight_tensor != 1.0).sum().item()}")

    def forward(self, logits, labels):
        """
        Apply vocabulary-level weighted cross-entropy loss to model outputs.
        Uses PyTorch's built-in weight parameter in CrossEntropyLoss.
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Move weight tensor to same device as logits
        weight_tensor = self.weight_tensor.to(logits.device)
        
        # Use PyTorch's built-in weighted cross-entropy
        loss_fct = nn.CrossEntropyLoss(
            weight=weight_tensor,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        # Flatten and compute loss
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        
        loss = loss_fct(flat_logits, flat_labels)
        
        return loss


# ---------- Custom Callbacks ----------
class ClassificationMetricsCallback(TrainerCallback):
    """
    Custom callback that replaces token-level metrics with classification metrics during training.
    """
    def __init__(self, tokenizer, compute_metrics_fn, eval_dataset=None):
        """
        Args:
            tokenizer: The tokenizer to decode model outputs
            compute_metrics_fn: The function to compute metrics
            eval_dataset: Optional evaluation dataset to use for periodic evaluation
        """
        self.tokenizer = tokenizer
        self.compute_metrics_fn = compute_metrics_fn
        self.eval_dataset = eval_dataset
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after each logging event."""
        if logs is None:
            return
            
        # Remove token accuracy metrics that aren't relevant for classification task
        for key in list(logs.keys()):
            if "accuracy" in key and "eval" not in key:
                logs.pop(key, None)
                
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Print classification metrics after each evaluation."""
        if metrics is None:
            return
            
        if "eval_accuracy" in metrics:
            metrics["classification_performance"] = f"F1={metrics.get('eval_f1_macro', 0):.4f}, Acc={metrics.get('eval_accuracy', 0):.4f}"


class WeightedSFTTrainer(SFTTrainer):
    """
    Trainer that uses vocabulary-level weighted cross-entropy loss.
    """
    def __init__(self, class_weights, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
        # Use provided tokenizer or get it from the trainer
        if tokenizer is None:
            tokenizer = self.tokenizer
            
        self.weighted_loss = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
            tokenizer=tokenizer
        )
        
        print(f"WeightedSFTTrainer initialized with vocabulary-level weighting")
        print(f"  Class weights: {class_weights} [nee (0), ja (1)]")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override to use vocabulary-level weighted cross-entropy loss
        """
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss = self.weighted_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# -------------- Main --------------
if __name__ == "__main__":
    args = parse_args()

    # ------- Seeds -------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # ------- CUDA memory management -------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"Available CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Current CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # ---------- Model & Tokenizer ----------
    model_name = "CohereLabs/aya-expanse-8b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")
    
    # Test tokenization of our target tokens
    ja_tokens = tokenizer("ja", add_special_tokens=False)
    nee_tokens = tokenizer("nee", add_special_tokens=False)
    print(f"'ja' tokenizes to: {ja_tokens}")
    print(f"'nee' tokenizes to: {nee_tokens}")
    
    # Ensure 'ja' and 'nee' tokenize to single tokens
    ja_token_ids = tokenizer("ja", add_special_tokens=False).input_ids
    nee_token_ids = tokenizer("nee", add_special_tokens=False).input_ids
    assert len(ja_token_ids) == 1, f"'ja' tokenized to multiple tokens: {ja_token_ids}"
    assert len(nee_token_ids) == 1, f"'nee' tokenized to multiple tokens: {nee_token_ids}"

    # ---------- Bits & Bytes Quantization Config ----------
    quantization_config = None
    if args.quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # NOTE: Vera said this conflicts with 4bit quantizing
        )

    # ---------- FlashAttention Fallback ----------
    attn_impl = None
    if args.use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("Successfully loaded Flash Attention 2.")
        except ImportError:
            print("WARNING: flash_attn not installed; disabling FlashAttention2.")
            attn_impl = None

    # ---------- Load Model ----------
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation=attn_impl,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,  # disable KV cache for training to save memory
        )
    except Exception as e:
        print(f"Error loading with 4-bit quantization: {e}")
        print("Trying to load with default precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
    
    if args.quantize_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---------- LoRA Setup ----------
    target_modules = (
        args.lora_target_modules.split(",")
        if args.lora_target_modules != "all-linear"
        else "all-linear"
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,  # or ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- Prepare Data ----------
    dataset = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
    train_df = pd.DataFrame(dataset["train"])
    sampled_train_df = sample_data(train_df, strategy=args.sampling)
    print(f"Sampling: {args.sampling} -> {len(sampled_train_df)} samples")
    train_ds = Dataset.from_pandas(sampled_train_df)
    eval_ds = dataset["validation"]
    
    tokenized_train = train_ds.map(lambda example: preprocess(example, tokenizer), remove_columns=train_ds.column_names, batched=False)
    
    # Clear memory after preprocessing training data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory after train preprocessing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    tokenized_eval = eval_ds.map(lambda example: preprocess(example, tokenizer), remove_columns=eval_ds.column_names, batched=False)
    # tokenized_test = test_ds.map(lambda example: preprocess(example, tokenizer), remove_columns=test_ds.column_names, batched=False)
    
    # Clear memory after all preprocessing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory after all preprocessing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ---------- Training Arguments ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        # per_device_eval_batch_size removed to save memory during evaluation
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=args.use_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},  
        optim="paged_adamw_8bit",  # OR adamw_torch_fused
        # Checkpoint saving based on steps
        save_strategy="epoch",  
        # save_steps=50,
        save_total_limit=1,  # keep only the 1 best checkpoint
        # Evaluation settings - DISABLED to save memory during training (to avoid CUDA OOM during evaluation)
        # eval_strategy="epoch", 
        # eval_steps=100,  
        eval_accumulation_steps=10,  # for memory optimization 
        # Best model handling - DISABLED to save memory
        # load_best_model_at_end=True,  # requires extra memory to store best model
        # metric_for_best_model="f1_macro", 
        # greater_is_better=True,  
        # Logging settings
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",  # disable wandb/tensorboard logging to save memory
        # Optimizer settings
        learning_rate=args.lr,
        weight_decay=args.wd,
        fp16=False,  # use float16 for training
        bf16=False,  # use bfloat16 for training
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        dataloader_pin_memory=False,  # reduces CPU memory usage
    )

    # ---------- Trainer & Train ----------
    # Create callbacks
    callbacks = [
        ClassificationMetricsCallback(tokenizer=tokenizer, compute_metrics_fn=compute_metrics),
        # MetricsLoggerCallback()
    ]
    
    # Add early stopping callback if enabled
    if args.early_stopping:
        print(f"Early stopping enabled with patience={args.patience}")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
            )  # Will monitor "eval_f1_macro" as specified in training_args
        )
    
    # Compute class weights if using weighted loss
    if args.loss_type == "weighted":
        print("Using weighted cross-entropy loss to handle class imbalance")
        
        # Get train dataset labels
        train_labels = [sample["label"] for sample in dataset["train"]]
        
        # Compute balanced class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_labels
        )
        
        print(f"Computed class weights: {class_weights} [0=not biased, 1=biased]")
        
        # Create weighted trainer
        trainer = WeightedSFTTrainer(
            class_weights=class_weights,
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            # eval_dataset=tokenized_eval,  # DISABLED to save memory by not loading eval dataset
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    else:
        # Use standard SFT Trainer
        print("Using standard cross-entropy loss")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            # eval_dataset=tokenized_eval,  # DISABLED to save memory by not loading eval dataset
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    print("Starting training")
    trainer.train()
    print("Training complete")
    
    # ---------- Evaluate ---------
    # print("\n===== VALIDATION SET EVALUATION =====")
    # val_metrics = trainer.evaluate()
    # best_f1 = val_metrics["eval_f1_macro"]
    # print(f"Best validation F1: {best_f1:.4f}")
    # print(f"Validation metrics: {val_metrics}")
    
    # ---------- Save ---------
    sampling_abbrev = STRAT_ABBREV[args.sampling]
    
    # Add loss type to model name
    loss_abbrev = ""
    if args.loss_type == "weighted":
        loss_abbrev = "_WL"  # WL for Weighted Loss
    
    model_dir = f"{args.output_dir}/aya-expanse-8b_{sampling_abbrev}{loss_abbrev}_seed{args.seed}"  #_f1_{best_f1:.4f}"
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    print(f"Model saved to {model_dir}")

    # # ---------- Test Metrics ----------
    # print("\n===== TEST SET EVALUATION =====")
    # test_pred = trainer.predict(tokenized_test)
    # test_metrics = compute_metrics(test_pred)
    
    # # Print formatted metrics comparison
    # print_loss_comparison(args.loss_type, test_metrics)
    
    # # Save test metrics to the model directory
    # metrics_file = os.path.join(model_dir, "test_metrics.txt")
    # with open(metrics_file, "w") as f:
    #     f.write(f"Loss Type: {args.loss_type}\n")
    #     if args.loss_type == "focal":
    #         f.write(f"Focal Gamma: {args.focal_gamma}\n")
    #     f.write(f"Sampling Strategy: {args.sampling}\n\n")
    #     for key, value in test_metrics.items():
    #         f.write(f"{key}: {value}\n")
    
    # print(f"Test metrics saved to {metrics_file}")
    
    # # ---------- Loss Comparison ----------
    # if args.loss_type in ["weighted", "focal"]:
    #     print_loss_comparison(args.loss_type, test_metrics)
