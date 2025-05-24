import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---------- Constants & Defaults ----------
STRAT_ABBREV = {
    "undersample": "us",
    "oversample":  "os",
    "balanced":    "bl",
    "normal":      "nm",
}

TRAIN_MAX_SEQ_LENGTH = 256  # reduced from 512 to save memory
DEFAULT_QUANTIZE_4BIT = True
DEFAULT_FLASH_ATTENTION = True 
DEFAULT_GRAD_ACC_STEPS = 4  # increased to reduce memory usage
USE_GRAD_CHECKPOINTING = True  # enable gradient checkpointing to save memory


# ---------- Argument Parser ----------
def parse_args():
    p = argparse.ArgumentParser("Fine-tune Aya for bias detection using LoRA")
    # data & optimization
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling", choices=STRAT_ABBREV, default="undersample")
    p.add_argument("--loss_type", type=str, choices=["standard", "weighted", "focal"], default="standard", 
                   help="Loss function type: standard CE, weighted CE, or focal loss")
    p.add_argument("--focal_gamma", type=float, default=2.0, 
                   help="Gamma parameter for focal loss (higher value = more focus on hard examples)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=8, help="per-device train batch size")
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

# ---------- Data Sampling ----------
def sample_data(df, strategy="undersample", oversample_factor=2, undersample_ratio=0.7, balanced_neg_ratio=0.5, random_state=42):
    biased = df[df.label == 1]
    unbiased = df[df.label == 0]

    if strategy == "undersample":
        # Undersample the unbiased (majority) class to match a specified ratio
        unbiased_sampled = unbiased.sample(frac=undersample_ratio, random_state=random_state)
        return pd.concat([biased, unbiased_sampled])

    elif strategy == "oversample":
        # Duplicate the biased (minority) class oversample_factor times
        return pd.concat([biased] * oversample_factor + [unbiased])

    elif strategy == "balanced":
        # Target 50% biased, 50% unbiased in the final dataset (or as specified by balanced_neg_ratio)
        target_total = len(df)  # preserve the original size
        unbiased_target = int(target_total * balanced_neg_ratio)  # target number of unbiased samples  
        biased_target = target_total - unbiased_target  # target number of biased samples 

        # Sample unbiased samples to match the target
        neg_sampled = unbiased.sample(n=unbiased_target, random_state=random_state)

        # Compute the number of times to duplicate all biased samples
        repeats = biased_target // len(biased)
        # Compute the remainder of biased samples to sample to reach target
        remainder = biased_target % len(biased)

        # Sample biased samples to match the target based on the number of repeats and remainder
        biased_repeated = pd.concat([biased] * repeats + [biased.sample(n=remainder, random_state=random_state)])
        return pd.concat([biased_repeated, neg_sampled])

    elif strategy == "normal":
        return df

    else:
        raise ValueError("Unsupported strategy.")

def preprocess(example):
    """
    Preprocess an example for training/evaluation.
    
    Converts the numeric label (0/1) to the textual format ("ja"/"nee") that the model will generate,
    then tokenizes the input and target, and creates the appropriate input_ids, attention_mask, and labels.
    """
    # complete the prompt with the example text
    prompt = prompt_template.format(text=example["text"])
    
    # convert binary label (0/1) to the text format "ja"/"nee" that the model will generate
    target = "ja" if example["label"] == 1 else "nee"
    
    # tokenize the prompt and target (separately)
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=TRAIN_MAX_SEQ_LENGTH, add_special_tokens=False)
    target_tokens = tokenizer(target, truncation=True, max_length=5, add_special_tokens=False)
    
    # combine tokenized prompt and target
    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]
    attn_mask = prompt_tokens["attention_mask"] + target_tokens["attention_mask"]
    
    # for causal LM training, we only want to predict the target tokens, not the prompt tokens
    # -100 is the ignore index for the loss function 
    labels = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"]
    
    # truncate sequences to maximum length
    input_ids = input_ids[:TRAIN_MAX_SEQ_LENGTH]
    attn_mask = attn_mask[:TRAIN_MAX_SEQ_LENGTH]
    labels = labels[:TRAIN_MAX_SEQ_LENGTH]
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attn_mask, 
        "labels": labels
    }


# ---------- Metrics ----------
def classify_answers(text_list):
    """
    Classify each example sentence as:
        1  if the last word is exactly "ja" (case-insensitive),
        0  if the last word is exactly "nee" (case-insensitive),
        -1  otherwise (invalid answer).

    Parameters:
        text_list: List of text strings (generated outputs).

    Returns:
        List of ints (1, 0, or -1) corresponding to each input string.
    """
    preds = []
    for text in text_list:
        words = text.strip().lower().split()
        
        # if the last word is "ja" or "nee", classify as 1 or 0 respectively, otherwise as invalid          
        if words[-1] == "ja":
            preds.append(1)  # biased
        elif words[-1] == "nee":
            preds.append(0)  # not biased
        else:
            preds.append(-1)  # invalid answer

    return preds

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # decode predictions and labels to text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # print a few examples for debugging
    if len(decoded_preds) > 0:
        print("\nExample predictions:")
        for i in range(min(3, len(decoded_preds))):
            print(f"Pred {i}: '{decoded_preds[i]}'")
            if i < len(decoded_labels):
                print(f"Label {i}: '{decoded_labels[i]}'")
    
    # convert text to numeric labels (1, 0, -1)
    pred_labels = classify_answers(decoded_preds)
    true_labels = classify_answers(decoded_labels)
    
    # count invalid predictions/labels
    invalid_preds = pred_labels.count(-1)
    invalid_labels = true_labels.count(-1)
    
    # log information about invalid predictions
    print(f"Invalid predictions: {invalid_preds}/{len(pred_labels)} ({invalid_preds/max(1,len(pred_labels))*100:.2f}%)")
    
    # get indices of samples with valid predictions and labels
    valid_indices = [i for i, (p, l) in enumerate(zip(pred_labels, true_labels)) 
                    if p != -1 and l != -1]
    
    # handle case with no valid predictions
    if not valid_indices:
        print("No valid predictions/labels found!")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "invalid_preds_percent": 100.0 if pred_labels else 0.0,
            "invalid_labels_percent": 100.0 if true_labels else 0.0
        }

    # extract valid predictions and labels
    valid_preds = [pred_labels[i] for i in valid_indices]
    valid_labels = [true_labels[i] for i in valid_indices]
    
    # Calculate binary classification metrics
    metrics = {
        "accuracy": accuracy_score(valid_labels, valid_preds),
        "f1_macro": f1_score(valid_labels, valid_preds, pos_label=1, average='macro'),
        "precision_macro": precision_score(valid_labels, valid_preds, pos_label=1, average='macro'),
        "recall_macro": recall_score(valid_labels, valid_preds, pos_label=1, average='macro'),
        "invalid_preds_percent": (invalid_preds / len(pred_labels) * 100) if pred_labels else 0.0,
        "invalid_labels_percent": (invalid_labels / len(true_labels) * 100) if true_labels else 0.0
    }
    
    # log counts of class distribution in valid predictions
    pos_preds = valid_preds.count(1)
    neg_preds = valid_preds.count(0)
    print(f"Valid predictions: #positive={pos_preds}, #negative={neg_preds}")

    return metrics


# ---------- Weighted Loss Implementation ----------
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=-100, tokenizer=None):
        super().__init__()
        self.class_weights = class_weights  # [weight_neg, weight_pos]
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
    
    def forward(self, logits, labels):
        """
        Apply weighted cross-entropy loss to model outputs.
        
        Args:
            logits: Model output logits of shape [batch_size, seq_len, vocab_size]
            labels: Target labels of shape [batch_size, seq_len]
            
        Returns:
            Weighted loss value
        """
        # Get vocab size from logits
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create a mapping from token IDs to class weights
        # Default weight is 1.0 for tokens that aren't "ja"/"nee"
        token_weights = torch.ones(vocab_size, device=logits.device)
        
        # Get token IDs for "ja" and "nee"
        # These represent the positive and negative classes
        ja_token_ids = self.tokenizer("ja", add_special_tokens=False).input_ids
        nee_token_ids = self.tokenizer("nee", add_special_tokens=False).input_ids
        
        # Set weights for class tokens
        for token_id in ja_token_ids:
            token_weights[token_id] = self.class_weights[1]  # positive class weight
        
        for token_id in nee_token_ids:
            token_weights[token_id] = self.class_weights[0]  # negative class weight
        
        # Reshape for CrossEntropyLoss
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        
        # Create loss function with token weights
        loss_fct = nn.CrossEntropyLoss(
            weight=token_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        # Compute loss
        loss = loss_fct(flat_logits, flat_labels)
        return loss


# ---------- Focal Loss Implementation ----------
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling both class imbalance and hard examples.
    
    Focal Loss adds a focusing parameter (gamma) that reduces the impact of well-classified examples
    and increases the impact of misclassified examples during training.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Where:
      - alpha_t is the class weight
      - gamma is the focusing parameter
      - p_t is the model's estimated probability for the target class
    """
    def __init__(self, class_weights, gamma=2.0, ignore_index=-100, tokenizer=None):
        super().__init__()
        self.class_weights = class_weights  # [weight_neg, weight_pos]
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
    
    def forward(self, logits, labels):
        """
        Apply focal loss to model outputs.
        
        Args:
            logits: Model output logits of shape [batch_size, seq_len, vocab_size]
            labels: Target labels of shape [batch_size, seq_len]
            
        Returns:
            Focal loss value
        """
        # Get vocab size from logits
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create a mapping from token IDs to class weights
        # Default weight is 1.0 for tokens that aren't "ja"/"nee"
        token_weights = torch.ones(vocab_size, device=logits.device)
        
        # Get token IDs for "ja" and "nee"
        # These represent the positive and negative classes
        ja_token_ids = self.tokenizer("ja", add_special_tokens=False).input_ids
        nee_token_ids = self.tokenizer("nee", add_special_tokens=False).input_ids
        
        # Set weights for class tokens
        for token_id in ja_token_ids:
            token_weights[token_id] = self.class_weights[1]  # positive class weight
        
        for token_id in nee_token_ids:
            token_weights[token_id] = self.class_weights[0]  # negative class weight
        
        # Reshape for loss calculation
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        
        # Filter out ignored indices
        mask = flat_labels != self.ignore_index
        flat_logits = flat_logits[mask]
        flat_labels = flat_labels[mask]
        
        if flat_labels.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute softmax probabilities
        log_softmax = nn.LogSoftmax(dim=1)(flat_logits)
        log_pt = log_softmax.gather(1, flat_labels.unsqueeze(1))
        pt = log_pt.exp()  # pt is the softmax probability for the target class
        
        # Get class weights for the actual targets
        label_weights = torch.zeros_like(flat_labels, dtype=torch.float, device=flat_labels.device)
        for i, label in enumerate(flat_labels):
            label_weights[i] = token_weights[label]
            
        # Compute focal modulating factor: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute weighted focal loss
        loss = -label_weights * focal_weight.view(-1) * log_pt.view(-1)
        return loss.mean()


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
            
        # Remove token accuracy metrics that aren't relevant for our classification task
        for key in list(logs.keys()):
            if "accuracy" in key and "eval" not in key:
                logs.pop(key, None)
                
        # Add a note about classification metrics
        if "loss" in logs and "eval_accuracy" not in logs:
            logs["classification_metrics_note"] = "Classification metrics available during evaluation steps"
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Print classification metrics after each evaluation."""
        if metrics is None:
            return
            
        if "eval_accuracy" in metrics:
            metrics["classification_performance"] = f"F1={metrics.get('eval_f1_macro', 0):.4f}, Acc={metrics.get('eval_accuracy', 0):.4f}"

class WeightedSFTTrainer(SFTTrainer):
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
        
        # Add the classification metrics callback
        if not any(isinstance(callback, ClassificationMetricsCallback) for callback in self.callback_handler.callbacks):
            self.add_callback(ClassificationMetricsCallback(
                tokenizer=tokenizer,
                compute_metrics_fn=self.compute_metrics if hasattr(self, "compute_metrics") else None,
                eval_dataset=self.eval_dataset
            ))
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override to use weighted loss
        """
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss = self.weighted_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

class FocalSFTTrainer(SFTTrainer):
    def __init__(self, class_weights, gamma=2.0, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        
        # Use provided tokenizer or get it from the trainer
        if tokenizer is None:
            tokenizer = self.tokenizer
            
        self.focal_loss = FocalLoss(
            class_weights=class_weights,
            gamma=gamma,
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
            tokenizer=tokenizer
        )
        
        # Add the classification metrics callback
        if not any(isinstance(callback, ClassificationMetricsCallback) for callback in self.callback_handler.callbacks):
            self.add_callback(ClassificationMetricsCallback(
                tokenizer=tokenizer,
                compute_metrics_fn=self.compute_metrics if hasattr(self, "compute_metrics") else None,
                eval_dataset=self.eval_dataset
            ))
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override to use focal loss
        """
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ---------- Custom Trainer Callback ----------
class MetricsLoggerCallback(TrainerCallback):
    """
    Custom callback to log classification metrics (accuracy, F1, precision, recall) 
    instead of token-level accuracy during training.
    """
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
            
        print("\n=== Classification Metrics ===")
        print(f"F1 Score: {metrics.get('eval_f1_macro', 0.0):.4f}")
        print(f"Accuracy: {metrics.get('eval_accuracy', 0.0):.4f}")
        print(f"Precision: {metrics.get('eval_precision_macro', 0.0):.4f}")
        print(f"Recall: {metrics.get('eval_recall_macro', 0.0):.4f}")
        print(f"Invalid predictions: {metrics.get('eval_invalid_preds_percent', 0.0):.2f}%")
        print("=============================\n")


def print_loss_comparison(loss_type, test_metrics):
    """Print a formatted comparison of this loss type's performance."""
    print(f"\n{'=' * 50}")
    print(f"LOSS TYPE: {loss_type.upper()}")
    if loss_type == "focal":
        print(f"Gamma: {args.focal_gamma}")
    print(f"{'=' * 50}")
    print(f"F1 Score (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"Precision:        {test_metrics['precision_macro']:.4f}")
    print(f"Recall:           {test_metrics['recall_macro']:.4f}")
    print(f"Invalid preds:    {test_metrics['invalid_preds_percent']:.2f}%")
    print(f"{'=' * 50}\n")


# -------------- Main --------------
if __name__ == "__main__":
    args = parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # model_dir = "/scratch-shared/scur1424/aya_models"
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_name = "CohereLabs/aya-expanse-8b"
    
    # ---------- Load Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            print("Can install with: pip install flash-attn --no-build-isolation")
            attn_impl = None

    # ---------- Load Model ----------
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # model_dir,
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
            # model_dir,
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
        target_modules=target_modules,  # ["q_proj", "v_proj", "k_proj", "o_proj"]
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
    test_ds = dataset["test"]

    # ---------- Prompt & Preprocess ----------
    prompt_template = (
        "Je bent een expert op het gebied van bias, stereotypering en discriminatie. "
        "Je taak is om zinnen te classificeren volgens de volgende definitie: een zin bevat bias "
        "als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat, "
        "die impliciet of expliciet kunnen zijn, en die gaan over een afgebakende groep personen, "
        "OF als de zin een woord bevat dat vermeden moet worden vanwege bias. "
        "Geef de volgende zin een label: 'ja' voor biased of 'nee' voor niet-biased. "
        "Antwoord uitsluitend met 'ja' (biased) of 'nee' (niet-biased). Geen verdere uitleg. "
        'De zin is: "{text}"'
    )

    tokenized_train = train_ds.map(preprocess, remove_columns=train_ds.column_names, batched=False)
    tokenized_eval = eval_ds.map(preprocess, remove_columns=eval_ds.column_names, batched=False)
    tokenized_test = test_ds.map(preprocess, remove_columns=test_ds.column_names, batched=False)

    # ---------- Training Arguments ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,  # NOTE: larger eval batch size
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=args.use_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="paged_adamw_32bit",  # OR adamw_torch_fused
        # Checkpoint saving based on steps
        save_strategy="steps",  
        save_steps=50,
        save_total_limit=1,  # keep only the 1 best checkpoint
        # Evaluation settings
        eval_strategy="steps",  # evaluate during training
        eval_steps=50, 
        # Best model handling
        load_best_model_at_end=True,  
        metric_for_best_model="f1_macro", 
        greater_is_better=True, 
        # Logging settings
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        # Optimizer settings
        learning_rate=args.lr,
        weight_decay=args.wd,
        fp16=False, 
        bf16=False,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        dataloader_pin_memory=False,  # reduces CPU memory usage
    )

    # ---------- Trainer & Train ----------
    # Create callbacks
    callbacks = [
        ClassificationMetricsCallback(tokenizer=tokenizer, compute_metrics_fn=compute_metrics),
        MetricsLoggerCallback()
    ]
    
    # Add early stopping callback if enabled
    if args.early_stopping:
        print(f"Early stopping enabled with patience={args.patience}")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
            )  # Will monitor "eval_f1_macro" as specified in training_args
        )
    
    # Compute class weights if using weighted/focal loss
    if args.loss_type in ["weighted", "focal"]:
        # Get train dataset labels
        train_labels = [sample["label"] for sample in dataset["train"]]
        
        # Compute balanced class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_labels
        )
        
        if args.loss_type == "weighted":
            print("Using weighted cross-entropy loss to handle class imbalance")
            print(f"Computed class weights: {class_weights} [0=not biased, 1=biased]")
            
            # Create weighted trainer
            trainer = WeightedSFTTrainer(
                class_weights=class_weights,
                tokenizer=tokenizer,
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
        else:  # focal loss
            print(f"Using focal loss (gamma={args.focal_gamma}) to handle class imbalance and hard examples")
            print(f"Computed class weights: {class_weights} [0=not biased, 1=biased]")
            
            # Create focal loss trainer
            trainer = FocalSFTTrainer(
                class_weights=class_weights,
                gamma=args.focal_gamma,
                tokenizer=tokenizer,
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
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
            eval_dataset=tokenized_eval,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    print("Starting training")
    trainer.train()
    print("Training complete")
    
    # ---------- Evaluate ---------
    print("\n===== VALIDATION SET EVALUATION =====")
    val_metrics = trainer.evaluate()
    best_f1 = val_metrics["eval_f1_macro"]
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Validation metrics: {val_metrics}")
    
    # ---------- Save ---------
    abbrev = STRAT_ABBREV[args.sampling]
    
    # Add loss type to model name
    loss_abbrev = ""
    if args.loss_type == "weighted":
        loss_abbrev = "_WL"  # WL for Weighted Loss
    elif args.loss_type == "focal":
        loss_abbrev = f"_FL{args.focal_gamma}"  # FL for Focal Loss with gamma value
    
    save_name = model_name.replace("/", "-")
    model_dir = f"{args.output_dir}/{save_name}{loss_abbrev}_{abbrev}_f1_{best_f1:.4f}"
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    print(f"Model saved to {model_dir}")

    # ---------- Test Metrics ----------
    print("\n===== TEST SET EVALUATION =====")
    test_pred = trainer.predict(tokenized_test)
    test_metrics = compute_metrics(test_pred)
    
    # Print formatted metrics comparison
    print_loss_comparison(args.loss_type, test_metrics)
    
    # Save test metrics to the model directory
    metrics_file = os.path.join(model_dir, "test_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Loss Type: {args.loss_type}\n")
        if args.loss_type == "focal":
            f.write(f"Focal Gamma: {args.focal_gamma}\n")
        f.write(f"Sampling Strategy: {args.sampling}\n\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Test metrics saved to {metrics_file}")
    
    # ---------- Loss Comparison ----------
    if args.loss_type in ["weighted", "focal"]:
        print_loss_comparison(args.loss_type, test_metrics)
