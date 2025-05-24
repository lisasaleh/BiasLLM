import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
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
    p.add_argument("--focal_loss", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=8, help="per-device train batch size")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay")
    p.add_argument("--patience", type=int, default=2, help="early stop patience")
    p.add_argument("--output_dir", type=str, default="./results_aya_lora")

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
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
        raise ValueError("Unsupported strategy.")

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
    
    # get indices of samples with valid predictions and labels
    valid_indices = [i for i, (p, l) in enumerate(zip(pred_labels, true_labels)) 
                    if p != -1 and l != -1]

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
    
    # ------- Load Tokenizer -------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------- Bits & Bytes Quantization Config -------
    quantization_config = None
    if args.quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # NOTE: Vera said this conflicts with 4bit quantizing
        )

    # ------- FlashAttention Fallback -------
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

    # ------- Load Model -------
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

    # ------- LoRA Setup -------
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

    # ------- Training Arguments -------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=args.use_grad_checkpointing,
        optim="paged_adamw_32bit",  # OR adamw_torch_fused
        save_steps=50,
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        learning_rate=args.lr,
        weight_decay=args.wd,
        fp16=False,  # Use bfloat16 for training
        bf16=True,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        dataloader_pin_memory=False,  # Reduces CPU memory usage
    )

    # ---------- Trainer & Train ----------
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        label_names=["input_ids"],
        # callbacks=callbacks,
    )

    print("Starting training")
    trainer.train()
    print("Training complete")
    
    # ---------- Evaluate ---------
    metrics = trainer.evaluate()
    best_f1 = metrics["eval_f1_macro"]
    print(f"Best validation F1: {best_f1:.4f}")
    
    # ---------- Save ---------
    abbrev = STRAT_ABBREV[args.sampling]
    fe = "_FE" if args.focal_loss else ""
    save_name = model_name.replace("/", "-")
    model_dir = f"{args.output_dir}/{save_name}{fe}_{abbrev}_f1_{best_f1:.4f}"
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    print(f"Model saved to {model_dir}")
    
    # ---------- Test Metrics ----------
    test_pred = trainer.predict(tokenized_test)
    test_metrics = compute_metrics(test_pred)
    print("Test metrics:", test_metrics)