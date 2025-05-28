"""
Evaluate Aya Expanse model (zero-shot or fine-tuned) for bias detection tasks 
on (validation and) test split(s) of the Dutch-Government-Data-for-Bias-detection (DGDB) dataset
using either the prompt used for fine-tuning via LoRA or a chain-of-thought (CoT) variant. 
"""

import os
import argparse
import random
import gc
import json


import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from utils.model_utils import train_prompt_template, cot_prompt_template, classify_answer_eval, compute_metrics


# Clear any existing CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

TRAIN_MAX_SEQ_LENGTH = 512

# ---------- Argument Parser ----------
def parse_args():
    p = argparse.ArgumentParser("Evaluate Aya LoRA checkpoint for bias detection - Fixed Version")
    p.add_argument("--checkpoint_path", default="zero-shot", type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bs", type=int, default=1, help="batch size - keep at 1 for memory")
    p.add_argument("--output_dir", type=str, default="./results_aya_lora")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    p.add_argument("--quantize_4bit", action="store_true", default=True)
    p.add_argument("--prompt", type=str, default="same", 
                   help="Prompt to evaluate on. If 'same', uses prompt used for training. If 'CoT', uses chain-of-thought prompt. Otherwise, uses custom prompt template provided as argument.")
    p.add_argument("--test_only", action="store_true", default=False,
                   help="If set, only evaluates on the test set, skipping validation.")
    return p.parse_args()


def evaluate_model_generation(model, tokenizer, dataset, prompt_template, max_samples=None):
    """Evaluate model using generation instead of trainer.evaluate()."""    
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    all_decoded_preds = []
    invalid_count = 0
    
    # Convert dataset to list for iteration
    if hasattr(dataset, 'to_pandas'):
        data_list = dataset.to_pandas().to_dict('records')
    else:
        data_list = list(dataset)
    data_list_size = len(data_list)
    
    if max_samples:
        # data_list = data_list[:max_samples]
        data_list = random.sample(data_list, min(max_samples, data_list_size))
    
    print(f"Evaluating on {len(data_list)}/{data_list_size} samples...")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(data_list, desc="Evaluating")):
            # Complete prompt
            prompt = prompt_template.format(text=example["text"])
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=TRAIN_MAX_SEQ_LENGTH)  # full prompt
            print(f"Input length: {inputs['input_ids'].shape[1]} tokens")  # Debugging: print input token length
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=6 if prompt_template == train_prompt_template else 200,  # Adjust based on prompt type
                    do_sample=False,  # Deterministic
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                            
            # Decode full output
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Full output length: {len(full_text.split())} tokens")  # Debugging: print output length in tokens
            # Extract only the generated part (after the prompt)
            prediction_text = full_text[len(prompt):].strip()
            print(f"Prediction length: {len(prediction_text.split())} words")  # Debugging: print decoded prediction

            # Classify prediction
            pred_label = classify_answer_eval(prediction_text)
            true_label = example["label"]
            
            all_predictions.append(pred_label)
            all_true_labels.append(true_label)
            all_decoded_preds.append(prediction_text)
            
            if pred_label == -1:
                invalid_count += 1
            
            # Clear memory periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    return all_predictions, all_true_labels, all_decoded_preds, invalid_count

# -------------- Main --------------
if __name__ == "__main__":
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "CohereLabs/aya-expanse-8b"
    
    # ------- Load Tokenizer -------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------- Load Model with Quantization -------
    print("Loading base model...")
    quantization_config = None
    if args.quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)


    # ---------- Load Model (from checkpoint or zero-shot) ----------
    if args.checkpoint_path == "zero-shot":
        print("Using zero-shot evaluation mode, no LoRA weights to load.")
    else:
        print("Loading LoRA weights from checkpoint...")
        try:
            model = PeftModel.from_pretrained(model, args.checkpoint_path)
            print("LoRA weights loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            exit(1)
            

    # Set to evaluation mode
    model.eval()

    # Clear memory after LoRA load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---------- Prepare Data ----------
    print("Loading datasets...")
    dataset = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
    
    # Only process validation and test sets
    if not args.test_only:
        eval_ds = dataset["validation"]
    test_ds = dataset["test"]

    # Limit samples if specified
    if args.max_samples:
        if not args.test_only:
            eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    # ---------- Prompt Template ----------
    if args.prompt == "same":
        prompt_template = train_prompt_template
    elif args.prompt.lower() == "cot":
        prompt_template = cot_prompt_template
    else:  
        # Custom prompt template provided by user
        prompt_template = args.prompt

    # ---------- Validation Evaluation ---------
    if args.test_only:
        print("Skipping validation evaluation as --test_only is set.")
        val_preds, val_true, val_decoded, val_invalid = [], [], [], 0
    else:
        print("\n===== VALIDATION SET EVALUATION =====")
        val_preds, val_true, val_decoded, val_invalid = evaluate_model_generation(
            model, tokenizer, eval_ds, prompt_template, args.max_samples
        )
    val_metrics = compute_metrics(val_preds, val_true, val_invalid)
    print("Validation metrics:", val_metrics)

    # Clear memory before test evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---------- Test Evaluation ----------
    print("\n===== TEST SET EVALUATION =====")
    test_preds, test_true, test_decoded, test_invalid = evaluate_model_generation(
        model, tokenizer, test_ds, prompt_template, args.max_samples
    )
    test_metrics = compute_metrics(test_preds, test_true, test_invalid)
    print("Test metrics:", test_metrics)

    # ---------- Save Results ----------
    results = {
        "checkpoint_path": args.checkpoint_path,
        "max_samples": args.max_samples, 
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    results_file = os.path.join(args.output_dir, f"eval_results_{args.checkpoint_path.split('/')[-1]}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("Evaluation complete!") 
