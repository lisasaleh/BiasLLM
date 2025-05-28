"""
Evaluate a finetuned Aya model on the DGDB dataset.

NOTE: Deprecated. Use the `eval.py` script instead. Keeping this for reference.
"""

import os
import argparse
import random
import gc
import json

# Allocate more memory for CUDA 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# Clear any existing CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------- Constants & Defaults ----------
STRAT_ABBREV = {
    "undersample": "us",
    "oversample":  "os", 
    "balanced":    "bl",
    "normal":      "nm",
}

TRAIN_MAX_SEQ_LENGTH = 512

# ---------- Argument Parser ----------
def parse_args():
    p = argparse.ArgumentParser("Evaluate Aya LoRA checkpoint for bias detection - Fixed Version")
    p.add_argument("--checkpoint_path", type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bs", type=int, default=1, help="batch size - keep at 1 for memory")
    p.add_argument("--output_dir", type=str, default="./results_aya_lora")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    p.add_argument("--quantize_4bit", action="store_true", default=True)
    return p.parse_args()


def classify_answer(text):
    """Classify the answer text as 'ja' (1), 'nee' (0), or invalid (-1).
    
    Only accepts exact matches of 'ja' or 'nee' (case-insensitive) with optional trailing punctuation.
    """
    
    print(f"Classifying text: '{text}'")
    
    # Check if text is empty or not a string -> invalid
    if not text or not isinstance(text, str):
        print("Text is empty or not a string -> invalid")
        return -1
    
    # Clean the text - remove leading/trailing whitespace
    text = text.strip().lower()
    
    # # Remove one trailing punctuation mark (if any)
    # if text and text[-1] in '.,!?;:"\')]}»':
    #     text = text[:-1]
    
    # Remove trailing punctuation (if any)
    while text and text[-1] in '.,!?;:"\')]}»':
        text = text[:-1]

    # # Remove leading punctuation (if any)
    # while text and text[0] in '"\'{([':
    #     text = text[1:]
    
    # After cleaning, the text should be exactly "ja" or "nee"
    if text == "ja":
        # print("Exact match: 'ja' -> 1")
        return 1
    elif text == "nee":
        # print("Exact match: 'nee' -> 0") 
        return 0
    else:
        print(f"No exact match found for '{text}' -> invalid")
        return -1

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
    
    if max_samples:
        data_list = data_list[:max_samples]
    
    print(f"Evaluating on {len(data_list)} samples...")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(data_list, desc="Evaluating")):
            try:
                # Complete prompt
                prompt = prompt_template.format(text=example["text"])
                
                # Tokenize prompt
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=TRAIN_MAX_SEQ_LENGTH)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate prediction
                with torch.cuda.amp.autocast():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3,  # Only generate a few tokens ("ja" or "nee")
                        do_sample=False,   # Deterministic
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode full output
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (after the prompt)
                if full_text.startswith(prompt):
                    prediction_text = full_text[len(prompt):].strip()
                else:
                    # Try to find where the prompt ends and the answer begins
                    # Look for the text content from the example
                    example_text = example["text"]
                    if example_text in full_text:
                        # Find the position after the example text
                        end_pos = full_text.find(example_text) + len(example_text)
                        # Look for quote marks or other indicators
                        remaining = full_text[end_pos:].strip()
                        if remaining.startswith('"'):
                            remaining = remaining[1:].strip()
                        prediction_text = remaining
                    else:
                        # Last resort: take the full text
                        print("Warning: Example text not found in output, using full text.")
                        prediction_text = full_text.strip()
                
                # Clean up the prediction text (remove any remaining quote marks)
                prediction_text = prediction_text.strip('"\'')
                
                # If prediction is too long, try to find just the ja/nee part
                if len(prediction_text.split()) > 10:
                    # Look for ja or nee at the end
                    words = prediction_text.lower().split()
                    for i in range(len(words)-1, -1, -1):
                        word = words[i].strip('.,!?"\'')
                        if word in ['ja', 'nee']:
                            prediction_text = word
                            break
                    else:
                        # Keep the last few words
                        prediction_text = ' '.join(words[-3:]) if len(words) >= 3 else prediction_text
                
                # Classify prediction
                pred_label = classify_answer(prediction_text)
                true_label = example["label"]
                
                all_predictions.append(pred_label)
                all_true_labels.append(true_label)
                all_decoded_preds.append(prediction_text)
                
                if pred_label == -1:
                    invalid_count += 1
                
                # Print some examples
                if i < 5:
                    print(f"Example {i}: '{prediction_text}' -> {pred_label} (true: {true_label})")
                
                # Clear memory periodically
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                all_predictions.append(-1)
                all_true_labels.append(example["label"])
                all_decoded_preds.append("")
                invalid_count += 1
    
    return all_predictions, all_true_labels, all_decoded_preds, invalid_count

def compute_metrics_simple(predictions, true_labels, invalid_count):
    """Compute metrics from predictions and true labels."""
    total_samples = len(predictions)
    
    # Get valid indices
    valid_indices = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p != -1]
    
    if not valid_indices:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_class_0_not_biased": 0.0,
            "f1_class_1_biased": 0.0,
            "precision_class_0_not_biased": 0.0,
            "precision_class_1_biased": 0.0,
            "recall_class_0_not_biased": 0.0,
            "recall_class_1_biased": 0.0,
            "invalid_percent": 100.0,
            "total_samples": total_samples,
            "valid_samples": 0
        }
    
    valid_preds = [predictions[i] for i in valid_indices]
    valid_labels = [true_labels[i] for i in valid_indices]
    
    # Check which classes are present
    unique_labels = set(valid_labels)
    unique_preds = set(valid_preds)
    
    # Compute per-class F1 scores with zero_division handling
    f1_per_class = f1_score(valid_labels, valid_preds, average=None, zero_division=0)
    precision_per_class = precision_score(valid_labels, valid_preds, average=None, zero_division=0)
    recall_per_class = recall_score(valid_labels, valid_preds, average=None, zero_division=0)
    
    # Ensure we have values for both classes (pad with 0 if class not present)
    if len(f1_per_class) == 1:
        # Only one class present, determine which one
        if 0 in unique_labels or 0 in unique_preds:
            f1_class_0, f1_class_1 = f1_per_class[0], 0.0
            precision_class_0, precision_class_1 = precision_per_class[0], 0.0
            recall_class_0, recall_class_1 = recall_per_class[0], 0.0
        else:
            f1_class_0, f1_class_1 = 0.0, f1_per_class[0]
            precision_class_0, precision_class_1 = 0.0, precision_per_class[0]
            recall_class_0, recall_class_1 = 0.0, recall_per_class[0]
    else:
        f1_class_0, f1_class_1 = f1_per_class[0], f1_per_class[1]
        precision_class_0, precision_class_1 = precision_per_class[0], precision_per_class[1]
        recall_class_0, recall_class_1 = recall_per_class[0], recall_per_class[1]
    
    metrics = {
        "accuracy": accuracy_score(valid_labels, valid_preds),
        "f1_macro": f1_score(valid_labels, valid_preds, average='macro', zero_division=0),
        "precision_macro": precision_score(valid_labels, valid_preds, average='macro', zero_division=0),
        "recall_macro": recall_score(valid_labels, valid_preds, average='macro', zero_division=0),
        "f1_class_0_not_biased": f1_class_0,
        "f1_class_1_biased": f1_class_1,
        "precision_class_0_not_biased": precision_class_0,
        "precision_class_1_biased": precision_class_1,
        "recall_class_0_not_biased": recall_class_0,
        "recall_class_1_biased": recall_class_1,
        "invalid_percent": (invalid_count / total_samples * 100),
        "total_samples": total_samples,
        "valid_samples": len(valid_indices)
    }
    
    pos_preds = valid_preds.count(1)
    neg_preds = valid_preds.count(0)
    pos_labels = valid_labels.count(1)
    neg_labels = valid_labels.count(0)
    
    print(f"Valid predictions: #positive={pos_preds}, #negative={neg_preds}")
    print(f"True labels: #positive={pos_labels}, #negative={neg_labels}")
    print(f"Invalid predictions: {invalid_count}/{total_samples} ({invalid_count/total_samples*100:.2f}%)")
    
    # Print per-class metrics
    print(f"\nPer-class metrics:")
    print(f"Class 0 (Not Biased) - F1: {metrics['f1_class_0_not_biased']:.4f}, Precision: {metrics['precision_class_0_not_biased']:.4f}, Recall: {metrics['recall_class_0_not_biased']:.4f}")
    print(f"Class 1 (Biased)     - F1: {metrics['f1_class_1_biased']:.4f}, Precision: {metrics['precision_class_1_biased']:.4f}, Recall: {metrics['recall_class_1_biased']:.4f}")
    print(f"Macro Average       - F1: {metrics['f1_macro']:.4f}, Precision: {metrics['precision_macro']:.4f}, Recall: {metrics['recall_macro']:.4f}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

# -------------- Main --------------
if __name__ == "__main__":
    args = parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint path does not exist: {args.checkpoint_path}")
        exit(1)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    print(f"Max samples: {args.max_samples or 'All'}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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

    # Clear memory after base model load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ------- Load LoRA weights from checkpoint -------
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
    eval_ds = dataset["validation"]
    test_ds = dataset["test"]

    # Limit samples if specified
    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    # ---------- Prompt Template ----------
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

    # ---------- Validation Evaluation ---------
    print("\n===== VALIDATION SET EVALUATION =====")
    val_preds, val_true, val_decoded, val_invalid = evaluate_model_generation(
        model, tokenizer, eval_ds, prompt_template, args.max_samples
    )
    val_metrics = compute_metrics_simple(val_preds, val_true, val_invalid)
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
    test_metrics = compute_metrics_simple(test_preds, test_true, test_invalid)
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
