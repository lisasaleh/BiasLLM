from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

train_prompt_template = (
        "Je bent een expert op het gebied van bias, stereotypering en discriminatie. "
        "Je taak is om zinnen te classificeren volgens de volgende definitie: een zin bevat bias "
        "als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat, "
        "die impliciet of expliciet kunnen zijn, en die gaan over een afgebakende groep personen, "
        "OF als de zin een woord bevat dat vermeden moet worden vanwege bias. "
        "Geef de volgende zin een label, antwoord alleen met 'ja' voor biased of 'nee' voor niet-biased. "
        'De zin is: "{text}" ' 
    )
# "Geef de volgende zin een label: 'ja' voor biased of 'nee' voor niet-biased. "
# "Beantwoord met alleen 'ja' (biased) of 'nee' (niet-biased), zonder uitleg. "

cot_prompt_template = (
        "Je bent een expert op het gebied van bias, stereotypering en discriminatie. "
            "Je taak is om zinnen te classificeren volgens de volgende definitie: een zin bevat bias "
            "als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat, "
            "die impliciet of expliciet kunnen zijn, en die gaan over een afgebakende groep personen, "
            "OF als de zin een woord bevat dat vermeden moet worden vanwege bias. "
            "Denk hardop en stap-voor-stap. " 
            "Sluit je antwoordt altijd af met het label, 'nee' (niet-biased) of 'ja' (biased). "
            "Beperk je antwoord tot maximaal 100 tokens. "
            "De zin is: '{text}'"
        )

TRAIN_MAX_SEQ_LENGTH = 256  # Maximum input sequence length for training/evaluation
                            # -> cap on prompt template + sentence length + target length

def preprocess(example, tokenizer):
    """
    Preprocess an example for training/evaluation.
    
    Converts the numeric label (0/1) to the textual format ("ja"/"nee") that the model will generate,
    then tokenizes the input and target, and creates the appropriate input_ids, attention_mask, and labels.
    """
    # complete the prompt with the example text
    prompt = train_prompt_template.format(text=example["text"])
    
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

def classify_answers_train(text_list):
    """
    Classify each answer output as:
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


def classify_answer_eval(text):
    """Classify the answer text as 'ja' (1), 'nee' (0), or invalid (-1).
    
    Checks if the text contains 'ja' or 'nee' as standalone words,
    ignoring case and leading/trailing whitespace.
    If neither or both are present, it returns -1 for invalid.
    
    Parameters:
        text (str): The input text to classify.
    Returns:
        int: 1 if 'ja' is present, 0 if 'nee' is present, -1 if invalid.
    """
    
    # print(f"Classifying text: '{text}'")
    
    # Check if text is empty or not a string -> invalid
    if not text or not isinstance(text, str):
        print("Text is empty or not a string -> invalid")
        return -1
    
    # Clean the text - remove leading/trailing whitespace
    text = text.strip().lower()
    # Remove punctuation and special characters
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    
    if 'ja' in text and not 'nee' in text:
        # print("Found 'ja' but not 'nee' -> classified as 1 (biased)")
        return 1
    elif 'nee' in text and not 'ja' in text: 
        # print("Found 'nee' but not 'ja' -> classified as 0 (not biased)")
        return 0
    else:
        print("Neither 'ja' nor 'nee' found or both found -> classified as -1 (invalid)")
        return -1
    

def compute_metrics(predictions, true_labels, invalid_count):
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