from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset, Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import random

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# data sampling utility
def sample_data(df, strategy="undersample", oversample_factor=2, undersample_ratio=0.7, balanced_neg_ratio=0.5, random_state=42):
    biased = df[df.label == 1]
    unbiased = df[df.label == 0]

    if strategy == "undersample":
        unbiased_sampled = unbiased.sample(frac=undersample_ratio, random_state=random_state)
        return pd.concat([biased, unbiased_sampled])
    elif strategy == "oversample":
        return pd.concat([biased] * oversample_factor + [unbiased])
    elif strategy == "balanced":
        target_total = len(df)
        unbiased_target = int(target_total * balanced_neg_ratio)
        biased_target = target_total - unbiased_target
        neg_sampled = unbiased.sample(n=unbiased_target, random_state=random_state)
        repeats = biased_target // len(biased)
        remainder = biased_target % len(biased)
        biased_repeated = pd.concat([biased] * repeats + [biased.sample(n=remainder, random_state=random_state)])
        return pd.concat([biased_repeated, neg_sampled])
    elif strategy == "normal":
        return df
    else:
        raise ValueError("Unsupported strategy.")

# metrics

def compute_metrics(pred):
    decoded_preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    map_lbl = {"biased": 1, "niet-biased": 0}
    y_true = [map_lbl.get(lbl, int(lbl)) for lbl in decoded_labels]
    y_pred = [map_lbl.get(p, int(p)) for p in decoded_preds]
    return {
        "accuracy": (np.array(y_true) == np.array(y_pred)).mean(),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro")
    }

# model & tokenizer
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")

def preprocess(example, makeitwords=True):
    # tokenize inputs
    model_inputs = tokenizer(
        example['text'], truncation=True, padding="max_length", max_length=512
    )
    # build label string
    label_str = "biased" if example["label"] == 1 else "niet-biased"
    # tokenize labels
    label_ids = tokenizer(
        label_str, truncation=True, padding="max_length", max_length=3
    )["input_ids"]
    # mask padding
    label_ids = [tok if tok != tokenizer.pad_token_id else -100 for tok in label_ids]
    model_inputs["labels"] = label_ids
    return model_inputs

# prepare datasets
train_df = dataset["train"].to_pandas()
train_df = sample_data(train_df, strategy='undersample').sample(frac=1, random_state=42).reset_index(drop=True)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_df = dataset["validation"].to_pandas()
val_ds = Dataset.from_pandas(val_df, preserve_index=False)
test_ds = dataset["test"]

tokenized_train = train_ds.map(preprocess)
tokenized_val = val_ds.map(preprocess)
tokenized_test = test_ds.map(preprocess)

for ds in (tokenized_train, tokenized_val, tokenized_test):
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# training arguments
taining_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.0,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # evaluate & save best
    eval_metrics = trainer.evaluate()
    best_f1 = eval_metrics["eval_f1_macro"]
    print(f"Best validation F1: {best_f1:.4f}")
    model_dir = f"alt_mt5-base-finetuned-f1_{best_f1:.4f}"
    trainer.save_model(model_dir)
    print(f"Model saved to {model_dir}")

    # test
    test_pred = trainer.predict(tokenized_test)
    test_metrics = compute_metrics(test_pred)
    print("Test metrics:", test_metrics)
