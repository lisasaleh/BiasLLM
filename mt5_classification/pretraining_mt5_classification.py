import torch
import torch.nn as nn
from transformers import MT5Tokenizer, MT5EncoderModel, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import argparse
from sklearn.utils.class_weight import compute_class_weight

STRAT_ABBREV = {
    "undersample": "us",
    "oversample":  "os",
    "balanced":    "bl",
    "normal":      "nm",
}

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MT5 encoder for bias detection (classification)")
    p.add_argument("--sampling", type=str, default="undersample",
                   choices=list(STRAT_ABBREV.keys()),
                   help="Data sampling strategy")
    p.add_argument("--epochs", type=int, default=12,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning Rate")
    p.add_argument("--bs", type=int, default=64, help="Batch Size")
    p.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bare_model", action='store_true')
    return p.parse_args()

def sample_data(df, strategy="undersample", oversample_factor=2, undersample_ratio=0.7, balanced_neg_ratio=0.5, random_state=None):
    if random_state is None:
        random_state = 42
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

def preprocess_classification(examples):
    # Tokenize input texts in batch
    model_inputs = tokenizer(
        examples['text'],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    # Use integer labels (0 or 1)
    model_inputs["labels"] = examples["label"]
    #convert to long tensor
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"]).long()
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"]).long()
    # Convert labels to long
    model_inputs["labels"] = torch.tensor(model_inputs["labels"]).long()
    return model_inputs

class MT5Classifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = MT5EncoderModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the first token ([CLS]-like) representation
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

def compute_metrics_classification(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    print(f"Predictions: {preds}")
    print(f"Labels: {labels}")
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro")
    rec = recall_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
    }

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5Classifier(model_name)
    dataset = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
    train_df = dataset["train"].to_pandas()
    train_df = sample_data(train_df, strategy=args.sampling, random_state=args.seed).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_df = dataset["validation"].to_pandas()
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = dataset["test"]
    tokenized_train = train_ds.map(
        preprocess_classification,
        batched=True,
        batch_size=64,
        num_proc=8,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        preprocess_classification,
        batched=True,
        batch_size=64,
        num_proc=8,
        remove_columns=val_ds.column_names,
    )
    tokenized_test = test_ds.map(
        preprocess_classification,
        batched=True,
        batch_size=64,
        num_proc=8,
        remove_columns=test_ds.column_names,
    )
    for ds in (tokenized_train, tokenized_val, tokenized_test):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        overwrite_output_dir=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_classification,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    if args.bare_model is False:
        print("Starting training...")
        trainer.train()
    print("Training complete.")
    eval_metrics = trainer.evaluate()
    print(f"Best validation F1: {eval_metrics['eval_f1_macro']:.4f}")
    if args.bare_model is False:
        abbrev = STRAT_ABBREV[args.sampling]
        safe_name = model_name.replace("/", "-")
        model_dir = f"{args.output_dir}/{safe_name}_clf_{abbrev}_f1_{args.sampling}_{eval_metrics['eval_f1_macro']:.4f}"
        print(f"Model saved to {model_dir}")
        model.encoder.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
    test_pred = trainer.predict(tokenized_test)
    test_metrics = compute_metrics_classification(test_pred)
    print("Test metrics:", test_metrics) 