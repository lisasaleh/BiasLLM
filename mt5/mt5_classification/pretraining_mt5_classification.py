import torch
import torch.nn as nn
from transformers import MT5Tokenizer, MT5EncoderModel, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import os

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
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--focal_loss", action="store_true", default=False,
                   help='Use focal loss to handle class imbalance')
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help='Gamma parameter for focal loss')
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
        # Use the mean of the
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

def compute_metrics_classification(pred):
    from sklearn.metrics import classification_report

    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    # Standard metrics
    acc = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    prec_macro = precision_score(labels, preds, average="macro")
    rec_macro = recall_score(labels, preds, average="macro")

    # Per-class F1 scores
    f1_per_class = f1_score(labels, preds, average=None)
    class_report = classification_report(labels, preds, output_dict=True)
    print("Classification Report:\n", classification_report(labels, preds))
    # Create dictionary with per-class F1s
    per_class_metrics = {
        f"f1_class_{i}": score for i, score in enumerate(f1_per_class)
    }

    # Combine all metrics
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        **per_class_metrics
    }

class FocalLoss(nn.Module):
    def __init__(self, alpha, device, gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float, device=device)
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
        
    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[labels]
        focal_factor = alpha_t * (1 - pt) ** self.gamma
        return (focal_factor * ce_loss).mean()

class ClassificationWithFocal(Trainer):
    def __init__(self, focal_alpha, focal_gamma, **kwargs):
        super().__init__(**kwargs)
        device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.focal_loss = FocalLoss(alpha=focal_alpha,
                                  device=device,
                                  gamma=focal_gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs['logits']
        labels = inputs['labels']
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
    
    # Compute class weights if using focal loss
    weights = None
    if args.focal_loss:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_df.label.values
        ).tolist()
    
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
    
    if args.focal_loss:
        trainer = ClassificationWithFocal(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_classification,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
            focal_alpha=weights,
            focal_gamma=args.focal_gamma
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_classification,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )
    
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    eval_metrics = trainer.evaluate()
    print(f"Best validation F1: {eval_metrics['eval_f1_macro']:.4f}")
    abbrev = STRAT_ABBREV[args.sampling]
    safe_name = model_name.replace("/", "-")
    model_dir = f"{args.output_dir}/{safe_name}_clf_{abbrev}_f1_{args.sampling}_{eval_metrics['eval_f1_macro']:.4f}"
    if args.focal_loss:
        model_dir += "_focal"
    print(f"Model saved to {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    print("Evaluating on test set...")
    test_pred = trainer.predict(tokenized_test)
    test_metrics = compute_metrics_classification(test_pred)
    print("Test metrics:", test_metrics) 
    with open(model_dir + "/test_metrics.txt", "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print("Test metrics saved to", model_dir + "/test_metrics.txt")
    print("Done.")
    # Save the model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")