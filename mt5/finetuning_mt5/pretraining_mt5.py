from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset, Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import argparse
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


# for saving the model
STRAT_ABBREV = {
    "undersample": "us",
    "oversample":  "os",
    "balanced":    "bl",
    "normal":      "nm",
}

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MT5 for bias detection")
    p.add_argument("--sampling", type=str, default="undersample",
                   choices=list(STRAT_ABBREV.keys()),
                   help="Data sampling strategy")
    p.add_argument("--epochs", type=int, default=12,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning Rate")
    p.add_argument("--bs", type=int, default=64, help="Batch Size")
    p.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    p.add_argument("--patience", type=int, default=2,
                   help="Early stopping patience")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focal_loss", action="store_true", default=False, 
                   help='Use a different loss that helps against unbalanced datasetes')
    return p.parse_args()


def classify_preds(list):
    """
    Classify each prediction string as:
      0  if it’s exactly “niet-biased” or “niet biased anywhere in the text” (case-insensitive),
      1  if it starts with “biased” (case-insensitive),
     -1  otherwise.

    Parameters:
        preds: List of prediction strings.

    Returns:
        List of ints (0, 1, or -1) corresponding to each input string.
    """
    pattern = re.compile(r'niet[- ]biased', flags=re.IGNORECASE)
    result = []
    for pred in list:
        txt = pred.strip()
        if pattern.fullmatch(txt):
            result.append(0)
        elif re.match(r'biased', txt, flags=re.IGNORECASE):
            result.append(1)
        else:
            result.append(-1)
    return result

# metrics
def compute_metrics(pred):
    # predictions can be raw logits or token IDs; ensure we have IDs
    preds = pred.predictions
    if isinstance(preds, np.ndarray) and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # decode to strings and normalize
    decoded_preds  = [p.strip().lower() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    decoded_labels = [l.strip().lower() for l in tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)]

    # map any string containing 'biased' → 1, anything else → 0
    y_pred = classify_preds(decoded_preds)
    y_true = classify_preds(decoded_labels)

    #  compute metrics
    acc  = (np.array(y_true) == np.array(y_pred)).mean()
    f1   = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    na_pred = y_pred.count(-1)
    na_lab = y_pred.count(-1)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "no_pred": na_pred,
        "no_label": na_lab
    }

# data sampling utility
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

# def preprocess_batch(batch):
#     # batch["text"] is a list of length B
#     # batch["label"] is a list of length B
#     # tokenize all texts in one go:
#     model_inputs = tokenizer(
#         batch["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#     )
#     # build label sequences for each example
#     labels = []
#     for lab in batch["label"]:
#         label_str = "biased" if lab == 1 else "niet-biased"
#         lab_ids = tokenizer(
#             label_str,
#             truncation=True,
#             padding="max_length",
#             max_length=6,
#         )["input_ids"]
#         # mask padding tokens
#         lab_ids = [tok if tok != tokenizer.pad_token_id else -100
#                    for tok in lab_ids]
#         labels.append(lab_ids)
#     # add to the dict and return
#     model_inputs["labels"] = labels
#     return model_inputs

class FocalLoss(nn.Module):
    def __init__(self, alpha, device, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float, device=device)   # list or scalar
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none",
                                    ignore_index=ignore_index)
    def forward(self, logits, labels):
        B,S,V = logits.shape
        logits = logits.view(-1, V)
        labels = labels.view(-1)
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[labels]    # per-token weight
        focal_factor = alpha_t * (1 - pt) ** self.gamma
        return (focal_factor * ce_loss).mean()

class Seq2SeqWithFocal(Seq2SeqTrainer):
    def __init__(self, focal_alpha, focal_gamma, **kwargs):
        super().__init__(**kwargs)
        device = self.model.device
        self.focal_loss = FocalLoss(alpha=focal_alpha,
                                    device=device,
                                    gamma=focal_gamma,
                                    ignore_index=self.model.config.pad_token_id)
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward
        outputs = model(**inputs)
        logits = outputs.logits  # (B, S, V)
        # flatten to (B*S, V) and (B*S,)
        vocab_size = logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(
            weight=weight_tensor,
            ignore_index=tokenizer.pad_token_id
        )
        loss = loss_fct(
            logits.view(-1, vocab_size),
            inputs["labels"].view(-1)
        )
        return (loss, outputs) if return_outputs else loss
    

if __name__ == "__main__":
    args = parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model & tokenizer
    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    dataset = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
    # prepare datasets
    train_df = dataset["train"].to_pandas()
    train_df = sample_data(train_df, strategy=args.sampling, random_state=args.seed).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_df = dataset["validation"].to_pandas()
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = dataset["test"]

    #the weights for the FL
    if args.focal_loss:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_df.label.values
        ).tolist()  # e.g. [w_neg, w_pos]
        weight_tensor = torch.tensor(weights, device=model.device)

    tokenized_train = train_ds.map(
        preprocess,
        num_proc=8,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        preprocess,
        num_proc=8,
        remove_columns=val_ds.column_names,
    )
    tokenized_test = test_ds.map(
        preprocess,
        num_proc=8,
        remove_columns=test_ds.column_names,
    )

    for ds in (tokenized_train, tokenized_val, tokenized_test):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Number of examples in tokenized_train: {len(tokenized_train)}")

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=2,
        fp16=True,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        predict_with_generate=True
            )

    # if args.focal_loss is True:
    #     trainer = Seq2SeqWithFocal(
    #         model=model,
    #         args=training_args,
    #         train_dataset=tokenized_train,
    #         eval_dataset=tokenized_val,
    #         tokenizer=tokenizer,
    #         compute_metrics=compute_metrics,
    #         callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    #         focal_alpha=weights.tolist(),
    #         focal_gamma=0,  # focusing parameter (set to 0 means just weighted CE )
    #     )
    # else:
    data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    print("Using Seq2SeqTrainer? ", not args.focal_loss)
    print("Instantiating Trainer class:", type(trainer))
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # evaluate & save best
    eval_metrics = trainer.evaluate()
    best_f1 = eval_metrics["eval_f1_macro"]
    print(f"Best validation F1: {best_f1:.4f}")
    abbrev = STRAT_ABBREV[args.sampling]
    safe_name = model_name.replace("/", "-")
    if args.focal_loss:
        model_dir = f"{args.output_dir}/{safe_name}_FE_f1_{best_f1:.4f}"
    else:
        model_dir = f"{args.output_dir}/{safe_name}_{abbrev}_f1_{best_f1:.4f}"
    print(f"Model saved to {model_dir}")
    trainer.save_model(model_dir)

    # test
    test_pred = trainer.predict(tokenized_test)
    test_metrics = compute_metrics(test_pred)
    print("Test metrics:", test_metrics)
