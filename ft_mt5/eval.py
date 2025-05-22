import argparse
import re
import pandas as pd
import os
import torch
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    MT5Config
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a pretrained mT5 on the Mila&Meentje bias-detection test set"
    )
    p.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Path or HF identifier of your fine-tuned mT5"
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for generation"
    )
    p.add_argument(
        "--max_length", type=int, default=3,
        help="Max length of the generated label (e.g. 'biased' or 'niet-biased')"
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--output_csv", type=str, default="test_predictions.csv",
        help="Where to dump the predictions"
    )
    return p.parse_args()

def load_local_mt5(model_dir, device):
    # 1) config & model weights
    cfg = MT5Config.from_json_file(os.path.join(model_dir, "config.json"))
    model = MT5ForConditionalGeneration(cfg)
    state = torch.load(os.path.join(model_dir, "pytorch_model.bin"),
                       map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    # 2) tokenizer (SentencePiece)
    # MT5Tokenizer can be initialized directly from the .model file:
    tok = MT5Tokenizer(
        sp_model_file=os.path.join(model_dir, "spiece.model"),
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>"
    )
    return model, tok

def classify_preds(preds):
    """
    Map each generated string to a label:
      0 if exactly 'niet-biased' (case-insensitive, dash or space),
      1 if it starts with 'biased',
     -1 otherwise.
    """
    pat  = re.compile(r'niet[- ]biased', flags=re.IGNORECASE)
    out  = []
    for p in preds:
        txt = p.strip().lower()
        if pat.fullmatch(txt):
            out.append(0)
        elif txt.startswith("biased"):
            out.append(1)
        else:
            out.append(-1)
    return out

def main():
    args = parse_args()
    # 1) Clean up the path: strip any "file://"
    model_dir = args.model_name_or_path
    if model_dir.startswith("file://"):
        model_dir = model_dir[len("file://"):]

    # 2) Expand and verify
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    print(f"Loading from local folder: {model_dir}")

    # 1) Load model & tokenizer
    model, tokenizer = load_local_mt5(model_dir, args.device)

    model.to(args.device).eval()

    # 2) Load test split
    ds = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
    test_ds = ds["test"]

    # 3) Preprocess & collator
    def preprocess(ex):
        # just encode the input text
        return tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    tokenized_test = test_ds.map(
        preprocess,
        batched=True,
        remove_columns=test_ds.column_names
    )
    tokenized_test.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )
    loader = DataLoader(
        tokenized_test,
        batch_size=args.batch_size,
        collate_fn=collator
    )

    # 4) Generate
    all_preds = []
    for batch in tqdm(loader, desc="Generating"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            num_beams=5,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        all_preds.extend(decoded)

    # 5) Decode & save
    raw_texts = test_ds["text"]
    pred_labels = classify_preds(all_preds)

    df = pd.DataFrame({
        "text": raw_texts,
        "prediction_str": all_preds,
        "prediction_label": pred_labels
    })
    df.to_csv(args.output_csv, index=False)
    print(f"→ Wrote {len(df)} predictions to {args.output_csv}")
    # also print the first 10
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
