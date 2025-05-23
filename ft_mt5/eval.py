import argparse, re, pandas as pd, torch
from datasets import load_dataset
from inseq import load_model
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Eval fine-tuned mT5 on bias dataset via inseq")
    p.add_argument(
        "--model_path", required=True,
        help="Path to your ft-mt5 checkpoint folder (config.json, pytorch_model.bin, spiece.model)"
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for generation"
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=5,
        help="How many tokens to generate"
    )
    p.add_argument(
        "--output_csv", type=str, default="test_predictions.csv",
        help="Where to save the predictions"
    )
    return p.parse_args()

def classify_preds(preds):
    pat = re.compile(r'niet[- ]biased', re.IGNORECASE)
    out = []
    for p in preds:
        t = p.strip().lower()
        if pat.fullmatch(t):         out.append(0)
        elif t.startswith("biased"): out.append(1)
        else:                        out.append(-1)
    return out

def main():
    args = parse_args()

    # 1) Load the dataset just like in training
    ds = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
    test_ds = ds["test"]

    # 2) Load your fine-tuned MT5 via inseq
    model_wrapper = load_model(args.model_path, "integrated_gradients")
    hf_model     = model_wrapper.model
    hf_tokenizer = model_wrapper.tokenizer
    device       = model_wrapper.device

    # 3) Tokenize & batch
    def prep(ex):
        return hf_tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)
    tokenized = test_ds.map(prep, batched=True)
    tokenized.set_format("torch", ["input_ids","attention_mask"])
    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        collate_fn=default_data_collator
    )

    # 4) Generate
    all_preds = []
    for batch in tqdm(loader, desc="Generating"):
        batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids","attention_mask"]}
        outs = hf_model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            num_beams=5,
            early_stopping=True
        )
        all_preds += hf_tokenizer.batch_decode(outs, skip_special_tokens=True)

    # 5) Classify & save
    labels = classify_preds(all_preds)
    df = pd.DataFrame({
        "text": test_ds["text"],
        "prediction_str": all_preds,
        "prediction_label": labels
    })
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} predictions to {args.output_csv}\n")
    print("First 10 predictions:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
