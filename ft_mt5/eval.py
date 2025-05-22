import argparse, os, re, torch, pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from inseq import load_model
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--split", choices=["train","validation","test"], default="test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--output_csv", default="predictions.csv")
    return p.parse_args()

def classify_preds(preds):
    pat = re.compile(r'niet[- ]biased', re.IGNORECASE)
    out = []
    for p in preds:
        t = p.strip().lower()
        if pat.fullmatch(t):      out.append(0)
        elif t.startswith("biased"): out.append(1)
        else:                     out.append(-1)
    return out

def main():
    args = parse_args()

    # 1) Download raw CSVs (avoids the fsspec '**' bug)
    TRAIN_CSV = hf_hub_download(
        repo_id="milenamileentje/Dutch-Government-Data-for-Bias-detection",
        filename="train.csv", repo_type="dataset",
        local_dir=".cache_bias", local_dir_use_symlinks=False
    )
    VAL_CSV = hf_hub_download(
        repo_id="milenamileentje/Dutch-Government-Data-for-Bias-detection",
        filename="validation.csv", repo_type="dataset",
        local_dir=".cache_bias", local_dir_use_symlinks=False
    )
    TEST_CSV = hf_hub_download(
        repo_id="milenamileentje/Dutch-Government-Data-for-Bias-detection",
        filename="test.csv", repo_type="dataset",
        local_dir=".cache_bias", local_dir_use_symlinks=False
    )

    # 2) Load the split as a Dataset
    data_files = {"train": TRAIN_CSV, "validation": VAL_CSV, "test": TEST_CSV}
    ds = load_dataset("csv", data_files=data_files)[args.split]

    # 3) Load your fine-tuned model via inseq
    model_wrapper = load_model(args.model_path, "integrated_gradients")
    hf_model     = model_wrapper.model
    hf_tokenizer = model_wrapper.tokenizer
    device       = model_wrapper.device

    # 4) Tokenize & batch
    def prep(ex):
        return hf_tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)
    tds = ds.map(prep, batched=True)
    tds.set_format("torch", ["input_ids","attention_mask"])
    collator = DataCollatorForSeq2Seq(hf_tokenizer, model=hf_model, label_pad_token_id=-100)
    loader = DataLoader(tds, batch_size=args.batch_size, collate_fn=collator)

    # 5) Generate
    raw_preds = []
    for batch in tqdm(loader, desc="Generating"):
        batch = {k:v.to(device) for k,v in batch.items()}
        outs  = hf_model.generate(**batch, max_new_tokens=args.max_new_tokens)
        raw_preds += hf_tokenizer.batch_decode(outs, skip_special_tokens=True)

    # 6) Save
    labels = classify_preds(raw_preds)
    df = pd.DataFrame({"text": ds["text"], "prediction_str": raw_preds, "prediction_label": labels})
    df.to_csv(args.output_csv, index=False)
    print("Wrote", len(df), "predictions to", args.output_csv)
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
