#!/usr/bin/env python
import argparse
from inseq import load_model

def parse_args():
    p = argparse.ArgumentParser(description="Generate with a fine-tuned mT5 via inseq")
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Path to your fine-tuned mkT5 folder (containing config.json, pytorch_model.bin, spiece.model)"
    )
    p.add_argument(
        "--split", choices=["train","test","validation"], default="test",
        help="Which split of the Mila&Meentje dataset to run on"
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=5,
        help="How many tokens to generate per example"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load via inseq (attribution_method is a required argument, 
    #    but we won't actually use the attributions here)
    model = load_model(
        args.model_path,
        attribution_method="integrated_gradients"
    )

    # 2) Grab the underlying HF model & tokenizer
    hf_model     = model.model
    hf_tokenizer = model.tokenizer
    device       = model.device

    # 3) Load the dataset split
    from datasets import load_dataset
    ds = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")[args.split]

    # 4) For each example, generate and print
    for i, sample in enumerate(ds):
        text = sample["text"]
        # tokenize + send to device
        inputs = hf_tokenizer(text, return_tensors="pt").to(device)
        # generate
        outs   = hf_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=5,
            early_stopping=True
        )
        pred = hf_tokenizer.batch_decode(outs, skip_special_tokens=True)[0]

        print(f"[{i:03d}] ▶️ {pred}")
        # break or limit how many you print:
        if i >= 9:
            break

if __name__ == "__main__":
    main()
