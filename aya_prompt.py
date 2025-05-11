from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import torch
import random
import pandas as pd
import re

def extract_label_loose(s: str):
    """Return the last '0' or '1' found in s, or None if not found."""
    matches = re.findall(r"[01]", s)
    return int(matches[-1]) if matches else None

# load model & tokenizer in 8-bit
model_name = "CohereForAI/aya-101"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto"
)

dataset = load_dataset(
    "milenamileentje/Dutch-Government-Data-for-Bias-detection",
    split="test"
)

templates = {
    # … your template_1 through template_5 definitions …
}

num_samples = 50
random.seed(42)

max_tokens = {
    "template_1": 5,
    "template_2": 5,
    "template_3": 5,
    "template_4": 300,
    "template_5": 5,
}

results = []
for tmpl_name, tmpl in templates.items():
    subset = random.sample(list(dataset), num_samples)
    for entry in subset:
        text       = entry["text"]
        true_label = int(entry["label"])

        prompt = tmpl.format(item=text)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        output = model.generate(**inputs, max_new_tokens=max_tokens[tmpl_name])
        full    = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # extract label & reasoning
        pred_label = extract_label_loose(full)
        # reasoning is everything before the last digit (if found), else full output
        if pred_label is not None:
            # find last occurrence
            idx = full.rfind(str(pred_label))
            reasoning = full[:idx].strip()
        else:
            reasoning = full

        results.append({
            "template":    tmpl_name,
            "text":        text,
            "true_label":  true_label,
            "pred_label":  pred_label,
            "reasoning":   reasoning
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))

# compute metrics
summary = []
for tmpl, group in df.groupby("template"):
    sub = group.dropna(subset=["pred_label"])
    trues = sub["true_label"]
    preds = sub["pred_label"].astype(int)
    n     = len(sub)
    if n == 0:
        summary.append({
            "template": tmpl, "n_evaluated": 0,
            "accuracy":"N/A","precision":"N/A","recall":"N/A","f1":"N/A"
        })
        continue
    acc = accuracy_score(trues, preds)
    p,r,f,_ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    summary.append({
        "template": tmpl,
        "n_evaluated": n,
        "accuracy": f"{acc:.2f}",
        "precision": f"{p:.2f}",
        "recall":    f"{r:.2f}",
        "f1":        f"{f:.2f}"
    })

print(pd.DataFrame(summary).to_markdown(index=False))

# inspect template_4
t4 = df[df["template"]=="template_4"]
no_label = t4[t4["pred_label"].isna()]
has_label = t4[t4["pred_label"].notna()]

print("\n=== template_4: NO valid label ===")
for _, row in no_label.iterrows():
    print(f"\nText: {row['text']}\nReasoning:\n{row['reasoning']}\n---")

print("\n=== template_4: random 5 with a label ===")
for _, row in has_label.sample(5, random_state=42).iterrows():
    print(f"\nText: {row['text']}\nReasoning:\n{row['reasoning']}\nPredicted label: {row['pred_label']}\n---")