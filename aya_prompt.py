from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import torch
import random
import pandas as pd

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
    "template_1": """Je bent een expert op het gebied van bias, stereotypering en discriminatie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Geef de volgende zin een label, antwoord alleen met 0 of 1.
Geef de zin het label 0 voor niet-biased en 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
""",
    "template_2": """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Geef de volgende zin een label, antwoord alleen met 0 of 1.
Geef de zin het label 0 voor niet-biased en 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
""",
    "template_3": """Je bent een expert in Nederlandse juridische en parlementaire documenten.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Geef de volgende zin een label, antwoord alleen met 0 of 1.
Geef de zin het label 0 voor niet-biased en 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
""",
    "template_4": """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Denk na en schrijf je stap-voor-stap redenatie uit voordat je antwoort.
Geef de volgende zin een label, 0 of 1.
Geef de zin het label 0 voor niet-biased en 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
""",
    "template_5": """Een zin moet als wordt herkend als een zin met bias als het aan een van de volgende voorwaarden voldoet:
1. De zin bevat een universeel ongepaste term, **OF**
2a. De zin vertoont bias, zoals stereotypering, uitsluiting, machtsongelijkheid of vooroordeel—impliciet of expliciet **EN**
2b. De bias is gericht op een specifieke groep personen.
Geef de zin het label 0 voor niet-biased en 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
"""
}

num_samples = 5
random.seed(42)

max_tokens = {
    "template_1": 5,
    "template_2": 5,
    "template_3": 5,
    "template_4": 150,
    "template_5": 5,
}

results = []
for tmpl_name, tmpl in templates.items():
    # shuffle & pick
    subset = random.sample(list(dataset), num_samples)
    for entry in subset:
        text = entry["text"]
        true_label = entry["label"]
        
        prompt = tmpl.format(item=text)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        output = model.generate(**inputs, max_new_tokens=max_tokens[tmpl_name])
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        if tmpl_name == "template_4":
            # assume reasoning + "\n1"  or "\n0" at the end
            *reasoning, pred_label = pred.rsplit(maxsplit=1)
            reasoning = " ".join(reasoning)
        else:
            reasoning = None
            pred_label = pred
        
        results.append({
            "template": tmpl_name,
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))

df = pd.DataFrame(results)

summary = []
for tmpl, group in df.groupby("template"):
    trues = group["true_label"].astype(int)
    preds = group["pred_label"].astype(int)
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    summary.append({
        "template": tmpl,
        "n_samples": len(group),
        "accuracy": f"{acc:.2f}",
        "precision": f"{prec:.2f}",
        "recall": f"{rec:.2f}",
        "f1": f"{f1:.2f}",
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_markdown(index=False))