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
    "template_1": """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Rederneer stap-voor-stap waarom deze zin wel of neit bias bevat.
Sluit je antwoordt altijd af met een label:
0 voor niet-biased of 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
""",
    
#     """Je bent een expert op het gebied van bias, stereotypering en discriminatie.
# Je taak is om zinnen te classificeren volgens de volgende definitie:
# een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
# die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
# OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
# Geef de volgende zin een label, antwoord alleen met 0 of 1.
# Geef de zin het label 0 voor niet-biased en 1 voor biased.
# De zin die beoordeeld moet worden: “{item}”
# """,
    "template_2": """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Rederneer stap-voor-stap waarom deze zin wel of niet bias bevat.
Sluit je antwoordt altijd af met het label, 0 (niet-biased) of 1 (biased).
De zin die beoordeeld moet worden: “{item}”
""",
     
#     """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
# Je taak is om zinnen te classificeren volgens de volgende definitie:
# een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
# die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
# OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
# Geef de volgende zin een label, antwoord alleen met 0 of 1.
# Geef de zin het label 0 voor niet-biased en 1 voor biased.
# De zin die beoordeeld moet worden: “{item}”
# """,
    "template_3": """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Denk hardop en stap-voor stap.
Sluit je antwoordt altijd af met het label, 0 (niet-biased) of 1 (biased).
De zin die beoordeeld moet worden: “{item}”
""",
    
#     """Je bent een expert in Nederlandse juridische en parlementaire documenten.
# Je taak is om zinnen te classificeren volgens de volgende definitie:
# een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
# die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
# OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
# Geef de volgende zin een label, antwoord alleen met 0 of 1.
# Geef de zin het label 0 voor niet-biased en 1 voor biased.
# De zin die beoordeeld moet worden: “{item}”
# """,
    "template_4": """Je bent een expert op het gebied van bias, stereotypering en discriminatie detectie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Denk na en schrijf je stap-voor-stap redenatie uit.
Sluit je antwoordt altijd af met het label, 0 (niet-biased) of 1 (biased).
De zin die beoordeeld moet worden: “{item}”
"""
#    "template_5": 
    
#     """Een zin moet als wordt herkend als een zin met bias als het aan een van de volgende voorwaarden voldoet:
# 1. De zin bevat een universeel ongepaste term, **OF**
# 2a. De zin vertoont bias, zoals stereotypering, uitsluiting, machtsongelijkheid of vooroordeel—impliciet of expliciet **EN**
# 2b. De bias is gericht op een specifieke groep personen.
# Geef de zin het label 0 voor niet-biased en 1 voor biased.
# De zin die beoordeeld moet worden: “{item}”
# """
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