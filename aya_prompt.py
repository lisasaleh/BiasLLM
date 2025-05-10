from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "CohereForAI/aya-101"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, load_in_8bit = True)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit = True, device_map="auto")

template = """Je bent een expert op het gebied van bias, stereotypering en discriminatie.
Je taak is om zinnen te classificeren volgens de volgende definitie:
een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat,
die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen,
OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
Geef de volgende zin een label, antwoord alleen met 0 of 1.
Geef de zin het label 0 voor niet-biased en 1 voor biased.
De zin die beoordeeld moet worden: “{item}”
Antwoord met alleen 0 of 1:
"""

items = [
    "Deze vrouw is te emotioneel om leiding te geven.",
    "Het huis heeft drie slaapkamers en een tuin.",
    "Ik ben mijn sleutels vergeten"
    "De immigrant had nog geen huis hier"
    # …add your test sentences here…
]

results = {}
for text in items:
    prompt = template.format(item=text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=5)
    label = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    results[text] = label

for txt, lbl in results.items():
    print(f"{lbl}  \t| {txt}")
