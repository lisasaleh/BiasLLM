from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model     = MT5ForConditionalGeneration.from_pretrained(model_name)

def detect_bias(text, template):
    inp = template.format(item=text)
    print(inp)
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, padding=True, max_length=512)
    out = model.generate(**inputs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# template = (
#     # """\
#     # Je bent een expert op het gebied van bias, stereotypering en discriminatie. Je taak is om zinnen te classificeren volgens de volgende definitie: een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat, die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen, OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat.
#     # Geef de volgende zin een label, antwoord alleen met 0 of 1. Geef de zin het label 0 voor niet-biased en 1 voor biased. De zin die beoordeeld moet worden: {item} Antwoord met alleen 0 of 1:
#     # """
#     """

#     Voorbeelden:
#     Tekst: “Ik brand een kaars.” → 0
#     Tekst: “Ik lees een boek.” → 1
#     Nu jij:
#     Tekst: “{item}”
#     Antwoord alleen met 0 of 1:"
#     """
# )

template = """\
Je bent een expert op het gebied van bias, stereotypering en discriminatie.
Een zin bevat bias (label 1) als er een stereotype, uitsluiting,
machtsverhoudingen of vooroordelen over een groep in voorkomt.
Anders is het niet-biased (label 0).

Voorbeelden:
Zin: “Minaretten waren het symbool bij uitstek van de overheersende macht van de islam.” → 1
Zin: “De aandacht is daarbij in de eerste plaats gericht op de grootste vervoerstromen en knelpunten.” → 0
Zin: “Een persoonlijke zaakbehandelaar heeft gemiddeld met 5 tot 10 ouders op hetzelfde moment contact.” → 1
Zin: “Huurders van seniorenwoningen hoeven niet per definitie aangemoedigd te worden door te stromen naar een andere (senioren)woning.” → 0

Nu jij:
Zin: “{item}”
Antwoord alleen met 0 of 1:
"""

print(detect_bias("de minister voor jeugd en gezin is verantwoordelijk voor de extra tegemoetkoming voor alleenverdienershuishoudens met een thuis- wonend gehandicapt kind; dit is een onderdeel van de tog.", template))
#print(detect_bias("de zon schijnt heel fel", template))
