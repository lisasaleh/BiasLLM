from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "your-huggingface-repo/aya-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

prompt = (
    "Je bent een bias-detector voor Nederlandse overheidsdocumenten.\n"
    "Beoordeel of de volgende tekst impliciet of expliciet bevooroordeeld is.\n"
    "Antwoord alleen met “0” voor geen bias of “1” voor bias, zonder extra tekst.\n\n"
    "Tekst:\n“" + sample_text + "”"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(output[0], skip_special_tokens=True))
