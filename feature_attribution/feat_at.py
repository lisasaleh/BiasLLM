from inseq import load_model

model_name        = "google/mt5-base"
model_path        = "../ft_mt5/results/google-mt5-base_os_f1_0.7492"
# load both models with the same attribution method
model_normal = load_model(model_name, method="integrated_gradients")
model_ft_mt5 = load_model(model_path, method="integrated_gradients")

examples = {
  "train": "mens (crm) oordeelde dat een onderwijsstichting geen verboden " 
           "onderscheid maakte door een meisje met down syndroom niet door te laten gaan op de reguliere school (oordeelnummer 2011–144).",
  "test":  "het traject is niet verplicht voor eu-migranten en turkse migranten, aangezien zij niet inburgeringsplichtig zijn."
}

for split, text in examples.items():
    # attribute with a small generative budget
    out_norm = model_normal.attribute(
        text,
        generation_args={"max_new_tokens": 5}
    )
    out_ft   = model_ft_mt5.attribute(
        text,
        generation_args={"max_new_tokens": 5}
    )

    # save to disk as JSON
    out_norm.to_json(f"attributions_normal_{split}.json")
    out_ft.to_json(  f"attributions_ft_{split}.json")
