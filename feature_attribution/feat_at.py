from inseq import SequenceAttributionPipeline, load_model
from transformers import MT5ForConditionalGeneration

model_name = "google/mt5-base"
model_path = "../ft_mt5/results/google-mt5-base_os_f1_0.7492"
model_ft_mt5 = load_model(model_path, "seq2seq")
model_normal = MT5ForConditionalGeneration.from_pretrained(model_name)
examples = {"train_dataset":'mens (crm) oordeelde dat een onderwijsstichting geen verboden onderscheid maakte door een meisje met down syndroom niet door te laten gaan op de reguliere school (oordeelnummer 2011–144).'
            , 'test_dataset':'het traject is niet verplicht voor eu-migranten en turkse migranten, aangezien zij niet inburgeringsplichtig zijn.'}
for i in examples:
    pipe_normal = SequenceAttributionPipeline.from_model(model_normal, method="integrated_gradients")
    attributions_normal=pipe_normal.attribute(examples[i], return_dict=True, generation_args={"max_new_tokens": 5})
    pipe_normal.save_attributions(examples[i], attributions_normal, "attributions_normal_mt5.json")
    pipe_ft_mt5=SequenceAttributionPipeline.from_model(model_ft_mt5
                                                       , method="integrated_gradients")
    attributions_ft=pipe_ft_mt5.attribute(examples[i], return_dict=True, generation_args={"max_new_tokens": 5})
    pipe_ft_mt5.save_attributions(examples[i], attributions_ft, "attributions_ft_mt5.json")