from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset, Dataset, load_metric
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate
import numpy as np
import re
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import torch.nn.functional as F
import pandas as pd


def sample_data(df, strategy="undersample", oversample_factor=2, undersample_ratio=0.7, balanced_neg_ratio=0.5, random_state=42):
    biased = df[df.label == 1]
    unbiased = df[df.label == 0]

    if strategy == "undersample":
        # Undersample the unbiased (majority) class to match a specified ratio
        unbiased_sampled = unbiased.sample(frac=undersample_ratio, random_state=random_state)
        return pd.concat([biased, unbiased_sampled])

    elif strategy == "oversample":
        # Duplicate the biased (minority) class oversample_factor times
        return pd.concat([biased] * oversample_factor + [unbiased])

    elif strategy == "balanced":
        # Target 50% biased, 50% unbiased in the final dataset (or as specified by balanced_neg_ratio)
        target_total = len(df)  # preserve the original size
        unbiased_target = int(target_total * balanced_neg_ratio)  # target number of unbiased samples  
        biased_target = target_total - unbiased_target  # target number of biased samples 

        # Sample unbiased samples to match the target
        neg_sampled = unbiased.sample(n=unbiased_target, random_state=random_state)

        # Compute the number of times to duplicate all biased samples
        repeats = biased_target // len(biased)
        # Compute the remainder of biased samples to sample to reach target
        remainder = biased_target % len(biased)

        # Sample biased samples to match the target based on the number of repeats and remainder
        biased_repeated = pd.concat([biased] * repeats + [biased.sample(n=remainder, random_state=random_state)])
        return pd.concat([biased_repeated, neg_sampled])

    elif strategy == "normal":
        return df

    else:
        raise ValueError("Unsupported strategy.")
    
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset("milenamileentje/Dutch-Government-Data-for-Bias-detection")
def preprocess(example, makeitwords=True):
    model_inputs = tokenizer(example['text'], truncation=True, padding="max_length", max_length=512)
    with tokenizer.as_target_tokenizer():
        if makeitwords:
            if example["label"] == 0:
                label = "niet-biased"
                labels = tokenizer(label, truncation=True, padding="max_length", max_length=512)
            elif example["label"] == 1: 
                label = "biased"
                labels = tokenizer(label, truncation=True, padding="max_length", max_length=512)
        else:
            labels = tokenizer(str(example["label"]), truncation=True, padding="max_length", max_length=512) 
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

#small_train_dataset = dataset["train"].select(range(1))
train_dataset = dataset["train"].to_pandas()
sampled_df = sample_data(train_dataset, strategy='undersample')
train_dataset = Dataset.from_pandas(sampled_df, preserve_index=False)
test_dataset = dataset["test"]
tokenized_test = test_dataset.map(preprocess)
tokenized_train = train_dataset.map(preprocess)
#small_val_dataset = dataset["validation"].select(range(1))
val_dataset = dataset["validation"].to_pandas()
tokenized_val = val_dataset.map(preprocess)
#tokenized_train = small_train_dataset.map(preprocess)
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#tokenized_val = small_val_dataset.map(preprocess)
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
accuracy = load_metric("accuracy")

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    #eval_strategy="epoch",
    learning_rate=5e-5,                     
    per_device_train_batch_size=1,         
    per_device_eval_batch_size=1,         
    num_train_epochs=1,                   
    weight_decay=0.0,                      
    save_strategy="epoch",                    
    logging_dir="./logs",
    logging_steps=1,                       
    report_to="none" 
    #load_best_model_at_end=True,
    #metric_for_best_model="accuracy",
    #generation_max_length=2
    )
training_args.generation_config = None 
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

if __name__ =="__main__":
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    trainer.save_model("mt5-base-finetuned")
    predictions = trainer.predict(test_dataset)
    generated_ids = predictions.predictions
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    result= accuracy.compute(predictions=decoded_preds, references=test_dataset["labels"])
    print("Accuracy: ", result)

    print("Model saved.")
