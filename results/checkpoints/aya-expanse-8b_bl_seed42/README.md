---
base_model: CohereLabs/aya-expanse-8b
library_name: peft
---

# Model Card for Aya Expanse 8B with LoRA Finetuning

<!-- Provide a quick summary of what the model is/does. -->

This model card documents a fine-tuned version of the open-source multilingual language model **Aya Expanse 8B**, adapted for **bias detection in Dutch governmental texts**. The model was fine-tuned using **Low-Rank Adaptation (LoRA)** to reduce memory consumption and training cost.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Aya Expanse 8B is a multilingual causal language model developed by Cohere for broad generative tasks. In this project, we apply LoRA-based parameter-efficient finetuning for a binary classification task: detecting biased vs. non-biased sentences in Dutch political and administrative texts, where examples are manually annotated for the presence of bias.

- **Developed by:** Cohere (base model), extended by our project team
- **Model type:** Causal Language Model (finetuned for classification)
- **Language(s) (NLP):** Multilingual (focus: Dutch)
- **License:** Apache 2.0 (Cohere)
- **Finetuned from model:** CohereLabs/aya-expanse-8b

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** (https://huggingface.co/CohereLabs/aya-expanse-8b)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Not intended for direct use without fine-tuning for classification purposes.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Suitable for tasks involving social bias detection in Dutch or other low-resource settings when combined with LoRA or prompt-tuning methods.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Not suitable for generating factually reliable or unbiased content.
- Not designed for high-stakes decision-making.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Model may reinforce existing social or institutional biases. Outputs depend on both pretraining and fine-tuning datasets.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("CohereLabs/aya-expanse-8b")
model = PeftModel.from_pretrained(base_model, "aya_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/aya-expanse-8b")
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- Dataset: Dutch governmental documents (public dataset: milenamileentje/Dutch-Government-Data-for-Bias-detection)
- Annotated labels: binary (ja = biased, nee = not-biased)

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

LoRA-based fine-tuning on top of the Aya model:

#### Preprocessing

<!-- Prompt-based formatting of inputs, binary labels (`ja` for biased, `nee` for not-biased), tokenized and truncated at 512 tokens. -->
- Tokenized prompts using instruction format
- Appended target labels to prompt (causal LM setup)

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->
- **Epochs:** 3
- **Batch size:** 2 (train) / 1 (eval)
- **Learning rate:** 1e-3
- **Gradient checkpointing:** Enabled
- **Adapter method:** LoRA (r=8, alpha=16, dropout=0.1)
- **Target modules modified via LoRA:** o_proj, k_proj, q_proj, up_proj, v_proj, gate_proj, down_proj

<!--#### Speeds, Sizes, Times [optional] -->

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

- **Dataset:** Test split of [Dutch-Government-Data-for-Bias-detection (DGDB)](https://huggingface.co/datasets/milenamileentje/Dutch-Government-Data-for-Bias-detection)

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The evaluation results were disaggregated by:
- **Sampling strategy:** undersampling, oversampling, balanced, no sampling
- **Prompt style:** training prompt vs chain-of-thought (CoT) prompt
- **Class performance:** separate metrics for "Biased" (class 1) and "Unbiased" (class 0)

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **Accuracy**: Overall correct classifications.
- **F1 Score (Macro)**: Mean F1 over both classes; used to balance class imbalance.
- **Precision (Macro)**: Precision across both classes.
- **Recall (Macro)**: Recall across both classes.
- **F1 Score per class**: Separate F1 scores for class 0 (Unbiased) and class 1 (Biased) to evaluate sensitivity to bias detection.

### Results

#### Zero-shot Results (no fine-tuning)

| Metric               | mT5     | Aya Expanse 8B |
|----------------------|---------|----------------|
| Accuracy             | 0.3696  | **0.7249**     |
| F1 (Macro)           | 0.3691  | **0.6617**     |
| Precision (Macro)    | 0.4819  | **0.6598**     |
| Recall (Macro)       | 0.4831  | **0.6640**     |
| F1 Class 0 (Unbiased)| 0.3064  | **0.8079**     |
| F1 Class 1 (Biased)  | 0.4113  | **0.5155**     |

#### LoRA-Fine-Tuned Aya (Balanced Sampling Strategy)

| Metric               | Value   |
|----------------------|---------|
| Accuracy             | 0.7862  |
| F1 (Macro)           | 0.6912  |
| Precision (Macro)    | 0.7390  |
| Recall (Macro)       | 0.6734  |
| F1 Class 0 (Unbiased)| 0.8625  |
| F1 Class 1 (Biased)  | 0.5200  |

#### Best Sampling Strategy Comparison (F1 Macro)

| Model (Strategy)        | F1 (Macro) |
|-------------------------|------------|
| BERTje (undersampling)  | **0.812**  |
| mT5 (undersampling)     | 0.769      |
| Aya + LoRA (balanced)   | 0.691      |
| GPT-3.5 (prompted)      | 0.559      |

#### Prompt Style Comparison (Aya)

| Prompt Type | Sampling     | Accuracy | F1 (Macro) | F1 Class 0 | F1 Class 1 |
|-------------|--------------|----------|------------|-------------|-------------|
| Training    | Balanced     | 0.7612   | **0.6545** | 0.8465      | 0.4624      |
| CoT         | Balanced     | 0.7606   | 0.6197     | 0.8512      | 0.3882      |


### Summary

The approach shows that multilingual LLMs can be effectively adapted to classification tasks in specific low-resource domains using parameter-efficient tuning methods.

## Model Examination

<!-- Relevant interpretability work for the model goes here -->

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA A100 80GB GPU  
- **Hours used:** ~0.4 GPU hours (3 epochs, ~7 min/epoch)  
- **Cloud Provider:** SURF Research Cloud
- **Compute Region:** Europe (Netherlands)  
- **Carbon Emitted:** ~0.04 kgCO₂eq (based on calculator estimate for A100 running for 24 minutes at high load)

## Technical Specifications

### Model Architecture and Objective

Aya Expanse 8B is an auto-regressive language model that uses an optimized transformer architecture, optimized for multilingual performance. Post-training includes supervised finetuning, preference training, and model merging.
For this task, the model was LoRA-fine-tuned for binary classification (biased vs. unbiased) using a prompt-based approach. The LoRA configuration used rank `r=8`, `lora_alpha=16`, and `lora_dropout=0.1`, and it targeted the main linear projection layers of the transformer.

### Compute Infrastructure

#### Hardware

- GPU: NVIDIA A100 80GB  
- CPU: N/A 
- RAM: 85–100 GB estimated (for Aya and dataset in memory)

#### Software

- Python 3.10  
- Transformers: v4.39.3  
- PEFT: v0.15.2  
- Datasets: v2.19.0  
- TRL (for SFTTrainer): v0.7.11  
- PyTorch: v2.2.2+cu121  
- CUDA: 12.1

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@misc{biasllm2025,
  title     = {{LLMs for Bias Detection: Open-Source Multilingual LLMs Can Detect Bias in Dutch Government Documents}},
  author    = {Adriana Haralambieva, Ina Klaric, Raya Mezeklieva, Weronika Orska, Lisa Saleh},
  year      = {2025},
  institution = {University of Amsterdam},
  note      = {ATICS Research Project},
  url       = {https://github.com/lisasaleh/BiasLLM}
}

**APA:**

Haralambieva, A., Klaric, I., Mezeklieva, R., Orska, W., & Saleh, L. (2025). LLMs for bias detection: Open-source multilingual LLMs can detect bias in Dutch government documents. University of Amsterdam. https://github.com/lisasaleh/BiasLLM

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning technique that adds trainable low-rank matrices to frozen pre-trained model weights.  
- **Balanced Sampling:** Ensuring equal class distribution during training.  
- **Zero-shot:** Model is evaluated without any fine-tuning on the target dataset.  
- **F1 (Macro):** The unweighted average of F1 scores per class.

## More Information
  
- Supervised by Vera Neplenbroek, Univeristy of Amsterdam

<!-- ## Model Card Authors [optional]

## Model Card Contact -->

### Framework versions

- PEFT 0.15.2
