# Bamba

<p align="center">
  <img src="/bamba.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/ibm-fms/Avengers-Mamba2-9B"> Bamba on Hugging Face</a>&nbsp | <a href="TODO"> Bamba Blog</a>&nbsp
<br>

Bamba is a repository for training and using [Bamba](https://huggingface.co/ibm-fms/Avengers-Mamba2-9B) models which are based on [Mamba](https://github.com/state-spaces/mamba) models.


## Installation

Besides [PyTorch](https://pytorch.org/), you would need a few [extra dependencies](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation) for
Mamba models.

We found some of these dependencies picky on PyTorch versions when doing pip install, so 
the best way is to build from source for all Mamba dependencies if you hit dependency 
issue with your env:
```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && pip install . && cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install . && cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && pip install . && cd ..
```


## Models

### Overview
TODO: add model card here

### Checkpoints
We have published our model checkpoints here: TODO: add mamba HF page once public


## Inference
You can utilize our newly contributed HF integration to run inference on our Bamba models:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-fms/Avengers-Mamba2-9B-hf")
tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Avengers-Mamba2-9B-hf")

message = ["TODO: find a prompt here"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

```


## Training

We trained our Bamba model with FSDP using our training repo [here](https://github.com/foundation-model-stack/fms-fsdp/tree/mamba-new).
Note that this training effort was started before FSDP2 and also long before we contributed
`Mamba2-Hybrid` to HF, so we were doing FSDP1 training with [official Mamba implementation](https://github.com/state-spaces/mamba).
For users trying to reproduce the training you now have much more options with our newly
contributed [HF-version of Mamba2-Hybrid]() (TODO: add link once live).


## Fine-tuning

The below example shows an example of using CodeAlpaca dataset to fine tune the bamba model. 
We will leverage [SFT TRainer](https://huggingface.co/docs/trl/en/sft_trainer#supervised-fine-tuning-trainer) for the same. 

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

# TO DO: add model path, once known from HF
model = AutoModelForCausalLM.from_pretrained("")
tokenizer = AutoTokenizer.from_pretrained("")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="/tmp"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
```

The below snippet shows how to do LoRA tuning with the same model.

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

# TO DO: add model path, once known from HF
model = AutoModelForCausalLM.from_pretrained("")
tokenizer = AutoTokenizer.from_pretrained("")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="/tmp"),
    peft_config=peft_config
)

trainer.train()
```

## Evaluation



