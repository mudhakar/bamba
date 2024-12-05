# Announcing Bamba -- Reduce Inference Latencies

<div style="text-align: center;">
  <img src="https://github.com/foundation-model-stack/bamba/blob/main/bamba.jpeg" alt="Bamba" width="400" height="400">
</div>

Standard `transformer` models are being adopted and deployed in production settings, where memory bandwidth bottleneck has emerged as a key challenge. The bottleneck is due to the per token decoding step which is very light on compute, but bottlenecked for short sequences by moving weights between memory and compute and for longer sequences moving KV-cache between memory and compute. As longer sequence models (e.g., Meta Llama3.1 is 128K, IBM Granite code v2 is 128K, Mistral large is 32k) are becoming popular due to the demands of applications, KV-cache bottleneck dominates. The key reason for KV-cache growth is the full attention layer, which results in linear growth in KV-cache with sequence length. While there are approaches to compress the KV-cache via lower precision, layer pruning, and compression, it does not fundamentally eliminate the problem. A new class of architectures for keeping KV-cache constant have emerged (e.g., Mamba2, DeltaNet, Linear attention) with the most promising of them being the Mamba layer. We have seen some proof points emerge in the last year (e.g., NVIDIA Mamba2, Codestral Mamba, Jamba, Samba, etc.).

We introduce Bamba, another proof point that improves on the existing SoTA Mamba models in its size and closes the gap further with SoTA transformer models. Inspired by AllenAI, in this collaboration between IBM, Princeton, and UIUC, we provide the entire lineage of data for training, multiple checkpoints, and the code for pretraining. We also enable the inference of this model in key OSS communities - Hugging Face `transformers`, `TRL`, `vLLM`, and `llama.cpp` to allow developers to use the model from the get go. We will share the details of our learnings when training this model and we welcome the community to help further close the gap with SoTA open source models and bring Mamba architecture to mainstream models and alleviate the KV-cache bottleneck.

## Evaluations

We break our evaluations into three parts:
1. Comparison with SoTA hybrid model of similar size as well as generic transformer models of similar size and number of tokens.
2. Comparison with SoTA transformer models of similar size.
3. Controlled ablation with transformer

TL;DR
We find that Bamba9B outperforms other similar sized Hybrid models and transformer models trained to the same number of tokens by 5-6 points on average across 8 key benchmarks. 
### Comparison with
Bamba outperforms similar sized Hybrid Mamba model from NVIDIA, outperforms the Olmo pure transformer model trained on the same data and Meta Llama2 7B, IBM Granite 7B trained to similar number of tokens.

| Benchmark score | Bamba 9B | NVIDIA Mamba2 Hybrid 8B | Olmo1.5 7B | Meta Llama2 7B | IBM Granite 7B |
|-----------------|----------|-------------------------|------------|----------------|----------------|
| MMLU (5-shot)   | _59.2_   | 53.6                   | 52         | 47             | 50             |
| Hellaswag       | _80.0_   | 77.69                  | 75.5       | 76             | 74             |
| Winogrande      | _73.6_   | 71.27                  | 69.8       | 69.0           | 67             |
| Piqa            | _81.77_  | 79.65                  | 77.5       | 79             | 79             |
| OpenbookQA      | 48       | 42.8                   | _50.0_     | 44             | 42             |
| ARC-C           | _56.1_   | 47.7                   | 42.5       | 46             | 44             |
| TruthfulQA      | _49.1_   | 38.72                  | 35.8       | 39             | 39             |
| **Average**     | _64.54_    | 58.49                  | 57.30      | 57.14          | 56.43          |


We also compare the model with SoTA OSS models of the same size and there are obvious benchmark gaps. However, we note that architecturally the changes are minimal (e.g., Meta Llama changed from MHA to GQA, IBM Granite v3 added `mup`), but the data quality has significantly improved resulting in better scores. We plan to incorporate the improved data in our future iterations of Bamba to further close the gap with SoTA OSS models.

| Benchmark score | Bamba 9B | Meta Llama 3.1 8B | IBM Granite v3 8B | Olmo2 7B |
|-----------------|------------------|------------------|------------------|----------|
| MMLU           | 59.2            | _66.7_           | 65.54           | 63.7     |
| MMLU PRO       |                 | _37.1_           | 33.27           | 31       |
| AGIEval        |                 | 47.8            | 34.45           | _50.4_   |
| Hellaswag      | 80              |                  | 83.61           | _83.8_   |
| Winogrande     | 73.6            | 60.5            | _80.9_          | 77.2     |
| SocialIQA      | 52.35           | 49.5            | _67.8_          |   51.33*       |
| Piqa           | 81.77           | 81              | _82.32_         |    81.07*      |
| OpenbookQA     | 48              | 45              | _46.8_          |    46.2*      |
| ARC-C          | 56.1            | 79.7            | 63.4            | _79.8_   |
| TruthfulQA     | 49.1            |                  | _52.89_         |    43.32*      |

While these results are promising, we invite the community to help improve the model further and identify any fundamental limitations in this inference efficient model.

## Inference efficiency


## Model architecture
We base our model architecture on the NVIDIA Hybrid Mamba2 with the following changes.
| Parameter | Bamba 9B | NVIDIA Hybrid Mamba2 8B |
|---------|-------|-------|
| Num layers | 32 | 29 |
| Num Mamba2 layers | 3 | 4 |
| MLP expansion factor | 3.5 | 4 |
| Vocab size | 128k | 256k |
| Non-embedding parameters | 8.8B | 8.6B |
| RoPE | yes | no |
| Gated linear units | yes | no |

We have a total of 8B parameters in the Mamba2 layers, 800M in full attention layers, and 1B in embeddings. The hidden state is 4K, GQA for full attention with 8 KV-heads and 32 heads, Mamba2 layer head dimension is 64, and convolution filter size is 4.

## Pre-Training
Pre-training Bamba was done in a phased manner, we performed ablation experiments at 1.8B model size and a few 100B tokens to determine the right learning rates and built on the previous community efforts - significant hyperparamters were borrowed from the Mamba2 paper and repository. Based on the promising results from this study, we scaled the model to 3B and 2T tokens using Dolma mix. We also trained a 3B transformer model following Meta Llama architecture with the same data mix and observed similar or better performance from the Bamba model.

Finally, we scaled the model to 9B size and leveraged PyTorch FSDP to train the model. 

For data, we use Dolma v1.7 with the data mix used illustrated in the below figure.

<img src="https://github.com/user-attachments/assets/0bc03608-fc3d-4886-b746-9839c52261d5" alt="Datamix" width="600" height="400">

We used a cosine learning rate schedule, with a peak learning rate of `3e−4`, a quadratic warmup over 2000 steps, decay factor of 0.033, and an ending learning rate of `1e−5` over 2T tokens. We use the AdamW optimizer with `β1` of 0.9 and `β2` of 0.95. We use a weight decay of 0.1, sequence length of 4096, and a global batch size of 1.5M tokens/batch.

We also performed a second phase training with high quality data from Hugging Face FineWeb-edu and Cosmopedia for an additional 200B tokens. We use a learning rate of 2e−5 and a cosine schedule to anneal the model, which helps improve the scores for our evaluation.


## Artifacts

