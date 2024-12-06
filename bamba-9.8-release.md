# Announcing Bamba

<p align="center">
  <img src="https://github.com/foundation-model-stack/bamba/blob/main/bamba.jpeg" alt="Bamba" width="400" height="400">
</p>

Standard `transformer` models are being adopted and deployed in production settings, where memory bandwidth bottleneck has emerged as a key challenge. The bottleneck is due to the per token decoding step which is very light on compute, but bottlenecked for short sequences by moving weights between memory and compute and for longer sequences moving KV-cache between memory and compute. As longer sequence models (e.g., Meta Llama3.1 is 128K, IBM Granite code v2 is 128K, Mistral large is 32k) are becoming popular due to the demands of applications, KV-cache bottleneck dominates. The key reason for KV-cache growth is the full attention layer, which results in linear growth in KV-cache with sequence length. While there are approaches to compress the KV-cache via lower precision, layer pruning, and compression, it does not fundamentally eliminate the problem. A new class of architectures for keeping KV-cache constant have emerged (e.g., Mamba2, DeltaNet, Linear attention) with the most promising of them being the Mamba layer. We have seen some proof points emerge in the last year (e.g., NVIDIA Mamba2, Codestral Mamba, Jamba, Samba, etc.).

We introduce Bamba, another proof point that improves on the existing SoTA Mamba models in its size and closes the gap further with SoTA transformer models. Inspired by AllenAI, in this collaboration between IBM, Princeton, and UIUC, we provide the entire lineage of data for training, multiple checkpoints, and the code for pretraining. We also enable the inference of this model in key OSS communities - Hugging Face `transformers`, `TRL`, `vLLM`, and `llama.cpp` to allow developers to use the model from the get go. We will share the details of our learnings when training this model and we welcome the community to help further close the gap with SoTA open source models and bring Mamba architecture to mainstream models and alleviate the KV-cache bottleneck.

## Evaluations

We break our evaluations into three parts:
1. Comparison with Mamba architecture based lnaguage models
2. Comparison with transformers trained to similar tokens
3. Comparison with SoTA transformer models of similar size

We use a local copy of the [Open LLM leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) for our benchmarks, whereas for other models, we use the output from the Live leaderboard (we could not run this for NVIDIA Hybird Mamba model as the model weights are not in Hugging Face transformers compatible format, hence we report the numbers from the paper).

TL;DR
We find that Bamba9B adds another proof point to similar models such as NVIDIA Mamba2, Zamba, and Falcon Mamba, while providing the entire data lineage. This will allow the community to surgically improve the model further. We also compare to similar sized transformer models and observe that our model outperforms since we use newer training techniques and better quality data. Compared to SoTA transformer models, we observe that we are on parity on various tasks and we believe that for those that we have gaps, it is due to lack of quality training data. We plan to continue training the current versions with newer datasets like Olmo2 mix and SFT datasets such as Tuluv3, agent instruct, and Anteater.

### Comparison with Hybrid Architectures
Several Mamba based architecture models have started coming up in the last 6months (e.g., NVIDIA Hybrid Mamba2, Codestral Mamba, Falcon Mamba, Zamba7Bv1) furthering the performance of these architectures and demonstrating their inference performance as well as closing the gap with quality. We compare 8 key benchmarks across Bamba, NVIDIA Hybrid model, Zamba, and Falcon Mamba. Falcon Mamba is a pure Mamba model, Zamba has shared attention layer for every 6 Mamba layers, and Bamba and NVIDIA are both Hybrid models with full attention layers interspersed with Mamba layer. While Falcon Mamba performs the best overall and has been trained to 5.5T tokens, there are open questions on how well copying tasks work on such pure Mamba models. Zamba was trained on fewer tokens (1T), but with a different Hybrid architecture. Bamba and NVIDIA Mamba are quite similar to each other (details on differences are summarized in the model architecture section), but Bamba is trained to 2.2T and NVIDIA Hybrid Mamba is trained to 3.5T tokens. The key point is that even with all these architectural variations and different number of tokens, Mamba based models are demonstrating competitive results. We are continuing to train the Bamba model with latest datasets and plan to release future checkpoints as the model gets better.

<p align="center">
  
| Benchmark score   | Bamba 9B   | NVIDIA Mamba2 Hybrid 8B | Zamba 7B   | Falcon Mamba 7B   |
|-------------------|------------|-------------------------|------------|-------------------|
| MMLU (5-shot)     | 60.77      | 53.6                   | 57.85      | **63.19**         |
| Hellaswag         | 81.8       | 77.69                  | **82.27**  | 80.82             |
| Winogrande        | 76.87      | 71.27                  | **79.32**  | 78.14             |
| Piqa              | 82.26      | 79.65                  | 82.21      | **83.62**         |
| OpenbookQA        | 47.6       | 42.8                   | 46.8       | **47.8**          |
| ARC-C             | 63.23      | 47.7                   | 55.38      | **63.4**          |
| TruthfulQA        | 49.21      | 38.72                  | 49.69      | **53.46**         |
| **Average**       | 65.96      | 58.78                  | 64.79      | **67.2**          |
</p>

## Comparison with transformers with similar token budget
We pick a few promiment models: Olmo 7B trained on identical data (2024), Meta Llama2 7B (2023), and IBM Granite 7B (2023), which have been trained to 2T tokens. While Olmo 7B outperforms Meta Llama2 and IBM Granite models across these 8 benchmarks, we note that with the same dataset, Bamba outperforms Olmo 7B. The main takeaway is that the Bamba model does well on the same dataset and similar token budget transformer models.

| Benchmark score   | Bamba 9B   | Olmo1.5 7B   | Meta Llama2 7B   | IBM Granite 7B   |
|-------------------|------------|--------------|------------------|------------------|
| MMLU (5-shot)     | **60.77**  | 53.39        | 46.87            | 49.02            |
| Hellaswag         | **81.8**   | 78.65        | 78.59            | 77.0             |
| Winogrande        | **76.87**  | 72.77        | 74.03            | 70.17            |
| Piqa              | **82.26**  | 78.4         | 79.0             | 80.14            |
| OpenbookQA        | 47.6       | **50.2**     | 44.0             | 40.8             |
| ARC-C             | **63.23**  | 48.5         | 53.07            | 49.91            |
| TruthfulQA        | **49.21**  | 36.0         | 38.76            | 38.7             |
| **Average**       | **65.96**  | 59.7         | 59.19            | 57.96            |

### Comparison with SoTA transformer models

Finally, we compare with 
We also compare the model with SoTA OSS models of the same size and there are obvious benchmark gaps. However, we note that architecturally the changes are minimal (e.g., Meta Llama changed from MHA to GQA, IBM Granite v3 added `mup`), but the data quality has significantly improved resulting in better scores. We plan to incorporate the improved data in our future iterations of Bamba to further close the gap with SoTA OSS models.

| Benchmark score | Bamba 9B | Meta Llama 3.1 8B | IBM Granite v3 8B | Olmo2 7B |
|-----------------|------------------|------------------|------------------|----------|
| MMLU           | 60.77            | 66.7           | 65.54           | 63.7     |
| MMLU PRO       |      25.77           | _37.1_           | 33.27           | 31       |
| BBH        |      40.16           | 47.8            | 34.45           | _50.4_   |
| Hellaswag      | 81.8              |                  | 83.61           | _83.8_   |
| Winogrande     | 76.87            | 60.5            | _80.9_          | 77.2     |
| SocialIQA      | 52.35           | 49.5            | _67.8_          |   51.33*       |
| Piqa           | 82.26           | 81              | _82.32_         |    81.07*      |
| OpenbookQA     | 47.6              | 45              | _46.8_          |    46.2*      |
| ARC-C          | 63.23            | 79.7            | 63.4            | _79.8_   |
| TruthfulQA     | 49.21            |                  | _52.89_         |    43.32*      |

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

<p align="center">
<img src="https://github.com/user-attachments/assets/0bc03608-fc3d-4886-b746-9839c52261d5" alt="Datamix" width="600" height="400">
</p>

We used a cosine learning rate schedule, with a peak learning rate of `3e−4`, a quadratic warmup over 2000 steps, decay factor of 0.033, and an ending learning rate of `1e−5` over 2T tokens. We use the AdamW optimizer with `β1` of 0.9 and `β2` of 0.95. We use a weight decay of 0.1, sequence length of 4096, and a global batch size of 1.5M tokens/batch.

We also performed a second phase training with high quality data from Hugging Face FineWeb-edu and Cosmopedia for an additional 200B tokens. We use a learning rate of 2e−5 and a cosine schedule to anneal the model, which helps improve the scores for our evaluation.


## Artifacts

