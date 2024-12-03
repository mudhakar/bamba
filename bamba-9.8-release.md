# Bamba -- Reducing Inference Latencies
Standard `transformer` models are being adopted and deployed in production settings, where memory bandwidth bottleneck has emerged as a key challenge. The bottleneck is due to the per token decoding step which is very light on compute, but bottlenecked for short sequences by moving weights between memory and compute and for longer sequences moving KV-cache between memory and compute. As longer sequence models (e.g., Meta Llama3.1 is 128K, IBM Granite code v2 is 128K, Mistral large is 32k) are becoming popular due to the demands of applications, KV-cache bottleneck dominates. The key reason for KV-cache growth is the full attention layer, which results in linear growth in KV-cache with sequence length. While there are approaches to compress the KV-cache via lower precision, layer pruning, and compression, it does not fundamentally eliminate the problem. A new class of architectures for keeping KV-cache constant have emerged (e.g., Mamba2, DeltaNet, Linear attention) with the most promising of them being the Mamba layer. We have seen some proof points emerge in the last year (e.g., NVIDIA Mamba2, Codestral Mamba, Jamba, Samba, etc.).

We introduce Bamba, another proof point that improves on the existing SoTA Mamba models in its size and closes the gap further with SoTA transformer models. Inspired by AllenAI, in this collaboration between IBM, Princeton, and UIUC, we provide the entire lineage of data for training, multiple checkpoints, and the code for pretraining. We also enable the inference of this model in key OSS communities - Hugging Face `transformers`, `TRL`, `vLLM`, and `llama.cpp` to allow developers to use the model from the get go. We will share the details of our learnings when training this model and we welcome the community to help further close the gap with SoTA open source models and bring Mamba architecture to mainstream models and alleviate the KV-cache bottleneck.

## Evaluations

| Benchmark score | Bamba 9.8B 2.2T | NVIDIA Mamba2 Hybrid 8B 3.5T | Olmo1.5 7B 2T |
|-----------------|------------------|-----------------------------|----------------|
| MMLU           | _59.2_          | 53.6                        | 52             |
| Hellaswag      | _80.0_          | 77.69                       | 75.5           |
| Winogrande     | _73.6_          | 71.27                       | 69.8           |
| SocialIQA      | 52.4         | n/a                         | n/a            |
| Piqa           | _81.77_         | 79.65                       | 77.5           |
| OpenbookQA     | 48              | 42.8                        | _50.0_         |
| ARC-C          | _56.1_          | 47.7                        | 42.5           |
| TruthfulQA     | _49.1_          | 38.72                       | 35.8           |


## Training

## 
