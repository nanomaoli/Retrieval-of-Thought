# RoT: Reasoning on Template

This project provides code for the paper "Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts" on math and science benchmarks (AIME 2023/2024/2025, AMC23, MATH500, GPQA) using vLLM for inference. 

Each runner script (`run_vllm_all_models_think*.py`) starts a vLLM server for the specified model, runs the evaluation pipeline against the configured datasets, and prints an evaluation summary at the end.

## Requirements

Python 3.10+, vLLM, PyTorch, sentence-transformers, openai, networkx, scikit-learn, psutil, pyyaml, tqdm.

A GPU with sufficient VRAM is required to serve models via vLLM.

## Usage

**Required arguments:** `--model` (HuggingFace model name) and `--max-num-seqs` (parallel sequences).

### Qwen3 (default, math datasets)

```bash
python3 -u run_vllm_all_models_think.py --model Qwen/Qwen3-14B --max-num-seqs 3
```

### Phi-4

```bash
python3 -u run_vllm_all_models_think_phi4.py --model microsoft/Phi-4-reasoning --max-num-seqs 3
```

### DeepSeek-R1

```bash
python3 -u run_vllm_all_models_think_r1.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max-num-seqs 3
```

### DeepSeek-R1 (DLER variant)

```bash
python3 -u run_vllm_all_models_think_dler.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max-num-seqs 3
```

### Instruct models (no thinking)

```bash
python3 -u run_vllm_all_models_instr.py
```


## Optional Flags

`--use_reranker` enables a Qwen3 cross-encoder reranker for graph retrieval. `--use_beam` enables beam-based graph search with the reranker. `--use_rag` uses RAG-based template retrieval instead of the knowledge graph. `--enable-lora --lora-path <path>` attaches a LoRA adapter. `--tp_size <N>` sets tensor parallel size for multi-GPU. `--vanilla_break` exits after running vanilla (no-graph) experiments only.

Example with reranker and LoRA:

```bash
python3 -u run_vllm_all_models_think.py --model Qwen/Qwen3-14B --max-num-seqs 3 --use_reranker --enable-lora --lora-path ./lora_weights --tp_size 2
```

## Eval Datasets

Problem files are stored in `eval_datasets/` as JSON. Each file contains a list of problems with `ID`, `Problem`, `Answer`, and optional `knowledge_tags` and `options` fields.

## Results

Results are written to `vllm_experiments/results/<dataset>/` with separate outputs for graph-based and vanilla runs. Logs go to `vllm_think_*.log` in the project root.

## Cite 

```
@inproceedings{
ahmed2026retrievalofthought,
title={Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts},
author={Ammar Ahmed and Azal Ahmad Khan and Ayaan Ahmad and Sheng Di and Zirui Liu and Ali Anwar},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Wy7NyScKlD}
}
```