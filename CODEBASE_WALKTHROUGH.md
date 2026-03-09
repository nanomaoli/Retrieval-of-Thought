# Retrieval-of-Thought Codebase Walkthrough

This guide is optimized for **fast onboarding**: understand the execution path first, then dive into retrieval logic, then model-specific prompting.


## 0) 10-minute quickstart (if you only have a short session)

1. Read `README.md` to understand supported run modes and flags.
2. Trace one execution path in `run_vllm_all_models_think.py`:
   - vLLM boot
   - eval subprocess launch
   - summary extraction and process teardown
3. Jump to `vllm_experiments/main.py` and follow only these functions/branches first:
   - argument parse + problem loading
   - graph/rag/vanilla branch select
   - template generation dispatch
   - `solve_problem(...)` call and result aggregation
4. Skim `vllm_experiments/src/graph_handler.py` for first-step node selection + traversal score.
5. Skim `vllm_experiments/src/llm_handler_qwen3_gpqa.py` for prompt construction and retry behavior.

If you complete the above, you will already understand most of the runtime behavior.

## 1) Big-picture mental model

The repository runs evaluation experiments where a vLLM-served model solves benchmark problems under multiple settings:

1. **Vanilla** (no template retrieval)
2. **Graph retrieval (RoT)**: retrieve a reasoning template from a knowledge graph and condition the LLM on it
3. **RAG retrieval**: retrieve a template from `rag_templates.json`
4. Optional **thinking intervention**, **reranker**, and **beam search** modes

At a high level, each run script:

- starts a vLLM server process,
- launches an evaluation script (`vllm_experiments/main*.py`),
- captures summary output,
- writes JSON results + logging + model metrics.

## 2) Start here: top-level runner scripts

Read in this order:

1. `README.md` (usage flags and run matrix)
2. `run_vllm_all_models_think.py` (main orchestration pattern)
3. Model variants: `run_vllm_all_models_think_phi4.py`, `run_vllm_all_models_think_r1.py`, `run_vllm_all_models_think_dler.py`
4. Non-thinking baseline: `run_vllm_all_models_instr.py`

What `run_vllm_all_models_think.py` does:

- Builds a `vllm serve ...` command with model/lora/tensor-parallel options.
- Optionally starts a reranker server (`Qwen3-Reranker-0.6B`) on port 8005.
- Calls `vllm_experiments/main.py` with CLI args for dataset, graph/rag toggles, and intervention mode.
- Runs the experiment grid over:
  - `vanilla in [False, True]`
  - `thinking_interven in [False, True]` (skips invalid vanilla+interven)
- Extracts and prints `--- Evaluation Summary ---` from subprocess output.

### Why this matters

If you can reason about `run_vllm_all_models_think.py`, you understand process lifecycle, kill/restart behavior, and experiment-level control flow.

## 3) Core evaluation pipeline (`vllm_experiments/main.py`)

This file is the **center of gravity**.

### 3.1 Parse config and load data

- CLI args come from `vllm_experiments/src/config.py`.
- Problems are read from JSON files in `eval_datasets/`.
- OpenAI-compatible client points to local vLLM endpoint (`http://localhost:8000/v1`).

### 3.2 Select retrieval mode

In `main.py`, retrieval branches are mutually exclusive by mode:

- **Graph mode (`--use_graph=True`)**
  - load GraphML graph
  - if reranker: precompute node text only
  - else: load embedding model + precompute node embeddings
- **RAG mode (`--use_rag=True`)**
  - load `rag_templates.json`
  - embed all templates once
- **Vanilla mode**
  - no retrieval template

### 3.3 Build per-problem tasks

For each problem:

- collect `ID`, `Problem`, `Answer`, optional `knowledge_tags`, optional multiple-choice `options`
- append options into prompt if present
- normalize task payload fields used by solver

### 3.4 For each task, retrieve template then solve

Template generation routes:

- `generate_dynamic_template(...)` for embedding-based graph retrieval
- `generate_dynamic_template_rerank(...)` for reranker-based retrieval
- `generate_dynamic_template_rerank_beam_single(...)` for reranker+beam
- `rag_top_1(...)` for RAG

Then `solve_problem(...)` sends prompt(s) to vLLM and returns:

- output text,
- token usage,
- extracted boxed answer,
- correctness flag.

### 3.5 Aggregate, report, and persist

- Computes accuracy over successful requests.
- Prints evaluation summary and token-per-problem diagnostics.
- Writes JSON to incremented output folder (`<path>_0`, `<path>_1`, ...).
- Pulls Prometheus metrics from vLLM (`curl localhost:8000/metrics`) and prints selected fields.

## 4) Retrieval internals (`vllm_experiments/src/graph_handler.py`)

Key ideas in graph retrieval:

1. **Metadata filtering first** (knowledge tags)
2. **First-step selection with `_s0` bonus**
3. **Iterative traversal** over graph successors
4. **Hybrid score** balancing query similarity + structural flow preference
5. **Termination guards** (thresholds, max length, no valid successor)

Important constants live in `vllm_experiments/src/VARS_gpqa.py` (or sibling VARS files for other setups):

- graph filename
- embedding model name
- similarity thresholds
- max template length
- first-step weighting
- token caps

### Reranker path

Reranker retrieval uses HTTP POST to local reranker service (`/score`) with batched query-document pairs, then applies the same start-node + traversal logic with reranker scores.

## 5) Prompting and answer extraction

### 5.1 LLM handler (`vllm_experiments/src/llm_handler_qwen3_gpqa.py`)

Responsibilities:

- construct system+user chat messages,
- tune generation parameters differently for thinking vs non-thinking paths,
- implement retry loop on API failures,
- inject retrieved template into a strict instruction scaffold,
- optionally perform “thinking intervention” continuation prompt style.

### 5.2 Utility functions (`vllm_experiments/src/utils.py`)

Key utilities to know:

- logging setup (`setup_logging`)
- answer extraction and validation (`extract_and_validate_answer_str` and numeric variant)
- metrics parsing/printing from Prometheus text format
- RAG helper (`rag_top_1`) for top-1 template retrieval

## 6) Data and artifacts map

- `eval_datasets/*.json`: benchmark inputs
- `knowledge_graph.graphml`, `gpqa_new_knowledge_graph.graphml`: graph retrieval assets
- `rag_templates.json` + `vllm_experiments/rag_templates.json`: RAG corpora
- `vllm_experiments/results/...`: experiment outputs
- root `vllm_think_*.log`: run logs

## 7) Efficient study plan (suggested 90-minute pass)

### Pass A (20 min): Control flow

- Read `README.md`.
- Trace one runner (`run_vllm_all_models_think.py`) from argument parse to subprocess call and summary extraction.

Goal: understand *when* servers start/stop and *what* flags change behavior.

### Pass B (30 min): Retrieval logic

- Read `vllm_experiments/main.py` retrieval branches.
- Read `vllm_experiments/src/graph_handler.py` focusing on:
  - first-step scoring,
  - traversal scoring,
  - stopping conditions,
  - reranker vs embedding path differences.

Goal: understand *how* a template is formed.

### Pass C (20 min): Prompting and scoring

- Read `llm_handler_qwen3_gpqa.py` prompt assembly and request retry policy.
- Read answer-extraction utilities in `utils.py`.

Goal: understand *how correctness is decided* and *what output format is required*.

### Pass D (20 min): Repro execution

Run a minimal experiment on one dataset and inspect one saved JSON result:

```bash
python3 -u run_vllm_all_models_think.py --model Qwen/Qwen3-14B --max-num-seqs 1
```

Then inspect the newest output under `vllm_experiments/results/gpqa/`.

## 8) Practical reading tips

- Don’t start with model-specific `main_phi4.py`/`main_r1.py` files; first master generic `main.py`.
- Keep a scratch table with columns: `flag`, `effect on retrieval`, `effect on prompt`, `effect on output filename`.
- When confused by behavior, search where a flag is consumed in both runner and `main.py`.

## 9) Common gotchas to notice while studying

- There are similar-but-separate files for different model families; behavior diverges subtly.
- Some constants are dataset/model specific (`VARS*.py`), so thresholds/token limits can differ by script.
- Result saving increments directory suffixes, so “latest run” is often in the highest `_N` folder.

---

If you want, the next step is I can generate a **one-page cheat sheet** of exactly which functions to breakpoint (or log) to trace a single problem end-to-end.

## 10) Read-by-question map (fast lookup)

- **“Where is the full run loop?”** → `run_vllm_all_models_think.py` main block + `run_vllm_model(...)`.
- **“Where does retrieval mode switch happen?”** → `vllm_experiments/main.py` (`use_graph` / `use_rag` / vanilla branches).
- **“How is the graph template selected?”** → `vllm_experiments/src/graph_handler.py` (start node + traversal scoring).
- **“How is the final prompt actually built?”** → `vllm_experiments/src/llm_handler_qwen3_gpqa.py` (`solve_problem`).
- **“How is correctness decided?”** → `vllm_experiments/src/utils.py` answer extraction/validation helpers.
- **“Where are run artifacts written?”** → `vllm_experiments/main.py` result-save block + `vllm_experiments/results/`.

