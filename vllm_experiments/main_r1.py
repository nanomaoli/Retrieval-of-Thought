import os
import json
import logging
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
from src.VARS import GRAPH_MODEL_NAME
import src.VARS as VARS
import subprocess
from src.utils import print_required_metrics, rag_top_1

# Import from local modules
from src.config import parse_arguments
from src.utils import setup_logging
from src.graph_handler import *
from src.llm_handler_r1 import solve_problem
import time

reranker_url = "http://127.0.0.1:8005/score"
reranker_model_name = "Qwen/Qwen3-Reranker-0.6B"
reranker_instruction = "Given a query containing a math problem, retrieve relevant reasoning steps that solves the problem in the query."


def main():
    """
    Main function to run the evaluation pipeline.
    """
    # 1. Setup: Parse arguments and set up logging
    args = parse_arguments()
    setup_logging(args.log_file)
    VARS.THINKING_INTERVEN = args.thinking_interven

    logging.info("####### Starting evaluation pipeline........................")
    logging.info(f"Using max_workers (threads): {args.max_workers}")

    THINKING = args.enable_thinking
    metric_file = args.metrics_file
    use_graph = args.use_graph
    use_rag = args.use_rag
    with open(args.problems_file, "r", encoding="utf-8") as f:
        problems = json.load(f)
    logging.info(
        f"Successfully loaded {len(problems)} problems from {args.problems_file}"
    )

    # 3. Initialize API Clients and Models
    try:
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        model_id = args.model_name
        logging.info(f"Using vLLM model: {model_id}")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return

    G = None
    embedding_model = None
    node_texts, node_embeddings_cache = [], {}
    if args.use_graph:
        print("Using knowledge graph for problem solving.")
        logging.info("Using knowledge graph for problem solving.")
        G = load_knowledge_graph()
        logging.info(
            f"Knowledge graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges."
        )
        if G:
            if args.use_reranker:
                logging.info("Using reranker for dynamic template generation.")
                node_texts, _ = precompute_node_text(G)
            else:
                embedding_model = SentenceTransformer(
                    GRAPH_MODEL_NAME,
                    trust_remote_code=True,
                    device="cuda:0",  # using CPU for now
                    model_kwargs={"torch_dtype": torch.float16},
                )
                logging.info("Sentence transformer model loaded successfully.")
                start = time.perf_counter()
                node_texts, node_embeddings_cache = precompute_node_embeddings(
                    G, embedding_model
                )
                end = time.perf_counter()
                logging.info(
                    f"Node embeddings precomputed in {end - start:.2f} seconds."
                )

    elif args.use_rag:
        logging.info("Using RAG for problem solving.")
        with open("rag_templates.json", "r") as f:
            templates = json.load(f)

        bi_encoder = SentenceTransformer(
            GRAPH_MODEL_NAME,
            trust_remote_code=True,
            device="cuda:0",  # using CPU for now
            model_kwargs={"torch_dtype": torch.float16},
        )
        corpus_embeddings = bi_encoder.encode(
            templates, batch_size=32, convert_to_tensor=True, show_progress_bar=True
        )

    else:
        logging.info("Knowledge graph is not being used for problem solving.")

    # 4. Prepare and Execute Problem-Solving Tasks
    tasks_to_submit = []
    for i, problem_data in enumerate(problems):
        problem_id = problem_data.get("ID", f"Problem_{i+1}")
        problem_text = problem_data.get("Problem")
        expected_answer = problem_data.get("Answer")

        if not all([problem_text, expected_answer is not None]):
            logging.warning(f"Skipping problem {problem_id} due to missing data.")
            continue

        tasks_to_submit.append(
            {
                "problem_id": problem_id,
                "problem_text": problem_text,
                "problem_tags": problem_data.get("knowledge_tags", []),
                # "problem_type": problem_data.get(
                #     "template_type",
                #     "Problem Solving Method",  # ⚠️ this could be an issue
                # ),
                "expected_answer_str": str(expected_answer),
                "thinking": THINKING,
                "thinking_interven": args.thinking_interven,
            }
        )

    if not tasks_to_submit:
        logging.error("No valid problems to process.")
        return

    results = []
    total_tokens = 0
    output_tokens_list = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        future_to_task = {}
        for task_data in tasks_to_submit:
            template = []
            time_to_gen_templ = 0
            if use_graph and G is not None and args.use_reranker and args.use_beam:
                # Generate a dynamic template using the knowledge graph
                template, time_to_gen_templ = (
                    generate_dynamic_template_rerank_beam_single(
                        task_data["problem_text"],
                        task_data["problem_tags"],
                        G,
                        node_texts,
                        reranker_url,
                        reranker_model_name,
                        reranker_instruction,
                    )
                )
            elif use_graph and G is not None and args.use_reranker:
                template, time_to_gen_templ = generate_dynamic_template_rerank(
                    task_data["problem_text"],
                    task_data["problem_tags"],
                    G,
                    node_texts,
                    reranker_url,
                    reranker_model_name,
                    reranker_instruction,
                )
            elif use_graph and G is not None:
                template, time_to_gen_templ = generate_dynamic_template(
                    task_data["problem_text"],
                    task_data["problem_tags"],
                    # task_data["problem_type"],
                    G,
                    embedding_model,
                    node_embeddings_cache,
                    node_texts,
                )
            elif use_rag:
                prob = f"{task_data['problem_text']}\nTags: {', '.join(i for i in task_data['problem_tags'])}"
                template, time_to_gen_templ = rag_top_1(
                    bi_encoder, prob, templates, corpus_embeddings
                )
                # logging.info(
                #     f"RAG selected template: {template} with total words {sum(len(t.split()) for t in template)}"
                # )
                template = template.split("\n")

            task_data["template"] = template
            task_data["time_to_gen_templ"] = time_to_gen_templ
            future = executor.submit(
                solve_problem,
                task_data["problem_id"],
                task_data["problem_text"],
                task_data["template"],
                task_data["thinking"],
                task_data["thinking_interven"],
                client,
                model_id,
                task_data["expected_answer_str"],
            )
            future_to_task[future] = task_data

        start_time = time.time()

        completed_count = 0
        total_tasks = len(tasks_to_submit)

        vllm_start_time = time.perf_counter()
        for future in concurrent.futures.as_completed(future_to_task):
            task_data = future_to_task[future]
            try:
                output_text, output_tokens, extracted_answer, messages, is_correct = (
                    future.result()
                )
                if output_text is None:
                    logging.warning(
                        f"No output text for problem {task_data['problem_id']}. Skipping."
                    )
                    continue
                total_tokens += output_tokens
                output_tokens_list.append(output_tokens)

                results.append(
                    {
                        "id": task_data["problem_id"],
                        "correct": is_correct,
                        "expected": task_data["expected_answer_str"],
                        "extracted": (
                            str(extracted_answer)
                            if extracted_answer is not None
                            else None
                        ),
                        "time_to_gen_templ": task_data.get("time_to_gen_templ", 0),
                        "tokens": output_tokens,
                        "output_text": output_text,
                        "input_text": task_data["problem_text"],
                        "template": task_data["template"],
                        "messages": messages,
                        "model": model_id,
                    }
                )
                logging.info(
                    f"Processed problem {task_data['problem_id']}: "
                    f"{'CORRECT' if is_correct else 'INCORRECT'}, "
                    f"Tokens: {output_tokens}, "
                    f"Expected Answer: {task_data['expected_answer_str']}, "
                    f"Extracted Answer: {extracted_answer}"
                )

            except Exception as e:
                logging.error(
                    f"An error occurred processing problem {task_data['problem_id']}: {e}",
                    exc_info=True,
                )
                results.append(
                    {
                        "id": task_data["problem_id"],
                        "error": str(e),
                    }
                )

            finally:
                completed_count += 1
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_count
                remaining_tasks = total_tasks - completed_count
                if remaining_tasks > 0:
                    estimated_remaining_time = avg_time_per_task * remaining_tasks
                    logging.info(
                        f"Progress: {completed_count}/{total_tasks} tasks completed. "
                        f"Elapsed: {elapsed_time:.2f}s. "
                        f"Estimated time remaining: {estimated_remaining_time:.2f}s."
                    )

        vllm_end_time = time.perf_counter()
        logging.info(
            f"Total vLLM request processing time: {vllm_end_time - vllm_start_time:.2f} seconds."
        )
    # 5. Summarize and Save Results
    num_processed = len(results)
    num_successful = sum(1 for r in results if "error" not in r)
    correct_count = sum(1 for r in results if r.get("correct"))

    if num_successful > 0:
        # Calculate average tokens only based on problems that didn't error
        successful_tokens = [r["tokens"] for r in results if "error" not in r]
        average_tokens = (
            sum(successful_tokens) / num_successful if num_successful > 0 else 0
        )
        accuracy = (correct_count / num_successful) * 100
    else:
        average_tokens = 0
        accuracy = 0

    if VARS.THINKING_INTERVEN and args.use_reranker and args.use_beam:
        print("\n--- Evaluation Summary --- (Thinking Intervention + Reranker + Beam) ")
        model_filename_part = (
            f"eval_results_lg_{model_id.replace('/', '_')}_interven_rerank_beam.json"
        )
    elif VARS.THINKING_INTERVEN and args.use_reranker:
        print("\n--- Evaluation Summary --- (Thinking Intervention + Reranker) ")
        model_filename_part = (
            f"eval_results_lg_{model_id.replace('/', '_')}_interven_rerank.json"
        )
    elif VARS.THINKING_INTERVEN:
        print("\n--- Evaluation Summary --- (Thinking Intervention) ")
        model_filename_part = (
            f"eval_results_lg_{model_id.replace('/', '_')}_interven.json"
        )
    elif use_graph:
        print("\n--- Evaluation Summary --- (Templ only) ")
        model_filename_part = f"eval_results_lg_{model_id.replace('/', '_')}_graph.json"
    elif use_rag:
        print("\n--- Evaluation Summary --- (RAG) ")
        model_filename_part = f"eval_results_lg_{model_id.replace('/', '_')}_rag.json"
    else:
        print("\n--- Evaluation Summary --- (vanilla) ")
        model_filename_part = f"eval_results_lg_{model_id.replace('/', '_')}.json"

    print(f"Total Problems Attempted: {num_processed}")
    print(f"Successfully Processed: {num_successful}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy (Successful Only): {accuracy:.2f}%")
    print(f"Average Output Tokens per Successful Problem: {average_tokens:.2f}")

    print("\nOutput Tokens per Problem:")
    # Ensure results and output_tokens_list have the same length before iterating
    if len(results) == len(output_tokens_list):
        for i, result_item in enumerate(results):
            problem_id_disp = result_item.get("id", f"Problem_{i+1}")
            tokens = output_tokens_list[i]
            status = "(Errored)" if "error" in result_item else ""
            print(f"Problem {problem_id_disp}: {tokens} tokens {status}")
    else:
        logging.error(
            "Error: Mismatch between results and token list lengths. Cannot display per-problem tokens."
        )

    # Find the next available results directory and filename
    i = 0
    base_path = args.path

    while True:
        # Construct a new path with a numeric suffix
        current_path = f"{base_path}_{i}"
        result_filename = os.path.join(current_path, model_filename_part)

        if not os.path.exists(result_filename):
            # If the file doesn't exist in this directory, we found our spot.
            os.makedirs(current_path, exist_ok=True)  # Create dir if it doesn't exist
            break

        # If the file exists, increment the counter and check the next directory
        i += 1

    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    time.sleep(5)  # Ensure the file is written before running the metrics command
    subprocess.run(
        f"curl -s http://localhost:8000/metrics > {metric_file}", shell=True, check=True
    )
    print_required_metrics(metric_file)
    print(f"\nResults saved to {result_filename}")


if __name__ == "__main__":
    main()
