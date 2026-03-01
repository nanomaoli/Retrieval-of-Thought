from sklearn.externals._packaging.version import parse
import yaml
import sys
import signal
import psutil
import subprocess
import time
import logging
import os
import argparse
import os


def get_vllm_cmd(
    model,
    max_model_len,
    max_num_seqs,
    enable_lora=False,
    lora_path=None,
    lora_model_name=None,
    use_reranker=False,
):
    if use_reranker:
        gpu_mem_util = 0.85
    else:
        gpu_mem_util = 0.91

    cmd = f"vllm serve {model} --max-model-len {max_model_len} --dtype bfloat16 --no-enable-prefix-caching --max-num-seqs {max_num_seqs} --enable-prompt-tokens-details --gpu-memory-utilization {gpu_mem_util} --reasoning-parser deepseek_r1 --tensor-parallel-size {TP_SIZE}"

    if enable_lora:
        if lora_path is None:
            raise ValueError("Lora path must be provided when enable_lora is True.")
        cmd += f" --enable-lora --lora-modules {lora_model_name}={lora_path}"
    return cmd


# run the command using subprocess and capture the output while also check process return code
def run_vllm_command(command):
    """
    Run a shell command in the background.
    The command's stdout and stderr will be inherited from the parent process.
    Returns the subprocess.Popen object representing the running process.
    """
    # Using shell=True as vllm_cmd is a formatted string command.
    # Ensure the command string is from a trusted source if shell=True is used.
    process = subprocess.Popen(command, shell=True)
    return process


def run_vllm_rerank_command():
    port = 8005
    reranker_url = f"http://127.0.0.1:{port}/score"
    reranker_model_name = "Qwen/Qwen3-Reranker-0.6B"
    hf_overrides_json = '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    compilation_json = '{"cudagraph_mode": "None"}'
    cmd = f"vllm serve {reranker_model_name} --port {port} --hf_overrides '{hf_overrides_json}' --compilation_config '{compilation_json}' --gpu-memory-utilization 0.11 --max-model-len 1024 --runner pooling --convert classify --no-enable-prefix-caching --no-enable-chunked-prefill"
    process = subprocess.Popen(cmd, shell=True)
    return process


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        print(f"Parent process {parent_pid} does not exist.")
        return
    children = parent.children(recursive=True)
    for chd in children:
        chd.send_signal(sig)
    # Also kill the parent process
    print(f"Killing parent process {parent_pid}")
    parent.send_signal(sig)


def parse_yaml(yaml_file):
    """
    Parse a YAML file and return its contents as a dictionary.
    """
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data


def run_python_script(script_path, args_dict):
    """
    Run a Python script with the given arguments.
    Returns the output, error, and return code of the script.
    """
    cmd = [
        sys.executable,
        script_path,
    ]  # Use sys.executable to ensure the same Python interpreter is used

    for key, value in args_dict.items():
        cmd.append(key)
        cmd.append(str(value))

    print(f"Running command: {' '.join(cmd)}")
    logging.info(f"Running (py) command: {' '.join(cmd)}")

    # Use PIPE for stdout/stderr to capture output
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = process.communicate()
    returncode = process.returncode

    return out, err, returncode


def run_vllm_model(
    model,
    max_model_len,
    max_num_seqs,
    PROBLEMS_FILE,
    path,
    py_file,
    log_filename,
    thinking_interven,
    use_graph,
    use_reranker=False,
    use_beam=False,
    use_rag=False,
    enable_lora=False,
    lora_path=None,
    lora_model_name=None,
):

    if use_reranker:
        process_rerank = run_vllm_rerank_command()
        time.sleep(VLLM_WAIT_TIME - 30)  # Give some time to start reranker

    vllm_cmd = get_vllm_cmd(
        model, max_model_len, max_num_seqs, enable_lora, lora_path, lora_model_name
    )
    process = run_vllm_command(vllm_cmd)
    time.sleep(VLLM_WAIT_TIME)  # Give it some time to start
    if enable_lora:
        if lora_model_name is not None:
            model = lora_model_name

    start = time.perf_counter()

    args_dict = {
        "--model_name": model,
        "--problems_file": PROBLEMS_FILE,
        "--path": path,
        "--log_file": log_filename,
        "--enable_thinking": THINKING,
        "--use_graph": use_graph,
        "--max_workers": max_num_seqs,
        "--metrics_file": metric_file,
        "--thinking_interven": thinking_interven,
        "--use_reranker": use_reranker,
        "--use_beam": use_beam,
        "--use_rag": use_rag,
    }

    out, err, returncode = run_python_script(
        py_file,
        args_dict,
    )
    # prints error if any
    if returncode != 0:
        print(f"Error running the script: {err}")
        logging.error(f"Error running the script: {err}")

    end = time.perf_counter()
    time_taken = end - start

    kill_child_processes(process.pid, signal.SIGTERM)
    time.sleep(30)  # Give vllm process some time to terminate
    if use_reranker:
        logging.info("Killing reranker process...")
        kill_child_processes(process_rerank.pid, signal.SIGTERM)
        time.sleep(20)  # Give rerank process some time to terminate

    # find the text in the output
    txt_find = "--- Evaluation Summary ---"
    start_index = out.find(txt_find)
    if start_index != -1:
        evaluation_summary = out[start_index:]
        print(evaluation_summary)
        return evaluation_summary, time_taken
    else:
        print(f"'{txt_find}' not found in the output.")
        print(f"Output: {out}")
        print(f"Error: {err}")
        print(f"Return code: {returncode}")
        raise ValueError(f"'{txt_find}' not found in the output.")


if __name__ == "__main__":

    THINKING = True
    VLLM_WAIT_TIME = 120  # seconds to wait for vllm to start
    MAX_MODEL_LEN = 20000

    metric_file = "some_metrics.txt"
    lora_path = None
    lora_model_name = None

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    log_filename = "vllm_think_dler.log"
    eval_dataset_dir = "eval_datasets"
    result_path = "vllm_experiments/results"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run VLLM models with thinking.")

    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model to run.",
        required=True,
    )

    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Use cross-encoder reranker instead of bi-encoder.",
    )

    parser.add_argument(
        "--use_beam",
        action="store_true",
        help="Use cross-encoder reranker instead of bi-encoder.",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Maximum number of sequences to process in parallel.",
        required=True,
    )

    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA for the model.",
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to the LoRA model.",
    )

    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Whether to use RAG for template retrieval.",
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        help="Tensor parallel size (number of GPUs to use).",
        default=1,
    )

    parser.add_argument(
        "--vanilla_break",
        action="store_true",
        help="Break after vanilla experiments.",
    )

    parser.add_argument(
        "--py_file",
        type=str,
        help="Path to the Python script to run for evaluation.",
        default="vllm_experiments/main_dler.py",
        required=False,
    )

    args = parser.parse_args()
    max_num_seqs = args.max_num_seqs
    enable_lora = args.enable_lora
    model = args.model
    max_model_len = MAX_MODEL_LEN
    use_rag = args.use_rag
    TP_SIZE = args.tp_size

    if enable_lora:
        if args.lora_path is None:
            raise ValueError("Lora path must be provided when enable_lora is True.")
        lora_path = args.lora_path
        lora_model_name = f"{model.split('/')[-1].split('.')[0]}-lora-{lora_path.split('/')[-1].replace('instr_fine_tuned_', '')}"

    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting VLLM model evaluation...")
    if args.use_reranker:
        logging.info("Using cross-encoder reranker for evaluation.")
    for vanilla in [False]:
        for thinking_interven in [False, True]:
            dataset_start = time.perf_counter()
            if vanilla and thinking_interven:
                logging.warning(
                    "Vanilla mode does not support thinking intervention. Skipping this configuration."
                )
                continue
            logging.info(
                f"Running VLLM models with vanilla={vanilla}, thinking_interven={thinking_interven}"
            )
            for dataset in [
                "aime_2024",
                "amc23",
                "aime_2025",
                "aime_2023",
            ]:

                # data = parse_yaml(f"{result_path}/{model_yaml}")
                PROBLEMS_FILE = f"{eval_dataset_dir}/{dataset}_problems.json"
                path = f"{result_path}/{dataset}/"
                py_file = "vllm_experiments/main_dler.py"

                use_graph = not vanilla
                if vanilla:  # no graph template
                    tag = "vanilla"
                    path = f"{result_path}/{dataset}/vanilla/"
                else:
                    tag = "graph"
                out_file = f"{result_path}/{dataset}/{tag}_think_output.txt"
                # Ensure the directory for the output file exists
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                logging.info(f"Running VLLM models for dataset: {dataset}")
                logging.info(f"Path: {path}")
                logging.info(f"Python file: {py_file}")
                # logging.info(f"Total experiments in Yaml File: {len(data)}")

                # for key, value in data.items():
                #     model = value.get("model", key)
                #     enable_lora = value.get("enable-lora", False)
                #     if enable_lora:
                #         lora_path = value.get("lora-path", None)
                #         if lora_path is None:
                #             raise ValueError(f"lora-path is not specified for model {model}.")
                #         lora_model_name = f"{model.split('/')[-1]}-lora-{lora_path.split('/')[-1].split('_')[-1]}"  # e.g., "r8"
                # value["max-model-len"]

                logging.info(
                    f"Model: {model}, Enable LoRA: {enable_lora}, LoRA Path: {lora_path if enable_lora else 'N/A'}, Max Model Length: {max_model_len}, Max Num Sequences: {max_num_seqs}"
                )
                logging.info(
                    f"vllm cmd: {get_vllm_cmd(model, max_model_len, max_num_seqs, enable_lora, lora_path, lora_model_name)}"
                )
                evaluation_summary, time_taken = run_vllm_model(
                    model,
                    max_model_len,
                    max_num_seqs,
                    PROBLEMS_FILE,
                    path,
                    py_file,
                    log_filename,
                    thinking_interven=thinking_interven,
                    use_graph=use_graph,
                    use_reranker=args.use_reranker,
                    use_beam=args.use_beam,
                    use_rag=args.use_rag,
                    enable_lora=enable_lora,
                    lora_path=lora_path if enable_lora else None,
                    lora_model_name=lora_model_name if enable_lora else None,
                )

                summary = (
                    f"\n----------------------------------------------------------------------------------\n"
                    f"Experiment setting: vanilla={vanilla} & RAG={use_rag} & thinking_interven={thinking_interven}\n"
                    f"Evaluation for model: {model} dataset: {dataset}.....\n"
                    f"{model}\n"
                    f"{evaluation_summary}\n"
                    f"vLLM inference Time taken: {time_taken:.2f} seconds\n"
                    f"Total time taken: {time.perf_counter() - dataset_start:.2f} seconds"
                )

                with open(out_file, "a") as f:
                    f.write(summary + "\n\n")

                log_summary = f"{summary}"
                logging.info(log_summary)

                time.sleep(10)

            if use_rag or args.vanilla_break:
                exit(0)
