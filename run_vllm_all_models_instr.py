import yaml
import os
import sys
import json
import argparse
import re
import signal
import psutil
import subprocess
import time
import argparse

def get_vllm_cmd(model, max_model_len, max_num_seqs):
    return f"vllm serve {model} --max-model-len {max_model_len} --dtype bfloat16 --no-enable-prefix-caching --max-num-seqs {max_num_seqs} --enable-prompt-tokens-details --gpu-memory-utilization 0.85 --max-seq-len-to-capture {max_model_len}"

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


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        print(f"Parent process {parent_pid} does not exist.")
        return
    children = parent.children(recursive=True)
    for chd in children:
        print(f"Killing child process {process.pid}")
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

def run_python_script(script_path, *args):
    """
    Run a Python script with arguments and return the output.
    """
    command = [sys.executable, script_path] + list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

if __name__ == "__main__":
    dataset = "aime_2024"

    data = parse_yaml(f"results/{dataset}/models.yaml")
    PROBLEMS_FILE = f"{dataset}_problems.json"
    path = f"results/{dataset}/"
    
    py_file = "main_vanilla_vllm_instruct.py"

    if "vanilla" in py_file:
        tag = "vanilla"
        path = f"results/{dataset}/vanilla/"
    else:
        tag = None

    out_file = f"results/{dataset}/{tag}_output.txt"
    print("Running VLLM models...")
    for key, value in data.items():
        model = key
        max_model_len = data[model]["max-model-len"]
        max_num_seqs = data[model]["max-num-seqs"]
        vllm_cmd = get_vllm_cmd(model, max_model_len, max_num_seqs)
        process = run_vllm_command(vllm_cmd)
        time.sleep(90)  # Give it some time to start
        start = time.perf_counter()
        out, err, returncode = run_python_script(
            py_file,
            "--hf_model_name",
            model,
            "--problems_file",
            PROBLEMS_FILE,
            "--path",
            path,
        )
        end  =  time.perf_counter()
        kill_child_processes(process.pid, signal.SIGTERM)
        # find the text in the output
        txt_find = "--- Evaluation Summary ---"
        start_index = out.find(txt_find)
        if start_index != -1:
            evaluation_summary = out[start_index:]
            print(evaluation_summary)
        else:
            print(f"'{txt_find}' not found in the output.")
            print(f"Output: {out}")
            print(f"Error: {err}")
            print(f"Return code: {returncode}")
        with open(out_file, "a") as f:
            f.write("-"* 20)
            f.write(f"{model}\n")
            f.write(evaluation_summary)
            f.write(f"Time taken: {end - start:.2f} seconds\n")
            f.write("\n\n")
        