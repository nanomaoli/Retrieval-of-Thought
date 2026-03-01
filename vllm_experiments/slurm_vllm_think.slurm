#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdgi-delta-gpu   # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=inference_qwen3_14b
#SBATCH --time=01:00:00      # hh:mm:ss for the job
#SBATCH -e inference_qwen3_14b-%j.err
#SBATCH -o inference_qwen3_14b-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest     # <- or closest

eval "$(micromamba shell hook --shell bash)"
micromamba activate cot_llm
export FORCE_TORCHRUN=1
module load cuda/12.6

cd /work/nvme/bdgi/ahmed1/cot_caching/graph
export HF_HOME="/work/nvme/bdgi/ahmed1/.cache"
export TRANSFORMERS_OFFLINE=1

python3 -u run_vllm_all_models_think.py
