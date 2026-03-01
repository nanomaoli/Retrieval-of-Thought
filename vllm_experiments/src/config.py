from argparse import ArgumentParser, ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


# --- Command Line Argument Parsing ---
def parse_arguments():
    parser = ArgumentParser(description="vLLM Math Solver")

    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to use for inference. ",
        required=True,
    )

    parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        required=True,
        help="Enable thinking mode for the model. Set to True or False.",
    )

    parser.add_argument(
        "--thinking_interven",
        type=str2bool,
        default=False,
        required=True,
        help="Enable thinking intervention. Set to True or False.",
    )

    parser.add_argument(
        "--use_graph",
        type=str2bool,
        required=True,
        help="Specify if the knowledge graph should be used.",
    )

    parser.add_argument(
        "--problems_file",
        type=str,
        required=True,
        help="Path to the JSON file containing problems.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        default="results",
        help="Path to save results.",
    )
    parser.add_argument(
        "--max_workers", type=int, required=True, help="Max concurrent workers."
    )

    parser.add_argument("--log_file", required=True, type=str, help="Log file name.")
    parser.add_argument(
        "--metrics_file",
        required=True,
        type=str,
        help="File to save metrics from vllm",
    )
    parser.add_argument(
        "--use_reranker",
        type=str2bool,
        help="Whether to use the reranker model for graph retrieval.",
    )

    parser.add_argument(
        "--use_beam",
        type=str2bool,
        help="Whether to use the reranker model for graph retrieval.",
    )

    parser.add_argument(
        "--use_rag",
        type=str2bool,
        help="Whether to use RAG for template retrieval.",
    )

    return parser.parse_args()
