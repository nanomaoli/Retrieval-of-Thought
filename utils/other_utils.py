import logging
import json
from os import DirEntry
import re
import pandas as pd


def load_problem_file(problem_file):
    # Update problems loading with the new PROBLEMS_FILE variable
    try:
        with open(PROBLEMS_FILE, "r") as f:
            problems = json.load(f)
        logging.info(
            f"Successfully loaded {len(problems)} problems from {PROBLEMS_FILE}"
        )
    except FileNotFoundError:
        logging.error(f"Error: '{PROBLEMS_FILE}' not found.")
        exit()
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from '{PROBLEMS_FILE}'.")
        exit()


def jsonl_to_json(jsonl_file):
    """
    Convert a JSONL file to a JSON file.
    """
    try:
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        with open(jsonl_file.replace(".jsonl", ".json"), "w") as json_file:
            json.dump(data, json_file, indent=4)
        logging.info(f"Successfully converted {jsonl_file} to JSON format.")
    except FileNotFoundError:
        logging.error(f"Error: '{jsonl_file}' not found.")
        exit()
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from '{jsonl_file}'.")
        exit()


def parse_evaluation_file(file_path):

    with open(file_path, "r") as file:
        content = file.read()

    # Split content by evaluation sections
    sections = content.split("--------------------")

    # Process each non-empty section
    for section in sections[1:]:
        if not section.strip():
            continue

        # Extract model name from the results line
        model_name_match = re.search(
            r"Results saved to .+/eval_results_(.+?)\.json", section
        )
        if model_name_match:
            model_name = model_name_match.group(1)
        else:
            model_name = "Unknown Model"

        # Extract accuracy
        accuracy_match = re.search(
            r"Accuracy \(Successful Only\): (\d+\.\d+)%", section
        )
        accuracy = accuracy_match.group(1) if accuracy_match else "N/A"

        # Extract average output tokens
        tokens_match = re.search(
            r"Average Output Tokens per Successful Problem: (\d+\.\d+)", section
        )
        avg_tokens = tokens_match.group(1) if tokens_match else "N/A"

        # Extract latency metrics
        prompt_tokens_sum_match = re.search(
            r"vllm:request_prompt_tokens_sum: (\d+\.\d+)", section
        )
        e2e_request_latency_seconds_sum = re.search(
            r"vllm:e2e_request_latency_seconds_sum: (\d+\.\d+)", section
        )
        count_match = re.search(
            r"vllm:e2e_request_latency_seconds_count: (\d+\.\d+)", section
        )
        time_taken = re.search(r"Time taken: (\d+\.\d+) seconds", section)

        if prompt_tokens_sum_match and count_match and e2e_request_latency_seconds_sum:
            e2e_count = float(count_match.group(1))
            avg_e2e_latency = (
                float(e2e_request_latency_seconds_sum.group(1)) / e2e_count
            )
            avg_prompt = float(prompt_tokens_sum_match.group(1)) / e2e_count
        else:
            raise ValueError("Missing required latency metrics in the section.")

        # Print the extracted information
        print(f"Model Name: {model_name}")
        print(f"Accuracy: {accuracy}%")
        print(f"Average Output Tokens: {avg_tokens}")
        print(f"Average E2E Latency: {avg_e2e_latency:.2f}")
        print(f"Average Prompt (input) Tokens: {avg_prompt:.2f}")
        print(
            f"Time Taken: {float(time_taken.group(1))} seconds"
            if time_taken
            else "Time Taken: N/A"
        )
        # Print in a format suitable for copy-pasting into a spreadsheet
        print(
            f"{model_name}\n{avg_e2e_latency:.2f},{avg_prompt:.2f},{float(avg_tokens):.2f},{float(accuracy):.2f}"
        )
        print("-" * 50)


def transform_aime_data(input_file, output_file):
    """
    Reads AIME problems from an input JSON file, transforms the format,
    and writes the result to an output JSON file.

    The transformation flattens the 'Metadata' object into the top level.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        transformed_data = []
        for item in data:
            # Extract metadata and remove the nested object
            metadata = item.pop("Metadata", {})

            # Create the new flattened structure
            new_item = {
                "ID": f"2025-I-{item['ID']}",
                "Problem": item["Problem"],
                "Solution": "",  # Add an empty Solution field like in the example
                "Answer": int(item["Answer"]),
                "template_type": metadata.get("template_type", ""),
                "knowledge_tags": metadata.get("knowledge_tags", []),
            }
            transformed_data.append(new_item)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully transformed {input_file} and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def create_sorted_dataframe_from_json(file_path):
    """
    Reads a JSON file, filters for entries with an 'id', sorts them,
    and creates a pandas DataFrame with specified columns.

    Args:
        file_path (str): The path to the input JSON file.

    Returns:
        pandas.DataFrame: A sorted DataFrame with 'correct', 'expected',
                          'extracted', 'tokens', and 'time_to_gen_templ'
                          columns, and 'id' as the index.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter for entries that have an 'id' key
    entries_with_id = [item for item in data if "id" in item]

    if not entries_with_id:
        logging.warning("No entries with 'id' found in the file.")
        return pd.DataFrame()

    # Sort entries by the 'id' attribute
    sorted_entries = sorted(entries_with_id, key=lambda x: x["id"])

    # Create a DataFrame from the sorted entries
    df = pd.DataFrame(sorted_entries)

    # Set 'id' as the index
    df.set_index("id", inplace=True)

    # Specify the columns for the final DataFrame
    columns = ["correct", "expected", "extracted", "tokens", "time_to_gen_templ"]

    # Filter the DataFrame to include only the specified columns
    # This also ensures the column order
    final_df = df[columns]

    return final_df


if __name__ == "__main__":
    import os
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate accuracy from JSON result files recursively in a directory."
    )
    parser.add_argument(
        "--dir",
        nargs="?",  # The argument is optional
        default=os.getcwd(),  # Default to the current working directory
        help="The directory to search for JSON files. Defaults to the current directory.",
    )
    args = parser.parse_args()
    directory = args.dir
    # go over txt files in the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith("txt"):
                file_path = os.path.join(root, filename)
                print(f"\n--- Parsing evaluation file: {file_path} ---\n")
                print("-" * 50)

                # get abs path
                file_path = os.path.abspath(file_path)
                parse_evaluation_file(file_path)
