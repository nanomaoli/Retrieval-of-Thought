from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import datasets
from datasets import load_dataset


SRC_PATH = Path("../eval_datasets/gpqa_problems.json")
OUT_PATH = Path("../eval_datasets/gpqa_problems_with_answers.json")


def flatten_splits(
    dataset_dict: datasets.Dataset | datasets.DatasetDict,
) -> datasets.Dataset:
    """
    Ensure we have a single Dataset regardless of whether load_dataset returns a Dataset
    or a DatasetDict containing multiple splits.
    """
    if isinstance(dataset_dict, datasets.Dataset):
        return dataset_dict
    return datasets.concatenate_datasets(list(dataset_dict.values()))


def build_answer_lookup(rows: Iterable[dict]) -> Dict[str, str]:
    """
    Create a mapping from Record ID -> Correct Answer.
    """
    lookup: Dict[str, str] = {}
    for row in rows:
        record_id = row.get("Record ID")
        answer = row.get("Correct Answer")
        if record_id is None or answer is None:
            continue
        lookup[record_id] = answer
    return lookup


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Source GPQA problems JSON not found at {SRC_PATH}")

    problems: List[dict] = json.loads(SRC_PATH.read_text())

    # Login with `huggingface-cli login` beforehand to access the dataset.
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main")
    flat_ds = flatten_splits(ds)
    answer_lookup = build_answer_lookup(flat_ds)

    missing_ids: List[str] = []
    for problem in problems:
        problem_id = problem.get("ID")
        answer = answer_lookup.get(problem_id)
        if answer is None:
            missing_ids.append(problem_id)
        problem["Correct Answer"] = answer

    OUT_PATH.write_text(json.dumps(problems, indent=2, ensure_ascii=False))

    print(f"Wrote {len(problems)} records to {OUT_PATH}")
    if missing_ids:
        print(f"Records without a matching answer: {len(missing_ids)}")
        print(f"First few missing IDs: {missing_ids[:5]}")


if __name__ == "__main__":
    main()
