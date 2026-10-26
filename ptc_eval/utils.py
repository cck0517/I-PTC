"""
BFCL-style helpers for loading JSONL benchmark files.
"""

import json
from pathlib import Path


def load_file(file_path: str | Path) -> list[dict]:
    """
    Load a JSONL file: one JSON object per line (same pattern as BFCL data README).
    """
    path = Path(file_path)
    result: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result.append(json.loads(line))
    return result


def load_dataset_entry(split: str) -> list[dict]:
    """
    Load prompt entries from ``data/{split}.jsonl``.

    Example: load_dataset_entry("main") -> reads main.jsonl
    """
    from ptc_eval.constants.eval_config import PROMPT_PATH

    return load_file(PROMPT_PATH / f"{split}.jsonl")


def load_ground_truth_entry(split: str) -> list[dict]:
    """
    Load ground-truth entries from ``data/possible_answer/{split}.jsonl``.
    """
    from ptc_eval.constants.eval_config import POSSIBLE_ANSWER_PATH

    return load_file(POSSIBLE_ANSWER_PATH / f"{split}.jsonl")
