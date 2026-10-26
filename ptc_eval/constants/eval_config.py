"""
Paths for PTC-Bench datasets (mirrors bfcl_eval.constants.eval_config layout).
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

PROMPT_PATH = PACKAGE_ROOT / "data"
MULTI_TURN_FUNC_DOC_PATH = PROMPT_PATH / "multi_turn_func_doc"
POSSIBLE_ANSWER_PATH = PROMPT_PATH / "possible_answer"
