"""Verifier for ptc_org_chart_most_teammates_0.

Extract JSON from the model answer and compare each field to the ground truth.
"""

from __future__ import annotations
from typing import Any
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )
    if isinstance(answer, list):
        answer = answer[0] if answer else {}

    checks: dict[str, bool] = {}
    for key, expected in ground_truth.items():
        checks[key] = answer.get(key) == expected

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"extracted_answer": answer, "ground_truth": ground_truth},
    )
