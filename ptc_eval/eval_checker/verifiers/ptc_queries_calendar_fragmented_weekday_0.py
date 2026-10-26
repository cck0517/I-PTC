"""Verifier for ptc_queries_calendar_fragmented_weekday_0.

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
    try:
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

        # day_of_week: exact match
        checks["day_of_week"] = answer.get("day_of_week") == ground_truth["day_of_week"]

        # total_free_blocks: exact match
        checks["total_free_blocks"] = answer.get("total_free_blocks") == ground_truth["total_free_blocks"]

        # avg_free_blocks_per_day: allow float comparison with tolerance
        gt_avg = ground_truth["avg_free_blocks_per_day"]
        ans_avg = answer.get("avg_free_blocks_per_day")
        if isinstance(ans_avg, (int, float)):
            checks["avg_free_blocks_per_day"] = abs(float(ans_avg) - gt_avg) < 0.15
        else:
            checks["avg_free_blocks_per_day"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": ground_truth},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
