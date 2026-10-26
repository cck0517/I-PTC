"""Verifier for ptc_multi_meeting_point_0.

Check that the model correctly identified the restaurant with the lowest
combined Uber cost and reported the individual and total costs.
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

        # Core check: correct restaurant
        checks["restaurant_name"] = (
            answer.get("restaurant_name") == ground_truth["restaurant_name"]
        )
        checks["business_id"] = (
            answer.get("business_id") == ground_truth["business_id"]
        )

        # Cost checks: allow ±$1.00 tolerance (model may compute slightly
        # differently due to rounding at intermediate steps)
        TOLERANCE = 1.0

        ans_my = answer.get("my_uber_cost")
        if isinstance(ans_my, (int, float)):
            checks["my_uber_cost"] = (
                abs(ans_my - ground_truth["my_uber_cost"]) <= TOLERANCE
            )
        else:
            checks["my_uber_cost"] = False

        ans_friend = answer.get("friend_uber_cost")
        if isinstance(ans_friend, (int, float)):
            checks["friend_uber_cost"] = (
                abs(ans_friend - ground_truth["friend_uber_cost"]) <= TOLERANCE
            )
        else:
            checks["friend_uber_cost"] = False

        ans_total = answer.get("total_combined_cost")
        if isinstance(ans_total, (int, float)):
            checks["total_combined_cost"] = (
                abs(ans_total - ground_truth["total_combined_cost"]) <= TOLERANCE
            )
        else:
            checks["total_combined_cost"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": {
                k: v for k, v in ground_truth.items() if k != "all_costs"
            }},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
