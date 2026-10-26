"""Verifier for ptc_multi_group_dinner_0.

Check that the model picked the restaurant minimizing total combined
Uber cost for 3 people.
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

        checks["restaurant_name"] = (
            answer.get("restaurant_name") == ground_truth["restaurant_name"]
        )
        checks["business_id"] = (
            answer.get("business_id") == ground_truth["business_id"]
        )

        TOLERANCE = 1.0

        # Individual costs
        ans_costs = answer.get("costs", {})
        gt_costs = ground_truth["costs"]
        for person, gt_cost in gt_costs.items():
            ans_c = ans_costs.get(person)
            if isinstance(ans_c, (int, float)):
                checks[f"cost_{person}"] = abs(ans_c - gt_cost) <= TOLERANCE
            else:
                checks[f"cost_{person}"] = False

        # Total cost
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
            details={"extracted_answer": answer, "ground_truth": ground_truth},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
