"""Verifier for ptc_yelp_uber_pareto_chinese_0.

Check that the model correctly performed Pareto-dominance filtering and
returned the highest-rated non-dominated Chinese restaurant.
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
        gt = ground_truth

        checks["name"] = answer.get("name") == gt["name"]
        checks["business_id"] = answer.get("business_id") == gt["business_id"]
        checks["rating"] = answer.get("rating") == gt["rating"]

        # Uber cost: ±$1.00 tolerance
        ans_cost = answer.get("uber_cost")
        if isinstance(ans_cost, (int, float)):
            checks["uber_cost"] = abs(ans_cost - gt["uber_cost"]) <= 1.0
        else:
            checks["uber_cost"] = False

        checks["non_dominated_count"] = (
            answer.get("non_dominated_count") == gt["non_dominated_count"]
        )

        # Check dominated list (order doesn't matter)
        ans_dom = set(answer.get("dominated_restaurants", []))
        gt_dom = set(gt["dominated_restaurants"])
        checks["dominated_list"] = ans_dom == gt_dom

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": gt},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
