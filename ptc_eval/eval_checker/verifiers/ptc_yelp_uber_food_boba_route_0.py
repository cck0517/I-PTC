"""Verifier for ptc_yelp_uber_food_boba_route_0.

Check that the model found the Chinese + boba pair and route order
that minimizes the home → A → B → home round-trip Uber cost.
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

        # Correct restaurants
        ans_chinese = answer.get("chinese_restaurant", {})
        ans_boba = answer.get("boba_shop", {})
        checks["chinese_id"] = (
            ans_chinese.get("business_id") == gt["chinese_restaurant"]["business_id"]
        )
        checks["chinese_name"] = (
            ans_chinese.get("name") == gt["chinese_restaurant"]["name"]
        )
        checks["boba_id"] = (
            ans_boba.get("business_id") == gt["boba_shop"]["business_id"]
        )
        checks["boba_name"] = (
            ans_boba.get("name") == gt["boba_shop"]["name"]
        )

        # Route order — both directions are acceptable since costs may be equal
        ans_route = answer.get("route_order", [])
        gt_route = gt["route_order"]
        gt_reversed = list(reversed(gt_route))
        checks["route_order"] = ans_route == gt_route or ans_route == gt_reversed

        # Total cost
        TOLERANCE = 1.5
        ans_total = answer.get("total_uber_cost")
        if isinstance(ans_total, (int, float)):
            checks["total_uber_cost"] = abs(ans_total - gt["total_uber_cost"]) <= TOLERANCE
        else:
            checks["total_uber_cost"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": gt},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
