"""Verifier for ptc_yelp_uber_supermarket_route_0.

Check that the model found the optimal store-visiting route minimizing
total Uber cost on a home → 3 stores → home circuit.
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
        gt_route = ground_truth["route"]

        # Route can be optimal in either direction (same cost due to symmetric distances)
        ans_route = answer.get("route", [])
        gt_reversed = list(reversed(gt_route))
        checks["route_correct"] = ans_route == gt_route or ans_route == gt_reversed

        # Total cost: ±$2.00 tolerance
        ans_total = answer.get("total_uber_cost")
        if isinstance(ans_total, (int, float)):
            checks["total_cost"] = abs(ans_total - ground_truth["total_uber_cost"]) <= 2.0
        else:
            checks["total_cost"] = False

        # Check legs exist and have correct count
        ans_legs = answer.get("legs", [])
        checks["leg_count"] = len(ans_legs) == 4

        # Verify leg costs sum to total (internal consistency)
        if ans_legs and isinstance(ans_total, (int, float)):
            leg_sum = sum(leg.get("cost", 0) for leg in ans_legs if isinstance(leg.get("cost"), (int, float)))
            checks["legs_sum_consistent"] = abs(leg_sum - ans_total) <= 0.5
        else:
            checks["legs_sum_consistent"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": ground_truth},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
