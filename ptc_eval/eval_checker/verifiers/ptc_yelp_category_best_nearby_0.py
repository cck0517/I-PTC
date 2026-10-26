"""Verifier for ptc_yelp_category_best_nearby_0.

Check that the model returned the correct best-rated restaurant per
category for all restaurants within 2 miles.
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
            # Try to convert list-of-dicts to dict keyed by category
            converted = {}
            for item in answer:
                cat = item.get("category")
                if cat:
                    converted[cat] = item
            answer = converted

        gt_cats = ground_truth["categories"]
        checks: dict[str, bool] = {}

        checks["category_count"] = len(answer) == len(gt_cats)

        for cat, gt_info in gt_cats.items():
            ans_info = answer.get(cat, {})
            checks[f"{cat}_name"] = ans_info.get("name") == gt_info["name"]
            checks[f"{cat}_rating"] = ans_info.get("rating") == gt_info["rating"]
            checks[f"{cat}_business_id"] = ans_info.get("business_id") == gt_info["business_id"]

        # Check no extra categories
        extra = set(answer.keys()) - set(gt_cats.keys())
        checks["no_extra_categories"] = len(extra) == 0

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": gt_cats},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
