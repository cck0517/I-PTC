"""Verifier for ptc_yelp_min_stops_coverage_0.

Check that the model found the minimum number of Uber stops to cover
dinner + boba + dessert, with the highest combined rating among ties.
"""

from __future__ import annotations
from typing import Any
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)


# The prompt asks for a "category" per place and says "any cuisine counts as
# dinner", so accept the prompt's own label ("dinner"), the generic
# "restaurant" tag, or any cuisine name used in the scenario.
_DINNER_CATEGORIES = {"dinner", "restaurant",
                      "chinese", "italian", "thai", "indian", "american",
                      "japanese", "mexican", "french", "burmese", "korean",
                      "vietnamese", "mediterranean", "seafood",
                      "bbq", "hawaiian", "southern"}


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

        checks["num_stops"] = answer.get("num_stops") == ground_truth["num_stops"]

        ans_rating = answer.get("total_rating")
        if isinstance(ans_rating, (int, float)):
            checks["total_rating"] = abs(ans_rating - ground_truth["total_rating"]) < 0.1
        else:
            checks["total_rating"] = False

        # Check all 3 categories are covered
        ans_places = answer.get("places", [])
        ans_cats = set()
        for p in ans_places:
            cat = (p.get("category") or "").lower()
            if cat in _DINNER_CATEGORIES:
                ans_cats.add("dinner")
            elif "boba" in cat:
                ans_cats.add("boba")
            elif "dessert" in cat:
                ans_cats.add("dessert")
        checks["covers_dinner"] = "dinner" in ans_cats
        checks["covers_boba"] = "boba" in ans_cats
        checks["covers_dessert"] = "dessert" in ans_cats

        # Check correct business IDs
        gt_ids = {p["business_id"] for p in ground_truth["places"]}
        ans_ids = {p.get("business_id") for p in ans_places}
        checks["correct_places"] = ans_ids == gt_ids

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": ground_truth},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
