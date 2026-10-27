"""Verifier for ptc_yelp_highest_rated_filtered_0.

Check that the model:
1. Returned a JSON with the correct business (name, rating, distance_miles, business_id).
2. Saved the correct business to saved_places (state check).
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

        # Check returned fields
        checks["business_id"] = answer.get("business_id") == ground_truth["business_id"]
        checks["name"] = answer.get("name") == ground_truth["name"]
        checks["rating"] = answer.get("rating") == ground_truth["rating"]

        # Distance: allow ±0.5 mile tolerance
        ans_dist = answer.get("distance_miles")
        if isinstance(ans_dist, (int, float)):
            checks["distance_miles"] = abs(ans_dist - ground_truth["distance_miles"]) <= 0.5
        else:
            checks["distance_miles"] = False

        # State check: the business must be saved. YelpAPI has two ways to
        # do this (save_business -> saved_places, bookmark_business ->
        # bookmarks); the prompt says "bookmarks", so accept either.
        if final_state:
            yelp_state = final_state.get("YelpAPI", {})
            saved = yelp_state.get("saved_places", {}) or {}
            bookmarks = yelp_state.get("bookmarks", []) or []
            target = ground_truth["saved_business_id"]
            checks["saved_to_bookmarks"] = target in saved or target in bookmarks
        else:
            checks["saved_to_bookmarks"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": ground_truth},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
