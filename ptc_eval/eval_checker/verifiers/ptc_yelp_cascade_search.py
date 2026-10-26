"""Verifier for ptc_yelp_cascade_search_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_PICK_FIELDS = ("business_id", "name", "rating", "review_count",
                "recent_rating", "recent_useful_sum")


def _pick_matches(got: Any, exp: Any) -> bool:
    if exp is None:
        return got is None
    if not isinstance(got, dict):
        return False
    for k in _PICK_FIELDS:
        if got.get(k) != exp[k]:
            return False
    return True


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    checks["tier_correct"] = answer.get("tier") == ground_truth["tier"]
    checks["pick_correct"] = _pick_matches(answer.get("pick"), ground_truth["pick"])

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
