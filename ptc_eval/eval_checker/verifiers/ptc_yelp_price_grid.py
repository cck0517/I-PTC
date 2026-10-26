"""Verifier for ptc_yelp_price_grid_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_LEVELS = ("$", "$$", "$$$", "$$$$")


def _tier_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("business_count", "total_review_count", "mean_rating"):
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

    got_grid = answer.get("by_price_level")
    exp_grid = ground_truth["by_price_level"]
    checks["has_grid_key"] = isinstance(got_grid, dict)
    if not isinstance(got_grid, dict):
        got_grid = {}
    for level in _LEVELS:
        checks[f"tier_{level}_correct"] = _tier_matches(got_grid.get(level), exp_grid[level])

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
