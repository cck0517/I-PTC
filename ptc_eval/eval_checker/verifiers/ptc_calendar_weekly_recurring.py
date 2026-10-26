"""Verifier for ptc_calendar_weekly_recurring_0 (title-category aggregation)."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("category", "total_events", "distinct_employees",
              "distinct_titles", "avg_duration_minutes"):
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

    checks["count_correct"] = answer.get("count") == ground_truth["count"]
    got = answer.get("categories")
    exp = ground_truth["categories"]
    checks["has_categories_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["categories_length_correct"] = len(got) == len(exp)
    entries_ok = checks["categories_length_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["categories_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
