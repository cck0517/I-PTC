"""Verifier for ptc_calendar_fragmentation_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("employee_id", "display_name", "event_count",
              "total_meeting_hours", "fragmentation_score"):
        if got.get(k) != exp[k]:
            return False
    return True


def _list_matches(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not _entry_matches(g, e):
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

    checks["most_fragmented_correct"] = _list_matches(
        answer.get("top3_most_fragmented"), ground_truth["top3_most_fragmented"])
    checks["least_fragmented_correct"] = _list_matches(
        answer.get("bottom3_least_fragmented"), ground_truth["bottom3_least_fragmented"])

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
