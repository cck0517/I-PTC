"""Verifier for ptc_yelp_review_threshold_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("business_id", "name", "review_count", "rating"):
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

    checks["threshold_correct"] = (
        answer.get("threshold_T") == ground_truth["threshold_T"])
    checks["count_correct"] = (
        answer.get("qualifying_count") == ground_truth["qualifying_count"])

    got = answer.get("qualifying_businesses")
    exp = ground_truth["qualifying_businesses"]
    checks["has_businesses_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["businesses_length_correct"] = len(got) == len(exp)
    entries_ok = checks["businesses_length_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["businesses_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
