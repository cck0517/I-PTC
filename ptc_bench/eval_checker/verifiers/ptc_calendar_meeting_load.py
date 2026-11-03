"""Verifier for ptc_calendar_meeting_load_0 (top-3 + drill-down to top partner)."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _partner_matches(got: Any, exp: Any) -> bool:
    if exp is None:
        return got is None
    if not isinstance(got, dict):
        return False
    for k in ("employee_id", "display_name", "shared_count"):
        if got.get(k) != exp[k]:
            return False
    return True


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("employee_id", "display_name", "accepted_event_count",
              "total_meeting_minutes"):
        if got.get(k) != exp[k]:
            return False
    return _partner_matches(got.get("top_partner"), exp["top_partner"])


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("top3")
    exp = ground_truth["top3"]
    checks["has_top3_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["top3_count_correct"] = len(got) == len(exp)
    entries_ok = checks["top3_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["top3_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
