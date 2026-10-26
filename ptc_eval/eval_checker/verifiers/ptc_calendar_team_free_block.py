"""Verifier for ptc_calendar_team_free_block_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got_att = answer.get("attendees")
    exp_att = ground_truth["attendees"]
    checks["attendees_correct"] = (
        isinstance(got_att, list) and list(got_att) == list(exp_att))

    got_block = answer.get("longest_block")
    exp_block = ground_truth["longest_block"]
    if isinstance(got_block, dict):
        checks["longest_block_correct"] = all(
            got_block.get(k) == exp_block[k]
            for k in ("start_utc", "end_utc", "duration_minutes"))
    else:
        checks["longest_block_correct"] = False

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
