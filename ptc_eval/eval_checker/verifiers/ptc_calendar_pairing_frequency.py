"""Verifier for ptc_calendar_pairing_frequency_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    g_pair = got.get("pair")
    if not isinstance(g_pair, list) or list(g_pair) != list(exp["pair"]):
        return False
    g_names = got.get("display_names")
    if not isinstance(g_names, list) or list(g_names) != list(exp["display_names"]):
        return False
    if got.get("shared_meeting_count") != exp["shared_meeting_count"]:
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

    got = answer.get("top5_pairs")
    exp = ground_truth["top5_pairs"]
    checks["has_top5_pairs_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["top5_count_correct"] = len(got) == len(exp)
    entries_ok = checks["top5_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["top5_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
