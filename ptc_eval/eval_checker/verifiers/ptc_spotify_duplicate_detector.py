"""Verifier for ptc_spotify_duplicate_detector_0.

Checks the sorted duplicate_groups list (count desc, artist asc, name
asc) + group_count scalar.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("artist") != exp["artist"]:
        return False
    if got.get("name") != exp["name"]:
        return False
    if got.get("count") != exp["count"]:
        return False
    got_tids = got.get("track_ids")
    if not isinstance(got_tids, list):
        return False
    return sorted(got_tids) == sorted(exp["track_ids"])


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("duplicate_groups")
    exp = ground_truth["duplicate_groups"]
    checks["has_duplicate_groups_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["groups_count_correct"] = len(got) == len(exp)
    entries_ok = checks["groups_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["groups_entries_correct"] = entries_ok

    checks["group_count_scalar_correct"] = (
        answer.get("group_count") == ground_truth["group_count"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
