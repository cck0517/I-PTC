"""Verifier for ptc_spotify_recently_played_orphans_0.

Checks sorted orphans list (name asc, artist asc) + orphan_count.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    return (
        got.get("track_id") == exp["track_id"]
        and got.get("name") == exp["name"]
        and got.get("artist") == exp["artist"]
    )


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("orphans")
    exp = ground_truth["orphans"]
    checks["has_orphans_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["orphans_count_correct"] = len(got) == len(exp)
    entries_ok = checks["orphans_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["orphans_entries_correct"] = entries_ok

    checks["orphan_count_scalar_correct"] = (
        answer.get("orphan_count") == ground_truth["orphan_count"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
