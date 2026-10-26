"""Verifier for ptc_spotify_track_ubiquity_0.

Checks the ordered top-10 tracks by user-playlist appearance count
(count desc, name asc, artist asc).
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("track_id", "name", "artist", "playlist_count"):
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

    got = answer.get("top10")
    exp = ground_truth["top10"]
    checks["has_top10_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["top10_count_correct"] = len(got) == len(exp)
    entries_ok = checks["top10_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["top10_entries_and_order_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
