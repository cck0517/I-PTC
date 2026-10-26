"""Verifier for ptc_spotify_artist_reach_0.

Checks the full sorted artists list. Each artist entry has the sorted
inner playlists list.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _playlists_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        for k in ("playlist_id", "playlist_name", "track_count"):
            if g.get(k) != e[k]:
                return False
    return True


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("artist") != exp["artist"]:
        return False
    if got.get("playlist_count") != exp["playlist_count"]:
        return False
    return _playlists_match(got.get("playlists"), exp["playlists"])


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("top10_artists")
    exp = ground_truth["top10_artists"]
    checks["has_top10_artists_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["top10_count_correct"] = len(got) == len(exp)
    entries_ok = checks["top10_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["top10_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
