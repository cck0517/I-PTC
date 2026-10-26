"""Verifier for ptc_spotify_artist_concentration_0.

Checks most_concentrated (top-3 share desc) and most_diverse (bottom-3
share asc). Share tolerance ±0.0005.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_S_TOL = 5e-4


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("playlist_id") != exp["playlist_id"]:
        return False
    if got.get("name") != exp["name"]:
        return False
    if got.get("top2_artists") != exp["top2_artists"]:
        return False
    s = _num(got.get("top2_track_share"))
    return s is not None and abs(s - exp["top2_track_share"]) <= _S_TOL


def _list_match(got: Any, exp: list) -> bool:
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

    checks["has_most_concentrated_key"] = isinstance(answer.get("most_concentrated"), list)
    checks["has_most_diverse_key"] = isinstance(answer.get("most_diverse"), list)
    checks["most_concentrated_correct"] = _list_match(
        answer.get("most_concentrated"), ground_truth["most_concentrated"]
    )
    checks["most_diverse_correct"] = _list_match(
        answer.get("most_diverse"), ground_truth["most_diverse"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
