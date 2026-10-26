"""Verifier for ptc_spotify_similarity_extremes_0.

Checks most_similar (top-3 by jaccard desc) and most_different
(bottom-3 by jaccard asc). Jaccard tolerance ±0.0005.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_J_TOL = 5e-4


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("pair") != exp["pair"]:
        return False
    if got.get("playlist_names") != exp["playlist_names"]:
        return False
    j = _num(got.get("jaccard"))
    return j is not None and abs(j - exp["jaccard"]) <= _J_TOL


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

    checks["has_most_similar_key"] = isinstance(answer.get("most_similar"), list)
    checks["has_most_different_key"] = isinstance(answer.get("most_different"), list)
    checks["most_similar_correct"] = _list_match(
        answer.get("most_similar"), ground_truth["most_similar"]
    )
    checks["most_different_correct"] = _list_match(
        answer.get("most_different"), ground_truth["most_different"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
