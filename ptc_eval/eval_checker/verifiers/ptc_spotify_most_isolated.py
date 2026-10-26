"""Verifier for ptc_spotify_most_isolated_0.

Checks the ordered bottom-3 by max_jaccard_to_other (asc). Jaccard
tolerance ±0.0005.
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
    if got.get("playlist_id") != exp["playlist_id"]:
        return False
    if got.get("name") != exp["name"]:
        return False
    j = _num(got.get("max_jaccard_to_other"))
    return j is not None and abs(j - exp["max_jaccard_to_other"]) <= _J_TOL


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("most_isolated")
    exp = ground_truth["most_isolated"]
    checks["has_most_isolated_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["most_isolated_count_correct"] = len(got) == len(exp)
    entries_ok = checks["most_isolated_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["most_isolated_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
