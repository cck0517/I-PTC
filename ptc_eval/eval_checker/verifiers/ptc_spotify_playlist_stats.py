"""Verifier for ptc_spotify_playlist_stats_0.

Checks top3_largest (count desc) and bottom3_smallest (count asc). Each
entry has playlist_id, name, track_count, total_duration_ms,
avg_duration_ms. avg tolerance ±1.0 ms.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_AVG_TOL = 1.0


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("playlist_id", "name", "track_count", "total_duration_ms"):
        if got.get(k) != exp[k]:
            return False
    a = _num(got.get("avg_duration_ms"))
    return a is not None and abs(a - exp["avg_duration_ms"]) <= _AVG_TOL


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

    checks["has_top3_key"] = isinstance(answer.get("top3_largest"), list)
    checks["has_bottom3_key"] = isinstance(answer.get("bottom3_smallest"), list)
    checks["top3_correct"] = _list_match(
        answer.get("top3_largest"), ground_truth["top3_largest"]
    )
    checks["bottom3_correct"] = _list_match(
        answer.get("bottom3_smallest"), ground_truth["bottom3_smallest"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
