"""Verifier for ptc_spotify_core_repertoire_0.

Checks top5_largest (sorted by count desc), core_tracks (intersection),
and core_count.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _pl_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("playlist_id", "name", "track_count"):
        if got.get(k) != exp[k]:
            return False
    return True


def _track_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("track_id", "name", "artist"):
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

    got_top = answer.get("top5_largest")
    exp_top = ground_truth["top5_largest"]
    checks["has_top5_key"] = isinstance(got_top, list)
    if not isinstance(got_top, list):
        got_top = []
    top_ok = len(got_top) == len(exp_top)
    if top_ok:
        for g, e in zip(got_top, exp_top):
            if not _pl_matches(g, e):
                top_ok = False; break
    checks["top5_largest_correct"] = top_ok

    got_core = answer.get("core_tracks")
    exp_core = ground_truth["core_tracks"]
    checks["has_core_tracks_key"] = isinstance(got_core, list)
    if not isinstance(got_core, list):
        got_core = []
    core_ok = len(got_core) == len(exp_core)
    if core_ok:
        for g, e in zip(got_core, exp_core):
            if not _track_matches(g, e):
                core_ok = False; break
    checks["core_tracks_correct"] = core_ok

    checks["core_count_correct"] = answer.get("core_count") == ground_truth["core_count"]

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
