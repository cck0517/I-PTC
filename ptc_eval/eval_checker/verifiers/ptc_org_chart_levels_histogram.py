"""Verifier for ptc_org_chart_levels_histogram_0.

Checks the ordered ``levels`` list — entry per level with count and
avg_direct_reports (2-dec tolerance).
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_AVG_TOL = 0.01


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("level") != exp["level"] or got.get("count") != exp["count"]:
        return False
    a = _num(got.get("avg_direct_reports"))
    return a is not None and abs(a - exp["avg_direct_reports"]) <= _AVG_TOL


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("levels")
    exp = ground_truth["levels"]
    checks["has_levels_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["levels_count_correct"] = len(got) == len(exp)
    entries_ok = checks["levels_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["levels_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
