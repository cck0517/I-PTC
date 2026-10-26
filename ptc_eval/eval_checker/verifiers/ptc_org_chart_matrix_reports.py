"""Verifier for ptc_org_chart_matrix_reports_0.

Checks the sorted matrix_reports list (total_overlap_days desc,
employee_name asc), each with overlap_intervals and total day count.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_DAY_TOL = 1


def _intervals_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if g.get("start") != e["start"] or g.get("end") != e["end"]:
            return False
        got_mids = g.get("manager_ids")
        if not isinstance(got_mids, list) or sorted(got_mids) != sorted(e["manager_ids"]):
            return False
    return True


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("employee_id") != exp["employee_id"]:
        return False
    if got.get("employee_name") != exp["employee_name"]:
        return False
    got_days = got.get("total_overlap_days")
    try:
        got_days_i = int(got_days)
    except (TypeError, ValueError):
        return False
    if abs(got_days_i - exp["total_overlap_days"]) > _DAY_TOL:
        return False
    return _intervals_match(got.get("overlap_intervals"), exp["overlap_intervals"])


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("matrix_reports")
    exp = ground_truth["matrix_reports"]
    checks["has_matrix_reports_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["matrix_reports_count_correct"] = len(got) == len(exp)
    entries_ok = checks["matrix_reports_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["matrix_reports_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
