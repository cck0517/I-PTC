"""Verifier for ptc_org_chart_boomerangs_0.

Checks the sorted boomerangs list (stint_count desc, employee_name asc,
manager_name asc). Each entry carries (employee_id, employee_name,
manager_id, manager_name, stint_count, stints).
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _stints_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if g.get("start") != e["start"] or g.get("end") != e["end"]:
            return False
    return True


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("employee_id", "employee_name", "manager_id", "manager_name",
             "stint_count"):
        if got.get(k) != exp[k]:
            return False
    return _stints_match(got.get("stints"), exp["stints"])


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("boomerangs")
    exp = ground_truth["boomerangs"]
    checks["has_boomerangs_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["boomerangs_count_correct"] = len(got) == len(exp)
    entries_ok = checks["boomerangs_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["boomerangs_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
