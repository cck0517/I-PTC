"""Verifier for ptc_org_chart_reorg_churn_0.

Checks the ordered top-5 (by distinct_manager_count desc, name asc) —
employee id/name, count, and the sorted manager_names list.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("employee_id") != exp["employee_id"]:
        return False
    if got.get("employee_name") != exp["employee_name"]:
        return False
    if got.get("distinct_manager_count") != exp["distinct_manager_count"]:
        return False
    got_names = got.get("manager_names")
    if not isinstance(got_names, list):
        return False
    return sorted(got_names) == sorted(exp["manager_names"])


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("top5")
    exp = ground_truth["top5"]
    checks["has_top5_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["top5_count_correct"] = len(got) == len(exp)
    entries_ok = checks["top5_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["top5_entries_and_order_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
