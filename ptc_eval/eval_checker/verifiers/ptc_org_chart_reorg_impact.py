"""Verifier for ptc_org_chart_reorg_impact_0.

Checks the sorted changes list (divergence_depth asc, employee_name asc)
plus the unchanged/changed counts.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("employee_id", "employee_name", "divergence_depth",
             "chain_before", "chain_after"):
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

    got = answer.get("changes")
    exp = ground_truth["changes"]
    checks["has_changes_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["changes_count_correct"] = len(got) == len(exp)
    entries_ok = checks["changes_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["changes_entries_correct"] = entries_ok

    checks["unchanged_count_correct"] = (
        answer.get("unchanged_count") == ground_truth["unchanged_count"]
    )
    checks["changed_count_correct"] = (
        answer.get("changed_count") == ground_truth["changed_count"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
