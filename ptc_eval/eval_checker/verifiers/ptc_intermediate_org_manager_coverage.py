"""Verifier for ptc_intermediate_org_manager_coverage_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "OrgChartAPIAlt", initial_config.get("OrgChartAPI", {}),
    )


def _compute_ground_truth(cfg: dict) -> dict:
    history = cfg["reporting_history"]
    emps_by_mgr: dict[str, set[str]] = {}
    mgr_names: dict[str, str] = {}
    for row in history:
        mid = row["manager_id"]
        emps_by_mgr.setdefault(mid, set()).add(row["employee_id"])
        mgr_names[mid] = row.get("manager_name", "")
    # pick max by distinct-count; tie-break by smallest manager_id (numeric id)
    def _mgr_num(mid: str) -> tuple[int, str]:
        try:
            return (int(mid.split("_")[-1]), mid)
        except (ValueError, IndexError):
            return (10**9, mid)
    ranked = sorted(
        emps_by_mgr.items(),
        key=lambda kv: (-len(kv[1]), _mgr_num(kv[0])),
    )
    best_mid, best_set = ranked[0]
    return {
        "manager_id": best_mid,
        "manager_name": mgr_names[best_mid],
        "distinct_employee_count": len(best_set),
        "distinct_employee_ids": sorted(best_set),
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0, details=details,
            errors=["Could not extract JSON object"],
        )
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected"] = expected

    checks["manager_id_correct"] = answer.get("manager_id") == expected["manager_id"]
    try:
        cnt = int(answer.get("distinct_employee_count"))
        checks["count_correct"] = cnt == expected["distinct_employee_count"]
    except (TypeError, ValueError):
        checks["count_correct"] = False
    got_ids = answer.get("distinct_employee_ids") or []
    if not isinstance(got_ids, list):
        got_ids = []
    checks["ids_correct"] = sorted(str(x) for x in got_ids) == expected["distinct_employee_ids"]

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
