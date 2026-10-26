"""Verifier for ptc_intermediate_org_teammate_degree_0."""

from __future__ import annotations

from datetime import date
from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "OrgChartAPIAlt", initial_config.get("OrgChartAPI", {}),
    )


def _intervals_overlap(a_s, a_e, b_s, b_e) -> bool:
    return a_s < b_e and b_s < a_e


def _compute_ground_truth(cfg: dict) -> dict:
    employees = cfg["employees"]
    history = cfg["reporting_history"]
    # rows_by_mgr[mgr_id] = list of (emp_id, start_date, end_date)
    rows_by_mgr: dict[str, list[tuple[str, date, date]]] = {}
    for row in history:
        mid = row["manager_id"]
        eid = row["employee_id"]
        s = date.fromisoformat(row["start_date"])
        e = date.fromisoformat(row["end_date"])
        rows_by_mgr.setdefault(mid, []).append((eid, s, e))

    teammates: dict[str, set[str]] = {e["employee_id"]: set() for e in employees}
    for rows in rows_by_mgr.values():
        for i in range(len(rows)):
            ei, si, eei = rows[i]
            for j in range(i + 1, len(rows)):
                ej, sj, eej = rows[j]
                if ei == ej:
                    continue
                if _intervals_overlap(si, eei, sj, eej):
                    teammates[ei].add(ej)
                    teammates[ej].add(ei)

    name_by_emp = {e["employee_id"]: e["employee_name"] for e in employees}
    def _ord(eid: str) -> tuple[int, str]:
        try:
            return (int(eid.split("_")[-1]), eid)
        except (ValueError, IndexError):
            return (10**9, eid)
    ranked = sorted(
        teammates.items(),
        key=lambda kv: (-len(kv[1]), _ord(kv[0])),
    )
    best_eid, best_set = ranked[0]
    return {
        "employee_id": best_eid,
        "employee_name": name_by_emp[best_eid],
        "teammate_count": len(best_set),
        "teammate_ids": sorted(best_set),
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

    checks["employee_id_correct"] = answer.get("employee_id") == expected["employee_id"]
    try:
        cnt = int(answer.get("teammate_count"))
        checks["count_correct"] = cnt == expected["teammate_count"]
    except (TypeError, ValueError):
        checks["count_correct"] = False
    got = answer.get("teammate_ids") or []
    if not isinstance(got, list):
        got = []
    checks["ids_correct"] = sorted(str(x) for x in got) == expected["teammate_ids"]

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
