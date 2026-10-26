"""Verifier for ptc_mt_org_chart_manager_churn_0.

Q2: The highest-assignments manager (churn hotspot), and the list of
    employees currently reporting to them (end_date == 'present') with each
    employee's own manager_turnover_count.
"""
from __future__ import annotations

from collections import defaultdict

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(config: dict) -> dict:
    cfg = config["OrgChartAPI"]
    history = cfg["reporting_history"]

    emp_distinct_mgrs: dict[str, set[str]] = defaultdict(set)
    mgr_assignments: dict[str, int] = defaultdict(int)
    mgr_names: dict[str, str] = {}
    emp_names: dict[str, str] = {}
    for row in history:
        emp_distinct_mgrs[row["employee_id"]].add(row["manager_id"])
        mgr_assignments[row["manager_id"]] += 1
        mgr_names[row["manager_id"]] = row["manager_name"]
        emp_names[row["employee_id"]] = row["employee_name"]

    churn_mgr = max(
        mgr_assignments,
        key=lambda m: (mgr_assignments[m], -ord(m[0]) if m else 0, m),
    )

    current_reports = []
    for row in history:
        if row["manager_id"] != churn_mgr:
            continue
        if row["end_date"] != "present":
            continue
        eid = row["employee_id"]
        current_reports.append({
            "employee_id": eid,
            "employee_name": emp_names[eid],
            "manager_turnover_count": len(emp_distinct_mgrs[eid]),
        })

    # Dedupe employees who might have duplicate "present" rows
    seen = set()
    deduped = []
    for r in current_reports:
        if r["employee_id"] in seen:
            continue
        seen.add(r["employee_id"])
        deduped.append(r)
    deduped.sort(key=lambda r: (-r["manager_turnover_count"], r["employee_id"]))

    return {
        "churn_manager_id": churn_mgr,
        "churn_manager_name": mgr_names[churn_mgr],
        "churn_manager_assignments": mgr_assignments[churn_mgr],
        "current_reports": deduped,
    }


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    rep_ans = answer.get("current_reports") or []
    rep_ans_ids = [r.get("employee_id") for r in rep_ans if isinstance(r, dict)]
    rep_exp_ids = [r["employee_id"] for r in expected["current_reports"]]
    checks = {
        "manager_id_correct":    answer.get("churn_manager_id") == expected["churn_manager_id"],
        "report_count_correct":  len(rep_ans) == len(expected["current_reports"]),
        "report_id_set":         set(rep_ans_ids) == set(rep_exp_ids),
    }

    exp_by_eid = {r["employee_id"]: r for r in expected["current_reports"]}
    turn_ok = True
    for r in rep_ans:
        if not isinstance(r, dict):
            turn_ok = False; break
        e = exp_by_eid.get(r.get("employee_id"))
        if e is None:
            continue
        if r.get("manager_turnover_count") != e["manager_turnover_count"]:
            turn_ok = False; break
    checks["turnover_counts_match"] = turn_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
