"""Verifier for ptc_mt_org_chart_drill_down_0 (multi-turn).

Q1: For every manager compute unique_employee_count / total_periods /
    reports(list of {employee_id, employee_name, start_date, end_date}),
    sorted by unique_employee_count desc.
Q2: For the top-3 managers by unique_employee_count:
      - avg_periods_per_employee = (# reporting rows for that manager) /
        (# distinct employees they supervised).
    Plus shared_employees: any employee who reported to more than one of
    the top-3 managers, with the list of those managers.

Expected JSON shape:
    top_managers: [{manager_id, manager_name, unique_employee_count,
                    avg_periods_per_employee}]  (sorted by unique desc)
    shared_employees: [{employee_id, employee_name, manager_ids}]
"""

from __future__ import annotations

from collections import defaultdict

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


def _compute_expected(config: dict) -> dict:
    cfg = config["OrgChartAPI"]
    history = cfg["reporting_history"]

    # Per manager: set of unique employee_ids + total reporting rows + name.
    by_mgr: dict[str, dict] = defaultdict(lambda: {
        "manager_name": None,
        "employees": set(),
        "total_periods": 0,
    })
    for row in history:
        mid = row["manager_id"]
        by_mgr[mid]["manager_name"] = row["manager_name"]
        by_mgr[mid]["employees"].add(row["employee_id"])
        by_mgr[mid]["total_periods"] += 1

    # Rank by unique count desc; tie-break by total_periods desc, then by id.
    ranked = sorted(
        by_mgr.items(),
        key=lambda kv: (-len(kv[1]["employees"]), -kv[1]["total_periods"], kv[0]),
    )
    top3 = ranked[:3]

    top_managers = []
    for mid, info in top3:
        unique_n = len(info["employees"])
        avg_periods = round(info["total_periods"] / unique_n, 2) if unique_n else 0.0
        top_managers.append({
            "manager_id": mid,
            "manager_name": info["manager_name"],
            "unique_employee_count": unique_n,
            "avg_periods_per_employee": avg_periods,
        })

    top3_ids = {mid for mid, _ in top3}

    # Shared employees: reported to >=2 of top3's managers.
    emp_to_mgrs: dict[str, tuple[str, set[str]]] = {}
    for row in history:
        if row["manager_id"] not in top3_ids:
            continue
        eid = row["employee_id"]
        if eid not in emp_to_mgrs:
            emp_to_mgrs[eid] = (row["employee_name"], set())
        emp_to_mgrs[eid][1].add(row["manager_id"])

    shared_employees = []
    for eid, (ename, mgrs) in emp_to_mgrs.items():
        if len(mgrs) >= 2:
            shared_employees.append({
                "employee_id": eid,
                "employee_name": ename,
                "manager_ids": sorted(mgrs),
            })
    shared_employees.sort(key=lambda r: r["employee_id"])

    return {
        "top_managers": top_managers,
        "shared_employees": shared_employees,
    }


def _approx_eq(a, b, tol=0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
    turn_answers: list[str | None] | None = None,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )

    expected = _compute_expected(initial_config)

    top_ans = answer.get("top_managers") or []
    top_exp = expected["top_managers"]
    checks: dict[str, bool] = {
        "top_managers_count":  len(top_ans) == len(top_exp),
    }

    exp_by_mid = {t["manager_id"]: t for t in top_exp}
    ans_ids = [t.get("manager_id") for t in top_ans if isinstance(t, dict)]
    checks["top_managers_set"] = set(ans_ids) == set(exp_by_mid)

    mgr_fields_ok = True
    for t in top_ans:
        if not isinstance(t, dict):
            mgr_fields_ok = False; break
        e = exp_by_mid.get(t.get("manager_id"))
        if e is None:
            continue
        if t.get("unique_employee_count") != e["unique_employee_count"]:
            mgr_fields_ok = False; break
        if not _approx_eq(t.get("avg_periods_per_employee"), e["avg_periods_per_employee"]):
            mgr_fields_ok = False; break
    checks["top_managers_fields"] = mgr_fields_ok

    # Shared-employee set match + consistent manager_ids set per employee.
    shared_ans = answer.get("shared_employees") or []
    shared_exp = expected["shared_employees"]
    exp_by_eid = {s["employee_id"]: s for s in shared_exp}
    ans_eids = [s.get("employee_id") for s in shared_ans if isinstance(s, dict)]
    checks["shared_count"] = len(shared_ans) == len(shared_exp)
    checks["shared_set"] = set(ans_eids) == set(exp_by_eid)

    shared_ok = True
    for s in shared_ans:
        if not isinstance(s, dict):
            shared_ok = False; break
        e = exp_by_eid.get(s.get("employee_id"))
        if e is None:
            continue
        if set(s.get("manager_ids") or []) != set(e["manager_ids"]):
            shared_ok = False; break
    checks["shared_manager_ids"] = shared_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
