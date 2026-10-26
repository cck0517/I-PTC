"""Verifier for ptc_mt_org_chart_boomerangs_0.

Q1: (employee, manager) groups where the employee LEFT and CAME BACK
    (>=30-day gap between consecutive closed stints).
Q2: top-3 magnetic_managers by distinct boomerang-employee count
    + total_gap_days (sum of >=30-day gaps) per Q1 boomerang.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import name_by_id, parse_date
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_boomerangs(cfg: dict) -> list[dict]:
    history = cfg["reporting_history"]
    nbi = name_by_id(cfg)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in history:
        groups[(r["employee_id"], r["manager_id"])].append(r)

    out = []
    for (eid, mid), rows in groups.items():
        rows = sorted(rows, key=lambda r: r["start_date"])
        gap_total = 0
        is_boomerang = False
        for i in range(len(rows) - 1):
            ri, rj = rows[i], rows[i + 1]
            if ri["end_date"]:
                end_i = parse_date(ri["end_date"])
                start_j = parse_date(rj["start_date"])
                gap = (start_j - end_i).days
                if gap >= 30:
                    is_boomerang = True
                    gap_total += gap
        if is_boomerang:
            mgr_name = rows[0].get("manager_name") or nbi.get(mid, "")
            emp_name = rows[0].get("employee_name") or nbi.get(eid, "")
            out.append({
                "employee_id": eid,
                "employee_name": emp_name,
                "manager_id": mid,
                "manager_name": mgr_name,
                "stint_count": len(rows),
                "total_gap_days": gap_total,
            })
    out.sort(key=lambda b: (-b["stint_count"], b["employee_name"], b["manager_name"]))
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    booms = _q1_boomerangs(cfg)

    by_mgr: dict[str, set[str]] = defaultdict(set)
    for b in booms:
        by_mgr[b["manager_id"]].add(b["employee_id"])
    magnetic = []
    for mid, emps in by_mgr.items():
        magnetic.append({
            "manager_id": mid,
            "manager_name": nbi.get(mid, ""),
            "distinct_boomerang_employees": len(emps),
        })
    magnetic.sort(key=lambda m: (-m["distinct_boomerang_employees"], m["manager_name"]))
    magnetic = magnetic[:3]

    boomerang_gap_days = [
        {"employee_id": b["employee_id"], "manager_id": b["manager_id"],
         "total_gap_days": b["total_gap_days"]}
        for b in booms
    ]
    return {
        "magnetic_managers": magnetic,
        "boomerang_gap_days": boomerang_gap_days,
    }


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)

    mag_ans = answer.get("magnetic_managers") or []
    mag_exp = expected["magnetic_managers"]
    checks: dict[str, bool] = {
        "magnetic_count": len(mag_ans) == len(mag_exp),
    }
    ans_mids = [m.get("manager_id") for m in mag_ans if isinstance(m, dict)]
    exp_mids = [m["manager_id"] for m in mag_exp]
    checks["magnetic_set"] = set(ans_mids) == set(exp_mids)
    checks["magnetic_order"] = ans_mids == exp_mids
    exp_by_mid = {m["manager_id"]: m for m in mag_exp}
    cnt_ok = True
    for m in mag_ans:
        if not isinstance(m, dict):
            cnt_ok = False; break
        e = exp_by_mid.get(m.get("manager_id"))
        if e is None:
            continue
        if m.get("distinct_boomerang_employees") != e["distinct_boomerang_employees"]:
            cnt_ok = False; break
    checks["magnetic_count_match"] = cnt_ok

    bgd_ans = answer.get("boomerang_gap_days") or []
    bgd_exp = expected["boomerang_gap_days"]
    checks["bgd_count"] = len(bgd_ans) == len(bgd_exp)
    exp_by_pair = {(b["employee_id"], b["manager_id"]): b for b in bgd_exp}
    gap_ok = True
    for b in bgd_ans:
        if not isinstance(b, dict):
            gap_ok = False; break
        e = exp_by_pair.get((b.get("employee_id"), b.get("manager_id")))
        if e is None:
            gap_ok = False; break
        if b.get("total_gap_days") != e["total_gap_days"]:
            gap_ok = False; break
    checks["bgd_total_gap_days_match"] = gap_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
