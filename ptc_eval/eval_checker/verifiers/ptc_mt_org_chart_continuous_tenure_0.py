"""Verifier for ptc_mt_org_chart_continuous_tenure_0.

Q1: top-10 longest continuous tenures with current manager on 2026-01-15
    (merging gaps <= 7 days).
Q2: top-3 sticky managers from those 10 (most retained, with mean tenure).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import (
    OPEN_END, active_manager_on, name_by_id, parse_date,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


SNAPSHOT = date(2026, 1, 15)


def _q1_top10(cfg: dict) -> list[dict]:
    history = cfg["reporting_history"]
    nbi = name_by_id(cfg)
    am = active_manager_on(history, SNAPSHOT)

    out = []
    for eid, (_, mid) in am.items():
        rows = sorted(
            [r for r in history if r["employee_id"] == eid and r["manager_id"] == mid],
            key=lambda r: r["start_date"],
        )
        merged: list[list[date]] = []
        for r in rows:
            s = parse_date(r["start_date"])
            e = parse_date(r["end_date"])
            if not merged:
                merged.append([s, e])
                continue
            last_s, last_e = merged[-1]
            if last_e == OPEN_END or (s - last_e).days <= 7:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
        cont = next(((ms, me) for ms, me in merged if ms <= SNAPSHOT <= me), None)
        if cont is None:
            continue
        out.append({
            "employee_id": eid,
            "employee_name": nbi.get(eid, ""),
            "current_manager_id": mid,
            "current_manager_name": nbi.get(mid, ""),
            "continuous_tenure_days": (SNAPSHOT - cont[0]).days,
            "tenure_start_date": cont[0].isoformat(),
        })
    out.sort(key=lambda t: (-t["continuous_tenure_days"], t["employee_name"]))
    return out[:10]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    top10 = _q1_top10(cfg)

    by_mgr: dict[str, list[dict]] = defaultdict(list)
    for t in top10:
        by_mgr[t["current_manager_id"]].append(t)

    sticky = []
    for mid, ts in by_mgr.items():
        retained = sorted(
            [{
                "employee_id": t["employee_id"],
                "employee_name": t["employee_name"],
                "continuous_tenure_days": t["continuous_tenure_days"],
            } for t in ts],
            key=lambda x: (-x["continuous_tenure_days"], x["employee_name"]),
        )
        sticky.append({
            "manager_id": mid,
            "manager_name": nbi.get(mid, ""),
            "retained_count": len(ts),
            "mean_tenure_days": round(
                sum(t["continuous_tenure_days"] for t in ts) / len(ts), 2
            ),
            "retained_employees": retained,
        })
    sticky.sort(key=lambda s: (-s["retained_count"], s["manager_name"]))
    return {"sticky_managers": sticky[:3]}


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

    sm_ans = answer.get("sticky_managers") or []
    sm_exp = expected["sticky_managers"]
    checks: dict[str, bool] = {
        "sticky_count": len(sm_ans) == len(sm_exp),
    }
    ans_mids = [m.get("manager_id") for m in sm_ans if isinstance(m, dict)]
    exp_mids = [m["manager_id"] for m in sm_exp]
    checks["sticky_set"] = set(ans_mids) == set(exp_mids)
    checks["sticky_order"] = ans_mids == exp_mids

    exp_by_mid = {m["manager_id"]: m for m in sm_exp}
    fields_ok = True
    for m in sm_ans:
        if not isinstance(m, dict):
            fields_ok = False; break
        e = exp_by_mid.get(m.get("manager_id"))
        if e is None:
            continue
        if m.get("retained_count") != e["retained_count"]:
            fields_ok = False; break
        try:
            if abs(float(m.get("mean_tenure_days", -1)) - e["mean_tenure_days"]) > 0.05:
                fields_ok = False; break
        except (TypeError, ValueError):
            fields_ok = False; break
        ans_emps = [r.get("employee_id") for r in (m.get("retained_employees") or [])
                    if isinstance(r, dict)]
        exp_emps = [r["employee_id"] for r in e["retained_employees"]]
        if set(ans_emps) != set(exp_emps):
            fields_ok = False; break
    checks["sticky_fields_match"] = fields_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
