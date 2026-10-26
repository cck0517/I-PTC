"""Verifier for ptc_mt_calendar_self_conflict_0.

Q1: per-employee overlapping events on their own calendar.
Q2: near-conflicts — consecutive non-overlapping events with gap < 5 min.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from ._calendar_helpers import employees_by_id, events_by_employee
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    emps = employees_by_id(cfg)
    by_emp = events_by_employee(cfg)

    rows = []
    for eid, evs in by_emp.items():
        # Sort by start ascending; tiebreak by end ascending (irrelevant for non-overlapping).
        evs_sorted = sorted(evs, key=lambda e: (e["start"], e["end"]))
        nc = []
        for i in range(len(evs_sorted) - 1):
            a, b = evs_sorted[i], evs_sorted[i + 1]
            a_end = datetime.fromisoformat(a["end"])
            b_start = datetime.fromisoformat(b["start"])
            if a_end > b_start:
                continue  # overlapping pair — covered by Q1, not a near-conflict
            gap = int((b_start - a_end).total_seconds() // 60)
            if 0 <= gap < 5:
                nc.append({
                    "prior_title": a["title"],
                    "next_title": b["title"],
                    "prior_end": a["end"],
                    "next_start": b["start"],
                    "gap_minutes": gap,
                })
        if nc:
            nc.sort(key=lambda x: (x["prior_end"], x["prior_title"]))
            rows.append({
                "employee_id": eid,
                "display_name": emps[eid]["display_name"],
                "near_conflict_count": len(nc),
                "near_conflicts": nc,
            })
    rows.sort(key=lambda r: (-r["near_conflict_count"], r["display_name"]))
    return {"top5_near_conflict_employees": rows[:5]}


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
    n_ans = answer.get("top5_near_conflict_employees") or []
    n_exp = expected["top5_near_conflict_employees"]
    checks: dict[str, bool] = {
        "count_match": len(n_ans) == len(n_exp),
    }
    ans_ids = [r.get("employee_id") for r in n_ans if isinstance(r, dict)]
    exp_ids = [r["employee_id"] for r in n_exp]
    checks["employee_set"] = set(ans_ids) == set(exp_ids)
    checks["employee_order"] = ans_ids == exp_ids

    exp_by_id = {r["employee_id"]: r for r in n_exp}
    cnt_ok = True
    pair_ok = True
    for r in n_ans:
        if not isinstance(r, dict):
            cnt_ok = pair_ok = False; break
        e = exp_by_id.get(r.get("employee_id"))
        if e is None:
            continue
        if r.get("near_conflict_count") != e["near_conflict_count"]:
            cnt_ok = False
        ans_pairs = {
            (nc.get("prior_end"), nc.get("next_start"), nc.get("gap_minutes"))
            for nc in (r.get("near_conflicts") or []) if isinstance(nc, dict)
        }
        exp_pairs = {
            (nc["prior_end"], nc["next_start"], nc["gap_minutes"])
            for nc in e["near_conflicts"]
        }
        if ans_pairs != exp_pairs:
            pair_ok = False
    checks["near_conflict_count_match"] = cnt_ok
    checks["near_conflict_pairs_match"] = pair_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
