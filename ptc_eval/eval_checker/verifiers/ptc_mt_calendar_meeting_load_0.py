"""Verifier for ptc_mt_calendar_meeting_load_0.

Q1: top-3 employees by total_meeting_minutes (accepted only) + top partner.
Q2: team_total_meeting_minutes, median_meeting_minutes, tier2 (ranks 4-6).
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from ._calendar_helpers import (
    employees_by_id, event_duration_minutes,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_employee_minutes(cfg: dict) -> list[dict]:
    emps = employees_by_id(cfg)
    minutes = defaultdict(int)
    for ev in cfg["events"]:
        if ev.get("rsvp") != "accepted":
            continue
        minutes[ev["employee_id"]] += event_duration_minutes(ev)
    rows = []
    for eid, e in emps.items():
        rows.append({
            "employee_id": eid,
            "display_name": e["display_name"],
            "total_meeting_minutes": minutes.get(eid, 0),
        })
    rows.sort(key=lambda r: (-r["total_meeting_minutes"], r["display_name"]))
    return rows


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    rows = _per_employee_minutes(cfg)
    minutes = [r["total_meeting_minutes"] for r in rows]
    return {
        "team_total_meeting_minutes": sum(minutes),
        "median_meeting_minutes": round(statistics.median(minutes), 2) if minutes else 0.0,
        "tier2": rows[3:6],
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
    checks: dict[str, bool] = {
        "team_total_meeting_minutes": (
            answer.get("team_total_meeting_minutes") == expected["team_total_meeting_minutes"]
        ),
        "median_meeting_minutes": (
            abs(float(answer.get("median_meeting_minutes", -1)) -
                expected["median_meeting_minutes"]) <= 0.05
        ),
    }
    t2_ans = answer.get("tier2") or []
    t2_exp = expected["tier2"]
    checks["tier2_count"] = len(t2_ans) == len(t2_exp)
    ans_ids = [t.get("employee_id") for t in t2_ans if isinstance(t, dict)]
    exp_ids = [t["employee_id"] for t in t2_exp]
    checks["tier2_set"] = set(ans_ids) == set(exp_ids)
    checks["tier2_order"] = ans_ids == exp_ids

    exp_by_id = {t["employee_id"]: t for t in t2_exp}
    minutes_ok = True
    for t in t2_ans:
        if not isinstance(t, dict):
            minutes_ok = False; break
        e = exp_by_id.get(t.get("employee_id"))
        if e is None:
            continue
        if t.get("total_meeting_minutes") != e["total_meeting_minutes"]:
            minutes_ok = False; break
    checks["tier2_minutes_match"] = minutes_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
