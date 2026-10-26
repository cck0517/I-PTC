"""Verifier for ptc_mt_calendar_rsvp_audit_0.

Q1: top-3 by declined_rate + top-3 by tentative_rate.
Q2: top-3 noisy_responders by combined_uncertain_rate + team_avg_decline_rate.
"""
from __future__ import annotations

from collections import defaultdict

from ._calendar_helpers import employees_by_id
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_employee_rates(cfg: dict) -> list[dict]:
    emps = employees_by_id(cfg)
    counts: dict[str, dict] = {
        eid: {"total": 0, "declined": 0, "tentative": 0} for eid in emps
    }
    for ev in cfg["events"]:
        d = counts.get(ev["employee_id"])
        if d is None:
            continue
        d["total"] += 1
        rsvp = ev.get("rsvp")
        if rsvp == "declined":
            d["declined"] += 1
        elif rsvp == "tentative":
            d["tentative"] += 1
    rows = []
    for eid, d in counts.items():
        total = d["total"]
        dr = round(d["declined"] / total, 4) if total else 0.0
        tr = round(d["tentative"] / total, 4) if total else 0.0
        rows.append({
            "employee_id": eid,
            "display_name": emps[eid]["display_name"],
            "total_events": total,
            "declined": d["declined"],
            "tentative": d["tentative"],
            "declined_rate": dr,
            "tentative_rate": tr,
            "combined_uncertain_rate": round(dr + tr, 4),
        })
    return rows


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    rows = _per_employee_rates(cfg)
    top3 = sorted(rows, key=lambda r: (-r["combined_uncertain_rate"], r["display_name"]))[:3]
    avg_decline = round(
        sum(r["declined_rate"] for r in rows) / len(rows), 4
    ) if rows else 0.0
    return {
        "top3_noisy_responders": top3,
        "team_avg_decline_rate": avg_decline,
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
    t_ans = answer.get("top3_noisy_responders") or []
    t_exp = expected["top3_noisy_responders"]
    checks: dict[str, bool] = {
        "count_match": len(t_ans) == len(t_exp),
        "team_avg_decline_rate": False,
    }
    try:
        checks["team_avg_decline_rate"] = (
            abs(float(answer.get("team_avg_decline_rate", -1)) -
                expected["team_avg_decline_rate"]) <= 0.0005
        )
    except (TypeError, ValueError):
        pass

    ans_ids = [t.get("employee_id") for t in t_ans if isinstance(t, dict)]
    exp_ids = [t["employee_id"] for t in t_exp]
    checks["employee_set"] = set(ans_ids) == set(exp_ids)
    checks["employee_order"] = ans_ids == exp_ids

    exp_by_id = {t["employee_id"]: t for t in t_exp}
    rate_ok = True
    for t in t_ans:
        if not isinstance(t, dict):
            rate_ok = False; break
        e = exp_by_id.get(t.get("employee_id"))
        if e is None:
            continue
        try:
            if abs(float(t.get("combined_uncertain_rate", -1)) -
                   e["combined_uncertain_rate"]) > 0.0005:
                rate_ok = False; break
        except (TypeError, ValueError):
            rate_ok = False; break
    checks["combined_rate_match"] = rate_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
