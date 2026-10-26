"""Verifier for ptc_mt_calendar_utc_heatmap_0.

Q1: 24-hour UTC histogram of event start hours + top-3 peak hours.
Q2: bottom-3 quietest hours, coverage_score, business_hours_share (UTC 09-16).
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

from ._calendar_helpers import employees_by_id, to_utc
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _histogram(cfg: dict) -> dict[int, int]:
    emps = employees_by_id(cfg)
    hist = {h: 0 for h in range(24)}
    for ev in cfg["events"]:
        emp = emps.get(ev["employee_id"])
        if not emp:
            continue
        tz = ZoneInfo(emp["timezone"])
        u = to_utc(ev["start"], tz)
        hist[u.hour] += 1
    return hist


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    hist = _histogram(cfg)
    total = sum(hist.values())
    bottom3 = sorted(
        [{"utc_hour": h, "meeting_count": c} for h, c in hist.items()],
        key=lambda x: (x["meeting_count"], x["utc_hour"]),
    )[:3]
    coverage = sum(1 for c in hist.values() if c > 0)
    bh_count = sum(c for h, c in hist.items() if 9 <= h <= 16)
    return {
        "bottom3_quietest_hours": bottom3,
        "coverage_score": coverage,
        "business_hours_share": round(bh_count / total, 4) if total else 0.0,
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

    b_ans = answer.get("bottom3_quietest_hours") or []
    b_exp = expected["bottom3_quietest_hours"]
    checks: dict[str, bool] = {
        "bottom3_count": len(b_ans) == len(b_exp),
        "coverage_score": answer.get("coverage_score") == expected["coverage_score"],
    }
    try:
        checks["business_hours_share"] = (
            abs(float(answer.get("business_hours_share", -1))
                - expected["business_hours_share"]) <= 0.0005
        )
    except (TypeError, ValueError):
        checks["business_hours_share"] = False

    ans_hours = [b.get("utc_hour") for b in b_ans if isinstance(b, dict)]
    exp_hours = [b["utc_hour"] for b in b_exp]
    checks["bottom3_set"] = set(ans_hours) == set(exp_hours)
    checks["bottom3_order"] = ans_hours == exp_hours

    exp_by_h = {b["utc_hour"]: b for b in b_exp}
    cnt_ok = True
    for b in b_ans:
        if not isinstance(b, dict):
            cnt_ok = False; break
        e = exp_by_h.get(b.get("utc_hour"))
        if e is None:
            continue
        if b.get("meeting_count") != e["meeting_count"]:
            cnt_ok = False; break
    checks["bottom3_counts_match"] = cnt_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
