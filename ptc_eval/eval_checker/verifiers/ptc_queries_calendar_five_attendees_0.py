"""Verifier for ptc_queries_calendar_five_attendees_0.

Check that the proposed 45-minute slot falls in next week and every attendee's
local time is between 10 AM and 3 PM.
"""

from __future__ import annotations
from datetime import date, time, timedelta
from zoneinfo import ZoneInfo
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, extract_utc_timestamps,
    utc_to_local, score_from_checks,
)
from datetime import datetime


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    cal_cfg = initial_config["EnterpriseCalendarAPI"]
    meta = cal_cfg["metadata"]
    employees = cal_cfg["employees"]

    timestamps = extract_utc_timestamps(final_answer or "")
    answer_json = extract_json_from_text(final_answer)

    start_utc_str = end_utc_str = None
    if len(timestamps) >= 2:
        start_utc_str, end_utc_str = timestamps[0], timestamps[1]
    elif answer_json and isinstance(answer_json, dict):
        start_utc_str = answer_json.get("meeting_start_utc", answer_json.get("start_utc"))
        end_utc_str = answer_json.get("meeting_end_utc", answer_json.get("end_utc"))

    if not start_utc_str or not end_utc_str:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract UTC start/end from answer"],
            details={"raw_answer": final_answer},
        )

    start_utc = datetime.fromisoformat(start_utc_str.rstrip("Z")).replace(tzinfo=ZoneInfo("UTC"))
    end_utc = datetime.fromisoformat(end_utc_str.rstrip("Z")).replace(tzinfo=ZoneInfo("UTC"))

    duration = (end_utc - start_utc).total_seconds() / 60
    checks["duration_45min"] = abs(duration - 45) < 1
    if not checks["duration_45min"]:
        errors.append(f"Duration is {duration:.0f} min, expected 45")

    week_start = date.fromisoformat(meta["current_week_start_date"]) + timedelta(days=7)
    week_end = week_start + timedelta(days=6)
    any_in_next_week = any(
        week_start <= utc_to_local(start_utc, emp["timezone"]).date() <= week_end
        for emp in employees
    )
    checks["in_next_week"] = any_in_next_week
    if not any_in_next_week:
        errors.append(f"Slot not in next week ({week_start}–{week_end})")

    for emp in employees:
        local_start = utc_to_local(start_utc, emp["timezone"])
        local_end = utc_to_local(end_utc, emp["timezone"])
        ok = local_start.time() >= time(10, 0) and local_end.time() <= time(15, 0)
        checks[f"{emp['display_name']}_10am_3pm"] = ok
        if not ok:
            errors.append(
                f"{emp['display_name']}: local {local_start.strftime('%H:%M')}–"
                f"{local_end.strftime('%H:%M')} outside 10:00–15:00"
            )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "extracted_start_utc": start_utc_str,
            "extracted_end_utc": end_utc_str,
            "gt_start_utc": ground_truth.get("meeting_start_utc"),
            "gt_end_utc": ground_truth.get("meeting_end_utc"),
        },
        errors=errors,
    )
