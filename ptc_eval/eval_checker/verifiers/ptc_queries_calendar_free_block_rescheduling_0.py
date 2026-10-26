"""Verifier for ptc_queries_calendar_free_block_rescheduling_0.

Check that Q2 Planning Sync was rescheduled with all attendees available,
within workday, and Jordan's longest free block is maximised.
"""

from __future__ import annotations
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_new_events, group_into_bookings,
    times_overlap, utc_to_local, free_blocks, score_from_checks,
)


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    if not final_state or "EnterpriseCalendarAPI" not in final_state:
        return VerificationResult(passed=False, score=0.0, errors=["No final calendar state"])

    cal_cfg = initial_config["EnterpriseCalendarAPI"]
    new_events = extract_new_events(initial_config, final_state)
    employees = final_state["EnterpriseCalendarAPI"]["employees"]
    emp_tz = {e["id"]: e["timezone"] for e in employees}

    bookings = group_into_bookings(new_events, employees)
    checks["has_new_booking"] = len(bookings) >= 1
    if not bookings:
        errors.append("No new booking found")
        return VerificationResult(passed=False, score=0.0, checks=checks, errors=errors)

    expected_attendees = {"jordan", "morgan", "taylor", "riley"}
    booking_ok = False
    booked_utc_start = None
    for (utc_time, title), evs in bookings.items():
        if {ev["employee_id"] for ev in evs} == expected_attendees:
            booking_ok = True
            booked_utc_start = utc_time
            break
    checks["correct_attendees"] = booking_ok
    if not booking_ok:
        errors.append(f"No booking with all 4 attendees: {expected_attendees}")

    if booked_utc_start:
        utc_dt = datetime.fromisoformat(booked_utc_start.rstrip("Z")).replace(tzinfo=ZoneInfo("UTC"))
        jordan_tz = emp_tz.get("jordan", "America/New_York")
        local_start = utc_to_local(utc_dt, jordan_tz)
        local_end = utc_to_local(utc_dt + timedelta(hours=1), jordan_tz)
        checks["within_workday"] = local_start.time() >= time(9, 0) and local_end.time() <= time(18, 0)
        if not checks["within_workday"]:
            errors.append(f"Booking {local_start.strftime('%H:%M')}–{local_end.strftime('%H:%M')} outside 9–18")

    original_title = "Q2 Planning Sync"
    initial_events = cal_cfg["events"]

    if booked_utc_start:
        conflict_found = False
        utc_dt = datetime.fromisoformat(booked_utc_start.rstrip("Z")).replace(tzinfo=ZoneInfo("UTC"))
        for emp_id in expected_attendees:
            tz_name = emp_tz[emp_id]
            new_s = utc_to_local(utc_dt, tz_name).strftime("%Y-%m-%dT%H:%M:%S")
            new_e = utc_to_local(utc_dt + timedelta(hours=1), tz_name).strftime("%Y-%m-%dT%H:%M:%S")
            for ev in initial_events:
                if ev["employee_id"] != emp_id or ev["title"] == original_title:
                    continue
                if times_overlap(new_s, new_e, ev["start"], ev["end"]):
                    conflict_found = True
                    errors.append(f"Conflict for {emp_id}: overlaps '{ev['title']}' ({ev['start']}–{ev['end']})")
        checks["no_conflicts"] = not conflict_found

    gt_longest = ground_truth.get("longest_free_block_minutes", 0)
    if booked_utc_start:
        utc_dt = datetime.fromisoformat(booked_utc_start.rstrip("Z")).replace(tzinfo=ZoneInfo("UTC"))
        jordan_tz_name = emp_tz.get("jordan", "America/New_York")
        the_day = utc_to_local(utc_dt, jordan_tz_name).date()
        jordan_events = [
            ev for ev in initial_events
            if ev["employee_id"] == "jordan"
            and ev["title"] != original_title
            and datetime.fromisoformat(ev["start"]).date() == the_day
        ]
        new_s = utc_to_local(utc_dt, jordan_tz_name).strftime("%Y-%m-%dT%H:%M:%S")
        new_e = utc_to_local(utc_dt + timedelta(hours=1), jordan_tz_name).strftime("%Y-%m-%dT%H:%M:%S")
        jordan_events.append({"start": new_s, "end": new_e, "title": "rescheduled"})
        blocks = free_blocks(jordan_events, the_day, time(9, 0), time(18, 0))
        longest = max((int((b[1] - b[0]).total_seconds() / 60) for b in blocks), default=0)
        checks["optimal_free_block"] = longest >= gt_longest
        if longest < gt_longest:
            errors.append(f"Longest free block is {longest} min, optimal is {gt_longest} min")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"booked_utc_start": booked_utc_start, "gt_longest_free_block": gt_longest},
        errors=errors,
    )
