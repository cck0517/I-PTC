"""Verifier for ptc_queries_calendar_recurring_weekly_sync_0.

Check 4 recurring weekly 1-hour bookings (Jordan + Alex) on the same day/time
across consecutive weeks, within workday, no conflicts, earliest valid slot.
"""

from __future__ import annotations
from datetime import datetime, time
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_new_events, group_into_bookings,
    check_no_overlap_all_employees, score_from_checks,
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

    new_events = extract_new_events(initial_config, final_state)
    employees = final_state["EnterpriseCalendarAPI"]["employees"]
    all_events = final_state["EnterpriseCalendarAPI"]["events"]
    bookings = group_into_bookings(new_events, employees)

    checks["four_bookings"] = len(bookings) == 4
    if len(bookings) != 4:
        errors.append(f"Expected 4 bookings, found {len(bookings)}")

    expected = {"jordan", "alex"}
    attendees_ok = True
    for (utc_time, title), evs in bookings.items():
        if {ev["employee_id"] for ev in evs} != expected:
            attendees_ok = False
            errors.append(f"Booking at {utc_time}: expected {expected}")
    checks["correct_attendees"] = attendees_ok

    dur_ok = True
    for ev in new_events:
        dur = (datetime.fromisoformat(ev["end"]) - datetime.fromisoformat(ev["start"])).total_seconds() / 60
        if abs(dur - 60) > 1:
            dur_ok = False
            errors.append(f"Event for {ev['employee_id']} is {dur:.0f} min, expected 60")
    checks["one_hour_duration"] = dur_ok

    jordan_new = sorted(
        (ev for ev in new_events if ev["employee_id"] == "jordan"),
        key=lambda e: e["start"],
    )

    workday_ok = True
    for ev in jordan_new:
        s, e = datetime.fromisoformat(ev["start"]), datetime.fromisoformat(ev["end"])
        if s.time() < time(9, 0) or e.time() > time(17, 0):
            workday_ok = False
            errors.append(f"Jordan event {ev['start']}–{ev['end']} outside 9:00–17:00")
    checks["within_workday"] = workday_ok

    if jordan_new:
        days_of_week = [datetime.fromisoformat(ev["start"]).weekday() for ev in jordan_new]
        times_of_day = [datetime.fromisoformat(ev["start"]).time() for ev in jordan_new]
        checks["same_day_of_week"] = len(set(days_of_week)) == 1
        checks["same_time_each_week"] = len(set(times_of_day)) == 1
        if not checks["same_day_of_week"]:
            errors.append(f"Different days of week: {days_of_week}")
        if not checks["same_time_each_week"]:
            errors.append(f"Different times: {times_of_day}")

        if len(jordan_new) == 4:
            dates = [datetime.fromisoformat(ev["start"]).date() for ev in jordan_new]
            checks["consecutive_weeks"] = all((dates[i + 1] - dates[i]).days == 7 for i in range(3))
            if not checks["consecutive_weeks"]:
                errors.append(f"Not 4 consecutive weeks: {dates}")

    overlap_errs = check_no_overlap_all_employees(all_events)
    checks["no_overlaps"] = len(overlap_errs) == 0
    errors.extend(overlap_errs)

    gt_slot = ground_truth.get("recurring_slot", {})
    gt_day = gt_slot.get("day_of_week")
    gt_start = gt_slot.get("start_time_jordan_local")
    if gt_day and gt_start and jordan_new:
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        actual_day = day_names[datetime.fromisoformat(jordan_new[0]["start"]).weekday()]
        actual_time = datetime.fromisoformat(jordan_new[0]["start"]).strftime("%H:%M")
        checks["earliest_slot"] = actual_day == gt_day and actual_time == gt_start
        if not checks["earliest_slot"]:
            errors.append(f"Slot is {actual_day} {actual_time}, earliest valid is {gt_day} {gt_start}")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"n_bookings": len(bookings), "gt_slot": gt_slot},
        errors=errors,
    )
