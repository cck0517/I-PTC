"""Verifier for ptc_queries_calendar_minimize_dead_time_0.

Check 3 × 45-min 1-on-1 bookings (Sam + Alice/Bob/Charlie) with minimal dead time.
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
    emp_tz = {e["id"]: e["timezone"] for e in employees}
    all_events = final_state["EnterpriseCalendarAPI"]["events"]
    bookings = group_into_bookings(new_events, employees)

    checks["three_bookings"] = len(bookings) == 3
    if len(bookings) != 3:
        errors.append(f"Expected 3 bookings, found {len(bookings)}")

    expected_others = {"alice", "bob", "charlie"}
    found_others: set[str] = set()
    sam_events_new: list[dict] = []

    for (utc_time, title), evs in bookings.items():
        attendee_ids = {ev["employee_id"] for ev in evs}
        if "sam" not in attendee_ids:
            errors.append(f"Booking at {utc_time}: Sam is not an attendee")
            continue
        found_others.update(attendee_ids - {"sam"})
        for ev in evs:
            if ev["employee_id"] == "sam":
                sam_events_new.append(ev)

    checks["correct_partners"] = found_others == expected_others
    if found_others != expected_others:
        errors.append(f"Expected partners {expected_others}, found {found_others}")

    dur_ok = True
    for ev in new_events:
        dur = (datetime.fromisoformat(ev["end"]) - datetime.fromisoformat(ev["start"])).total_seconds() / 60
        if abs(dur - 45) > 1:
            dur_ok = False
            errors.append(f"Event for {ev['employee_id']} is {dur:.0f} min, expected 45")
    checks["45min_duration"] = dur_ok

    workday_ok = True
    for ev in sam_events_new:
        s, e = datetime.fromisoformat(ev["start"]), datetime.fromisoformat(ev["end"])
        if s.time() < time(9, 0) or e.time() > time(17, 0):
            workday_ok = False
            errors.append(f"Sam event {ev['start']}–{ev['end']} outside 9:00–17:00")
    checks["within_workday"] = workday_ok

    overlap_errs = check_no_overlap_all_employees(all_events)
    checks["no_overlaps"] = len(overlap_errs) == 0
    errors.extend(overlap_errs)

    gt_same_day = ground_truth.get("same_day", True)
    if sam_events_new:
        days = {ev["start"][:10] for ev in sam_events_new}
        checks["same_day"] = len(days) == 1
        if not checks["same_day"] and gt_same_day:
            errors.append(f"Meetings on {len(days)} days, same-day was feasible")

    gt_dead_time = ground_truth.get("total_dead_time_minutes", 0)
    if sam_events_new:
        sorted_new = sorted(sam_events_new, key=lambda e: e["start"])
        dead = 0
        for i in range(len(sorted_new) - 1):
            if sorted_new[i]["start"][:10] == sorted_new[i + 1]["start"][:10]:
                gap = (datetime.fromisoformat(sorted_new[i + 1]["start"])
                       - datetime.fromisoformat(sorted_new[i]["end"])).total_seconds() / 60
                dead += max(0, gap)
        checks["minimal_dead_time"] = dead <= gt_dead_time + 1
        if dead > gt_dead_time + 1:
            errors.append(f"Dead time is {dead:.0f} min, optimal is {gt_dead_time} min")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"n_bookings": len(bookings), "found_partners": sorted(found_others), "gt_dead_time": gt_dead_time},
        errors=errors,
    )
