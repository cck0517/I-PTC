"""Verifier for ptc_queries_calendar_compact_oneononones_0.

Check 4 × 30-min 1-on-1 bookings (Riley + Alice/Bob/Charlie/Diana) with
minimum days used, minimum idle time between meetings, and earliest start
as tiebreaker.
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

    # Check exactly 4 bookings
    checks["four_bookings"] = len(bookings) == 4
    if len(bookings) != 4:
        errors.append(f"Expected 4 bookings, found {len(bookings)}")

    # Check each booking has Riley + one partner
    expected_partners = {"alice", "bob", "charlie", "diana"}
    found_partners: set[str] = set()
    riley_events_new: list[dict] = []

    for (utc_time, title), evs in bookings.items():
        attendee_ids = {ev["employee_id"] for ev in evs}
        if "riley" not in attendee_ids:
            errors.append(f"Booking at {utc_time}: Riley is not an attendee")
            continue
        partners = attendee_ids - {"riley"}
        found_partners.update(partners)
        for ev in evs:
            if ev["employee_id"] == "riley":
                riley_events_new.append(ev)

    checks["correct_partners"] = found_partners == expected_partners
    if found_partners != expected_partners:
        errors.append(f"Expected partners {expected_partners}, found {found_partners}")

    # Check 30-min duration
    dur_ok = True
    for ev in new_events:
        dur = (datetime.fromisoformat(ev["end"]) - datetime.fromisoformat(ev["start"])).total_seconds() / 60
        if abs(dur - 30) > 1:
            dur_ok = False
            errors.append(f"Event for {ev['employee_id']} is {dur:.0f} min, expected 30")
    checks["30min_duration"] = dur_ok

    # No overlaps
    overlap_errs = check_no_overlap_all_employees(all_events)
    checks["no_overlaps"] = len(overlap_errs) == 0
    errors.extend(overlap_errs)

    # Check number of days used
    gt_num_days = ground_truth.get("num_days", 1)
    if riley_events_new:
        days = {ev["start"][:10] for ev in riley_events_new}
        checks["min_days"] = len(days) <= gt_num_days
        if len(days) > gt_num_days:
            errors.append(f"Used {len(days)} days, optimal is {gt_num_days}")

    # Check idle time
    gt_idle = ground_truth.get("total_idle_time_minutes", 0)
    if riley_events_new:
        sorted_new = sorted(riley_events_new, key=lambda e: e["start"])
        idle = 0
        for i in range(len(sorted_new) - 1):
            if sorted_new[i]["start"][:10] == sorted_new[i + 1]["start"][:10]:
                gap = (datetime.fromisoformat(sorted_new[i + 1]["start"])
                       - datetime.fromisoformat(sorted_new[i]["end"])).total_seconds() / 60
                idle += max(0, gap)
        checks["minimal_idle_time"] = idle <= gt_idle + 1
        if idle > gt_idle + 1:
            errors.append(f"Idle time is {idle:.0f} min, optimal is {gt_idle} min")

    # Check correct day
    gt_day = ground_truth.get("day")
    if gt_day and riley_events_new:
        days = {ev["start"][:10] for ev in riley_events_new}
        checks["correct_day"] = gt_day in days
        if gt_day not in days:
            errors.append(f"Expected day {gt_day}, meetings on {days}")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "n_bookings": len(bookings),
            "found_partners": sorted(found_partners),
            "gt_idle": gt_idle,
            "gt_day": gt_day,
        },
        errors=errors,
    )
