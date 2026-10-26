"""Verifier for ptc_queries_calendar_interview_scheduling_0.

Check 6 interview bookings: each candidate gets 2 with different interviewers,
no overlaps, lunch break preserved, correct dates, 1-hour duration.
"""

from __future__ import annotations
from collections import defaultdict
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

    checks["six_bookings"] = len(bookings) == 6
    if len(bookings) != 6:
        errors.append(f"Expected 6 bookings, found {len(bookings)}")

    candidates = {"amy", "ben", "chloe"}
    interviewers = {"anna", "brian", "carol", "david", "emma", "frank"}
    candidate_interviews: dict[str, list[str]] = defaultdict(list)

    for (utc_time, title), evs in bookings.items():
        attendee_ids = {ev["employee_id"] for ev in evs}
        cands = attendee_ids & candidates
        inters = attendee_ids & interviewers
        if len(cands) == 1 and len(inters) == 1:
            candidate_interviews[cands.pop()].append(inters.pop())
        else:
            errors.append(f"Booking at {utc_time}: could not identify 1 candidate + 1 interviewer")

    for cand in candidates:
        n = len(candidate_interviews.get(cand, []))
        checks[f"{cand}_has_2_interviews"] = n == 2
        if n != 2:
            errors.append(f"Candidate {cand} has {n} interviews, expected 2")

    for cand in candidates:
        inters = candidate_interviews.get(cand, [])
        if len(inters) == 2:
            checks[f"{cand}_diff_interviewers"] = inters[0] != inters[1]
            if inters[0] == inters[1]:
                errors.append(f"Candidate {cand} has same interviewer twice: {inters[0]}")

    overlap_errs = check_no_overlap_all_employees(all_events)
    checks["no_overlaps"] = len(overlap_errs) == 0
    errors.extend(overlap_errs)

    lunch_ok = True
    for ev in new_events:
        s = datetime.fromisoformat(ev["start"])
        e = datetime.fromisoformat(ev["end"])
        lunch_s = datetime.combine(s.date(), time(12, 0))
        lunch_e = datetime.combine(s.date(), time(13, 0))
        if s < lunch_e and e > lunch_s:
            lunch_ok = False
            errors.append(f"Lunch break violated for {ev['employee_id']}: {ev['title']} at {ev['start']}–{ev['end']}")
    checks["lunch_break_preserved"] = lunch_ok

    valid_dates = {"2025-04-08", "2025-04-09"}
    dates_ok = all(ev["start"][:10] in valid_dates for ev in new_events)
    checks["correct_dates"] = dates_ok
    if not dates_ok:
        errors.append(f"Events on wrong dates: {set(ev['start'][:10] for ev in new_events) - valid_dates}")

    dur_ok = True
    for ev in new_events:
        dur = (datetime.fromisoformat(ev["end"]) - datetime.fromisoformat(ev["start"])).total_seconds() / 60
        if abs(dur - 60) > 1:
            dur_ok = False
            errors.append(f"Event '{ev['title']}' for {ev['employee_id']} is {dur:.0f} min, expected 60")
    checks["one_hour_duration"] = dur_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"n_bookings": len(bookings), "candidate_interviews": dict(candidate_interviews)},
        errors=errors,
    )
