"""Verifier for ptc_queries_calendar_interview_matching_0.

Check 4 interview bookings: 2 candidates × 2 interviewers each, bipartite
matching, no overlaps, 1-hour duration, minimal total gap between each
candidate's two interviews.
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

    # Check exactly 4 bookings
    checks["four_bookings"] = len(bookings) == 4
    if len(bookings) != 4:
        errors.append(f"Expected 4 bookings, found {len(bookings)}")

    candidates = {"amy", "ben"}
    interviewers = {"iris", "jake", "karen", "leo"}
    candidate_interviews: dict[str, list[dict]] = defaultdict(list)
    interviewer_used: dict[str, int] = defaultdict(int)

    for (utc_time, title), evs in bookings.items():
        attendee_ids = {ev["employee_id"] for ev in evs}
        cands = attendee_ids & candidates
        inters = attendee_ids & interviewers
        if len(cands) == 1 and len(inters) == 1:
            cand = cands.pop()
            inter = inters.pop()
            candidate_interviews[cand].append({
                "interviewer": inter,
                "utc_time": utc_time,
                "events": evs,
            })
            interviewer_used[inter] += 1
        else:
            errors.append(f"Booking at {utc_time}: could not identify 1 candidate + 1 interviewer")

    # Each candidate has exactly 2 interviews
    for cand in candidates:
        n = len(candidate_interviews.get(cand, []))
        checks[f"{cand}_has_2_interviews"] = n == 2
        if n != 2:
            errors.append(f"Candidate {cand} has {n} interviews, expected 2")

    # Each candidate has different interviewers
    for cand in candidates:
        ints = [i["interviewer"] for i in candidate_interviews.get(cand, [])]
        if len(ints) == 2:
            checks[f"{cand}_diff_interviewers"] = ints[0] != ints[1]
            if ints[0] == ints[1]:
                errors.append(f"Candidate {cand} has same interviewer twice: {ints[0]}")

    # Each interviewer used exactly once
    all_once = all(interviewer_used.get(i, 0) == 1 for i in interviewers)
    checks["each_interviewer_once"] = all_once
    if not all_once:
        errors.append(f"Interviewer usage: {dict(interviewer_used)}")

    # No overlaps
    overlap_errs = check_no_overlap_all_employees(all_events)
    checks["no_overlaps"] = len(overlap_errs) == 0
    errors.extend(overlap_errs)

    # 1-hour duration
    dur_ok = True
    for ev in new_events:
        dur = (datetime.fromisoformat(ev["end"]) - datetime.fromisoformat(ev["start"])).total_seconds() / 60
        if abs(dur - 60) > 1:
            dur_ok = False
            errors.append(f"Event for {ev['employee_id']} is {dur:.0f} min, expected 60")
    checks["one_hour_duration"] = dur_ok

    # Within workday 9-17
    workday_ok = True
    for ev in new_events:
        s = datetime.fromisoformat(ev["start"])
        e = datetime.fromisoformat(ev["end"])
        if s.time() < time(9, 0) or e.time() > time(17, 0):
            workday_ok = False
            errors.append(f"Event {ev['start']}–{ev['end']} outside 9:00–17:00 for {ev['employee_id']}")
    checks["within_workday"] = workday_ok

    # Check total gap is optimal
    gt_gap = ground_truth.get("total_gap_minutes", 0)
    total_gap = 0
    for cand in candidates:
        interviews = candidate_interviews.get(cand, [])
        if len(interviews) == 2:
            # Get candidate's event times
            cand_times = []
            for interview in interviews:
                for ev in interview["events"]:
                    if ev["employee_id"] == cand:
                        cand_times.append((ev["start"], ev["end"]))
            if len(cand_times) == 2:
                cand_times.sort()
                gap = (datetime.fromisoformat(cand_times[1][0]) - datetime.fromisoformat(cand_times[0][1])).total_seconds() / 60
                total_gap += max(0, gap)

    checks["minimal_gap"] = total_gap <= gt_gap + 1
    if total_gap > gt_gap + 1:
        errors.append(f"Total gap is {total_gap:.0f} min, optimal is {gt_gap} min")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "n_bookings": len(bookings),
            "candidate_interviews": {
                k: [i["interviewer"] for i in v] for k, v in candidate_interviews.items()
            },
            "total_gap_minutes": total_gap,
            "gt_gap_minutes": gt_gap,
        },
        errors=errors,
    )
