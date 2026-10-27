"""Verifier for ptc_queries_calendar_interview_scheduling_0.

Check 6 interview bookings: each candidate gets 2 with different interviewers,
no overlaps involving the new bookings, lunch break preserved, correct dates,
1-hour duration.

The initial calendars already contain overlapping events that the agent has
no way to remove, so only overlaps that involve at least one newly booked
event count. New events are grouped by (UTC start, title); when several
interviews share a start time and title, the group is split into
candidate-interviewer pairs, choosing a pairing that gives each candidate two
different interviewers if one exists.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from datetime import datetime, time

from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_new_events, group_into_bookings,
    score_from_checks, times_overlap,
)

_CANDIDATES = {"amy", "ben", "chloe"}
_INTERVIEWERS = {"anna", "brian", "carol", "david", "emma", "frank"}


def _key(ev: dict) -> tuple:
    return (ev["employee_id"], ev["start"], ev["end"], ev["title"])


def _overlaps_with_new(all_events: list[dict], new_events: list[dict]) -> list[str]:
    new_keys = {_key(e) for e in new_events}
    by_emp: dict[str, list[dict]] = defaultdict(list)
    for ev in all_events:
        by_emp[ev["employee_id"]].append(ev)
    errors = []
    for emp_id, evs in by_emp.items():
        for a, b in itertools.combinations(evs, 2):
            if _key(a) not in new_keys and _key(b) not in new_keys:
                continue
            if times_overlap(a["start"], a["end"], b["start"], b["end"]):
                errors.append(f"Overlap for {emp_id}: '{a['title']}' ({a['start']}-{a['end']}) "
                              f"vs '{b['title']}' ({b['start']}-{b['end']})")
    return errors


def _assign(groups: list[tuple[list[str], list[str]]]) -> dict[str, list[str]]:
    """Pair candidates with interviewers inside each group. Groups with one
    candidate and one interviewer are fixed; larger groups are tried in every
    pairing, keeping the one that gives the most candidates two different
    interviewers."""
    choices = [list(itertools.permutations(inters)) for _, inters in groups]
    best: dict[str, list[str]] | None = None
    best_score = -1
    for combo in itertools.product(*choices):
        assign: dict[str, list[str]] = defaultdict(list)
        for (cands, _), perm in zip(groups, combo):
            for c, i in zip(cands, perm):
                assign[c].append(i)
        score = sum(1 for c in _CANDIDATES if len(assign[c]) == 2 and assign[c][0] != assign[c][1])
        if score > best_score:
            best, best_score = assign, score
    return best or {}


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

    groups: list[tuple[list[str], list[str]]] = []
    n_interviews = 0
    for (utc_time, title), evs in group_into_bookings(new_events, employees).items():
        ids = sorted({ev["employee_id"] for ev in evs})
        cands = [i for i in ids if i in _CANDIDATES]
        inters = [i for i in ids if i in _INTERVIEWERS]
        if cands and len(cands) == len(inters) and len(cands) + len(inters) == len(ids):
            groups.append((cands, inters))
            n_interviews += len(cands)
        else:
            errors.append(f"Booking at {utc_time} ('{title}'): could not split into "
                          f"one-candidate/one-interviewer interviews ({ids})")
    checks["six_bookings"] = n_interviews == 6 and len(errors) == 0
    if n_interviews != 6:
        errors.append(f"Expected 6 interviews, found {n_interviews}")

    candidate_interviews = _assign(groups)
    for cand in sorted(_CANDIDATES):
        n = len(candidate_interviews.get(cand, []))
        checks[f"{cand}_has_2_interviews"] = n == 2
        if n != 2:
            errors.append(f"Candidate {cand} has {n} interviews, expected 2")
    for cand in sorted(_CANDIDATES):
        inters = candidate_interviews.get(cand, [])
        if len(inters) == 2:
            checks[f"{cand}_diff_interviewers"] = inters[0] != inters[1]
            if inters[0] == inters[1]:
                errors.append(f"Candidate {cand} has same interviewer twice: {inters[0]}")

    overlap_errs = _overlaps_with_new(all_events, new_events)
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
            errors.append(f"Lunch break violated for {ev['employee_id']}: {ev['title']} at {ev['start']}-{ev['end']}")
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
        details={"n_interviews": n_interviews, "candidate_interviews": dict(candidate_interviews)},
        errors=errors,
    )
