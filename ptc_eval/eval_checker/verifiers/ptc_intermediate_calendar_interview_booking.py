"""Verifier for ptc_intermediate_calendar_interview_booking_0.

Looser structural checker: we verify the agent proposed an assignment that
(a) covers all 4x3=12 candidate-interviewer pairs, (b) places each within
the valid window, (c) introduces no overlap conflicts for any participant
(including existing events), and (d) the resulting final_state contains
the booked events.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .core import VerificationResult, extract_json_from_text, score_from_checks


CANDIDATES = ["Cand Alpha", "Cand Beta", "Cand Gamma", "Cand Delta"]
INTERVIEWERS = ["Rita Khan", "Sam Oro", "Uri Vas"]
VALID_DATES = {"2026-05-06", "2026-05-07"}
VALID_HOURS = set(range(9, 17))  # 9..16 (start hour)


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "EnterpriseCalendarAPIAlt",
        initial_config.get("EnterpriseCalendarAPI", {}),
    )


def _emp_by_name(cfg: dict) -> dict:
    return {e["display_name"]: e for e in cfg["employees"]}


def _overlap(a_s, a_e, b_s, b_e) -> bool:
    return a_s < b_e and b_s < a_e


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    cfg = _cfg(initial_config)
    emp_by_name = _emp_by_name(cfg)
    base_events = list(cfg["events"])

    scheduled = answer.get("scheduled")
    if not isinstance(scheduled, list):
        scheduled = []

    pairs_covered = set()
    participant_slots: dict[str, list[tuple[datetime, datetime]]] = {}
    valid_structurally = True

    for row in scheduled:
        if not isinstance(row, dict):
            valid_structurally = False
            continue
        cand = row.get("candidate")
        intv = row.get("interviewer")
        d = row.get("date")
        try:
            h = int(row.get("start_hour"))
        except (TypeError, ValueError):
            valid_structurally = False
            continue
        if cand not in CANDIDATES or intv not in INTERVIEWERS:
            valid_structurally = False
            continue
        if d not in VALID_DATES or h not in VALID_HOURS:
            valid_structurally = False
            continue
        pairs_covered.add((cand, intv))
        for name in (cand, intv):
            emp = emp_by_name[name]
            tz = ZoneInfo(emp["timezone"])
            s = datetime.fromisoformat(f"{d}T{h:02d}:00:00").replace(tzinfo=tz)
            e = s + timedelta(hours=1)
            participant_slots.setdefault(name, []).append((s, e))

    checks["all_pairs_covered"] = (
        {(c, i) for c in CANDIDATES for i in INTERVIEWERS} == pairs_covered
    )
    checks["structural_valid"] = valid_structurally and len(scheduled) == 12

    # Conflict check: intra-scheduled + against existing events
    no_conflicts = True
    for name, slots in participant_slots.items():
        emp = emp_by_name[name]
        tz = ZoneInfo(emp["timezone"])
        # existing events
        existing = []
        for ev in base_events:
            if ev["employee_id"] != emp["id"]:
                continue
            es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
            ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
            existing.append((es, ee))
        # pairwise within scheduled slots
        for i in range(len(slots)):
            s_i, e_i = slots[i]
            for j in range(i + 1, len(slots)):
                s_j, e_j = slots[j]
                if _overlap(s_i, e_i, s_j, e_j):
                    no_conflicts = False
                    break
            for es, ee in existing:
                if _overlap(s_i, e_i, es, ee):
                    no_conflicts = False
                    break
    checks["no_conflicts"] = no_conflicts

    try:
        total = int(answer.get("total_booked"))
        checks["total_booked_is_12"] = total == 12
    except (TypeError, ValueError):
        checks["total_booked_is_12"] = False

    # Final state check: each proposed slot corresponds to an event in
    # state.EnterpriseCalendarAPI.events with the same participants.
    state = (final_state or {}).get("EnterpriseCalendarAPI", {})
    state_events = state.get("events", [])
    booked_found = 0
    for row in scheduled:
        if not isinstance(row, dict):
            continue
        d = row.get("date")
        try:
            h = int(row.get("start_hour"))
        except (TypeError, ValueError):
            continue
        start_key = f"{d}T{h:02d}:00:00"
        for ev in state_events:
            if ev.get("start", "").startswith(start_key):
                booked_found += 1
                break
    checks["events_in_final_state"] = booked_found >= 12

    details["pairs_covered_count"] = len(pairs_covered)
    details["booked_found"] = booked_found

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
