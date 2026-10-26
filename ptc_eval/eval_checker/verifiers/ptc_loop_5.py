"""Verifier for ptc_loop_5.

Create a calendar event 'Sprint Planning' on 2026-04-05 10:00-11:00 at
'Conference Room A' and add 5 attendees: Sarah, Mike, Jenny, Tom, Lisa.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks

EXPECTED_TITLE = "Sprint Planning"
EXPECTED_START = "2026-04-05 10:00"
EXPECTED_END = "2026-04-05 11:00"
EXPECTED_LOCATION = "Conference Room A"
EXPECTED_ATTENDEES = {"sarah", "mike", "jenny", "tom", "lisa"}


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    calendar = (final_state or {}).get("PersonalCalendarAPI", {})
    events = calendar.get("events", {})

    # Find the Sprint Planning event
    target_event = None
    for evt in events.values():
        if isinstance(evt, dict) and "sprint planning" in evt.get("title", "").lower():
            target_event = evt
            break

    checks["event_created"] = target_event is not None

    if target_event is None:
        checks["correct_title"] = False
        checks["correct_start"] = False
        checks["correct_end"] = False
        checks["correct_location"] = False
        checks["has_5_attendees"] = False
        checks["all_attendees_present"] = False
    else:
        checks["correct_title"] = target_event.get("title", "").lower() == EXPECTED_TITLE.lower()
        checks["correct_start"] = EXPECTED_START in (target_event.get("start_time") or target_event.get("start", ""))
        checks["correct_end"] = EXPECTED_END in (target_event.get("end_time") or target_event.get("end", ""))

        location = (target_event.get("location") or "").lower()
        checks["correct_location"] = "conference room a" in location

        attendees = target_event.get("attendees", [])
        attendee_lower = {a.lower() for a in attendees}
        checks["has_5_attendees"] = len(attendees) >= 5

        # Check each expected attendee is present (names may have various formats)
        found_count = 0
        for name in EXPECTED_ATTENDEES:
            if any(name in a for a in attendee_lower):
                found_count += 1
        checks["all_attendees_present"] = found_count == len(EXPECTED_ATTENDEES)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "found_event": target_event,
            "expected_attendees": sorted(EXPECTED_ATTENDEES),
        },
    )
