"""Verifier for ptc_loop_3.

Delete calendar events that have 'optional' in their description.
Dynamically computes which events should be deleted from initial_config.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Compute expected sets from initial_config
    initial_events = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
    should_delete = set()
    should_remain = set()
    for eid, evt in initial_events.items():
        desc = evt.get("description", "")
        if "optional" in desc.lower():
            should_delete.add(eid)
        else:
            should_remain.add(eid)

    calendar = (final_state or {}).get("PersonalCalendarAPI", {})
    events = calendar.get("events", {})
    remaining_ids = set(events.keys())

    # Check that optional events are deleted
    for evt_id in sorted(should_delete):
        checks[f"deleted_{evt_id}"] = evt_id not in remaining_ids

    # Check that non-optional events remain
    for evt_id in sorted(should_remain):
        checks[f"kept_{evt_id}"] = evt_id in remaining_ids

    # Exact set match
    checks["exact_remaining_events"] = remaining_ids == should_remain

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "remaining_event_ids": sorted(remaining_ids),
            "expected_remaining": sorted(should_remain),
            "expected_deleted": sorted(should_delete),
        },
    )
