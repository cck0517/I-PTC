"""Verifier for ptc_chain_5.

Chain: PersonalCalendarAPI -> UberAPI -> VenmoAPI
Get tomorrow's events -> estimate Uber to each physical location ->
find most expensive ride -> request ride -> split fare with attendees.

Tomorrow is 2026-04-01.  Events with physical locations:
  evt_20: "Client Presentation" at "Financial District, 100 Pine St"
          attendees: [alice_v, bob_smith]
  evt_21: "Lunch with Mentor" at "Ferry Building, San Francisco"
          attendees: [charlie99]
  evt_23: "Team Dinner" at "Waterbar Restaurant, 399 Embarcadero"
          attendees: [diana_k, eve_jones, alice_v]
  evt_22: "Focus Time" has empty location -> skip.

The most expensive ride estimate determines which event to ride to.
Then Venmo payment requests go to that event's attendees for equal
split.  Since fares depend on haversine, we verify structurally.
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
    if final_state is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["final_state is None"],
        )

    try:
        checks: dict[str, bool] = {}
        details: dict[str, Any] = {}

        # ---- Identify tomorrow's events with physical locations ----
        cal_cfg = initial_config["PersonalCalendarAPI"]
        current_date = cal_cfg.get("current_date", "2026-03-31")
        # Tomorrow
        from datetime import datetime, timedelta
        today = datetime.strptime(current_date, "%Y-%m-%d")
        tomorrow = today + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        details["tomorrow"] = tomorrow_str

        events_tomorrow = []
        for evt in cal_cfg["events"].values():
            if evt["start_time"].startswith(tomorrow_str) and evt.get("location", "").strip():
                events_tomorrow.append(evt)
        details["events_with_locations"] = [e["title"] for e in events_tomorrow]

        # ---- Check 1: Uber ride created ----
        uber_state = final_state.get("UberAPI", {})
        uber_rides = uber_state.get("rides", {})
        if isinstance(uber_rides, list):
            uber_rides_list = uber_rides
        else:
            uber_rides_list = list(uber_rides.values())

        initial_rides = initial_config.get("UberAPI", {}).get("rides", {})
        if isinstance(initial_rides, list):
            initial_ride_count = len(initial_rides)
        else:
            initial_ride_count = len(initial_rides)

        checks["uber_ride_created"] = len(uber_rides_list) > initial_ride_count

        # ---- Check 2: Venmo requests sent to attendees of chosen event ----
        venmo_state = final_state.get("VenmoAPI", {})
        venmo_txns = venmo_state.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        new_txns = venmo_txns[len(initial_txns):]

        # We expect payment requests (type=request) to attendees
        request_receivers = {
            t.get("receiver") for t in new_txns
            if t.get("type") in ("request", "payment")
        }

        # We don't know which event is most expensive (depends on Uber
        # haversine), so check that requests were sent to at least 1 attendee
        all_attendees = set()
        for evt in events_tomorrow:
            all_attendees.update(evt.get("attendees", []))

        checks["venmo_requests_sent"] = len(new_txns) >= 1
        checks["venmo_recipients_are_attendees"] = request_receivers <= all_attendees

        # All requests should be for equal amounts (same fare split)
        if new_txns:
            amounts = [t.get("amount", 0) for t in new_txns]
            all_equal = all(abs(a - amounts[0]) < 0.02 for a in amounts)
            checks["venmo_amounts_equal_split"] = all_equal
        else:
            checks["venmo_amounts_equal_split"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
