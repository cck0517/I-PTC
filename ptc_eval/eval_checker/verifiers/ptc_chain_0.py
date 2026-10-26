"""Verifier for ptc_chain_0.

Chain: YelpAPI -> UberAPI -> PersonalCalendarAPI -> VenmoAPI
Find highest-rated Italian restaurant -> request Uber ride -> create
calendar event -> send Venmo payment request.

Expected: Trattoria Roma (biz_10, rating 4.8) is the top Italian
restaurant.  Uber ride to its address, calendar event for dinner there,
Venmo request of $50 to alice_v.
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

        # ---- Derive expected restaurant from initial_config ----
        yelp_cfg = initial_config["YelpAPI"]
        italian = [
            b for b in yelp_cfg["businesses"].values()
            if any("italian" in c.lower() for c in b.get("categories", []))
        ]
        italian.sort(key=lambda b: b["rating"], reverse=True)
        expected_biz = italian[0]  # Trattoria Roma, 4.8
        expected_name = expected_biz["name"]
        expected_address = expected_biz["address"]
        details["expected_restaurant"] = expected_name
        details["expected_address"] = expected_address

        # ---- Check 1: Uber ride created toward restaurant address ----
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

        # ---- Check 2: Calendar event created with restaurant name ----
        cal_state = final_state.get("PersonalCalendarAPI", {})
        cal_events = cal_state.get("events", {})
        if isinstance(cal_events, list):
            final_events = cal_events
        else:
            final_events = list(cal_events.values())

        initial_cal = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
        if isinstance(initial_cal, list):
            initial_event_ids = {e["id"] for e in initial_cal}
        else:
            initial_event_ids = set(initial_cal.keys())

        new_events = [
            e for e in final_events
            if e.get("id") not in initial_event_ids
        ]

        restaurant_in_event = any(
            expected_name.lower() in e.get("title", "").lower()
            or expected_name.lower() in e.get("description", "").lower()
            or expected_name.lower() in e.get("location", "").lower()
            for e in new_events
        )
        checks["calendar_event_created"] = len(new_events) >= 1
        checks["calendar_event_mentions_restaurant"] = restaurant_in_event

        # ---- Check 3: Venmo transaction to alice_v for $50 ----
        venmo_state = final_state.get("VenmoAPI", {})
        venmo_txns = venmo_state.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        new_txns = venmo_txns[len(initial_txns):]

        alice_request = any(
            t.get("receiver") == "alice_v"
            and abs(t.get("amount", 0) - 50.0) < 0.02
            for t in new_txns
        )
        checks["venmo_request_alice_50"] = alice_request

        # The task says "send a Venmo payment *request*" — verify
        # request_payment was used (type == "request"), not send_payment.
        alice_is_request = any(
            t.get("receiver") == "alice_v"
            and t.get("type") == "request"
            and abs(t.get("amount", 0) - 50.0) < 0.02
            for t in new_txns
        )
        checks["venmo_used_request_not_payment"] = alice_is_request

        # Check the note mentions the restaurant
        alice_note_ok = any(
            t.get("receiver") == "alice_v"
            and expected_name.lower() in t.get("note", "").lower()
            for t in new_txns
        )
        checks["venmo_note_mentions_restaurant"] = alice_note_ok

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
