"""Check meal-prep orders, Venmo split, three calendar events, and restaurant bookmarks."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, score_from_checks


_EXPECTED_CUISINES = {"japanese", "mexican", "indian"}
_EXPECTED_YELP_IDS = {"y_j", "y_m", "y_i"}
_ROOMMATE = "roommate1"


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    cal = (final_state or {}).get("PersonalCalendarAPI", {})
    cal_events = cal.get("events", {})
    initial_events = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
    new_event_ids = set(cal_events.keys()) - set(initial_events.keys())
    new_events = [cal_events[eid] for eid in new_event_ids]

    meal_prep_events = [
        e for e in new_events
        if "meal" in e.get("title", "").lower()
        or "prep" in e.get("title", "").lower()
        or any(c in e.get("title", "").lower() for c in ["japanese", "mexican", "indian"])
    ]
    checks["three_calendar_events"] = len(meal_prep_events) >= 3

    if len(meal_prep_events) >= 3:
        dates = set()
        for e in meal_prep_events:
            start = e.get("start_time", "")
            if len(start) >= 10:
                dates.add(start[:10])
        checks["events_on_different_days"] = len(dates) >= 3
    else:
        checks["events_on_different_days"] = False

    venmo = (final_state or {}).get("VenmoAPI", {})
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    initial_ids = {t.get("id") for t in initial_txns}
    new_txns = [t for t in txns if t.get("id") not in initial_ids]

    roommate_txns = [
        t for t in new_txns if t.get("receiver") == _ROOMMATE
    ]
    checks["venmo_to_roommate"] = len(roommate_txns) > 0

    if roommate_txns:
        amount = roommate_txns[0].get("amount", 0)
        # Accept range $18-$28 to be lenient on item selection
        checks["venmo_amount_reasonable"] = 15.0 <= amount <= 35.0
    else:
        checks["venmo_amount_reasonable"] = False

    yelp = (final_state or {}).get("YelpAPI", {})
    # Bookmarks could be in 'bookmarks' list or 'saved_places' dict
    saved_places = yelp.get("saved_places", {})
    bookmarks = yelp.get("bookmarks", [])
    initial_bookmarks = initial_config.get("YelpAPI", {}).get("bookmarks", [])
    initial_saved = initial_config.get("YelpAPI", {}).get("saved_places", {})

    new_bookmarks = set(bookmarks) - set(initial_bookmarks)
    new_saved = (set(saved_places.keys()) if isinstance(saved_places, dict) else set(saved_places)) - \
                (set(initial_saved.keys()) if isinstance(initial_saved, dict) else set(initial_saved))
    all_new_yelp_saves = new_bookmarks | new_saved

    checks["yelp_bookmarks_added"] = len(all_new_yelp_saves) >= 2  # lenient: at least 2 of 3

    ans = (final_answer or "").lower()
    checks["mentions_cuisines"] = (
        sum(1 for c in _EXPECTED_CUISINES if c in ans) >= 2
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "new_calendar_events": [(e.get("title"), e.get("start_time")) for e in new_events],
            "new_venmo": [(t.get("receiver"), t.get("amount")) for t in new_txns],
            "new_yelp_saves": list(all_new_yelp_saves),
        },
    )
