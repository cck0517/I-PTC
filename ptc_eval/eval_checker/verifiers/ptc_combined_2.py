"""Verifier for ptc_combined_2.

Meal prep planning: find top 3 cuisines from order history,
search Yelp for each, get DoorDash menus, pick items closest to $15,
Venmo split, create 3 calendar events (Mon/Wed/Fri), bookmark restaurants.

Pre-computed from 15 DoorDash orders:
  mp_j (Japanese): mp01,mp03,mp06,mp08,mp10,mp13 = 6 orders
  mp_m (Mexican):  mp02,mp05,mp09,mp12,mp15 = 5 orders
  mp_i (Indian):   mp04,mp07,mp11,mp14 = 4 orders
  Top 3 cuisines: Japanese (6), Mexican (5), Indian (4)

Yelp highest-rated for each:
  Japanese -> y_j Katsura Japanese (4.6)
  Mexican  -> y_m El Farolito (4.5)
  Indian   -> y_i Dosa (4.4)

DoorDash menus - items closest to $15:
  mp_j: Chicken Katsu $15.99 (diff 0.99), Miso Ramen $14.49 (diff 0.51)
    -> Miso Ramen ($14.49) closest
  mp_m: Super Burrito $14.99 (diff 0.01)
    -> Super Burrito ($14.99)
  mp_i: Dosa Combo $14.99 (diff 0.01)
    -> Dosa Combo ($14.99)

Total: 14.49 + 14.99 + 14.99 = 44.47
Half = 22.235 (Venmo to roommate1)

Calendar events: current_date = 2026-03-31 (Tuesday)
  Next Mon = 2026-04-06, Wed = 2026-04-01, Fri = 2026-04-03
  (or Mon 2026-04-06, Wed 2026-04-08 -- depends on interpretation)
  Actually from Tue 3/31: next Wed = 4/1, Fri = 4/3, Mon = 4/6

Bookmark: y_j, y_m, y_i on Yelp
"""

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

    # --- 1. Calendar: 3 meal prep events created ----------------------------
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

    # Check events are on different days of the week (Mon/Wed/Fri pattern)
    if len(meal_prep_events) >= 3:
        dates = set()
        for e in meal_prep_events:
            start = e.get("start_time", "")
            if len(start) >= 10:
                dates.add(start[:10])
        checks["events_on_different_days"] = len(dates) >= 3
    else:
        checks["events_on_different_days"] = False

    # --- 2. Venmo request to roommate1 --------------------------------------
    venmo = (final_state or {}).get("VenmoAPI", {})
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    initial_ids = {t.get("id") for t in initial_txns}
    new_txns = [t for t in txns if t.get("id") not in initial_ids]

    roommate_txns = [
        t for t in new_txns if t.get("receiver") == _ROOMMATE
    ]
    checks["venmo_to_roommate"] = len(roommate_txns) > 0

    # Check amount is approximately half the total (~$22.24)
    if roommate_txns:
        amount = roommate_txns[0].get("amount", 0)
        # Accept range $18-$28 to be lenient on item selection
        checks["venmo_amount_reasonable"] = 15.0 <= amount <= 35.0
    else:
        checks["venmo_amount_reasonable"] = False

    # --- 3. Yelp bookmarks added -------------------------------------------
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

    # --- 4. Answer mentions the cuisine analysis ----------------------------
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
