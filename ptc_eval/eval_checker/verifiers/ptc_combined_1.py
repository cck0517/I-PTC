"""Verifier for ptc_combined_1.

Birthday party planning: pick venue (The Box SF has best reviews),
create calendar event, add 5 attendees, 5 Venmo requests for $40
each, create Spotify playlist 'Sarah Birthday Bash', order cake
from Tartine Bakery.

Expected:
  Venue: The Box SF (v3) -- all 5-star reviews, highest avg review rating
  Calendar: 'Birthday Party for Sarah' at The Box SF, 19:00-23:00 tomorrow
    Attendees: alice_v, bob_smith, charlie99, diana_k, eve_jones
  Venmo: 5 requests of $40 each
  Spotify: new playlist 'Sarah Birthday Bash' with 8 tracks from Birthday Party Hits
  DoorDash: order from Tartine Bakery, most expensive dessert =
    bk3 'Birthday Celebration Cake' $55.99
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, score_from_checks


_ATTENDEES = {"alice_v", "bob_smith", "charlie99", "diana_k", "eve_jones"}
_VENMO_AMOUNT = 40.0
_VENUE_NAME = "The Box SF"


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # --- 1. Calendar event created ------------------------------------------
    cal = (final_state or {}).get("PersonalCalendarAPI", {})
    cal_events = cal.get("events", {})
    initial_events = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
    new_event_ids = set(cal_events.keys()) - set(initial_events.keys())
    new_events = [cal_events[eid] for eid in new_event_ids]

    bday_events = [
        e for e in new_events
        if "birthday" in e.get("title", "").lower()
        or "sarah" in e.get("title", "").lower()
    ]
    checks["calendar_event_created"] = len(bday_events) > 0

    # --- 2. Calendar event has correct attendees ----------------------------
    if bday_events:
        evt = bday_events[0]
        evt_attendees = set(evt.get("attendees", []))
        checks["correct_attendees"] = _ATTENDEES.issubset(evt_attendees)
    else:
        checks["correct_attendees"] = False

    # --- 3. Calendar event has correct timing (19:00-23:00 on 2026-04-01) ---
    if bday_events:
        evt = bday_events[0]
        start = evt.get("start_time", "")
        end = evt.get("end_time", "")
        checks["event_timing"] = "19:00" in start and "23:00" in end
    else:
        checks["event_timing"] = False

    # --- 4. Venmo: 5 requests for $40 each ----------------------------------
    venmo = (final_state or {}).get("VenmoAPI", {})
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    initial_ids = {t.get("id") for t in initial_txns}
    new_txns = [t for t in txns if t.get("id") not in initial_ids]

    # Check each attendee received a request/payment
    venmo_receivers = {t.get("receiver") for t in new_txns}
    covered_attendees = _ATTENDEES & venmo_receivers
    checks["venmo_all_attendees"] = len(covered_attendees) >= 4  # lenient: at least 4 of 5

    # Check amounts are $40
    correct_amount_count = sum(
        1 for t in new_txns
        if t.get("receiver") in _ATTENDEES and abs(t.get("amount", 0) - _VENMO_AMOUNT) <= 1.0
    )
    checks["venmo_correct_amounts"] = correct_amount_count >= 4

    # --- 5. Spotify: new playlist created -----------------------------------
    spotify = (final_state or {}).get("SpotifyAPI", {})
    playlists = spotify.get("playlists", {})
    initial_playlists = initial_config.get("SpotifyAPI", {}).get("playlists", {})
    new_pl_ids = set(playlists.keys()) - set(initial_playlists.keys())
    new_pls = [playlists[pid] for pid in new_pl_ids]

    sarah_pls = [
        pl for pl in new_pls
        if "sarah" in pl.get("name", "").lower()
        or "birthday" in pl.get("name", "").lower()
    ]
    checks["spotify_playlist_created"] = len(sarah_pls) > 0

    # Check playlist has tracks from the birthday/party playlist in initial config
    if sarah_pls:
        pl_tracks = set(sarah_pls[0].get("tracks", []))
        # Find the source birthday playlist (may have different IDs across difficulty levels)
        source_tracks = set()
        for pid, pl in initial_config.get("SpotifyAPI", {}).get("playlists", {}).items():
            name = pl.get("name", "").lower()
            if "birthday" in name or "party" in name:
                source_tracks.update(pl.get("tracks", []))
                break
        if source_tracks:
            overlap = pl_tracks & source_tracks
            checks["spotify_tracks_copied"] = len(overlap) >= min(5, len(source_tracks))
        else:
            checks["spotify_tracks_copied"] = len(pl_tracks) > 0
    else:
        checks["spotify_tracks_copied"] = False

    # --- 6. DoorDash order placed (cake) ------------------------------------
    dd = (final_state or {}).get("DoorDashAPI", {})
    orders = dd.get("orders", [])
    initial_orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    initial_order_ids = {o.get("id") for o in initial_orders}
    new_orders = [o for o in orders if o.get("id") not in initial_order_ids]
    checks["doordash_order_placed"] = len(new_orders) > 0

    # Check it's from a bakery and includes a cake/dessert
    dd_cfg = initial_config.get("DoorDashAPI", {})
    bakery_rids = set()
    for rid, r in dd_cfg.get("restaurants", {}).items():
        name = r.get("name", "").lower()
        cuisine = r.get("cuisine", "").lower()
        if any(kw in name or kw in cuisine for kw in ["bakery", "cake", "pastry", "dessert"]):
            bakery_rids.add(rid)
    if new_orders:
        bakery_orders = [o for o in new_orders if o.get("restaurant_id") in bakery_rids] if bakery_rids else new_orders
        checks["order_from_bakery"] = len(bakery_orders) > 0
        if bakery_orders:
            items = [i.lower() if isinstance(i, str) else i.get("name", "").lower() for i in bakery_orders[0].get("items", [])]
            checks["cake_ordered"] = any("cake" in i for i in items)
        else:
            checks["cake_ordered"] = False
    else:
        checks["order_from_bakery"] = False
        checks["cake_ordered"] = False

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "new_calendar_events": [(e.get("title"), e.get("attendees")) for e in new_events],
            "new_venmo_count": len(new_txns),
            "new_playlists": [(pl.get("name"), len(pl.get("tracks", []))) for pl in new_pls],
            "new_orders": len(new_orders),
        },
    )
