"""Check the Thai dinner event, appetizer order, Venmo requests, and saved playlist."""

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

    cal = (final_state or {}).get("PersonalCalendarAPI", {})
    cal_events = cal.get("events", {})
    initial_events = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
    new_event_ids = set(cal_events.keys()) - set(initial_events.keys())
    new_events = [cal_events[eid] for eid in new_event_ids]

    thai_events = [
        e for e in new_events
        if "thai" in e.get("title", "").lower()
        or "dinner" in e.get("title", "").lower()
    ]
    checks["calendar_event_created"] = len(thai_events) > 0

    if thai_events:
        start = thai_events[0].get("start_time", "")
        # Accept any time at 19:00 or later
        checks["event_after_yoga"] = "19:" in start or "20:" in start or "21:" in start
    else:
        checks["event_after_yoga"] = False

    dd = (final_state or {}).get("DoorDashAPI", {})
    orders = dd.get("orders", [])
    initial_orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    initial_order_ids = {o.get("id") for o in initial_orders}
    new_orders = [o for o in orders if o.get("id") not in initial_order_ids]
    checks["doordash_order_placed"] = len(new_orders) > 0

    dd_cfg = initial_config.get("DoorDashAPI", {})
    thai_rids = set()
    for rid, r in dd_cfg.get("restaurants", {}).items():
        if "thai" in r.get("name", "").lower() or "thai" in r.get("cuisine", "").lower():
            thai_rids.add(rid)
    if new_orders and thai_rids:
        checks["order_from_thai"] = any(
            o.get("restaurant_id") in thai_rids for o in new_orders
        )
    else:
        checks["order_from_thai"] = len(new_orders) > 0

    venmo = (final_state or {}).get("VenmoAPI", {})
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    initial_ids = {t.get("id") for t in initial_txns}
    new_txns = [t for t in txns if t.get("id") not in initial_ids]

    alice_txn = any(
        t.get("receiver") == "alice_v" for t in new_txns
    )
    bob_txn = any(
        t.get("receiver") == "bob_smith" for t in new_txns
    )
    checks["venmo_alice"] = alice_txn
    checks["venmo_bob"] = bob_txn

    for name, receiver in [("venmo_alice_amount", "alice_v"), ("venmo_bob_amount", "bob_smith")]:
        receiver_txns = [t for t in new_txns if t.get("receiver") == receiver]
        if receiver_txns:
            checks[name] = any(
                abs(t.get("amount", 0) - 35.0) <= 1.0 for t in receiver_txns
            )
        else:
            checks[name] = False

    spotify = (final_state or {}).get("SpotifyAPI", {})
    saved = spotify.get("saved_playlists", [])
    initial_saved = initial_config.get("SpotifyAPI", {}).get("saved_playlists", [])
    new_saved = set(saved) - set(initial_saved)
    checks["spotify_playlist_saved"] = len(new_saved) > 0

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "new_calendar_events": [e.get("title") for e in new_events],
            "new_orders": len(new_orders),
            "new_venmo_txns": [(t.get("receiver"), t.get("amount")) for t in new_txns],
            "new_saved_playlists": list(new_saved),
        },
    )
