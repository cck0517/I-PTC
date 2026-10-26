"""Verifier for ptc_chain_7.

Chain: SpotifyAPI -> YelpAPI -> PersonalCalendarAPI -> UberAPI -> VenmoAPI
Most played artist from recently_played -> search Yelp events ->
create calendar event -> estimate Uber -> maybe book -> Venmo split.

recently_played: [t30, t31, t32, t33, t34, t30, t35, t31]
Artist counts:
  The Weeknd: t30(x2), t31(x2), t33, t34 = 6 plays
  Dua Lipa: t32, t35 = 2 plays
Most played: The Weeknd

Yelp event: "The Weeknd - After Hours Til Dawn Tour" on 2026-04-15
at Chase Center, 1 Warriors Way, San Francisco, CA.

Calendar event should be created for this show.
Uber from 300 Folsom St to Chase Center.  If < $30, book it.
Venmo request to charlie99 for half the fare.
"""

from __future__ import annotations

from collections import Counter
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

        # ---- Derive expected artist ----
        spotify_cfg = initial_config["SpotifyAPI"]
        recently_played = spotify_cfg.get("recently_played", [])
        artist_counts: Counter = Counter()
        for tid in recently_played:
            track = spotify_cfg["tracks"].get(tid)
            if track:
                artist_counts[track["artist"]] += 1
        most_played_artist = artist_counts.most_common(1)[0][0] if artist_counts else ""
        details["most_played_artist"] = most_played_artist
        details["artist_play_counts"] = dict(artist_counts)

        # ---- Expected Yelp event ----
        yelp_cfg = initial_config["YelpAPI"]
        matching_events = [
            e for e in yelp_cfg.get("events", [])
            if most_played_artist.lower() in e.get("name", "").lower()
            or most_played_artist.lower() in e.get("description", "").lower()
        ]
        expected_event = matching_events[0] if matching_events else None
        if expected_event:
            details["expected_yelp_event"] = expected_event["name"]
            details["expected_venue"] = expected_event.get("location", "")

        # ---- Check 1: Calendar event created for the show ----
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

        checks["calendar_event_created"] = len(new_events) >= 1

        # Check the event references the artist or show
        event_mentions_artist = any(
            most_played_artist.lower() in e.get("title", "").lower()
            or most_played_artist.lower() in e.get("description", "").lower()
            or (expected_event and expected_event["name"].lower() in e.get("title", "").lower())
            for e in new_events
        )
        checks["calendar_event_mentions_artist"] = event_mentions_artist

        # ---- Check 2: Uber ride (booked if < $30) ----
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

        # With default pricing for a short SF distance, fare should be < $30
        checks["uber_ride_booked"] = len(uber_rides_list) > initial_ride_count

        # ---- Check 3: Venmo request to charlie99 for half the fare ----
        venmo_state = final_state.get("VenmoAPI", {})
        venmo_txns = venmo_state.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        new_txns = venmo_txns[len(initial_txns):]

        charlie_txn = any(
            t.get("receiver") == "charlie99"
            for t in new_txns
        )
        checks["venmo_request_to_charlie"] = charlie_txn

        # Check note mentions the artist
        charlie_note_ok = any(
            t.get("receiver") == "charlie99"
            and most_played_artist.lower() in t.get("note", "").lower()
            for t in new_txns
        )
        checks["venmo_note_mentions_artist"] = charlie_note_ok

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
