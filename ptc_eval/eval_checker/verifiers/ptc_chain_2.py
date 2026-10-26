"""Verifier for ptc_chain_2.

Chain: SpotifyAPI -> PersonalCalendarAPI -> UberAPI
Search 'workout energy' playlists -> get tracks -> compute total
duration -> if >45 min create calendar event -> request Uber ride.

Expected: Playlist "Workout Energy Mix" (pl_10) has tracks
t10(245000)+t11(312000)+t12(297000)+t13(326000)+t14(292000)+t15(233000)
= 1705000ms = 1705s = 28.42min.  This is LESS than 45 minutes, so NO
calendar event should be created.  Uber ride should still be requested.
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

        # ---- Compute expected total duration from config ----
        spotify_cfg = initial_config["SpotifyAPI"]
        # The top "workout energy" playlist
        workout_playlists = [
            pl for pl in spotify_cfg["playlists"].values()
            if "workout" in pl["name"].lower() and "energy" in pl["name"].lower()
        ]
        workout_playlists.sort(key=lambda p: len(p.get("tracks", [])), reverse=True)
        playlist = workout_playlists[0]  # "Workout Energy Mix"
        track_ids = playlist["tracks"]

        total_ms = sum(
            spotify_cfg["tracks"][tid]["duration_ms"]
            for tid in track_ids
            if tid in spotify_cfg["tracks"]
        )
        total_minutes = total_ms / 60000.0
        should_create_event = total_minutes > 45
        details["total_duration_ms"] = total_ms
        details["total_duration_minutes"] = round(total_minutes, 2)
        details["should_create_calendar_event"] = should_create_event

        # ---- Check 1: Calendar event created only if duration > 45 min ----
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

        if should_create_event:
            checks["calendar_event_correct"] = len(new_events) >= 1
            # Check event title
            gym_event = any(
                "gym" in e.get("title", "").lower()
                for e in new_events
            )
            checks["calendar_event_title_gym"] = gym_event
        else:
            # Should NOT have created a calendar event
            checks["calendar_event_correct"] = len(new_events) == 0

        # ---- Check 2: Uber ride requested ----
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

        # The task says "request the ride" regardless of calendar event
        checks["uber_ride_requested"] = len(uber_rides_list) > initial_ride_count

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
