"""Check the venue-inspired playlist and book the ride iff its canonical fare is under $25."""

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

        yelp_cfg = initial_config["YelpAPI"]
        venues = [
            b for b in yelp_cfg["businesses"].values()
            if any("live music" in c.lower() or "venue" in c.lower()
                   for c in b.get("categories", []))
        ]
        venues.sort(key=lambda b: b["rating"], reverse=True)
        expected_venue = venues[0]  # The Fillmore
        expected_venue_name = expected_venue["name"]
        details["expected_venue"] = expected_venue_name

        reviews = yelp_cfg.get("reviews", {}).get(expected_venue["id"], [])
        # "Khruangbin" is mentioned in the first review
        expected_artist = "Khruangbin"
        details["expected_artist"] = expected_artist

        spotify_cfg = initial_config["SpotifyAPI"]
        artist_tracks = [
            t for t in spotify_cfg["tracks"].values()
            if t.get("artist", "").lower() == expected_artist.lower()
        ]
        expected_track_ids = {t["id"] for t in artist_tracks}
        details["expected_track_count"] = len(artist_tracks)

        spotify_state = final_state.get("SpotifyAPI", {})
        playlists = spotify_state.get("playlists", {})
        initial_playlists = spotify_cfg.get("playlists", {})

        new_playlists = {
            pid: pl for pid, pl in playlists.items()
            if pid not in initial_playlists
        }

        checks["playlist_created"] = len(new_playlists) >= 1

        playlist_name_ok = any(
            expected_venue_name.lower() in pl.get("name", "").lower()
            for pl in new_playlists.values()
        )
        checks["playlist_name_has_venue"] = playlist_name_ok

        playlist_has_tracks = False
        for pl in new_playlists.values():
            pl_tracks = set(pl.get("tracks", []))
            if pl_tracks & expected_track_ids:
                playlist_has_tracks = True
                if expected_track_ids <= pl_tracks:
                    checks["playlist_has_all_tracks"] = True
                else:
                    checks["playlist_has_all_tracks"] = False
                break
        if not playlist_has_tracks:
            checks["playlist_has_all_tracks"] = False
        checks["playlist_has_artist_tracks"] = playlist_has_tracks

        # Compare booking behavior with the canonical API estimate.
        from ptc_bench.eval_checker.multi_turn_eval.func_source_code.uber_api import UberAPI

        venue_address = expected_venue.get("address", "")
        canonical_fare = UberAPI._compute_fare("55 Mission St", venue_address, "UberX")
        details["canonical_fare"] = canonical_fare
        should_book = canonical_fare < 25.0

        uber_state = final_state.get("UberAPI", {})
        uber_rides = uber_state.get("rides", {})
        uber_rides_list = uber_rides if isinstance(uber_rides, list) else list(uber_rides.values())
        initial_rides = initial_config.get("UberAPI", {}).get("rides", {})
        initial_ride_count = len(initial_rides) if isinstance(initial_rides, (list, dict)) else 0

        ride_was_booked = len(uber_rides_list) > initial_ride_count
        details["ride_was_booked"] = ride_was_booked
        details["should_book"] = should_book
        checks["uber_decision_honors_conditional"] = ride_was_booked == should_book

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
