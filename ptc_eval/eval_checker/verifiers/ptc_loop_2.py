"""Verifier for ptc_loop_2.

Remove all Drake songs from the 'Road Trip' playlist.
Dynamically computes Drake tracks and expected remaining from initial_config.
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

    # Compute expected from initial_config
    spotify_cfg = initial_config.get("SpotifyAPI", {})
    tracks_db = spotify_cfg.get("tracks", {})

    # Find the Road Trip playlist
    road_trip_id = None
    for pid, pl in spotify_cfg.get("playlists", {}).items():
        if pl.get("name", "").lower() == "road trip":
            road_trip_id = pid
            break
    if not road_trip_id:
        road_trip_id = "pl_1"  # fallback

    initial_tracks = set(spotify_cfg.get("playlists", {}).get(road_trip_id, {}).get("tracks", []))
    drake_tracks = {tid for tid in initial_tracks if tracks_db.get(tid, {}).get("artist") == "Drake"}
    expected_remaining = initial_tracks - drake_tracks

    spotify = (final_state or {}).get("SpotifyAPI", {})
    playlists = spotify.get("playlists", {})
    road_trip = playlists.get(road_trip_id, {})
    final_tracks = set(road_trip.get("tracks", []))

    # Check Drake tracks are removed
    for tid in sorted(drake_tracks):
        checks[f"removed_{tid}"] = tid not in final_tracks

    # Check non-Drake tracks remain
    for tid in sorted(expected_remaining):
        checks[f"kept_{tid}"] = tid in final_tracks

    # Exact set match
    checks["exact_track_list"] = final_tracks == expected_remaining

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "final_tracks": sorted(final_tracks),
            "expected_tracks": sorted(expected_remaining),
            "drake_tracks": sorted(drake_tracks),
        },
    )
