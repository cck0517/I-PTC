"""Check the Best Of Short Playlists seeds, recommendations, and duration analysis.

The seeds are the first tracks from playlists below the median duration.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, score_from_checks


_EXPECTED_SEED_TRACKS = {"ta1", "td1"}
_EXPECTED_PLAYLIST_NAME = "best of short playlists"


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    spotify = (final_state or {}).get("SpotifyAPI", {})
    playlists = spotify.get("playlists", {})
    initial_playlists = initial_config.get("SpotifyAPI", {}).get("playlists", {})

    new_pl_ids = set(playlists.keys()) - set(initial_playlists.keys())
    checks["new_playlist_created"] = len(new_pl_ids) > 0

    best_of_pl = None
    for pid in new_pl_ids:
        pl = playlists[pid]
        if _EXPECTED_PLAYLIST_NAME in pl.get("name", "").lower():
            best_of_pl = pl
            break

    checks["correct_playlist_name"] = best_of_pl is not None

    if best_of_pl:
        pl_tracks = set(best_of_pl.get("tracks", []))
        checks["contains_seed_tracks"] = _EXPECTED_SEED_TRACKS.issubset(pl_tracks)
    else:
        checks["contains_seed_tracks"] = False

    if best_of_pl:
        pl_tracks = set(best_of_pl.get("tracks", []))
        # Recommendation tracks should be beyond the 2 seed tracks
        non_seed = pl_tracks - _EXPECTED_SEED_TRACKS
        # We expect 5 recommended tracks; accept >= 3 as partial success
        checks["recommended_tracks_added"] = len(non_seed) >= 3
    else:
        checks["recommended_tracks_added"] = False

    if best_of_pl:
        total_tracks = len(best_of_pl.get("tracks", []))
        checks["total_track_count"] = total_tracks >= 5
    else:
        checks["total_track_count"] = False

    ans = (final_answer or "").lower()
    checks["mentions_duration_analysis"] = (
        "duration" in ans or "minutes" in ans or "median" in ans
        or "short" in ans
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "new_playlist": best_of_pl,
            "new_playlist_ids": list(new_pl_ids),
        },
    )
