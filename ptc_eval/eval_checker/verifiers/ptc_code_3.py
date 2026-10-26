"""Verifier for ptc_code_3.

Spotify playlist analysis: compute total duration per playlist,
sort by duration, find median, identify short playlists (below median),
create 'Best Of Short Playlists', add the first track from each
short playlist, then add 5 recommended tracks.

Pre-computed durations (in ms):
  pl_a "Morning Vibes":  240000 + 185000 + 215000 = 640000 (10.67 min)
  pl_b "Deep Focus":     127000+347000+315000+404000+139000+186000+305000+194000
                        = 2017000 (33.62 min)
  pl_c "Road Trip":      265000+197000+270000+180000+157000 = 1069000 (17.82 min)
  pl_d "Quick Pump":     245000+213000 = 458000 (7.63 min)
  pl_e "Jazz Evening":   562000+324000+327000+822000+350000+295000+478000
                        = 3158000 (52.63 min)

Sorted by duration (longest to shortest):
  pl_e (52.63), pl_b (33.62), pl_c (17.82), pl_a (10.67), pl_d (7.63)

Median (5 playlists -> middle value = 3rd): pl_c = 17.82 min

Short playlists (below median): pl_a (10.67), pl_d (7.63)

First (most-played) track from each:
  pl_a -> ta1 ("Sunrise")
  pl_d -> td1 ("Eye of the Tiger")

New playlist "Best Of Short Playlists" should contain: ta1, td1
Then 5 recommended tracks added (rec1-rec5 are available as
they're not in any playlist).

State checks:
  - A new playlist exists with name "Best Of Short Playlists" (or close)
  - It contains ta1 and td1 as seed tracks
  - It has some recommended tracks (total >= 5 additional)
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

    # --- 1. Check a new playlist was created --------------------------------
    new_pl_ids = set(playlists.keys()) - set(initial_playlists.keys())
    checks["new_playlist_created"] = len(new_pl_ids) > 0

    # --- 2. Check the new playlist has the right name -----------------------
    best_of_pl = None
    for pid in new_pl_ids:
        pl = playlists[pid]
        if _EXPECTED_PLAYLIST_NAME in pl.get("name", "").lower():
            best_of_pl = pl
            break

    checks["correct_playlist_name"] = best_of_pl is not None

    # --- 3. Check it contains the seed tracks (ta1 and td1) ----------------
    if best_of_pl:
        pl_tracks = set(best_of_pl.get("tracks", []))
        checks["contains_seed_tracks"] = _EXPECTED_SEED_TRACKS.issubset(pl_tracks)
    else:
        checks["contains_seed_tracks"] = False

    # --- 4. Check recommended tracks were added (at least some) -------------
    if best_of_pl:
        pl_tracks = set(best_of_pl.get("tracks", []))
        # Recommendation tracks should be beyond the 2 seed tracks
        non_seed = pl_tracks - _EXPECTED_SEED_TRACKS
        # We expect 5 recommended tracks; accept >= 3 as partial success
        checks["recommended_tracks_added"] = len(non_seed) >= 3
    else:
        checks["recommended_tracks_added"] = False

    # --- 5. Check that the total track count is reasonable (2 seeds + 5 recs = 7)
    if best_of_pl:
        total_tracks = len(best_of_pl.get("tracks", []))
        checks["total_track_count"] = total_tracks >= 5
    else:
        checks["total_track_count"] = False

    # --- 6. Check the answer mentions duration analysis ---------------------
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
