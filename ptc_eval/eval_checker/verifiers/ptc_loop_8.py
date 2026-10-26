"""Verifier for ptc_loop_8.

Read-only task: Get all Spotify playlists and their details (track count,
total duration). Dynamically computes expected values from initial_config.
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

    answer = (final_answer or "").lower()

    # Compute expected playlists from initial_config
    spotify_cfg = initial_config.get("SpotifyAPI", {})
    playlists = spotify_cfg.get("playlists", {})
    tracks_db = spotify_cfg.get("tracks", {})
    user_pls = spotify_cfg.get("user_playlists", list(playlists.keys()))

    expected = {}
    for pid in user_pls:
        pl = playlists.get(pid, {})
        name = pl.get("name", "")
        track_ids = pl.get("tracks", [])
        track_count = len(track_ids)
        total_ms = sum(tracks_db.get(tid, {}).get("duration_ms", 0) for tid in track_ids)
        expected[name] = {"track_count": track_count, "total_duration_ms": total_ms}

    for name, info in expected.items():
        # Check playlist name is mentioned in the answer
        checks[f"mentions_{name}"] = name.lower() in answer
        # Check track count is mentioned
        checks[f"{name}_track_count"] = str(info["track_count"]) in answer

    # State should be unchanged (read-only)
    if final_state:
        spotify = final_state.get("SpotifyAPI", {})
        initial_spotify = initial_config.get("SpotifyAPI", {})
        playlists_unchanged = True
        for pl_id in initial_spotify.get("playlists", {}):
            init_tracks = initial_spotify["playlists"][pl_id].get("tracks", [])
            final_tracks = spotify.get("playlists", {}).get(pl_id, {}).get("tracks", [])
            if init_tracks != final_tracks:
                playlists_unchanged = False
        checks["state_unchanged"] = playlists_unchanged

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "expected_playlists": expected,
            "answer_preview": (final_answer or "")[:500],
        },
    )
