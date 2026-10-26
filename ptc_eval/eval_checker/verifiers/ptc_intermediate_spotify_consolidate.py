"""Verifier for ptc_intermediate_spotify_consolidate_0.

Derives the expected top-25 tracks by popularity across all 6 source
playlists (the playlists present in initial_config — the model must not
have to name them itself since the task lists the IDs).  Checks both the
final_answer (top_25 array + unique_track_count + new_playlist_id) and
the final_state (new playlist actually created and contains the 25
tracks in the ranked order).
"""

from __future__ import annotations

from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


PLAYLIST_NAME = "Consolidated Favorites"
TOP_K = 25


def _spotify_cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "SpotifyAPIAlt", initial_config.get("SpotifyAPI", {}),
    )


def _popularity(track: dict) -> float:
    play_count = int(track.get("play_count", 0) or 0)
    listener_rating = float(track.get("listener_rating", 0) or 0)
    return round(min(play_count, 500_000) / 10_000 + listener_rating * 10, 1)


def _compute_ground_truth(cfg: dict) -> dict:
    playlists = cfg.get("playlists", {})
    tracks = cfg.get("tracks", {})

    # Exclude any playlist whose name is the target "Consolidated Favorites"
    # — SpotifyAPI._load_scenario aliases the playlists dict by reference,
    # so after create_playlist runs the initial_config gets mutated to
    # include the new playlist.  Filtering by name recovers the original
    # source set unambiguously.
    source_ids = sorted(
        pid for pid, pl in playlists.items()
        if pl.get("name") != PLAYLIST_NAME
    )
    seen: set[str] = set()
    for pid in source_ids:
        for tid in playlists[pid].get("tracks", []):
            seen.add(tid)

    ranked = []
    for tid in seen:
        if tid not in tracks:
            continue
        t = tracks[tid]
        ranked.append({
            "track_id": tid,
            "name": t.get("name", ""),
            "artist": t.get("artist", ""),
            "popularity_index": _popularity(t),
        })
    ranked.sort(key=lambda r: (-r["popularity_index"], r["track_id"]))
    top = ranked[:TOP_K]
    return {
        "unique_track_count": len(seen),
        "top_25": top,
        "top_25_ids": [r["track_id"] for r in top],
        "source_playlist_ids": source_ids,
    }


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]

    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0, checks={}, details=details,
            errors=["Could not extract JSON object from final answer"],
        )

    cfg = _spotify_cfg(initial_config)
    expected = _compute_ground_truth(cfg)
    details["expected_unique_track_count"] = expected["unique_track_count"]
    details["expected_top_25_ids"] = expected["top_25_ids"]

    try:
        got_cnt = int(answer.get("unique_track_count"))
        checks["unique_track_count_correct"] = (
            got_cnt == expected["unique_track_count"]
        )
    except (TypeError, ValueError):
        checks["unique_track_count_correct"] = False

    got_top = answer.get("top_25") or []
    if not isinstance(got_top, list):
        got_top = []
    checks["top_25_length_correct"] = len(got_top) == TOP_K

    ids_ok = True
    fields_ok = bool(expected["top_25"])
    for i, exp in enumerate(expected["top_25"]):
        if i >= len(got_top) or not isinstance(got_top[i], dict):
            ids_ok = False
            fields_ok = False
            break
        got = got_top[i]
        if got.get("track_id") != exp["track_id"]:
            ids_ok = False
        try:
            if abs(float(got.get("popularity_index")) - exp["popularity_index"]) > 0.2:
                fields_ok = False
        except (TypeError, ValueError):
            fields_ok = False
    checks["top_25_order_correct"] = ids_ok
    checks["top_25_popularity_correct"] = fields_ok

    # Verify final_state: new playlist exists, named "Consolidated
    # Favorites", and contains the 25 expected track IDs in order.
    # NOTE: Spotify's _load_scenario aliases the playlists dict from
    # initial_config, so we can't detect "new" by set-diff.  Identify the
    # created playlist by its expected name.
    state = (final_state or {}).get("SpotifyAPI", {})
    state_playlists = state.get("playlists", {})
    new_pl = None
    for pid, pl in state_playlists.items():
        if pl.get("name") == PLAYLIST_NAME and pid not in expected["source_playlist_ids"]:
            new_pl = pl
            break
    if new_pl is None:
        checks["new_playlist_created"] = False
        checks["new_playlist_tracks_correct"] = False
    else:
        checks["new_playlist_created"] = True
        tracks_in_new = list(new_pl.get("tracks", []))
        checks["new_playlist_tracks_correct"] = (
            tracks_in_new == expected["top_25_ids"]
        )
        details["new_playlist_id"] = new_pl.get("id")
        details["new_playlist_tracks"] = tracks_in_new

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks, details=details,
    )
