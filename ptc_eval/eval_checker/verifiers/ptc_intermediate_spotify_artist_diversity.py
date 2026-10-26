"""Verifier for ptc_intermediate_spotify_artist_diversity_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


PLAYLIST_NAME = "Artist Diversity Sampler"
SOURCE_PL = "pl_source"


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "SpotifyAPIAlt", initial_config.get("SpotifyAPI", {}),
    )


def _pop(t: dict) -> float:
    pc = int(t.get("play_count", 0) or 0)
    lr = float(t.get("listener_rating", 0) or 0)
    return round(min(pc, 500_000) / 10_000 + lr * 10, 1)


def _compute_ground_truth(cfg: dict) -> dict:
    tracks = cfg["tracks"]
    source_tids = cfg["playlists"][SOURCE_PL]["tracks"]
    by_artist: dict[str, dict] = {}
    for tid in source_tids:
        t = tracks[tid]
        a = t["artist"]
        p = _pop(t)
        cur = by_artist.get(a)
        if cur is None or p > cur["popularity_index"] or (
            p == cur["popularity_index"] and tid < cur["track_id"]
        ):
            by_artist[a] = {
                "track_id": tid, "artist": a, "popularity_index": p,
            }
    selected = sorted(
        by_artist.values(),
        key=lambda r: (-r["popularity_index"], r["track_id"]),
    )
    return {
        "artist_count": len(by_artist),
        "selected_tracks": selected,
        "selected_ids": [r["track_id"] for r in selected],
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected_ids"] = expected["selected_ids"]
    try:
        checks["artist_count_correct"] = int(answer.get("artist_count")) == expected["artist_count"]
    except (TypeError, ValueError):
        checks["artist_count_correct"] = False
    got = answer.get("selected_tracks") or []
    if not isinstance(got, list):
        got = []
    got_ids = [str(x.get("track_id")) for x in got if isinstance(x, dict)]
    checks["selected_order_correct"] = got_ids == expected["selected_ids"]
    pops_ok = bool(got)
    for g, e in zip(got, expected["selected_tracks"]):
        if not isinstance(g, dict):
            pops_ok = False
            break
        try:
            if abs(float(g.get("popularity_index")) - e["popularity_index"]) > 0.2:
                pops_ok = False
            if g.get("artist") != e["artist"]:
                pops_ok = False
        except (TypeError, ValueError):
            pops_ok = False
    checks["selected_fields_correct"] = pops_ok
    # New playlist in final_state
    state = (final_state or {}).get("SpotifyAPI", {})
    pls = state.get("playlists", {})
    new_pl = None
    for pid, pl in pls.items():
        if pl.get("name") == PLAYLIST_NAME and pid != SOURCE_PL:
            new_pl = pl
            break
    if new_pl is None:
        checks["new_playlist_created"] = False
        checks["new_playlist_tracks_correct"] = False
    else:
        checks["new_playlist_created"] = True
        checks["new_playlist_tracks_correct"] = (
            list(new_pl.get("tracks", [])) == expected["selected_ids"]
        )
        details["new_playlist_id"] = new_pl.get("id")
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
