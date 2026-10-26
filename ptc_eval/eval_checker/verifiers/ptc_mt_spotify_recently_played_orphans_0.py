"""Verifier for ptc_mt_spotify_recently_played_orphans_0.

Q1: orphans = recently-played tracks not in any user playlist.
Q2: per-artist orphan summary + existing playlists for that artist.
"""
from __future__ import annotations

from collections import defaultdict

from ._spotify_helpers import user_playlists
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    tracks = cfg.get("tracks", {})

    user_pls = user_playlists(cfg)
    in_user: set[str] = set()
    for pl in user_pls:
        in_user.update(pl.get("tracks", []))

    rp_distinct = []
    seen = set()
    for tid in cfg.get("recently_played", []):
        if tid in seen:
            continue
        seen.add(tid)
        rp_distinct.append(tid)

    orphans = [tid for tid in rp_distinct if tid not in in_user]
    orphan_artists: dict[str, list[str]] = defaultdict(list)
    for tid in orphans:
        t = tracks.get(tid)
        if not t:
            continue
        orphan_artists[t["artist"]].append(tid)

    # For each artist, find user playlists containing OTHER (non-orphan) tracks by them.
    # Reuse: scan user_playlists once for each artist's existing presence.
    existing_by_artist: dict[str, set[str]] = defaultdict(set)
    for pl in user_pls:
        for tid in pl.get("tracks", []):
            t = tracks.get(tid)
            if not t:
                continue
            if t["artist"] in orphan_artists:
                # Only count tracks that are NOT orphans (orphans are by definition
                # not in any user playlist, so this branch always picks up only
                # the artist's *non-orphan* tracks living in user playlists).
                if tid not in orphans:
                    existing_by_artist[t["artist"]].add(pl["id"])

    summary = []
    artists_without_home = []
    for artist in orphan_artists:
        existing = existing_by_artist.get(artist, set())
        summary.append({
            "artist": artist,
            "orphan_track_count": len(orphan_artists[artist]),
            "existing_playlist_ids": sorted(existing),
        })
        if not existing:
            artists_without_home.append(artist)
    summary.sort(key=lambda x: (-x["orphan_track_count"], x["artist"]))

    return {
        "orphan_artist_summary": summary,
        "artists_without_existing_home": sorted(artists_without_home),
    }


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)

    s_ans = answer.get("orphan_artist_summary") or []
    s_exp = expected["orphan_artist_summary"]
    checks: dict[str, bool] = {
        "summary_count": len(s_ans) == len(s_exp),
    }
    ans_artists = [s.get("artist") for s in s_ans if isinstance(s, dict)]
    exp_artists = [s["artist"] for s in s_exp]
    checks["artist_set"] = set(ans_artists) == set(exp_artists)
    checks["artist_order"] = ans_artists == exp_artists

    exp_by_artist = {s["artist"]: s for s in s_exp}
    fields_ok = True
    for s in s_ans:
        if not isinstance(s, dict):
            fields_ok = False; break
        e = exp_by_artist.get(s.get("artist"))
        if e is None:
            continue
        if s.get("orphan_track_count") != e["orphan_track_count"]:
            fields_ok = False; break
        if set(s.get("existing_playlist_ids") or []) != set(e["existing_playlist_ids"]):
            fields_ok = False; break
    checks["fields_match"] = fields_ok

    awh_ans = answer.get("artists_without_existing_home") or []
    awh_exp = expected["artists_without_existing_home"]
    checks["without_home_match"] = set(awh_ans) == set(awh_exp)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
