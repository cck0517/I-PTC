"""Verifier for ptc_mt_spotify_deep_cuts_0.

Q2: Top-10 artists by total_duration_minutes. For each, the LEAST popular
    distinct track they've played (argmin popularity).
"""
from __future__ import annotations

from collections import defaultdict

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(config: dict) -> list[dict]:
    cfg = config["SpotifyAPI"]
    tracks = cfg["tracks"]
    rp = cfg["recently_played"]

    # Per artist: play_count, total_duration (ms), distinct_tracks
    artist_plays: dict[str, list[str]] = defaultdict(list)  # artist -> list[track_id]
    for tid in rp:
        t = tracks.get(tid)
        if not t:
            continue
        artist_plays[t["artist"]].append(tid)

    rows = []
    for artist, plays in artist_plays.items():
        distinct = set(plays)
        total_ms = sum(tracks[tid]["duration_ms"] for tid in plays)
        rows.append({
            "artist": artist,
            "play_count": len(plays),
            "total_duration_minutes": round(total_ms / 60000, 2),
            "distinct_track_count": len(distinct),
            "distinct_track_ids": distinct,
        })
    rows.sort(key=lambda r: (-r["total_duration_minutes"], r["artist"]))
    top10 = rows[:10]

    deep_cuts = []
    for r in top10:
        # Pick the LEAST popular distinct track played for this artist
        tids = list(r["distinct_track_ids"])
        tids.sort(key=lambda tid: (tracks[tid]["popularity"], tid))
        pick = tids[0]
        play_count_this = sum(1 for x in rp if x == pick)
        deep_cuts.append({
            "artist": r["artist"],
            "track_id": pick,
            "track_title": tracks[pick]["name"],
            "popularity": tracks[pick]["popularity"],
            "play_count_this_track": play_count_this,
        })
    deep_cuts.sort(key=lambda r: (r["popularity"], r["artist"]))
    return deep_cuts


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if isinstance(answer, dict):
        for k in ("deep_cuts", "tracks", "results"):
            if isinstance(answer.get(k), list):
                answer = answer[k]; break
        else:
            answer = [answer]
    if not isinstance(answer, list):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON array"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    ans_ids = [a.get("track_id") for a in answer if isinstance(a, dict)]
    exp_ids = [e["track_id"] for e in expected]
    checks = {
        "count":     len(answer) == len(expected),
        "track_set": set(ans_ids) == set(exp_ids),
    }
    # Check artist→track_id mapping matches
    exp_by_artist = {e["artist"]: e["track_id"] for e in expected}
    mapping_ok = True
    for a in answer:
        if not isinstance(a, dict):
            mapping_ok = False; break
        exp_tid = exp_by_artist.get(a.get("artist"))
        if exp_tid is None:
            continue
        if a.get("track_id") != exp_tid:
            mapping_ok = False; break
    checks["artist_to_track_mapping"] = mapping_ok
    pops = [a.get("popularity") for a in answer if isinstance(a, dict)]
    checks["sorted_popularity_asc"] = all(
        pops[i] <= pops[i + 1] for i in range(len(pops) - 1)
    ) if len(pops) >= 2 else True

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
