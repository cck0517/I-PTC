"""Verifier for ptc_mt_spotify_top_artists_0.

Q1: top-5 artists by unique_track_count across user_playlists.
Q2: for each top-5 artist, list their distinct tracks + sum of unique counts.
"""
from __future__ import annotations

from ._spotify_helpers import artist_to_tracks
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_top5(cfg: dict) -> list[dict]:
    a2t = artist_to_tracks(cfg)
    rows = [{"artist": a, "unique_track_count": len(t)} for a, t in a2t.items()]
    rows.sort(key=lambda r: (-r["unique_track_count"], r["artist"]))
    return rows[:5]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    tracks = cfg.get("tracks", {})
    a2t = artist_to_tracks(cfg)
    top5 = _q1_top5(cfg)

    out = []
    for r in top5:
        tids = a2t[r["artist"]]
        track_list = sorted(
            ({"track_id": tid, "name": tracks[tid]["name"]} for tid in tids
             if tid in tracks),
            key=lambda t: (t["name"], t["track_id"]),
        )
        out.append({
            "artist": r["artist"],
            "unique_track_count": r["unique_track_count"],
            "tracks": track_list,
        })
    return {
        "artist_tracks": out,
        "total_unique_tracks": sum(r["unique_track_count"] for r in top5),
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

    at_ans = answer.get("artist_tracks") or []
    at_exp = expected["artist_tracks"]
    checks: dict[str, bool] = {
        "count_match": len(at_ans) == len(at_exp),
        "total_unique_tracks": (
            answer.get("total_unique_tracks") == expected["total_unique_tracks"]
        ),
    }
    ans_artists = [a.get("artist") for a in at_ans if isinstance(a, dict)]
    exp_artists = [a["artist"] for a in at_exp]
    checks["artist_set"] = set(ans_artists) == set(exp_artists)
    checks["artist_order"] = ans_artists == exp_artists

    exp_by_artist = {a["artist"]: a for a in at_exp}
    track_ok = True
    for a in at_ans:
        if not isinstance(a, dict):
            track_ok = False; break
        e = exp_by_artist.get(a.get("artist"))
        if e is None:
            continue
        if a.get("unique_track_count") != e["unique_track_count"]:
            track_ok = False; break
        ans_tids = {t.get("track_id") for t in (a.get("tracks") or [])
                    if isinstance(t, dict)}
        exp_tids = {t["track_id"] for t in e["tracks"]}
        if ans_tids != exp_tids:
            track_ok = False; break
    checks["tracks_match"] = track_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
