"""Verifier for ptc_mt_spotify_playlist_stats_0.

Q1: top3_largest + bottom3_smallest user playlists by track_count.
Q2: aggregates over ALL user playlists + most_binge_worthy (largest avg_duration).
"""
from __future__ import annotations

from ._spotify_helpers import user_playlists
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_playlist(cfg: dict) -> list[dict]:
    tracks = cfg.get("tracks", {})
    out = []
    for pl in user_playlists(cfg):
        tids = pl.get("tracks", [])
        n = len(tids)
        total_ms = sum(tracks[t]["duration_ms"] for t in tids if t in tracks)
        avg = round(total_ms / n, 2) if n else 0.0
        out.append({
            "playlist_id": pl["id"],
            "name": pl["name"],
            "track_count": n,
            "total_duration_ms": total_ms,
            "avg_duration_ms": avg,
        })
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    pp = _per_playlist(cfg)
    aggregates = {
        "total_track_count": sum(p["track_count"] for p in pp),
        "total_duration_ms": sum(p["total_duration_ms"] for p in pp),
        "playlist_count": len(pp),
    }
    has_tracks = [p for p in pp if p["track_count"] >= 1]
    if has_tracks:
        bw = sorted(has_tracks, key=lambda p: (-p["avg_duration_ms"], p["name"]))[0]
    else:
        bw = None
    return {"aggregates": aggregates, "most_binge_worthy": bw}


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

    a_ans = answer.get("aggregates") or {}
    a_exp = expected["aggregates"]
    checks: dict[str, bool] = {
        "total_track_count":  a_ans.get("total_track_count") == a_exp["total_track_count"],
        "total_duration_ms":  a_ans.get("total_duration_ms") == a_exp["total_duration_ms"],
        "playlist_count":     a_ans.get("playlist_count") == a_exp["playlist_count"],
    }

    bw_ans = answer.get("most_binge_worthy")
    bw_exp = expected["most_binge_worthy"]
    if bw_exp is None:
        checks["most_binge_worthy"] = bw_ans is None
    else:
        checks["most_binge_worthy"] = (
            isinstance(bw_ans, dict)
            and bw_ans.get("playlist_id") == bw_exp["playlist_id"]
            and bw_ans.get("track_count") == bw_exp["track_count"]
            and bw_ans.get("total_duration_ms") == bw_exp["total_duration_ms"]
        )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
