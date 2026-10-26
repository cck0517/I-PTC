"""Verifier for ptc_mt_spotify_core_repertoire_0.

Q1: top-5 largest user playlists + core_tracks (intersection of all 5).
Q2: almost_core_tracks (in EXACTLY 4 of the 5 playlists), with missing playlist.
"""
from __future__ import annotations

from collections import Counter

from ._spotify_helpers import user_playlists
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_top5(cfg: dict) -> list[dict]:
    pls = user_playlists(cfg)
    rows = []
    for pl in pls:
        rows.append({
            "playlist_id": pl["id"],
            "name": pl["name"],
            "track_count": len(pl.get("tracks", [])),
            "track_ids": set(pl.get("tracks", [])),
        })
    rows.sort(key=lambda r: (-r["track_count"], r["name"]))
    return rows[:5]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    tracks = cfg.get("tracks", {})
    top5 = _q1_top5(cfg)

    presence: Counter = Counter()
    where: dict[str, set[str]] = {}
    name_by_id = {p["playlist_id"]: p["name"] for p in top5}
    for p in top5:
        for tid in p["track_ids"]:
            presence[tid] += 1
            where.setdefault(tid, set()).add(p["playlist_id"])

    almost_core = []
    for tid, c in presence.items():
        if c != 4:
            continue
        t = tracks.get(tid)
        if not t:
            continue
        missing_pid = next(p["playlist_id"] for p in top5 if p["playlist_id"] not in where[tid])
        almost_core.append({
            "track_id": tid,
            "name": t["name"],
            "artist": t["artist"],
            "missing_playlist_id": missing_pid,
            "missing_playlist_name": name_by_id[missing_pid],
        })
    almost_core.sort(key=lambda r: (r["name"], r["artist"], r["track_id"]))
    return {"almost_core": almost_core, "almost_core_count": len(almost_core)}


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

    ac_ans = answer.get("almost_core") or []
    ac_exp = expected["almost_core"]
    checks: dict[str, bool] = {
        "count_match":      answer.get("almost_core_count") == expected["almost_core_count"],
        "list_count_match": len(ac_ans) == len(ac_exp),
    }
    ans_tids = [t.get("track_id") for t in ac_ans if isinstance(t, dict)]
    exp_tids = [t["track_id"] for t in ac_exp]
    checks["track_set"] = set(ans_tids) == set(exp_tids)

    exp_by_tid = {t["track_id"]: t for t in ac_exp}
    miss_ok = True
    for t in ac_ans:
        if not isinstance(t, dict):
            miss_ok = False; break
        e = exp_by_tid.get(t.get("track_id"))
        if e is None:
            continue
        if t.get("missing_playlist_id") != e["missing_playlist_id"]:
            miss_ok = False; break
    checks["missing_playlist_match"] = miss_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
