"""Verifier for ptc_mt_spotify_artist_reach_0.

Q1: top-10 artists by playlist_count (>=3) across user_playlists.
Q2: top-3 user playlists by how many of those top-10 artists they contain.
"""
from __future__ import annotations

from collections import defaultdict

from ._spotify_helpers import (
    artist_to_playlists, playlist_name, user_playlists,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_top10(cfg: dict) -> list[dict]:
    a2p = artist_to_playlists(cfg)
    rows = [{"artist": a, "playlist_count": len(p), "playlist_ids": p}
            for a, p in a2p.items() if len(p) >= 3]
    rows.sort(key=lambda r: (-r["playlist_count"], r["artist"]))
    return rows[:10]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    top10 = _q1_top10(cfg)

    pl_counts: dict[str, set[str]] = defaultdict(set)
    for r in top10:
        for pid in r["playlist_ids"]:
            pl_counts[pid].add(r["artist"])

    rows = []
    for pid, arts in pl_counts.items():
        rows.append({
            "playlist_id": pid,
            "playlist_name": playlist_name(cfg, pid),
            "top10_artist_count": len(arts),
            "top10_artists": sorted(arts),
        })
    rows.sort(key=lambda r: (-r["top10_artist_count"], r["playlist_name"]))
    return {"playlists_by_top10_reach": rows[:3]}


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

    pa_ans = answer.get("playlists_by_top10_reach") or []
    pa_exp = expected["playlists_by_top10_reach"]
    checks: dict[str, bool] = {
        "count_match": len(pa_ans) == len(pa_exp),
    }
    ans_pids = [p.get("playlist_id") for p in pa_ans if isinstance(p, dict)]
    exp_pids = [p["playlist_id"] for p in pa_exp]
    checks["playlist_set"] = set(ans_pids) == set(exp_pids)
    checks["playlist_order"] = ans_pids == exp_pids

    exp_by_pid = {p["playlist_id"]: p for p in pa_exp}
    arts_ok = True
    cnt_ok = True
    for p in pa_ans:
        if not isinstance(p, dict):
            arts_ok = cnt_ok = False; break
        e = exp_by_pid.get(p.get("playlist_id"))
        if e is None:
            continue
        if p.get("top10_artist_count") != e["top10_artist_count"]:
            cnt_ok = False
        if set(p.get("top10_artists") or []) != set(e["top10_artists"]):
            arts_ok = False
    checks["top10_artist_count_match"] = cnt_ok
    checks["top10_artists_match"] = arts_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
