"""Verifier for ptc_mt_spotify_playlist_overlap_0.

Q2: Tracks appearing in more than one of the user's playlists.
"""
from __future__ import annotations

from collections import defaultdict

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(config: dict) -> list[dict]:
    cfg = config["SpotifyAPI"]
    user_pls = cfg.get("user_playlists", [])
    playlists = cfg["playlists"]
    tracks = cfg["tracks"]

    appearances: dict[str, set[str]] = defaultdict(set)
    for pid in user_pls:
        if pid not in playlists:
            continue
        for tid in playlists[pid].get("tracks", []):
            appearances[tid].add(pid)

    out = []
    for tid, pids in appearances.items():
        if len(pids) < 2:
            continue
        if tid not in tracks:
            continue
        t = tracks[tid]
        out.append({
            "track_id": tid,
            "title": t.get("name"),
            "artist": t.get("artist"),
            "popularity": t.get("popularity"),
            "appearance_count": len(pids),
            "appears_in": sorted(playlists[p]["name"] for p in pids),
        })
    out.sort(key=lambda r: (-r["appearance_count"], -r["popularity"], r["track_id"]))
    return out


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if isinstance(answer, dict):
        for k in ("tracks", "results"):
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
    checks: dict[str, bool] = {
        "count":   len(answer) == len(expected),
        "id_set":  set(ans_ids) == set(exp_ids),
    }

    # Match appearance_count values
    exp_by_id = {e["track_id"]: e for e in expected}
    counts_ok = True
    for a in answer:
        if not isinstance(a, dict):
            counts_ok = False; break
        e = exp_by_id.get(a.get("track_id"))
        if e is None:
            continue
        if a.get("appearance_count") != e["appearance_count"]:
            counts_ok = False; break
    checks["appearance_counts_match"] = counts_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
