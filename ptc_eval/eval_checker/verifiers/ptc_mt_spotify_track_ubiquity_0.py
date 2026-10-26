"""Verifier for ptc_mt_spotify_track_ubiquity_0.

Q1: top-10 tracks by playlist_count across user_playlists.
Q2: for each, list the playlists it appears in.
"""
from __future__ import annotations

from ._spotify_helpers import track_to_playlists, user_playlists
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_top10(cfg: dict) -> list[dict]:
    tracks = cfg.get("tracks", {})
    t2p = track_to_playlists(cfg)
    rows = []
    for tid, pls in t2p.items():
        t = tracks.get(tid)
        if not t:
            continue
        rows.append({
            "track_id": tid,
            "name": t["name"],
            "artist": t["artist"],
            "playlist_count": len(pls),
        })
    rows.sort(key=lambda r: (-r["playlist_count"], r["name"], r["artist"]))
    return rows[:10]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    top10 = _q1_top10(cfg)
    t2p = track_to_playlists(cfg)
    pl_by_id = {pl["id"]: pl for pl in user_playlists(cfg)}

    out = []
    for t in top10:
        pls = sorted(
            ({"playlist_id": pid, "playlist_name": pl_by_id[pid]["name"]}
             for pid in t2p[t["track_id"]]),
            key=lambda p: p["playlist_name"],
        )
        out.append({
            "track_id": t["track_id"],
            "name": t["name"],
            "artist": t["artist"],
            "playlists": pls,
        })
    return {"track_playlists": out}


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

    tp_ans = answer.get("track_playlists") or []
    tp_exp = expected["track_playlists"]
    checks: dict[str, bool] = {
        "count_match": len(tp_ans) == len(tp_exp),
    }
    ans_tids = [t.get("track_id") for t in tp_ans if isinstance(t, dict)]
    exp_tids = [t["track_id"] for t in tp_exp]
    checks["track_set"] = set(ans_tids) == set(exp_tids)
    checks["track_order"] = ans_tids == exp_tids

    exp_by_tid = {t["track_id"]: t for t in tp_exp}
    pl_ok = True
    for t in tp_ans:
        if not isinstance(t, dict):
            pl_ok = False; break
        e = exp_by_tid.get(t.get("track_id"))
        if e is None:
            continue
        ans_pids = {p.get("playlist_id") for p in (t.get("playlists") or [])
                    if isinstance(p, dict)}
        exp_pids = {p["playlist_id"] for p in e["playlists"]}
        if ans_pids != exp_pids:
            pl_ok = False; break
    checks["playlists_set_match"] = pl_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
