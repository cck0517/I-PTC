"""Verifier for ptc_mt_spotify_artist_concentration_0.

Q1: 3 most_concentrated + 3 most_diverse user playlists by top2_track_share.
Q2: full per-playlist artist breakdown for the 3 most_concentrated.
"""
from __future__ import annotations

from collections import Counter

from ._spotify_helpers import user_playlists
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_playlist_artist_counts(cfg: dict) -> dict[str, tuple[str, Counter]]:
    """{playlist_id: (playlist_name, Counter(artist -> track_count))}."""
    tracks = cfg.get("tracks", {})
    out: dict[str, tuple[str, Counter]] = {}
    for pl in user_playlists(cfg):
        c = Counter()
        for tid in pl.get("tracks", []):
            t = tracks.get(tid)
            if t:
                c[t["artist"]] += 1
        out[pl["id"]] = (pl["name"], c)
    return out


def _q1_most_concentrated(cfg: dict) -> list[dict]:
    pp = _per_playlist_artist_counts(cfg)
    rows = []
    for pid, (name, counts) in pp.items():
        total = sum(counts.values())
        if total < 1:
            continue
        sorted_counts = sorted(
            counts.items(),
            key=lambda x: (-x[1], x[0]),
        )
        top2 = sorted_counts[:2]
        share = round(sum(c for _, c in top2) / total, 4)
        rows.append({
            "playlist_id": pid,
            "name": name,
            "total_tracks": total,
            "top2_track_share": share,
            "_counts": counts,
        })
    rows.sort(key=lambda r: (-r["top2_track_share"], r["name"]))
    return rows[:3]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    most_conc = _q1_most_concentrated(cfg)
    out = []
    for r in most_conc:
        artist_counts = sorted(
            ({"artist": a, "track_count": c} for a, c in r["_counts"].items()),
            key=lambda x: (-x["track_count"], x["artist"]),
        )
        out.append({
            "playlist_id": r["playlist_id"],
            "name": r["name"],
            "total_tracks": r["total_tracks"],
            "artist_counts": artist_counts,
        })
    return {"concentration_breakdown": out}


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

    cb_ans = answer.get("concentration_breakdown") or []
    cb_exp = expected["concentration_breakdown"]
    checks: dict[str, bool] = {
        "count_match": len(cb_ans) == len(cb_exp),
    }
    ans_pids = [c.get("playlist_id") for c in cb_ans if isinstance(c, dict)]
    exp_pids = [c["playlist_id"] for c in cb_exp]
    checks["playlist_set"] = set(ans_pids) == set(exp_pids)
    checks["playlist_order"] = ans_pids == exp_pids

    exp_by_pid = {c["playlist_id"]: c for c in cb_exp}
    breakdown_ok = True
    for c in cb_ans:
        if not isinstance(c, dict):
            breakdown_ok = False; break
        e = exp_by_pid.get(c.get("playlist_id"))
        if e is None:
            continue
        if c.get("total_tracks") != e["total_tracks"]:
            breakdown_ok = False; break
        ans_pairs = {
            (ac.get("artist"), ac.get("track_count"))
            for ac in (c.get("artist_counts") or []) if isinstance(ac, dict)
        }
        exp_pairs = {(ac["artist"], ac["track_count"]) for ac in e["artist_counts"]}
        if ans_pairs != exp_pairs:
            breakdown_ok = False; break
    checks["artist_counts_match"] = breakdown_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
