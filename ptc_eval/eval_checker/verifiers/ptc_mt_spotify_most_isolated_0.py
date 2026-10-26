"""Verifier for ptc_mt_spotify_most_isolated_0.

Q1: 3 most-isolated user playlists by max_jaccard_to_other.
Q2: for each, identify the closest neighbor (argmax jaccard).
"""
from __future__ import annotations

from ._spotify_helpers import (
    jaccard, playlist_name, track_id_set_per_playlist, user_playlists,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _max_jaccard_per_playlist(sets: dict[str, set[str]]) -> dict[str, tuple[float, str]]:
    """{playlist_id: (max_jaccard, neighbor_id)}; ties broken by neighbor_id asc."""
    out: dict[str, tuple[float, str]] = {}
    pids = list(sets.keys())
    for a in pids:
        best = (-1.0, "")
        for b in pids:
            if a == b:
                continue
            j = jaccard(sets[a], sets[b])
            if (j > best[0]) or (j == best[0] and (best[1] == "" or b < best[1])):
                best = (j, b)
        out[a] = (best[0] if best[0] >= 0 else 0.0, best[1])
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    sets = track_id_set_per_playlist(cfg)
    mj = _max_jaccard_per_playlist(sets)

    pls = user_playlists(cfg)
    name_by_id = {pl["id"]: pl["name"] for pl in pls}

    rows = []
    for pid, (m, neighbor) in mj.items():
        rows.append({
            "playlist_id": pid,
            "name": name_by_id.get(pid, ""),
            "max_jaccard_to_other": round(m, 4),
            "neighbor_id": neighbor,
            "neighbor_name": name_by_id.get(neighbor, ""),
        })
    rows.sort(key=lambda r: (r["max_jaccard_to_other"], r["name"]))
    most_isolated_3 = rows[:3]
    return {
        "closest_neighbors": [
            {
                "playlist_id": r["playlist_id"],
                "name": r["name"],
                "neighbor_id": r["neighbor_id"],
                "neighbor_name": r["neighbor_name"],
                "jaccard": r["max_jaccard_to_other"],
            }
            for r in most_isolated_3
        ]
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

    cn_ans = answer.get("closest_neighbors") or []
    cn_exp = expected["closest_neighbors"]
    checks: dict[str, bool] = {
        "count_match": len(cn_ans) == len(cn_exp),
    }
    ans_pids = [c.get("playlist_id") for c in cn_ans if isinstance(c, dict)]
    exp_pids = [c["playlist_id"] for c in cn_exp]
    checks["playlist_set"] = set(ans_pids) == set(exp_pids)
    checks["playlist_order"] = ans_pids == exp_pids

    exp_by_pid = {c["playlist_id"]: c for c in cn_exp}
    neigh_ok = True
    j_ok = True
    for c in cn_ans:
        if not isinstance(c, dict):
            neigh_ok = j_ok = False; break
        e = exp_by_pid.get(c.get("playlist_id"))
        if e is None:
            continue
        if c.get("neighbor_id") != e["neighbor_id"]:
            neigh_ok = False
        try:
            if abs(float(c.get("jaccard", -1)) - e["jaccard"]) > 0.0005:
                j_ok = False
        except (TypeError, ValueError):
            j_ok = False
    checks["neighbor_id_match"] = neigh_ok
    checks["jaccard_match"] = j_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
