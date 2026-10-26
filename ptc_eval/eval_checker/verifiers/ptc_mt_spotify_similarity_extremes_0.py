"""Verifier for ptc_mt_spotify_similarity_extremes_0.

Q1: top-3 most-similar + bottom-3 most-different user playlist pairs by Jaccard.
Q2: shared tracks for each of the 3 most-similar pairs.
"""
from __future__ import annotations

from itertools import combinations

from ._spotify_helpers import (
    jaccard, playlist_name, track_id_set_per_playlist,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_most_similar(cfg: dict) -> tuple[list[dict], dict[str, set[str]]]:
    sets = track_id_set_per_playlist(cfg)
    pids = sorted(sets.keys())
    pairs = []
    for a, b in combinations(pids, 2):
        j = round(jaccard(sets[a], sets[b]), 4)
        pairs.append({
            "pair": [a, b],
            "playlist_names": [playlist_name(cfg, a), playlist_name(cfg, b)],
            "jaccard": j,
        })
    pairs.sort(key=lambda p: (-p["jaccard"], p["playlist_names"]))
    return pairs[:3], sets


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    most_similar, sets = _q1_most_similar(cfg)
    out = []
    for p in most_similar:
        a, b = p["pair"]
        shared = sorted(sets[a] & sets[b])
        out.append({
            "pair": p["pair"],
            "shared_count": len(shared),
            "shared_track_ids": shared,
        })
    return {"shared_tracks_per_pair": out}


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

    sp_ans = answer.get("shared_tracks_per_pair") or []
    sp_exp = expected["shared_tracks_per_pair"]
    checks: dict[str, bool] = {
        "count_match": len(sp_ans) == len(sp_exp),
    }
    ans_pairs = [tuple(p.get("pair") or []) for p in sp_ans if isinstance(p, dict)]
    exp_pairs = [tuple(p["pair"]) for p in sp_exp]
    checks["pair_set"] = set(ans_pairs) == set(exp_pairs)
    checks["pair_order"] = ans_pairs == exp_pairs

    exp_by_pair = {tuple(p["pair"]): p for p in sp_exp}
    shared_ok = True
    cnt_ok = True
    for p in sp_ans:
        if not isinstance(p, dict):
            shared_ok = cnt_ok = False; break
        e = exp_by_pair.get(tuple(p.get("pair") or []))
        if e is None:
            continue
        if p.get("shared_count") != e["shared_count"]:
            cnt_ok = False
        if set(p.get("shared_track_ids") or []) != set(e["shared_track_ids"]):
            shared_ok = False
    checks["shared_count_match"] = cnt_ok
    checks["shared_track_ids_match"] = shared_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
