"""Verifier for ptc_mt_spotify_duplicate_detector_0.

Q1: groups of duplicate tracks by (artist, name) with >=2 distinct track_ids.
Q2: per-version playlist spread + count of split groups.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from ._spotify_helpers import track_to_playlists, user_playlists
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_groups(cfg: dict) -> list[dict]:
    tracks = cfg.get("tracks", {})
    union: set[str] = set()
    for pl in user_playlists(cfg):
        union.update(pl.get("tracks", []))
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tid in union:
        t = tracks.get(tid)
        if not t:
            continue
        groups[(t["artist"], t["name"])].append(tid)
    out = []
    for (artist, name), tids in groups.items():
        if len(tids) < 2:
            continue
        out.append({
            "artist": artist,
            "name": name,
            "track_ids": sorted(tids),
            "count": len(tids),
        })
    out.sort(key=lambda g: (-g["count"], g["artist"], g["name"]))
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["SpotifyAPI"]
    t2p = track_to_playlists(cfg)
    groups = _q1_groups(cfg)

    spread = []
    split_count = 0
    for g in groups:
        dist = []
        per_version_sets: list[set[str]] = []
        for tid in g["track_ids"]:
            pls = sorted(t2p.get(tid, set()))
            dist.append({"track_id": tid, "playlist_ids": pls})
            per_version_sets.append(set(pls))
        # split = no two versions share any playlist (all pairs disjoint)
        is_split = all(
            not (a & b) for a, b in combinations(per_version_sets, 2)
        ) if len(per_version_sets) >= 2 else False
        if is_split:
            split_count += 1
        spread.append({
            "artist": g["artist"],
            "name": g["name"],
            "distribution": dist,
        })
    return {"dup_spread": spread, "split_groups": split_count}


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

    ds_ans = answer.get("dup_spread") or []
    ds_exp = expected["dup_spread"]
    checks: dict[str, bool] = {
        "count_match": len(ds_ans) == len(ds_exp),
        "split_groups": answer.get("split_groups") == expected["split_groups"],
    }
    ans_keys = [(g.get("artist"), g.get("name")) for g in ds_ans if isinstance(g, dict)]
    exp_keys = [(g["artist"], g["name"]) for g in ds_exp]
    checks["group_set"] = set(ans_keys) == set(exp_keys)
    checks["group_order"] = ans_keys == exp_keys

    exp_by_key = {(g["artist"], g["name"]): g for g in ds_exp}
    dist_ok = True
    for g in ds_ans:
        if not isinstance(g, dict):
            dist_ok = False; break
        e = exp_by_key.get((g.get("artist"), g.get("name")))
        if e is None:
            continue
        ans_dist = {
            (d.get("track_id"), tuple(sorted(d.get("playlist_ids") or [])))
            for d in (g.get("distribution") or []) if isinstance(d, dict)
        }
        exp_dist = {
            (d["track_id"], tuple(sorted(d["playlist_ids"])))
            for d in e["distribution"]
        }
        if ans_dist != exp_dist:
            dist_ok = False; break
    checks["distribution_match"] = dist_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
