"""Verifier for ptc_mt_yelp_breadth_then_avg_0 (multi-turn).

Q1 (parity): top-5 users by distinct businesses reviewed (>=3).
Q2: top-3 user pairs (from those 5) by intersection size of business sets.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from ._yelp_helpers import businesses, reviews_by_user, round4
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_top5(initial_config: dict) -> list[dict]:
    """Reproduce Q1's top-5 super-users."""
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    rbu = reviews_by_user(cfg)

    rows = []
    for uid, revs in rbu.items():
        biz_ids = sorted({r["business_id"] for r in revs})
        if len(biz_ids) < 3:
            continue
        names = sorted({bizs[b]["name"] for b in biz_ids if b in bizs})
        mean = round4(sum(r["rating"] for r in revs) / len(revs))
        rows.append({
            "user_id": uid,
            "distinct_business_count": len(biz_ids),
            "mean_rating": mean,
            "business_ids": biz_ids,
            "business_names": names,
        })
    rows.sort(key=lambda r: (-r["distinct_business_count"], -r["mean_rating"], r["user_id"]))
    return rows[:5]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    top5 = _q1_top5(initial_config)

    pairs_data = []
    for u, v in combinations(top5, 2):
        u_set = set(u["business_ids"])
        v_set = set(v["business_ids"])
        common = u_set & v_set
        names = sorted({bizs[b]["name"] for b in common if b in bizs})
        pair_sorted = sorted([u["user_id"], v["user_id"]])
        pairs_data.append({
            "pair": pair_sorted,
            "overlap_size": len(common),
            "shared_business_names": names,
        })
    pairs_data.sort(key=lambda r: (-r["overlap_size"], r["pair"]))
    top3 = pairs_data[:3]
    max_overlap = top3[0]["overlap_size"] if top3 else 0
    return {"top3_overlap": top3, "max_overlap": max_overlap}


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if list(g.get("pair") or []) != e["pair"]:
            return False
        if g.get("overlap_size") != e["overlap_size"]:
            return False
        if list(g.get("shared_business_names") or []) != e["shared_business_names"]:
            return False
    return True


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
    checks = {
        "top3_overlap_correct": _list_match(answer.get("top3_overlap"),
                                            expected["top3_overlap"]),
        "max_overlap_correct": answer.get("max_overlap") == expected["max_overlap"],
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
