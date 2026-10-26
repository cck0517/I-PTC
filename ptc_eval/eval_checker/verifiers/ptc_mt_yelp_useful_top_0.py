"""Verifier for ptc_mt_yelp_useful_top_0 (multi-turn).

Q1 (parity): top-5 reviews by useful_count across all reviews.
Q2: regroup the same flat reviews by user_id; top-5 users by total useful_count.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import reviews, round4
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    by_user: dict[str, dict] = {}
    for r in reviews(cfg).values():
        uid = r["user_id"]
        s = by_user.setdefault(uid, {"total": 0, "count": 0})
        s["total"] += r["useful_count"]
        s["count"] += 1

    rows = []
    for uid, s in by_user.items():
        rows.append({
            "user_id": uid,
            "total_useful_count": s["total"],
            "review_count": s["count"],
            "mean_useful_per_review": round4(s["total"] / s["count"]) if s["count"] else 0.0,
        })
    rows.sort(key=lambda r: (-r["total_useful_count"], -r["review_count"], r["user_id"]))
    return {"top5_users": rows[:5]}


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    keys = ("user_id", "total_useful_count", "review_count", "mean_useful_per_review")
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        for k in keys:
            if g.get(k) != e[k]:
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
    checks = {"top5_users_correct": _list_match(answer.get("top5_users"), expected["top5_users"])}
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
