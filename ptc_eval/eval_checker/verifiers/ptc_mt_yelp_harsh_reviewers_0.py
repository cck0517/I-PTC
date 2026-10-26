"""Verifier for ptc_mt_yelp_harsh_reviewers_0 (multi-turn).

Q1 (parity): users with >=3 reviews and mean rating <=2.5.
Q2: opposite end (generous, mean >=4.5) plus top-5 most prolific (still >=3).
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import reviews_by_user, round4
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    rbu = reviews_by_user(cfg)

    qualified = []
    for uid, revs in rbu.items():
        if len(revs) < 3:
            continue
        mean = round4(sum(r["rating"] for r in revs) / len(revs))
        qualified.append({
            "user_id": uid,
            "review_count": len(revs),
            "mean_rating": mean,
        })

    generous = sorted(
        [r for r in qualified if r["mean_rating"] >= 4.0],
        key=lambda r: (-r["mean_rating"], r["user_id"]),
    )
    prolific = sorted(qualified, key=lambda r: (-r["review_count"], r["user_id"]))[:5]
    return {"generous_reviewers": generous, "top5_most_prolific": prolific}


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    keys = ("user_id", "review_count", "mean_rating")
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
    checks = {
        "generous_correct": _list_match(answer.get("generous_reviewers"),
                                        expected["generous_reviewers"]),
        "prolific_correct": _list_match(answer.get("top5_most_prolific"),
                                        expected["top5_most_prolific"]),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
