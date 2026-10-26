"""Verifier for ptc_mt_yelp_cascade_search_0 (multi-turn).

Q1 (parity): cascade through strict -> relaxed -> none, return single pick.
Q2: top-3 strict picks + top-3 relaxed picks + counts.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import (
    RECENT_CUTOFF,
    businesses,
    filter_recent,
    reviews_by_business,
    round4,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_business_aggregates(initial_config: dict) -> list[dict]:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    rbb = reviews_by_business(cfg)
    out = []
    for bid, b in bizs.items():
        revs = rbb.get(bid, [])
        rec = filter_recent(revs)
        if not rec:
            continue
        rec_mean = round4(sum(r["rating"] for r in rec) / len(rec))
        rec_useful = sum(r["useful_count"] for r in rec)
        out.append({
            "business_id": bid,
            "name": b["name"],
            "rating": b["rating"],
            "review_count": b["review_count"],
            "recent_rating": rec_mean,
            "recent_useful_sum": rec_useful,
        })
    return out


def _compute_expected(initial_config: dict) -> dict:
    rows = _per_business_aggregates(initial_config)
    strict = [r for r in rows
              if r["rating"] >= 4.5 and r["review_count"] >= 8 and r["recent_rating"] >= 4.5]
    relaxed = [r for r in rows
               if r["rating"] >= 4.0 and r["review_count"] >= 8 and r["recent_rating"] >= 4.0]
    strict.sort(key=lambda r: (-r["recent_useful_sum"], r["name"]))
    relaxed.sort(key=lambda r: (-r["recent_useful_sum"], r["name"]))
    return {
        "strict_picks": strict[:3],
        "relaxed_picks": relaxed[:3],
        "strict_count": len(strict),
        "relaxed_count": len(relaxed),
    }


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    keys = ("business_id", "name", "rating", "review_count", "recent_rating", "recent_useful_sum")
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
        "strict_picks_correct": _list_match(answer.get("strict_picks"),
                                            expected["strict_picks"]),
        "relaxed_picks_correct": _list_match(answer.get("relaxed_picks"),
                                             expected["relaxed_picks"]),
        "strict_count_correct": answer.get("strict_count") == expected["strict_count"],
        "relaxed_count_correct": answer.get("relaxed_count") == expected["relaxed_count"],
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
