"""Verifier for ptc_mt_yelp_recent_drift_0 (multi-turn).

Q1 (single-turn parity, not separately scored): top-3 Italian businesses by
absolute drift between all-time and last-90-day mean review rating.

Q2 (final answer scored here): signed-direction split among Italian
businesses with at least one recent review (created_at >= 2026-01-27):
gainers, losers, stable_count.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import (
    RECENT_CUTOFF,
    businesses,
    filter_recent,
    reviews_by_business,
    round2,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    rbb = reviews_by_business(cfg)

    rows = []
    for bid, b in bizs.items():
        if "Italian" not in b.get("categories", []):
            continue
        all_revs = rbb.get(bid, [])
        rec_revs = filter_recent(all_revs)
        if not rec_revs:
            continue  # Q2 restricts to those with >=1 recent review
        all_mean = round2(sum(r["rating"] for r in all_revs) / len(all_revs)) if all_revs else 0.0
        rec_mean = round2(sum(r["rating"] for r in rec_revs) / len(rec_revs))
        signed = round2(rec_mean - all_mean)
        rows.append({
            "business_id": bid,
            "name": b["name"],
            "all_time_rating": all_mean,
            "recent_rating": rec_mean,
            "signed_drift": signed,
        })

    gainers = sorted(
        [r for r in rows if r["signed_drift"] > 0],
        key=lambda r: (-r["signed_drift"], r["name"]),
    )[:3]
    losers = sorted(
        [r for r in rows if r["signed_drift"] < 0],
        key=lambda r: (r["signed_drift"], r["name"]),
    )[:3]
    stable_count = sum(1 for r in rows if r["signed_drift"] == 0)

    return {"gainers": gainers, "losers": losers, "stable_count": stable_count}


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    keys = ("business_id", "name", "all_time_rating", "recent_rating", "signed_drift")
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
        "gainers_correct": _list_match(answer.get("gainers"), expected["gainers"]),
        "losers_correct": _list_match(answer.get("losers"), expected["losers"]),
        "stable_count_correct": answer.get("stable_count") == expected["stable_count"],
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
