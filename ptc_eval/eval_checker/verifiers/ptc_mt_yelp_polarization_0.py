"""Verifier for ptc_mt_yelp_polarization_0 (multi-turn).

Q1 (parity): top-3 most polarized businesses (sample stddev of stored review
ratings; min 5 reviews) across {Italian, Sushi, Mexican, Cafe, Bakery}.

Q2: per-category most-polarized + bottom-3 globally least polarized.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import (
    SF_FOOD_CATS,
    businesses,
    has_categories,
    reviews_by_business,
    round4,
    sample_stddev,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    rbb = reviews_by_business(cfg)

    qualified: list[dict] = []
    for bid, b in bizs.items():
        if not has_categories(b, SF_FOOD_CATS):
            continue
        revs = rbb.get(bid, [])
        if len(revs) < 5:
            continue
        std = round4(sample_stddev([r["rating"] for r in revs]))
        qualified.append({
            "business_id": bid,
            "name": b["name"],
            "categories": b.get("categories", []),
            "review_count": len(revs),
            "rating_stddev": std,
        })

    per_category: dict[str, dict | None] = {}
    for cat in SF_FOOD_CATS:
        in_cat = [r for r in qualified if cat in r["categories"]]
        if not in_cat:
            per_category[cat] = None
            continue
        in_cat.sort(key=lambda r: (-r["rating_stddev"], r["name"]))
        top = in_cat[0]
        per_category[cat] = {
            "business_id": top["business_id"],
            "name": top["name"],
            "review_count": top["review_count"],
            "rating_stddev": top["rating_stddev"],
        }

    bottom = sorted(qualified, key=lambda r: (r["rating_stddev"], r["name"]))[:3]
    bottom3 = [
        {"business_id": r["business_id"], "name": r["name"],
         "review_count": r["review_count"], "rating_stddev": r["rating_stddev"]}
        for r in bottom
    ]
    return {"per_category": per_category, "bottom3_consensus": bottom3}


def _entry_match(got: Any, exp: dict | None) -> bool:
    if exp is None:
        return got is None
    if not isinstance(got, dict):
        return False
    for k in ("business_id", "name", "review_count", "rating_stddev"):
        if got.get(k) != exp[k]:
            return False
    return True


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not _entry_match(g, e):
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

    pc_got = answer.get("per_category") or {}
    pc_exp = expected["per_category"]
    per_cat_ok = all(_entry_match(pc_got.get(cat), pc_exp[cat]) for cat in SF_FOOD_CATS)

    checks = {
        "per_category_correct": per_cat_ok,
        "bottom3_correct": _list_match(answer.get("bottom3_consensus"),
                                       expected["bottom3_consensus"]),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
