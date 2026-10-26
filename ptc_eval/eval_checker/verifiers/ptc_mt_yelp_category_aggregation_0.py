"""Verifier for ptc_mt_yelp_category_aggregation_0 (multi-turn).

Q2 asks for all restaurants in the top-rated category, sorted by rating
descending.  We recompute from the config.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _compute_expected(config: dict) -> tuple[str, list[dict]]:
    """Return (best_category, list of restaurants in that category sorted by rating desc)."""
    profile = config["profile"]
    ulat = profile["location"]["latitude"]
    ulon = profile["location"]["longitude"]

    by_category: dict[str, list[dict]] = defaultdict(list)
    for bid, b in config["businesses"].items():
        dist = _haversine(ulat, ulon, b["latitude"], b["longitude"])
        if dist > 5.0:
            continue
        cuisine = next(
            (c for c in b.get("categories", []) if c != "restaurant"),
            None,
        )
        if cuisine is None:
            continue
        by_category[cuisine].append({
            "business_id": bid,
            "name": b["name"],
            "rating": b["rating"],
            "review_count": b["review_count"],
            "distance_miles": round(dist, 2),
        })

    # Find category with highest average rating
    best_cat = None
    best_avg = -1.0
    for cat, restaurants in by_category.items():
        avg = sum(r["rating"] for r in restaurants) / len(restaurants)
        if avg > best_avg:
            best_avg = avg
            best_cat = cat

    # Sort that category's restaurants by rating desc
    results = sorted(
        by_category[best_cat],
        key=lambda x: (-x["rating"], x["name"]),
    )
    return best_cat, results


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )

    if isinstance(answer, dict):
        answer = [answer]

    cfg = initial_config["YelpAPI"]
    best_cat, expected = _compute_expected(cfg)

    checks: dict[str, bool] = {}
    checks["correct_count"] = len(answer) == len(expected)

    # Check that the right restaurants are returned
    expected_ids = {e["business_id"] for e in expected}
    answer_ids = {a.get("business_id") for a in answer}
    checks["correct_restaurant_set"] = answer_ids == expected_ids

    # Check ordering by rating desc
    if len(answer) >= 2:
        ratings = [a.get("rating", 0) for a in answer]
        checks["sorted_by_rating_desc"] = all(
            ratings[i] >= ratings[i + 1] for i in range(len(ratings) - 1)
        )
    else:
        checks["sorted_by_rating_desc"] = True

    # Check individual fields
    exp_by_id = {e["business_id"]: e for e in expected}
    for ans in answer:
        bid = ans.get("business_id", "")
        if bid in exp_by_id:
            exp = exp_by_id[bid]
            checks[f"{bid}_name"] = ans.get("name") == exp["name"]
            checks[f"{bid}_rating"] = ans.get("rating") == exp["rating"]

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "extracted_answer": answer,
            "expected_category": best_cat,
            "expected": expected,
        },
    )
