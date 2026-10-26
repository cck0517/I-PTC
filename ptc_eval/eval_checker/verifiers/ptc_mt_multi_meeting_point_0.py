"""Verifier for ptc_mt_multi_meeting_point_0 (multi-turn).

Q1 (parity): pick the restaurant minimizing combined Uber cost from 2
origins (me at 742 Evergreen Terrace SF, friend at 1234 Broadway Oakland)
across 5 named restaurants.

Q2: full ranking + most-equitable winner (min |my - friend|).
"""
from __future__ import annotations

from typing import Any

from ._uber_helpers import find_business, geocode, uber_cost
from .core import VerificationResult, extract_json_from_text, score_from_checks


ORIGINS = [
    ("my", "742 Evergreen Terrace, San Francisco, CA"),
    ("friend", "1234 Broadway, Oakland, CA"),
]

RESTAURANTS = [
    "Bella Vista",
    "Lake Chalet",
    "Chez Panisse",
    "Burma Superstar",
    "Grocery Cafe",
]


def _compute_expected(initial_config: dict) -> dict:
    yelp = initial_config["YelpAPI"]
    maps = initial_config["GoogleMapsAPI"]
    pricing = initial_config["UberAPI"]["pricing"]

    o_my = geocode(maps, ORIGINS[0][1])
    o_friend = geocode(maps, ORIGINS[1][1])
    if o_my is None or o_friend is None:
        raise ValueError("Geocode miss in initial_config")

    rows = []
    for rname in RESTAURANTS:
        b = find_business(yelp, rname)
        if b is None:
            continue
        dest = (float(b["latitude"]), float(b["longitude"]))
        my_c = uber_cost(o_my, dest, pricing)
        fr_c = uber_cost(o_friend, dest, pricing)
        rows.append({
            "restaurant_name": b["name"],
            "business_id": b["business_id"],
            "my_uber_cost": my_c,
            "friend_uber_cost": fr_c,
            "total_combined_cost": round(my_c + fr_c, 2),
        })

    rows.sort(key=lambda r: (r["total_combined_cost"], r["restaurant_name"]))

    def equity_key(r):
        return (round(abs(r["my_uber_cost"] - r["friend_uber_cost"]), 2),
                r["total_combined_cost"], r["restaurant_name"])
    eq = min(rows, key=equity_key)
    most_equitable = {
        "restaurant_name": eq["restaurant_name"],
        "business_id": eq["business_id"],
        "cost_difference": round(abs(eq["my_uber_cost"] - eq["friend_uber_cost"]), 2),
    }
    return {"full_ranking": rows, "most_equitable": most_equitable}


def _approx(a, b, tol=0.05) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _row_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("restaurant_name") != exp["restaurant_name"]:
        return False
    if got.get("business_id") != exp["business_id"]:
        return False
    for k in ("my_uber_cost", "friend_uber_cost", "total_combined_cost"):
        if not _approx(got.get(k), exp[k]):
            return False
    return True


def _ranking_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    return all(_row_match(g, e) for g, e in zip(got, exp))


def _eq_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("restaurant_name") != exp["restaurant_name"]:
        return False
    if got.get("business_id") != exp["business_id"]:
        return False
    return _approx(got.get("cost_difference"), exp["cost_difference"])


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
        "full_ranking_correct": _ranking_match(answer.get("full_ranking"),
                                               expected["full_ranking"]),
        "most_equitable_correct": _eq_match(answer.get("most_equitable"),
                                            expected["most_equitable"]),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
