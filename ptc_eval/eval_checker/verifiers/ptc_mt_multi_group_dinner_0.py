"""Verifier for ptc_mt_multi_group_dinner_0 (multi-turn).

Q1 (parity): pick the restaurant minimizing total combined Uber cost across
3 people (me at 200 Folsom St SF, Mia at 456 Grand Ave Oakland, Jake at
789 Ashby Ave Berkeley) over 6 named restaurants.

Q2: full ranking + fairness winner (minimax individual cost).
"""
from __future__ import annotations

from typing import Any

from ._uber_helpers import find_business, geocode, uber_cost
from .core import VerificationResult, extract_json_from_text, score_from_checks


PEOPLE = [
    ("me", "200 Folsom St, San Francisco, CA"),
    ("Mia", "456 Grand Ave, Oakland, CA"),
    ("Jake", "789 Ashby Ave, Berkeley, CA"),
]

RESTAURANTS = [
    "The Italian Place",
    "Tokyo Garden",
    "El Mexicano",
    "Golden Wok",
    "Le Petit Bistro",
    "Spice of India",
]


def _compute_expected(initial_config: dict) -> dict:
    yelp = initial_config["YelpAPI"]
    maps = initial_config["GoogleMapsAPI"]
    pricing = initial_config["UberAPI"]["pricing"]

    origins = {}
    for person, addr in PEOPLE:
        ll = geocode(maps, addr)
        if ll is None:
            raise ValueError(f"Geocode miss: {addr}")
        origins[person] = ll

    rows = []
    for rname in RESTAURANTS:
        b = find_business(yelp, rname)
        if b is None:
            continue
        dest = (float(b["latitude"]), float(b["longitude"]))
        costs = {p: uber_cost(origins[p], dest, pricing) for p, _ in PEOPLE}
        rows.append({
            "restaurant_name": b["name"],
            "business_id": b["business_id"],
            "costs": costs,
            "total_combined_cost": round(sum(costs.values()), 2),
        })

    rows.sort(key=lambda r: (r["total_combined_cost"], r["restaurant_name"]))

    # Fairness: minimize max(costs.values()); tiebreak total asc, name asc
    def fairness_key(r):
        return (max(r["costs"].values()), r["total_combined_cost"], r["restaurant_name"])
    fair = min(rows, key=fairness_key)
    fairness_winner = {
        "restaurant_name": fair["restaurant_name"],
        "business_id": fair["business_id"],
        "max_individual_cost": round(max(fair["costs"].values()), 2),
    }
    return {"full_ranking": rows, "fairness_winner": fairness_winner}


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
    gc = got.get("costs", {})
    if not isinstance(gc, dict):
        return False
    for person in ("me", "Mia", "Jake"):
        if not _approx(gc.get(person), exp["costs"][person]):
            return False
    if not _approx(got.get("total_combined_cost"), exp["total_combined_cost"]):
        return False
    return True


def _ranking_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    return all(_row_match(g, e) for g, e in zip(got, exp))


def _fairness_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("restaurant_name") != exp["restaurant_name"]:
        return False
    if got.get("business_id") != exp["business_id"]:
        return False
    return _approx(got.get("max_individual_cost"), exp["max_individual_cost"])


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
        "fairness_winner_correct": _fairness_match(answer.get("fairness_winner"),
                                                   expected["fairness_winner"]),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
