"""Verifier for ptc_mt_yelp_uber_route_reopt_0 (multi-turn).

Q1: Chinese restaurants + boba shops within 5 miles, each with distance,
    wait, and Uber cost from home, as two lists sorted by uber_cost asc.
Q2: Pick the cheapest Chinese restaurant and cheapest boba shop, but if
    wait > 15 min, skip to the next cheapest with wait <= 15.  Report
    both picks, my total route cost (home->chinese->boba->home), plus the
    friend's Uber cost from (37.7850, -122.4000) to the chosen boba shop.

Expected JSON keys:
    chinese: {business_id, name, rating, uber_cost}
    boba:    {business_id, name, rating, uber_cost}
    my_total_route_cost, friend_boba_cost, combined_total.
"""

from __future__ import annotations

import math

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)

FRIEND_LAT = 37.7850
FRIEND_LON = -122.4000


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    a1, o1, a2, o2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = a2 - a1, o2 - o1
    a = math.sin(dlat / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(dlon / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 2)


def _uber_cost(uber_cfg: dict, dist: float) -> float:
    p = uber_cfg["pricing"]
    speed = p.get("average_speed_mph", 25)
    duration_min = dist / speed * 60
    raw = (p.get("base_fare", 2.5)
           + dist * p.get("per_mile_rate", 1.75)
           + duration_min * p.get("per_minute_rate", 0.35))
    return round(max(p.get("minimum_fare", 8.0), raw) * p.get("surge_multiplier", 1.0), 2)


def _cheapest_with_wait_cap(candidates: list[dict]) -> dict | None:
    """Candidates are already sorted by uber_cost asc; pick first wait<=15."""
    for c in candidates:
        if c["wait_minutes"] <= 15:
            return c
    return None


def _compute_expected(config: dict) -> dict | None:
    yelp = config["YelpAPI"]
    uber = config["UberAPI"]
    profile = yelp["profile"]
    ulat, ulon = profile["location"]["latitude"], profile["location"]["longitude"]

    chinese: list[dict] = []
    boba: list[dict] = []
    for bid, b in yelp["businesses"].items():
        cats = b.get("categories", [])
        dist_home = _haversine(ulat, ulon, b["latitude"], b["longitude"])
        if dist_home > 5.0:
            continue
        entry = {
            "business_id": bid,
            "name": b["name"],
            "rating": b["rating"],
            "wait_minutes": b.get("current_wait_minutes", 0),
            "dist_home": dist_home,
            "lat": b["latitude"], "lon": b["longitude"],
            "uber_cost": _uber_cost(uber, dist_home),
        }
        if "Chinese" in cats:
            chinese.append(entry)
        if any(c in cats for c in ("Bubble Tea", "Boba")):
            boba.append(entry)

    chinese.sort(key=lambda r: r["uber_cost"])
    boba.sort(key=lambda r: r["uber_cost"])

    c_pick = _cheapest_with_wait_cap(chinese)
    b_pick = _cheapest_with_wait_cap(boba)
    if c_pick is None or b_pick is None:
        return None

    # Route legs: home -> chinese -> boba -> home.
    leg_home_c = c_pick["uber_cost"]
    leg_c_b = _uber_cost(uber, _haversine(c_pick["lat"], c_pick["lon"], b_pick["lat"], b_pick["lon"]))
    leg_b_home = _uber_cost(uber, _haversine(b_pick["lat"], b_pick["lon"], ulat, ulon))
    my_total = round(leg_home_c + leg_c_b + leg_b_home, 2)
    friend_cost = _uber_cost(uber, _haversine(FRIEND_LAT, FRIEND_LON, b_pick["lat"], b_pick["lon"]))
    combined = round(my_total + friend_cost, 2)

    return {
        "chinese": {
            "business_id": c_pick["business_id"],
            "name": c_pick["name"],
            "rating": c_pick["rating"],
            "uber_cost": c_pick["uber_cost"],
        },
        "boba": {
            "business_id": b_pick["business_id"],
            "name": b_pick["name"],
            "rating": b_pick["rating"],
            "uber_cost": b_pick["uber_cost"],
        },
        "my_total_route_cost": my_total,
        "friend_boba_cost": friend_cost,
        "combined_total": combined,
    }


def _approx_eq(a, b, tol=1.0) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
    turn_answers: list[str | None] | None = None,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )

    expected = _compute_expected(initial_config)
    if expected is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["No valid pair: both Chinese and boba need at least one wait<=15 option"],
            details={"expected": None, "answer": answer},
        )

    checks: dict[str, bool] = {}
    c_ans = answer.get("chinese") or {}
    b_ans = answer.get("boba") or {}
    checks["chinese_id_correct"] = c_ans.get("business_id") == expected["chinese"]["business_id"]
    checks["boba_id_correct"]    = b_ans.get("business_id")    == expected["boba"]["business_id"]
    # Cost tolerance: rounding across Uber API calls can shift by a few cents.
    checks["my_total_close"]     = _approx_eq(answer.get("my_total_route_cost"), expected["my_total_route_cost"], tol=1.0)
    checks["friend_close"]       = _approx_eq(answer.get("friend_boba_cost"),    expected["friend_boba_cost"],    tol=0.5)
    checks["combined_close"]     = _approx_eq(answer.get("combined_total"),      expected["combined_total"],      tol=1.5)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
