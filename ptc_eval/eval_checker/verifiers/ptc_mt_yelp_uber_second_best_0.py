"""Verifier for ptc_mt_yelp_uber_second_best_0 (multi-turn).

Q1: Every restaurant rated >=4.0, with distance, wait, Uber cost from user,
    sorted by uber_cost ascending.
Q2: From those results, filter to wait_minutes < 20 and re-rank by rating
    descending, top 5.  JSON array with keys:
        business_id, name, rating, wait_minutes, uber_cost_dollars.
"""

from __future__ import annotations

import math

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


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


def _compute_expected(config: dict) -> list[dict]:
    yelp = config["YelpAPI"]
    uber = config["UberAPI"]
    profile = yelp["profile"]
    ulat, ulon = profile["location"]["latitude"], profile["location"]["longitude"]

    results = []
    for bid, b in yelp["businesses"].items():
        if b.get("rating", 0) < 4.0:
            continue
        wait = b.get("current_wait_minutes", 0)
        if wait >= 20:
            continue
        dist = _haversine(ulat, ulon, b["latitude"], b["longitude"])
        results.append({
            "business_id": bid,
            "name": b["name"],
            "rating": b["rating"],
            "wait_minutes": wait,
            "uber_cost_dollars": _uber_cost(uber, dist),
        })

    results.sort(key=lambda r: (-r["rating"], r["uber_cost_dollars"]))
    return results[:5]


def _approx_eq(a, b, tol=0.05) -> bool:
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
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )
    if isinstance(answer, dict):
        for k in ("restaurants", "results", "items"):
            if isinstance(answer.get(k), list):
                answer = answer[k]
                break
        else:
            answer = [answer]

    expected = _compute_expected(initial_config)

    checks: dict[str, bool] = {
        "correct_count": len(answer) == len(expected),
    }
    ans_ids = [a.get("business_id") for a in answer if isinstance(a, dict)]
    exp_ids = [e["business_id"] for e in expected]
    checks["correct_set"] = set(ans_ids) == set(exp_ids)

    ratings = [a.get("rating") for a in answer if isinstance(a, dict)]
    checks["sorted_rating_desc"] = all(
        ratings[i] >= ratings[i + 1] for i in range(len(ratings) - 1)
    ) if len(ratings) >= 2 else True

    field_ok = True
    by_id = {e["business_id"]: e for e in expected}
    for a in answer:
        if not isinstance(a, dict):
            field_ok = False; break
        e = by_id.get(a.get("business_id"))
        if e is None:
            continue
        if not _approx_eq(a.get("rating"), e["rating"], tol=0.01):
            field_ok = False; break
        if a.get("wait_minutes") != e["wait_minutes"]:
            field_ok = False; break
        if not _approx_eq(a.get("uber_cost_dollars"), e["uber_cost_dollars"], tol=0.5):
            field_ok = False; break
    checks["fields_match"] = field_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer, "expected_count": len(expected)},
    )
