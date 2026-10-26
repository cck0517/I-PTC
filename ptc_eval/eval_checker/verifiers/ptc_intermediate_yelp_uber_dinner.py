"""Verifier for ptc_intermediate_yelp_uber_dinner_0.

Reproduces the same haversine + Uber-pricing math the APIs use, derives
the set of restaurants that qualify under the 4 filters (currently open,
rating >= 4.0, Uber cost < $25, Uber duration < 20 min), and checks the
agent's top-3 against that derived ground truth.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


RATING_MIN = 4.0
COST_MAX = 25.0
DURATION_MAX = 20


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return round(2 * R * math.asin(math.sqrt(a)), 2)


def _uber_estimate(dist_mi: float, pricing: dict) -> tuple[float, int]:
    speed = pricing.get("average_speed_mph", 25)
    duration_min = dist_mi / speed * 60
    base = pricing.get("base_fare", 2.50)
    per_mile = pricing.get("per_mile_rate", 1.75)
    per_min = pricing.get("per_minute_rate", 0.35)
    minimum = pricing.get("minimum_fare", 8.00)
    surge = pricing.get("surge_multiplier", 1.0)
    raw = base + dist_mi * per_mile + duration_min * per_min
    cost = round(max(minimum, raw) * surge, 2)
    return cost, round(duration_min)


def _is_open_now(biz: dict, current_time_iso: str) -> bool:
    ct = datetime.fromisoformat(current_time_iso)
    day_name = ct.strftime("%A")
    hours = biz.get("opening_hours", {}).get(day_name)
    if hours is None or hours == "Closed":
        return False
    open_t = datetime.strptime(hours["open"], "%H:%M").time()
    close_t = datetime.strptime(hours["close"], "%H:%M").time()
    return open_t <= ct.time() < close_t


def _yelp_cfg(initial_config: dict) -> dict:
    return initial_config.get("YelpAPI", initial_config.get("YelpAPIAlt", {}))


def _uber_cfg(initial_config: dict) -> dict:
    return initial_config.get("UberAPIAlt", initial_config.get("UberAPI", {}))


def _compute_qualified(initial_config: dict) -> list[dict]:
    yelp = _yelp_cfg(initial_config)
    uber = _uber_cfg(initial_config)
    user_loc = yelp["profile"]["location"]
    current_time = yelp["profile"]["current_time"]
    pricing = uber.get("pricing", {})

    qualified = []
    for biz in yelp["businesses"].values():
        if not _is_open_now(biz, current_time):
            continue
        if biz.get("rating", 0) < RATING_MIN:
            continue
        dist = _haversine(
            user_loc["latitude"], user_loc["longitude"],
            biz["latitude"], biz["longitude"],
        )
        cost, duration = _uber_estimate(dist, pricing)
        if cost >= COST_MAX or duration >= DURATION_MAX:
            continue
        qualified.append({
            "business_id": biz["business_id"],
            "name": biz["name"],
            "rating": biz["rating"],
            "uber_cost_dollars": cost,
            "uber_duration_minutes": duration,
        })
    qualified.sort(key=lambda r: (-r["rating"], r["business_id"]))
    return qualified


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]

    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0, checks={}, details=details,
            errors=["Could not extract JSON object from final answer"],
        )

    qualified = _compute_qualified(initial_config)
    expected_top_3 = qualified[:3]
    yelp = _yelp_cfg(initial_config)
    total_candidates = len(yelp.get("businesses", {}))

    try:
        cand = int(answer.get("candidates_considered"))
        checks["candidates_considered_correct"] = cand == total_candidates
    except (TypeError, ValueError):
        checks["candidates_considered_correct"] = False

    try:
        q = int(answer.get("qualified_count"))
        checks["qualified_count_correct"] = q == len(qualified)
    except (TypeError, ValueError):
        checks["qualified_count_correct"] = False

    top = answer.get("top_3", [])
    if not isinstance(top, list):
        top = []

    checks["top_3_length_correct"] = len(top) == len(expected_top_3)

    ids_ok = True
    fields_ok = bool(expected_top_3)
    for i, exp in enumerate(expected_top_3):
        if i >= len(top) or not isinstance(top[i], dict):
            ids_ok = False
            fields_ok = False
            break
        got = top[i]
        if got.get("business_id") != exp["business_id"]:
            ids_ok = False
        try:
            if abs(float(got.get("uber_cost_dollars")) - exp["uber_cost_dollars"]) > 0.5:
                fields_ok = False
            if abs(int(got.get("uber_duration_minutes")) - exp["uber_duration_minutes"]) > 1:
                fields_ok = False
            if abs(float(got.get("rating")) - exp["rating"]) > 1e-6:
                fields_ok = False
        except (TypeError, ValueError):
            fields_ok = False
    checks["top_3_ids_correct"] = ids_ok
    checks["top_3_fields_correct"] = fields_ok

    details["expected_top_3"] = expected_top_3
    details["expected_qualified_count"] = len(qualified)
    details["expected_candidates"] = total_candidates

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks, details=details,
        errors=errors,
    )
