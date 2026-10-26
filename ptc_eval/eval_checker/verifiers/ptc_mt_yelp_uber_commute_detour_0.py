"""Verifier for ptc_mt_yelp_uber_commute_detour_0.

Q2: Shops where extra_cost <= $4 AND is_open AND rating >= 4.0, sorted by
    rating desc, tie-break by extra_cost asc.
"""
from __future__ import annotations

import math
from datetime import datetime

from .core import VerificationResult, extract_json_from_text, score_from_checks

OFFICE_LAT = 37.7926
OFFICE_LON = -122.3955


def _haversine(lat1, lon1, lat2, lon2) -> float:
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


def _is_open(biz: dict, current_time_iso: str) -> bool:
    ct = datetime.fromisoformat(current_time_iso)
    day = ct.strftime("%A")
    hours = biz.get("opening_hours", {}).get(day)
    if not hours or hours == "Closed":
        return False
    ot = datetime.strptime(hours["open"], "%H:%M").time()
    ct2 = datetime.strptime(hours["close"], "%H:%M").time()
    return ot <= ct.time() < ct2


def _compute_expected(config: dict) -> list[dict]:
    yelp = config["YelpAPI"]
    uber = config["UberAPI"]
    profile = yelp["profile"]
    hlat, hlon = profile["location"]["latitude"], profile["location"]["longitude"]
    now = profile["current_time"]

    direct_cost = _uber_cost(uber, _haversine(hlat, hlon, OFFICE_LAT, OFFICE_LON))
    out = []
    for bid, b in yelp["businesses"].items():
        if "Coffee" not in b.get("categories", []):
            continue
        d_home = _haversine(hlat, hlon, b["latitude"], b["longitude"])
        d_off = _haversine(b["latitude"], b["longitude"], OFFICE_LAT, OFFICE_LON)
        detour_cost = round(_uber_cost(uber, d_home) + _uber_cost(uber, d_off), 2)
        extra = round(detour_cost - direct_cost, 2)
        op = _is_open(b, now)
        rating = b.get("rating", 0)
        if extra > 4.0:
            continue
        if not op:
            continue
        if rating < 4.0:
            continue
        out.append({
            "business_id": bid, "name": b["name"], "rating": rating,
            "wait_minutes": b.get("current_wait_minutes", 0),
            "extra_cost": extra, "is_open": True,
        })
    out.sort(key=lambda r: (-r["rating"], r["extra_cost"], r["business_id"]))
    return out


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if isinstance(answer, dict):
        for k in ("results", "shops", "items"):
            if isinstance(answer.get(k), list):
                answer = answer[k]; break
        else:
            answer = [answer]
    if not isinstance(answer, list):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON array"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    ans_ids = [a.get("business_id") for a in answer if isinstance(a, dict)]
    exp_ids = [e["business_id"] for e in expected]
    checks = {
        "count":   len(answer) == len(expected),
        "id_set":  set(ans_ids) == set(exp_ids),
    }
    ratings = [a.get("rating") for a in answer if isinstance(a, dict)]
    checks["sorted_rating_desc"] = all(
        ratings[i] >= ratings[i + 1] for i in range(len(ratings) - 1)
    ) if len(ratings) >= 2 else True

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
