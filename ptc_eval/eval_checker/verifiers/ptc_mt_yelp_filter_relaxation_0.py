"""Verifier for ptc_mt_yelp_filter_relaxation_0 (multi-turn).

Q1: Group ALL restaurants within 5 miles by cuisine, with per-category
    open_count / avg_rating / avg_wait_minutes, sorted by avg_rating desc.
Q2: From those results, show all *open* Italian restaurants with wait<20,
    sorted by rating desc.  JSON array with keys:
        business_id, name, rating, wait_minutes, distance_miles, is_open.
"""

from __future__ import annotations

import math
from datetime import datetime

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
    profile = config["profile"]
    ulat, ulon = profile["location"]["latitude"], profile["location"]["longitude"]
    now = profile["current_time"]

    results = []
    for bid, b in config["businesses"].items():
        if "Italian" not in b.get("categories", []):
            continue
        dist = _haversine(ulat, ulon, b["latitude"], b["longitude"])
        if dist > 5.0:
            continue
        wait = b.get("current_wait_minutes", 0)
        if wait >= 20:
            continue
        if not _is_open(b, now):
            continue
        results.append({
            "business_id": bid,
            "name": b["name"],
            "rating": b["rating"],
            "wait_minutes": wait,
            "distance_miles": dist,
            "is_open": True,
        })

    # Sort by rating desc, tie-break by name asc for determinism.
    results.sort(key=lambda r: (-r["rating"], r["name"]))
    return results


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

    expected = _compute_expected(initial_config["YelpAPI"])

    checks: dict[str, bool] = {
        "correct_count": len(answer) == len(expected),
    }
    ans_ids = [a.get("business_id") for a in answer if isinstance(a, dict)]
    exp_ids = [e["business_id"] for e in expected]
    checks["correct_set"] = set(ans_ids) == set(exp_ids)

    # Ratings should be monotonically non-increasing (Q2 asks: sorted by rating desc).
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
        if a.get("is_open") is not True:
            field_ok = False; break
    checks["fields_match"] = field_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer, "expected_count": len(expected)},
    )
