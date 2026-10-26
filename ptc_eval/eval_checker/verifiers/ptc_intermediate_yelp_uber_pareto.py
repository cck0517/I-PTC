"""Verifier for ptc_intermediate_yelp_uber_pareto_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks
from ..multi_turn_eval.func_source_code.uber_api import UberAPI


HOME = (37.7897, -122.3972)


def _yelp_cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "YelpAPI", initial_config.get("YelpAPIAlt", {}),
    )


def _compute_ground_truth(initial_config: dict) -> dict:
    yelp = _yelp_cfg(initial_config)
    businesses = yelp["businesses"]
    api = UberAPI()
    api._load_scenario({"pricing": {}})
    rows = []
    for bid, b in businesses.items():
        est = api.get_ride_estimate(
            start_latitude=HOME[0], start_longitude=HOME[1],
            end_latitude=b["latitude"], end_longitude=b["longitude"],
            ride_type="UberX",
        )
        cost = float(est.get("estimated_cost_dollars") or est.get("estimated_fare", 0.0))
        dur = int(est.get("estimated_duration_minutes") or est.get("estimated_duration", 0))
        rows.append({
            "business_id": bid,
            "name": b["name"],
            "rating": float(b["rating"]),
            "uber_cost": round(cost, 2),
            "uber_duration_min": dur,
        })

    def _dominates(x, y) -> bool:
        """x dominates y if weakly better on all 3 and strictly better on one."""
        better_rating = x["rating"] >= y["rating"]
        better_cost = x["uber_cost"] <= y["uber_cost"]
        better_dur = x["uber_duration_min"] <= y["uber_duration_min"]
        strict = (x["rating"] > y["rating"] or x["uber_cost"] < y["uber_cost"]
                  or x["uber_duration_min"] < y["uber_duration_min"])
        return better_rating and better_cost and better_dur and strict

    frontier = []
    for r in rows:
        if not any(_dominates(other, r) for other in rows if other is not r):
            frontier.append(r)
    frontier.sort(key=lambda r: r["business_id"])
    return {"frontier": frontier, "total_considered": len(rows)}


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(initial_config)
    details["expected_ids"] = [r["business_id"] for r in expected["frontier"]]

    try:
        checks["total_considered_correct"] = int(answer.get("total_considered")) == expected["total_considered"]
    except (TypeError, ValueError):
        checks["total_considered_correct"] = False

    got = answer.get("frontier") or []
    if not isinstance(got, list):
        got = []
    got_ids = sorted(str(r.get("business_id")) for r in got if isinstance(r, dict))
    checks["frontier_ids_correct"] = got_ids == [r["business_id"] for r in expected["frontier"]]

    numeric_ok = True
    exp_by_id = {r["business_id"]: r for r in expected["frontier"]}
    for r in got:
        if not isinstance(r, dict):
            continue
        exp = exp_by_id.get(r.get("business_id"))
        if exp is None:
            continue
        try:
            if abs(float(r.get("rating")) - exp["rating"]) > 0.01:
                numeric_ok = False
            if abs(float(r.get("uber_cost")) - exp["uber_cost"]) > 0.1:
                numeric_ok = False
            if int(r.get("uber_duration_min")) != exp["uber_duration_min"]:
                numeric_ok = False
        except (TypeError, ValueError):
            numeric_ok = False
    checks["frontier_numeric_correct"] = numeric_ok
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
