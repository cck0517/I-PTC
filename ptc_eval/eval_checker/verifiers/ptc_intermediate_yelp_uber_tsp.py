"""Verifier for ptc_intermediate_yelp_uber_tsp_0.

The TSP must be recomputed using the UberAPI's pricing function so the
verifier stays consistent with whatever fare formula the benchmark uses.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks
from ..multi_turn_eval.func_source_code.uber_api import UberAPI


HOME = (37.7897, -122.3972)


def _yelp_cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "YelpAPI", initial_config.get("YelpAPIAlt", {}),
    )


def _uber_estimate(cost_fn, sll, ell, rt="UberX") -> dict:
    return cost_fn(
        start_latitude=sll[0], start_longitude=sll[1],
        end_latitude=ell[0], end_longitude=ell[1],
        ride_type=rt,
    )


def _compute_ground_truth(initial_config: dict) -> dict:
    yelp = _yelp_cfg(initial_config)
    businesses = yelp["businesses"]
    # Build a fresh, non-flaky UberAPI with the same pricing scenario.
    api = UberAPI()
    api._load_scenario({"pricing": {}})
    coords = {bid: (b["latitude"], b["longitude"]) for bid, b in businesses.items()}
    ids = sorted(businesses.keys())
    # Precompute all pair costs: home↔biz and biz↔biz
    cache: dict[tuple, float] = {}
    def _cost(a: tuple, b: tuple) -> float:
        key = (a, b)
        if key in cache:
            return cache[key]
        res = _uber_estimate(api.get_ride_estimate, a, b)
        c = res.get("estimated_cost_dollars") or res.get("estimated_fare", 0.0)
        cache[key] = round(float(c), 2)
        return cache[key]

    best_total = None
    best_order = None
    best_legs: list[float] = []
    for perm in permutations(ids):
        # home → perm[0] → ... → perm[-1] → home
        legs = []
        total = 0.0
        prev = HOME
        for bid in perm:
            c = _cost(prev, coords[bid])
            legs.append(c)
            total += c
            prev = coords[bid]
        legs.append(_cost(prev, HOME))
        total += legs[-1]
        total = round(total, 2)
        if best_total is None or total < best_total or (
            total == best_total and list(perm) < list(best_order or ())
        ):
            best_total = total
            best_order = list(perm)
            best_legs = [round(x, 2) for x in legs]
    return {
        "order": best_order,
        "total_cost_dollars": best_total,
        "leg_costs": best_legs,
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(initial_config)
    details["expected"] = expected
    got_order = answer.get("order") or []
    if not isinstance(got_order, list):
        got_order = []
    got_order = [str(x) for x in got_order]
    checks["order_correct"] = got_order == expected["order"]
    try:
        checks["total_cost_correct"] = abs(
            float(answer.get("total_cost_dollars")) - expected["total_cost_dollars"]
        ) < 0.05
    except (TypeError, ValueError):
        checks["total_cost_correct"] = False
    got_legs = answer.get("leg_costs") or []
    if not isinstance(got_legs, list):
        got_legs = []
    try:
        checks["leg_costs_correct"] = (
            len(got_legs) == len(expected["leg_costs"]) and
            all(abs(float(a) - b) < 0.05 for a, b in zip(got_legs, expected["leg_costs"]))
        )
    except (TypeError, ValueError):
        checks["leg_costs_correct"] = False
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
