"""Verifier for ptc_loop_6.

Rate all delivered DoorDash orders 4 stars. Dynamically discovers
delivered orders from initial_config. Only delivered orders can be rated.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks

EXPECTED_RATING = 4


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Discover delivered orders from initial_config
    initial_orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    delivered_ids = [o.get("id", "") for o in initial_orders
                     if isinstance(o, dict) and o.get("status") == "delivered"]

    doordash = (final_state or {}).get("DoorDashAPI", {})
    orders = doordash.get("orders", [])

    # Build lookup by order id
    order_map: dict[str, dict] = {}
    for order in orders:
        if isinstance(order, dict):
            order_map[order.get("id", "")] = order

    # Each delivered order should be rated
    for order_id in delivered_ids:
        order = order_map.get(order_id, {})
        checks[f"{order_id}_rated_{EXPECTED_RATING}"] = order.get("rating") == EXPECTED_RATING

    # All original orders still present
    all_initial_ids = [o.get("id", "") for o in initial_orders if isinstance(o, dict)]
    checks["all_orders_present"] = all(oid in order_map for oid in all_initial_ids)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "delivered_order_ids": delivered_ids,
            "order_ratings": {
                oid: order_map.get(oid, {}).get("rating") for oid in delivered_ids
            },
        },
    )
