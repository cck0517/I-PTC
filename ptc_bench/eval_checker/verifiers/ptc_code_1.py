"""Check the cheapest-average sushi order after weighted ranking.

Top-three restaurant s6 has no DoorDash menu. Accept ds3 (ranked top
three) or ds5 (cheapest among available menus), with their top-priced items.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, score_from_checks


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    dd = (final_state or {}).get("DoorDashAPI", {})
    orders = dd.get("orders", [])
    initial_orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    initial_order_ids = {o.get("id") for o in initial_orders}
    new_orders = [o for o in orders if o.get("id") not in initial_order_ids]
    checks["order_placed"] = len(new_orders) > 0

    if new_orders:
        ordered_rest = new_orders[0].get("restaurant_id", "")
        checks["correct_restaurant"] = ordered_rest in ("ds5", "ds3")
    else:
        checks["correct_restaurant"] = False

    if new_orders:
        ordered_items = new_orders[0].get("items", [])
        checks["two_items_ordered"] = len(ordered_items) == 2
    else:
        checks["two_items_ordered"] = False

    if new_orders:
        items_lower = {i.lower() for i in new_orders[0].get("items", [])}
        ds5_top2 = {"chirashi bowl", "dragon roll"}
        ds3_top2 = {"nigiri set", "chef's special roll"}
        checks["correct_items"] = (
            items_lower == ds5_top2 or items_lower == ds3_top2
            or items_lower.issubset(ds5_top2 | ds3_top2)
        )
    else:
        checks["correct_items"] = False

    ans = (final_answer or "").lower()
    checks["mentions_ranking"] = (
        "sushi ran" in ans or "hinata" in ans or "saru" in ans
        or "score" in ans or "rank" in ans or "weighted" in ans
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={"new_orders": new_orders if new_orders else []},
    )
