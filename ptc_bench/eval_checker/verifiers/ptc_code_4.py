"""Check the higher-rated frequent/spend restaurant order and half-cost Venmo split.

Prime Steakhouse wins by rating; its Grilled Salmon is closest to the
historical average order value.
"""

from __future__ import annotations

import re
from typing import Any

from .core import VerificationResult, score_from_checks


_MOST_FREQUENT = "Pho Garden"
_MOST_SPENT = "Prime Steakhouse"
_HIGHER_RATED_REST_ID = "dr_ps"
_EXPECTED_ITEM = "Grilled Salmon"
_EXPECTED_VENMO_RECEIVER = "eve_jones"
_TOLERANCE = 2.0


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
    checks["doordash_order_placed"] = len(new_orders) > 0

    if new_orders:
        checks["order_from_higher_rated"] = any(
            o.get("restaurant_id") == _HIGHER_RATED_REST_ID for o in new_orders
        )
    else:
        checks["order_from_higher_rated"] = False

    if new_orders:
        all_items = []
        for o in new_orders:
            all_items.extend(o.get("items", []))
        items_lower = {i.lower() for i in all_items}
        checks["correct_item_ordered"] = (
            "grilled salmon" in items_lower
            or any("salmon" in i for i in items_lower)
        )
    else:
        checks["correct_item_ordered"] = False

    venmo = (final_state or {}).get("VenmoAPI", {})
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    initial_ids = {t.get("id") for t in initial_txns}
    new_txns = [t for t in txns if t.get("id") not in initial_ids]
    eve_txns = [
        t for t in new_txns
        if t.get("receiver") == _EXPECTED_VENMO_RECEIVER
    ]
    checks["venmo_to_eve"] = len(eve_txns) > 0

    if eve_txns and new_orders:
        ps_orders = [o for o in new_orders if o.get("restaurant_id") == _HIGHER_RATED_REST_ID]
        if ps_orders:
            order_total = ps_orders[0].get("total", 0)
            expected_half = order_total / 2
            venmo_amount = eve_txns[0].get("amount", 0)
            checks["venmo_amount_correct"] = abs(venmo_amount - expected_half) <= _TOLERANCE
        else:
            order_total = new_orders[0].get("total", 0)
            expected_half = order_total / 2
            venmo_amount = eve_txns[0].get("amount", 0)
            checks["venmo_amount_correct"] = abs(venmo_amount - expected_half) <= _TOLERANCE
    else:
        checks["venmo_amount_correct"] = False

    ans = (final_answer or "").lower()
    checks["mentions_analysis"] = (
        ("pho garden" in ans or "most frequent" in ans)
        and ("prime steakhouse" in ans or "most spent" in ans or "highest" in ans)
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "new_orders": new_orders,
            "new_venmo": [t for t in new_txns],
            "expected_most_frequent": _MOST_FREQUENT,
            "expected_most_spent": _MOST_SPENT,
        },
    )
