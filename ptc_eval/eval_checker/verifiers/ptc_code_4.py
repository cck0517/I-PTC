"""Verifier for ptc_code_4.

Analyse DoorDash order history, find most frequent restaurant and
most-spent restaurant, compare Yelp ratings, order from higher-rated,
Venmo split.

Pre-computed from 12 orders:
  dr_pg (Pho Garden):    o01,o03,o05,o07,o09,o11 = 6 orders
      totals: 17.48+18.48+17.48+13.48+17.48+25.47 = 109.87
  dr_ps (Prime Steakhouse): o02,o06,o10 = 3 orders
      totals: 58.98+51.98+77.97 = 188.93
  dr_tc (Thai Cottage):  o04,o08,o12 = 3 orders
      totals: 16.98+17.98+16.98 = 51.94

Most frequent: dr_pg (Pho Garden) - 6 orders
Most spent:    dr_ps (Prime Steakhouse) - $188.93

Yelp ratings:
  y_pg (Pho Garden):        4.4
  y_ps (Prime Steakhouse):  4.7

Higher-rated = Prime Steakhouse (4.7)

Total spent on all = 109.87 + 188.93 + 51.94 = 350.74
Average order value = 350.74 / 12 = 29.23

Item from Prime Steakhouse closest to avg order value ($29.23):
  ps1 Filet Mignon $52.99
  ps2 NY Strip $45.99
  ps3 Caesar Salad $16.99
  ps4 Lobster Bisque $18.99
  ps5 Grilled Salmon $34.99  <-- closest to $29.23 (diff = 5.76)
  ps6 Creme Brulee $12.99

Order: Grilled Salmon (ps5) to '250 King St, San Francisco'
Total with delivery: 34.99 + 5.99 = 40.98
Venmo request to eve_jones for half = 40.98 / 2 = 20.49

We check:
  1. DoorDash order placed from dr_ps
  2. Venmo transaction/request to eve_jones
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

    # --- 1. DoorDash order placed -------------------------------------------
    dd = (final_state or {}).get("DoorDashAPI", {})
    orders = dd.get("orders", [])
    initial_orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    initial_order_ids = {o.get("id") for o in initial_orders}
    new_orders = [o for o in orders if o.get("id") not in initial_order_ids]
    checks["doordash_order_placed"] = len(new_orders) > 0

    # --- 2. Order from Prime Steakhouse (dr_ps) ----------------------------
    if new_orders:
        checks["order_from_higher_rated"] = any(
            o.get("restaurant_id") == _HIGHER_RATED_REST_ID for o in new_orders
        )
    else:
        checks["order_from_higher_rated"] = False

    # --- 3. Correct item ordered (Grilled Salmon or close to avg price) ----
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

    # --- 4. Venmo request/payment to eve_jones ------------------------------
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

    # --- 5. Venmo amount is approximately half the order total ---------------
    if eve_txns and new_orders:
        # Find the order total from Prime Steakhouse
        ps_orders = [o for o in new_orders if o.get("restaurant_id") == _HIGHER_RATED_REST_ID]
        if ps_orders:
            order_total = ps_orders[0].get("total", 0)
            expected_half = order_total / 2
            venmo_amount = eve_txns[0].get("amount", 0)
            checks["venmo_amount_correct"] = abs(venmo_amount - expected_half) <= _TOLERANCE
        else:
            # Check against any order
            order_total = new_orders[0].get("total", 0)
            expected_half = order_total / 2
            venmo_amount = eve_txns[0].get("amount", 0)
            checks["venmo_amount_correct"] = abs(venmo_amount - expected_half) <= _TOLERANCE
    else:
        checks["venmo_amount_correct"] = False

    # --- 6. Answer mentions the analysis results ----------------------------
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
