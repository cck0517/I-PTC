"""Verifier for ptc_code_1.

Weighted scoring of sushi restaurants, get DoorDash menus for top 3,
compare average menu price, order from the cheapest-average restaurant.

Scoring formula: (rating * 2) + log(review_count) - (price_level * 0.5)

Precomputed scores (using natural log):
  s1: 4.8*2 + ln(1200) - 4*0.5 = 9.6 + 7.090 - 2.0 = 14.690
  s2: 4.7*2 + ln(800)  - 4*0.5 = 9.4 + 6.685 - 2.0 = 14.085
  s3: 4.6*2 + ln(600)  - 3*0.5 = 9.2 + 6.397 - 1.5 = 14.097
  s4: 4.5*2 + ln(450)  - 3*0.5 = 9.0 + 6.109 - 1.5 = 13.609
  s5: 4.4*2 + ln(350)  - 2*0.5 = 8.8 + 5.858 - 1.0 = 13.658
  s6: 4.3*2 + ln(900)  - 2*0.5 = 8.6 + 6.802 - 1.0 = 14.402
  s7: 4.2*2 + ln(280)  - 2*0.5 = 8.4 + 5.635 - 1.0 = 13.035
  s8: 4.0*2 + ln(200)  - 1*0.5 = 8.0 + 5.298 - 0.5 = 12.798
  s9: 3.8*2 + ln(150)  - 1*0.5 = 7.6 + 5.011 - 0.5 = 12.111
  s10:3.5*2 + ln(80)   - 1*0.5 = 7.0 + 4.382 - 0.5 = 10.882

Ranking: s1 (14.69), s6 (14.40), s3 (14.10), s2 (14.09), ...
Top 3: s1, s6, s3

DoorDash mapping:
  s1 -> ds1 (Sushi Ran):  menu avg = (45+28+18+6+8+16)/6 = 121/6 = 20.17
  s6 -> no DoorDash entry! (ds5 is Hinata Sushi = s5)
     But s3 -> ds3 (Saru Sushi Bar): menu avg = (22+32+14+9+7+6)/6 = 90/6 = 15.00
  The model has to figure out mapping. DoorDash only has ds1, ds3, ds5.

Since s6 is not on DoorDash, models may pick the available top 3
from DoorDash: ds1 (s1), ds3 (s3), ds5 (s5).

Average prices:
  ds1: 121/6 = 20.17
  ds3: 90/6 = 15.00
  ds5: (16+19+10+7+12+5)/6 = 69/6 = 11.50

Cheapest average = ds5 (Hinata Sushi, avg $11.50)
Two highest-priced items from ds5: Chirashi Bowl ($19) and Dragon Roll ($16)
Order: items hs2 and hs1 to '425 Mission St, San Francisco'

We verify: DoorDash order placed (any order with the right items).
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

    # --- 1. Check that a DoorDash order was placed --------------------------
    dd = (final_state or {}).get("DoorDashAPI", {})
    orders = dd.get("orders", [])
    initial_orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    initial_order_ids = {o.get("id") for o in initial_orders}
    new_orders = [o for o in orders if o.get("id") not in initial_order_ids]
    checks["order_placed"] = len(new_orders) > 0

    # --- 2. Check that the order was placed from the correct restaurant -----
    # The cheapest-average restaurant among those with DoorDash menus
    # should be ds5 (Hinata Sushi). But we're lenient: if the model
    # picks ds3 or ds5, both are reasonable depending on interpretation.
    if new_orders:
        ordered_rest = new_orders[0].get("restaurant_id", "")
        # ds5 is the ideal answer; ds3 is also acceptable
        checks["correct_restaurant"] = ordered_rest in ("ds5", "ds3")
    else:
        checks["correct_restaurant"] = False

    # --- 3. Check that the order has 2 items --------------------------------
    if new_orders:
        ordered_items = new_orders[0].get("items", [])
        checks["two_items_ordered"] = len(ordered_items) == 2
    else:
        checks["two_items_ordered"] = False

    # --- 4. Check that the items are the highest-priced from the chosen
    #    restaurant. For ds5: Chirashi Bowl ($19) and Dragon Roll ($16).
    #    For ds3: Nigiri Set ($32) and Chef's Special Roll ($22).
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

    # --- 5. Check the answer mentions the scoring / ranking -----------------
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
