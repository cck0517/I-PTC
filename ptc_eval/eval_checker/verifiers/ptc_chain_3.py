"""Verifier for ptc_chain_3.

Chain: VenmoAPI -> YelpAPI -> DoorDashAPI
Compute food spending from Venmo -> search cheap restaurants -> order
cheapest main from highest-rated among top 3 cheapest.

Expected Venmo food transactions (notes containing food/dinner/lunch):
  txn_1: $35.00 "Dinner at Thai place"
  txn_3: $22.50 "Lunch split"
  txn_4: $15.00 "food delivery split"
  txn_6: $18.75 "food court snacks"
  txn_8: $28.00 "dinner split at Olive Garden"
  Total food: $119.25

Top 3 cheapest restaurants (price_level=1):
  biz_30: Taqueria El Buen Sabor (rating 4.3, price_level 1)
  biz_31: Pho Saigon (rating 4.6, price_level 1)
  biz_32: Joe's Pizza (rating 4.1, price_level 1)

Highest-rated among those 3: Pho Saigon (4.6) -> DoorDash dd_r11
Cheapest main on Pho Saigon menu: Banh Mi ($8.99)
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
    if final_state is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["final_state is None"],
        )

    try:
        checks: dict[str, bool] = {}
        details: dict[str, Any] = {}

        # ---- Compute expected food spending from Venmo config ----
        venmo_cfg = initial_config["VenmoAPI"]
        food_keywords = ["food", "dinner", "lunch"]
        food_total = 0.0
        for txn in venmo_cfg["transactions"]:
            note = txn.get("note", "").lower()
            if any(kw in note for kw in food_keywords):
                food_total += txn["amount"]
        details["expected_food_total"] = food_total

        # ---- Determine expected restaurant & item ----
        yelp_cfg = initial_config["YelpAPI"]
        # Sort by price_level ascending, take top 3. price_level may be stored
        # as int (1..4) or as a dollar-sign string ("$".."$$$$"); normalize.
        def _pl_rank(b):
            pl = b.get("price_level")
            if isinstance(pl, int):
                return pl
            if isinstance(pl, str) and pl and set(pl) <= {"$"}:
                return len(pl)
            return 999
        all_biz = list(yelp_cfg["businesses"].values())
        all_biz.sort(key=_pl_rank)
        top_3_cheapest = all_biz[:3]
        # Highest-rated among those 3
        top_3_cheapest.sort(key=lambda b: b["rating"], reverse=True)
        expected_biz = top_3_cheapest[0]  # Pho Saigon (4.6)
        details["expected_restaurant"] = expected_biz["name"]

        # Find DoorDash restaurant matching
        dd_cfg = initial_config["DoorDashAPI"]
        dd_restaurant_id = None
        for rid, r in dd_cfg["restaurants"].items():
            if r["name"].lower() == expected_biz["name"].lower():
                dd_restaurant_id = rid
                break
        details["expected_doordash_id"] = dd_restaurant_id

        # Get cheapest main from that restaurant's menu
        if dd_restaurant_id and dd_restaurant_id in dd_cfg["menus"]:
            menu = dd_cfg["menus"][dd_restaurant_id]
            mains = [item for item in menu if item["category"] == "main"]
            mains.sort(key=lambda i: i["price"])
            expected_item = mains[0] if mains else None
        else:
            expected_item = None
        if expected_item:
            details["expected_item"] = expected_item["name"]
            details["expected_item_price"] = expected_item["price"]

        # ---- Check 1: DoorDash order placed ----
        dd_state = final_state.get("DoorDashAPI", {})
        dd_orders = dd_state.get("orders", [])
        initial_orders = dd_cfg.get("orders", [])
        checks["doordash_order_placed"] = len(dd_orders) > len(initial_orders)

        # Check order from correct restaurant
        new_orders = dd_orders[len(initial_orders):]
        if expected_item and dd_restaurant_id:
            order_from_correct = any(
                o.get("restaurant_id") == dd_restaurant_id
                for o in new_orders
            )
            checks["order_from_correct_restaurant"] = order_from_correct

            # Check order contains the cheapest main
            order_has_item = any(
                expected_item["name"].lower() in " ".join(
                    item.lower() if isinstance(item, str) else item.get("name", "").lower()
                    for item in o.get("items", [])
                )
                for o in new_orders
            )
            checks["order_has_cheapest_main"] = order_has_item

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
