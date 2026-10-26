"""Verifier for ptc_chain_9.

Chain: YelpAPI -> DoorDashAPI
Check bookmarked restaurants' review ratings -> remove if average
review rating dropped below 4.0 -> order from fastest delivery
remaining.

Initial bookmarks: [biz_60, biz_61, biz_62, biz_63]

Review average ratings:
  biz_60 (Golden Dragon): (3+4+3)/3 = 3.33 -> BELOW 4.0, remove
  biz_61 (La Taqueria):   (5+4)/2   = 4.50 -> keep
  biz_62 (Burma Superstar): (5+4)/2 = 4.50 -> keep
  biz_63 (House of Pancakes): (3+4+3)/3 = 3.33 -> BELOW 4.0, remove

Remaining bookmarks: [biz_61, biz_62]

DoorDash delivery times:
  dd_r40 La Taqueria:    20 min (fastest)
  dd_r41 Burma Superstar: 35 min

Order from La Taqueria (dd_r40). Highest-rated (is_popular) menu item:
  t1: Super Burrito $13.99 (popular)
  t2: Carne Asada Tacos $11.99 (popular)
Both are popular.  "Highest-rated menu item" -> is_popular items;
Super Burrito is the first popular item, or the most expensive popular
one. We accept either popular item.
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

        # ---- Derive expected bookmark removals ----
        yelp_cfg = initial_config["YelpAPI"]
        initial_bookmarks = list(yelp_cfg.get("bookmarks", []))
        reviews_data = yelp_cfg.get("reviews", {})

        should_remove = []
        should_keep = []
        for biz_id in initial_bookmarks:
            biz_reviews = reviews_data.get(biz_id, [])
            if biz_reviews:
                avg_rating = sum(r["rating"] for r in biz_reviews) / len(biz_reviews)
            else:
                # No reviews -> keep (can't determine if dropped)
                avg_rating = 5.0
            if avg_rating < 4.0:
                should_remove.append(biz_id)
            else:
                should_keep.append(biz_id)

        details["should_remove"] = should_remove
        details["should_keep"] = should_keep

        # ---- Check 1: Correct bookmarks removed ----
        yelp_state = final_state.get("YelpAPI", {})
        final_bookmarks = set(yelp_state.get("bookmarks", []))

        for biz_id in should_remove:
            checks[f"removed_{biz_id}"] = biz_id not in final_bookmarks

        for biz_id in should_keep:
            checks[f"kept_{biz_id}"] = biz_id in final_bookmarks

        # ---- Determine fastest delivery among remaining ----
        dd_cfg = initial_config["DoorDashAPI"]
        remaining_names = [
            yelp_cfg["businesses"][bid]["name"].lower() for bid in should_keep
        ]

        fastest_restaurant_id = None
        fastest_time = float("inf")
        for rid, r in dd_cfg["restaurants"].items():
            if r["name"].lower() in remaining_names:
                t = r.get("estimated_delivery_time", 999)
                if t < fastest_time:
                    fastest_time = t
                    fastest_restaurant_id = rid

        details["fastest_restaurant_id"] = fastest_restaurant_id
        if fastest_restaurant_id:
            details["fastest_restaurant_name"] = dd_cfg["restaurants"][fastest_restaurant_id]["name"]

        # ---- Check 2: DoorDash order placed from fastest restaurant ----
        dd_state = final_state.get("DoorDashAPI", {})
        dd_orders = dd_state.get("orders", [])
        initial_orders = dd_cfg.get("orders", [])
        new_orders = dd_orders[len(initial_orders):]

        checks["doordash_order_placed"] = len(new_orders) >= 1

        if fastest_restaurant_id:
            order_from_fastest = any(
                o.get("restaurant_id") == fastest_restaurant_id
                for o in new_orders
            )
            checks["order_from_fastest_restaurant"] = order_from_fastest

        # Check order contains a popular (highest-rated) item
        if fastest_restaurant_id:
            menu = dd_cfg["menus"].get(fastest_restaurant_id, [])
            popular_items = [i for i in menu if i.get("is_popular")]
            popular_names = {i["name"].lower() for i in popular_items}

            order_has_popular = False
            for o in new_orders:
                if o.get("restaurant_id") == fastest_restaurant_id:
                    for item in o.get("items", []):
                        item_name = item.lower() if isinstance(item, str) else item.get("name", "").lower()
                        if item_name in popular_names:
                            order_has_popular = True
                            break
            checks["order_has_popular_item"] = order_has_popular

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
