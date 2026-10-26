"""Verifier for ptc_chain_6.

Chain: DoorDashAPI -> VenmoAPI -> PersonalCalendarAPI
Search 'pizza' -> get menus from top 3 -> find cheapest Margherita ->
order 3 of it -> split cost with roommates -> create calendar event.

Top 3 pizza restaurants (sorted by rating):
  dd_r20: Tony's Pizza Napoletana (4.8) - Margherita $18.99
  dd_r21: Pizzeria Delfina (4.6)       - Margherita $15.99
  dd_r22: Domino's (3.5)               - Margherita $10.99
  (dd_r23 Sushi House doesn't match pizza)

Cheapest Margherita: Domino's (dd_r22) at $10.99.
Order 3 x $10.99 = $32.97 + delivery fee $0.99 = $33.96 total.
Split 3 ways (user + 2 roommates): $33.96 / 3 = $11.32 each.
Venmo requests to roommate1 and roommate2 for ~$11.32.
Calendar event 'Pizza Night' created.
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

        dd_cfg = initial_config["DoorDashAPI"]

        # ---- Derive expected values from config ----
        # Search for pizza restaurants
        pizza_restaurants = []
        for rid, r in dd_cfg["restaurants"].items():
            if "pizza" in r.get("cuisine", "").lower() or "pizza" in r["name"].lower():
                pizza_restaurants.append((rid, r))
        pizza_restaurants.sort(key=lambda x: x[1]["rating"], reverse=True)
        top_3 = pizza_restaurants[:3]
        top_3_ids = [rid for rid, _ in top_3]
        details["top_3_pizza"] = [r["name"] for _, r in top_3]

        # Find cheapest Margherita across top 3
        cheapest_margherita = None
        cheapest_restaurant_id = None
        for rid in top_3_ids:
            menu = dd_cfg["menus"].get(rid, [])
            for item in menu:
                if "margherita" in item["name"].lower():
                    if cheapest_margherita is None or item["price"] < cheapest_margherita["price"]:
                        cheapest_margherita = item
                        cheapest_restaurant_id = rid

        if cheapest_margherita:
            delivery_fee = dd_cfg["restaurants"][cheapest_restaurant_id]["delivery_fee"]
            item_total = 3 * cheapest_margherita["price"]
            expected_total = round(item_total + delivery_fee, 2)
            per_person = round(expected_total / 3, 2)
            details["cheapest_margherita"] = cheapest_margherita["name"]
            details["cheapest_restaurant"] = dd_cfg["restaurants"][cheapest_restaurant_id]["name"]
            details["order_total"] = expected_total
            details["per_person"] = per_person

        # ---- Check 1: DoorDash order placed ----
        dd_state = final_state.get("DoorDashAPI", {})
        dd_orders = dd_state.get("orders", [])
        initial_orders = dd_cfg.get("orders", [])
        new_orders = dd_orders[len(initial_orders):]

        checks["doordash_order_placed"] = len(new_orders) >= 1

        # Check order from correct restaurant
        if cheapest_restaurant_id:
            order_from_correct = any(
                o.get("restaurant_id") == cheapest_restaurant_id
                for o in new_orders
            )
            checks["order_from_cheapest_margherita_restaurant"] = order_from_correct

        # ---- Check 2: Venmo requests to roommates ----
        venmo_state = final_state.get("VenmoAPI", {})
        venmo_txns = venmo_state.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        new_txns = venmo_txns[len(initial_txns):]

        roommate_receivers = {t.get("receiver") for t in new_txns}
        checks["venmo_to_roommate1"] = "roommate1" in roommate_receivers
        checks["venmo_to_roommate2"] = "roommate2" in roommate_receivers

        # Check amounts are approximately equal split
        if cheapest_margherita:
            for rm in ["roommate1", "roommate2"]:
                rm_txns = [t for t in new_txns if t.get("receiver") == rm]
                if rm_txns:
                    amount = rm_txns[0].get("amount", 0)
                    checks[f"venmo_{rm}_amount_correct"] = abs(amount - per_person) < 1.0
                else:
                    checks[f"venmo_{rm}_amount_correct"] = False

        # ---- Check 3: Calendar event created ----
        cal_state = final_state.get("PersonalCalendarAPI", {})
        cal_events = cal_state.get("events", {})
        if isinstance(cal_events, list):
            final_events = cal_events
        else:
            final_events = list(cal_events.values())

        initial_cal = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
        if isinstance(initial_cal, list):
            initial_event_ids = {e["id"] for e in initial_cal}
        else:
            initial_event_ids = set(initial_cal.keys())

        new_events = [
            e for e in final_events
            if e.get("id") not in initial_event_ids
        ]

        checks["calendar_event_created"] = len(new_events) >= 1
        pizza_night_event = any(
            "pizza" in e.get("title", "").lower()
            for e in new_events
        )
        checks["calendar_event_title_pizza"] = pizza_night_event

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
