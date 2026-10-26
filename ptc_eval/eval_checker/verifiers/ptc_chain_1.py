"""Verifier for ptc_chain_1.

Chain: PersonalCalendarAPI -> YelpAPI -> DoorDashAPI -> VenmoAPI
Find calendar event with restaurant location -> search Yelp -> order
first appetizer from DoorDash -> Venmo payment to bob_smith.

Expected: "Dinner with Team" at "Sakura Japanese Bistro". First
appetizer is Edamame ($6.99). DoorDash order placed, Venmo payment to
bob_smith matching the order total.
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

        # ---- Derive expected values from initial_config ----
        # Find the calendar event with a restaurant name in its location
        cal_cfg = initial_config.get("PersonalCalendarAPI", {})
        dd_cfg = initial_config["DoorDashAPI"]

        # Find the DoorDash restaurant that matches a calendar event location
        target_rid = None
        for rid, rest in dd_cfg["restaurants"].items():
            rest_name = rest.get("name", "").lower()
            for evt in cal_cfg.get("events", {}).values():
                loc = evt.get("location", "").lower()
                if rest_name and rest_name in loc:
                    target_rid = rid
                    break
            if target_rid:
                break

        # Fallback: use first restaurant with a menu
        if not target_rid:
            for rid in dd_cfg.get("menus", {}):
                if dd_cfg["menus"][rid]:
                    target_rid = rid
                    break

        menu = dd_cfg["menus"].get(target_rid, [])
        appetizers = [item for item in menu if item.get("category") == "appetizer"]
        first_appetizer = appetizers[0] if appetizers else {}
        expected_item_name = first_appetizer.get("name", "")
        delivery_fee = dd_cfg["restaurants"].get(target_rid, {}).get("delivery_fee", 0)
        expected_total = round(first_appetizer.get("price", 0) + delivery_fee, 2)
        details["expected_item"] = expected_item_name
        details["expected_total"] = expected_total

        # ---- Check 1: DoorDash order placed ----
        dd_state = final_state.get("DoorDashAPI", {})
        dd_orders = dd_state.get("orders", [])
        initial_orders = dd_cfg.get("orders", [])

        checks["doordash_order_placed"] = len(dd_orders) > len(initial_orders)

        # Check the order contains the first appetizer
        new_orders = dd_orders[len(initial_orders):]
        order_has_appetizer = any(
            expected_item_name.lower() in " ".join(
                item.lower() if isinstance(item, str) else item.get("name", "").lower()
                for item in o.get("items", [])
            )
            for o in new_orders
        )
        checks["order_contains_first_appetizer"] = order_has_appetizer

        # ---- Check 2: Venmo payment to bob_smith ----
        venmo_state = final_state.get("VenmoAPI", {})
        venmo_txns = venmo_state.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        new_txns = venmo_txns[len(initial_txns):]

        bob_payment = any(
            t.get("receiver") == "bob_smith"
            for t in new_txns
        )
        checks["venmo_payment_to_bob"] = bob_payment

        # Check that the Venmo amount matches order total (within tolerance)
        bob_amount_matches = any(
            t.get("receiver") == "bob_smith"
            and abs(t.get("amount", 0) - expected_total) < 0.50
            for t in new_txns
        )
        checks["venmo_amount_matches_order"] = bob_amount_matches

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
