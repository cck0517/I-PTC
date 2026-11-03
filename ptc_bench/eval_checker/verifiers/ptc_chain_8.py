"""Check the three most expensive catering items, Dinner Party event, and four-way split."""

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

        yelp_cfg = initial_config["YelpAPI"]
        # Top catering with rating > 4.0
        catering_biz = [
            b for b in yelp_cfg["businesses"].values()
            if any("catering" in c.lower() for c in b.get("categories", []))
            and b.get("rating", 0) > 4.0
        ]
        catering_biz.sort(key=lambda b: b["rating"], reverse=True)
        expected_caterer = catering_biz[0]  # Golden Gate Catering
        details["expected_caterer"] = expected_caterer["name"]

        dd_cfg = initial_config["DoorDashAPI"]
        dd_restaurant_id = None
        for rid, r in dd_cfg["restaurants"].items():
            if r["name"].lower() == expected_caterer["name"].lower():
                dd_restaurant_id = rid
                break
        if not dd_restaurant_id:
            for rid, r in dd_cfg["restaurants"].items():
                if expected_caterer["name"].lower() in r["name"].lower() or r["name"].lower() in expected_caterer["name"].lower():
                    dd_restaurant_id = rid
                    break
        if not dd_restaurant_id:
            # Fallback: first restaurant with a menu
            for rid in dd_cfg.get("menus", {}):
                if dd_cfg["menus"][rid]:
                    dd_restaurant_id = rid
                    break
        details["doordash_restaurant_id"] = dd_restaurant_id

        menu = dd_cfg["menus"].get(dd_restaurant_id, [])
        menu_sorted = sorted(menu, key=lambda i: i.get("price", 0), reverse=True)
        top_3_items = menu_sorted[:3]
        expected_item_ids = {item.get("item_id", "") for item in top_3_items}
        expected_item_names = [item.get("name", "") for item in top_3_items]
        item_total = sum(item.get("price", 0) for item in top_3_items)
        delivery_fee = dd_cfg["restaurants"].get(dd_restaurant_id, {}).get("delivery_fee", 0)
        expected_order_total = round(item_total + delivery_fee, 2)
        attendees = ["alice_v", "bob_smith", "diana_k"]
        per_person = round(expected_order_total / (len(attendees) + 1), 2)

        details["expected_items"] = expected_item_names
        details["expected_order_total"] = expected_order_total
        details["expected_per_person"] = per_person

        dd_state = final_state.get("DoorDashAPI", {})
        dd_orders = dd_state.get("orders", [])
        initial_orders = dd_cfg.get("orders", [])
        new_orders = dd_orders[len(initial_orders):]

        checks["doordash_order_placed"] = len(new_orders) >= 1

        if dd_restaurant_id:
            order_from_correct = any(
                o.get("restaurant_id") == dd_restaurant_id
                for o in new_orders
            )
            checks["order_from_correct_restaurant"] = order_from_correct

        for o in new_orders:
            if o.get("restaurant_id") == dd_restaurant_id:
                ordered_names = set()
                for item in o.get("items", []):
                    if isinstance(item, str):
                        ordered_names.add(item.lower())
                    else:
                        ordered_names.add(item.get("name", "").lower())
                expected_lower = {n.lower() for n in expected_item_names}
                checks["order_has_3_most_expensive"] = expected_lower <= ordered_names
                break
        else:
            checks["order_has_3_most_expensive"] = False

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

        dinner_party_event = any(
            "dinner" in e.get("title", "").lower() and "party" in e.get("title", "").lower()
            for e in new_events
        )
        checks["calendar_event_title_dinner_party"] = dinner_party_event

        for e in new_events:
            if "dinner" in e.get("title", "").lower():
                event_attendees = set(e.get("attendees", []))
                checks["calendar_has_attendees"] = set(attendees) <= event_attendees
                break
        else:
            checks["calendar_has_attendees"] = False

        venmo_state = final_state.get("VenmoAPI", {})
        venmo_txns = venmo_state.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        new_txns = venmo_txns[len(initial_txns):]

        for attendee in attendees:
            attendee_txn = any(
                t.get("receiver") == attendee
                for t in new_txns
            )
            checks[f"venmo_to_{attendee}"] = attendee_txn

        for attendee in attendees:
            attn_txns = [t for t in new_txns if t.get("receiver") == attendee]
            if attn_txns:
                amount = attn_txns[0].get("amount", 0)
                checks[f"venmo_{attendee}_amount_correct"] = abs(amount - per_person) < 2.0
            else:
                checks[f"venmo_{attendee}_amount_correct"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, details=details,
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"Verifier error: {e}"],
        )
