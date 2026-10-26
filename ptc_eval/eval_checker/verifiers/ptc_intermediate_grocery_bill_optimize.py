"""Verifier for ptc_intermediate_grocery_bill_optimize_0.

Minimum-cost assignment of items→stores, where each store incurs a
delivery fee iff used, and gift-card balance offsets the final subtotal
at that store (cap = gift card balance per store).
"""

from __future__ import annotations

from itertools import product
from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "GroceryDeliveryAPIAlt", initial_config.get("GroceryDeliveryAPI", {}),
    )


def _cost_for_assignment(assignment: dict, stores: dict, gift_cards: dict,
                         price_of: dict) -> float:
    by_store: dict[str, float] = {}
    for item, sid in assignment.items():
        if sid is None:
            continue
        by_store[sid] = by_store.get(sid, 0.0) + price_of[(item, sid)]
    total = 0.0
    for sid, items_sum in by_store.items():
        fee = float(stores[sid]["delivery_fee"])
        subtotal = items_sum + fee
        gc = float(gift_cards.get(sid, {}).get("balance", 0.0))
        out_of_pocket = max(0.0, subtotal - gc)
        total += out_of_pocket
    return round(total, 2)


def _compute_ground_truth(cfg: dict) -> dict:
    stores = cfg["stores"]
    gift_cards = cfg["gift_cards"]
    shopping_list = cfg["shopping_list"]
    # price map: (item_name, store_id) → price where store carries it.
    price_of: dict[tuple[str, str], float] = {}
    available_stores: dict[str, list[str]] = {}
    for sid, s in stores.items():
        inv = s.get("inventory", [])
        for entry in inv:
            nm = entry["item_name"]
            price_of[(nm, sid)] = float(entry["price"])
            available_stores.setdefault(nm, []).append(sid)
    # For feasibility, each item must have ≥1 available store.
    for item in shopping_list:
        if not available_stores.get(item):
            return {}
    # Brute-force over small cartesian (8 stores × 20 items → 8^20 is too big).
    # Reduce: each item can go to the stores that carry it (at most ~5).
    # Prune via greedy enum + bound. Here 20 items × avg 4 options ≈ 4^20 still
    # too big. Use DP over "stores_used subset".
    store_ids = sorted(stores.keys())
    n_stores = len(store_ids)
    idx = {sid: i for i, sid in enumerate(store_ids)}
    # For each subset of stores, compute min per-item price when restricted to
    # that subset — sum gives items total. Then add per-store delivery fee −
    # capped gift card.
    best_cost = float("inf")
    best_assignment: dict[str, str] = {}
    best_stores_used = None

    # Enumerate subsets (2^8 = 256). For each subset, greedy per-item assignment
    # minimizes per-item cost, then we compute the optimal reassignment WITHIN
    # the subset.  Actually: items only incur their price, and subset decides
    # which stores' delivery fees + gift cards apply.  For a fixed subset S,
    # per-item optimum price in S is deterministic (independent of other items).
    for mask in range(1, 1 << n_stores):
        subset = [store_ids[i] for i in range(n_stores) if mask >> i & 1]
        assignment: dict[str, str] = {}
        item_total_by_store: dict[str, float] = {s: 0.0 for s in subset}
        feasible = True
        for item in shopping_list:
            best_price = None
            best_sid = None
            for sid in subset:
                if (item, sid) in price_of:
                    p = price_of[(item, sid)]
                    if best_price is None or p < best_price or (
                        p == best_price and sid < best_sid
                    ):
                        best_price = p
                        best_sid = sid
            if best_sid is None:
                feasible = False
                break
            assignment[item] = best_sid
            item_total_by_store[best_sid] += best_price
        if not feasible:
            continue
        # Now compute out-of-pocket: for each USED store (receives ≥1 item),
        # subtotal = items_sum + delivery_fee; gift card caps at balance.
        oop = 0.0
        used = set()
        for sid, items_sum in item_total_by_store.items():
            if items_sum == 0.0:
                continue  # store in subset but not used
            used.add(sid)
            fee = float(stores[sid]["delivery_fee"])
            subtotal = items_sum + fee
            gc = float(gift_cards.get(sid, {}).get("balance", 0.0))
            oop += max(0.0, subtotal - gc)
        oop = round(oop, 2)
        cand_key = (oop, len(used), sorted(used))
        if best_stores_used is None or cand_key < best_stores_used:
            best_stores_used = cand_key
            best_cost = oop
            best_assignment = dict(assignment)

    used = sorted({sid for sid in best_assignment.values()})
    return {
        "assignment": best_assignment,
        "stores_used": used,
        "total_cost_dollars": best_cost,
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected_cost"] = expected.get("total_cost_dollars")
    details["expected_stores_used"] = expected.get("stores_used")
    if not expected:
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["No feasible assignment"])
    # Accept any assignment that achieves the same total cost (the optimum may
    # not be unique).  But require stores_used and total_cost to match.
    try:
        checks["total_cost_correct"] = abs(
            float(answer.get("total_cost_dollars")) - expected["total_cost_dollars"]
        ) < 0.05
    except (TypeError, ValueError):
        checks["total_cost_correct"] = False
    got_stores = answer.get("stores_used") or []
    if not isinstance(got_stores, list):
        got_stores = []
    checks["stores_used_correct"] = sorted(str(x) for x in got_stores) == expected["stores_used"]
    # Assignment consistency: items in agent's assignment all map to stores_used
    assign_got = answer.get("assignment") or {}
    if not isinstance(assign_got, dict):
        assign_got = {}
    checks["assignment_consistent"] = (
        set(assign_got.keys()) == set(expected["assignment"].keys())
        and all(v in set(got_stores) for v in assign_got.values())
    )
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
