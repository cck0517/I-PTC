"""Verifier for ptc_grocery_gift_card_0.

The model must assign each item on the shopping list to a store that
carries it, minimizing total out-of-pocket cost after applying gift
cards.  We compute a greedy-optimal solution and compare.
"""

from __future__ import annotations

from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)

TOLERANCE = 0.50  # dollar tolerance


def _solve(config: dict) -> float:
    """Compute optimal out-of-pocket cost using greedy per-item assignment.

    For each item, assign it to the store that minimizes marginal OOP cost
    (considering gift cards). This is a heuristic but produces near-optimal
    results for typical grocery scenarios.
    """
    shopping = config["shopping_list"]
    stores = config["stores"]
    gift_cards = dict(config.get("gift_cards", {}))

    # Track spending per store
    store_spending: dict[str, float] = {}
    assignment: dict[str, str] = {}

    for item in shopping:
        best_sid = None
        best_marginal = float("inf")

        for sid, s in stores.items():
            inv = s.get("inventory", {})
            if item not in inv or not inv[item].get("in_stock", True):
                continue

            price = inv[item]["price"]
            current_spend = store_spending.get(sid, 0.0)

            if current_spend == 0:
                # First item from this store — include delivery fee
                new_spend = price + s.get("delivery_fee", 0)
            else:
                new_spend = current_spend + price

            gc = gift_cards.get(sid, 0.0)
            new_oop = max(0.0, new_spend - gc)
            old_oop = max(0.0, current_spend - gc) if current_spend > 0 else 0.0
            marginal = new_oop - old_oop

            if marginal < best_marginal:
                best_marginal = marginal
                best_sid = sid
                best_new_spend = new_spend

        if best_sid is not None:
            store_spending[best_sid] = best_new_spend
            assignment[item] = best_sid

    # Compute total OOP
    total_oop = 0.0
    for sid, spend in store_spending.items():
        gc = gift_cards.get(sid, 0.0)
        total_oop += max(0.0, spend - gc)

    return round(total_oop, 2)


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )
    if isinstance(answer, list):
        answer = answer[0] if answer else {}

    cfg = initial_config.get("GroceryDeliveryAPIAlt", initial_config.get("GroceryDeliveryAPI", {}))
    optimal_oop = _solve(cfg)

    checks: dict[str, bool] = {}

    # Check total_out_of_pocket (lenient — within tolerance of our greedy solution)
    ans_oop = answer.get("total_out_of_pocket")
    if isinstance(ans_oop, (int, float)):
        # Accept if model's answer is at most TOLERANCE worse than greedy
        checks["total_out_of_pocket"] = ans_oop <= optimal_oop + TOLERANCE
    else:
        checks["total_out_of_pocket"] = False

    # Check that assignments cover every item
    assignments = answer.get("assignments", [])
    assigned_items = set()
    for a in assignments:
        for item in a.get("items", []):
            assigned_items.add(item)
    checks["covers_all_items"] = assigned_items >= set(cfg["shopping_list"])

    # Verify per-store arithmetic is internally consistent
    arith_ok = True
    for a in assignments:
        ic = a.get("item_cost", 0)
        df = a.get("delivery_fee", 0)
        st = a.get("store_total", 0)
        gc = a.get("gift_card_applied", 0)
        oop = a.get("out_of_pocket", 0)
        if not isinstance(st, (int, float)):
            arith_ok = False
            continue
        if abs(st - (ic + df)) > 0.01:
            arith_ok = False
        if abs(oop - max(0, st - gc)) > 0.01:
            arith_ok = False
    checks["arithmetic_consistent"] = arith_ok

    # Verify items are assigned to stores that actually carry them
    valid_assignment = True
    for a in assignments:
        sid = a.get("store_id", "")
        store = cfg["stores"].get(sid)
        if store is None:
            valid_assignment = False
            break
        for item in a.get("items", []):
            inv = store.get("inventory", {})
            if item not in inv or not inv[item].get("in_stock", True):
                valid_assignment = False
                break
    checks["valid_store_assignment"] = valid_assignment

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "extracted_answer": answer,
            "optimal_out_of_pocket": optimal_oop,
        },
    )
