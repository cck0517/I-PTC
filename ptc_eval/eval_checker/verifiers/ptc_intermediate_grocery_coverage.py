"""Verifier for ptc_intermediate_grocery_coverage_0.

Ground truth is computed by enumerating minimum store covers and then,
among ties, picking the cover that maximizes total gift-card applied.
Assignments from items to chosen stores are also derived (each item goes
to the chosen store that carries it; when multiple chosen stores carry
the same item, the tie is broken by sorted store_id for determinism).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


def _grocery_cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "GroceryDeliveryAPIAlt",
        initial_config.get("GroceryDeliveryAPI", {}),
    )


def _compute_ground_truth(cfg: dict) -> dict:
    shopping = cfg["shopping_list"]
    stores = cfg["stores"]
    gift_cards = cfg.get("gift_cards", {})

    # Items each store covers (only in-stock items).
    coverage: dict[str, set[str]] = {}
    for sid, s in stores.items():
        inv = s.get("inventory", {})
        coverage[sid] = {
            it for it, d in inv.items()
            if it in shopping and d.get("in_stock", True)
        }

    shopping_set = set(shopping)

    # Find minimum k such that some k-subset of stores covers all items.
    min_k = None
    for k in range(1, len(stores) + 1):
        for combo in combinations(coverage.keys(), k):
            if set().union(*(coverage[s] for s in combo)) >= shopping_set:
                min_k = k
                break
        if min_k is not None:
            break

    def _assign(combo: tuple[str, ...]) -> dict[str, str]:
        """Assign each item to the lowest-sorted chosen store that carries it."""
        assignment: dict[str, str] = {}
        sorted_combo = sorted(combo)
        for item in shopping:
            for sid in sorted_combo:
                if item in coverage[sid]:
                    assignment[item] = sid
                    break
        return assignment

    def _gift_card_total(combo: tuple[str, ...], assignment: dict[str, str]) -> float:
        total = 0.0
        per_store_bill: dict[str, float] = {}
        for item, sid in assignment.items():
            price = stores[sid]["inventory"][item]["price"]
            per_store_bill[sid] = per_store_bill.get(sid, 0.0) + price
        for sid in combo:
            if sid in per_store_bill:
                bill = per_store_bill[sid] + stores[sid].get("delivery_fee", 0.0)
                total += min(gift_cards.get(sid, 0.0), bill)
        return round(total, 2)

    best_combo: tuple[str, ...] | None = None
    best_total = -1.0
    best_assignment: dict[str, str] = {}
    for combo in combinations(coverage.keys(), min_k):
        if set().union(*(coverage[s] for s in combo)) < shopping_set:
            continue
        assignment = _assign(combo)
        total = _gift_card_total(combo, assignment)
        if total > best_total or (
            total == best_total
            and (best_combo is None or sorted(combo) < sorted(best_combo))
        ):
            best_total = total
            best_combo = combo
            best_assignment = assignment

    assert best_combo is not None
    return {
        "min_k": min_k,
        "stores_selected": sorted(best_combo),
        "assignment": best_assignment,
        "total_gift_card_applied": best_total,
    }


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]

    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0, checks={}, details=details,
            errors=["Could not extract JSON object from final answer"],
        )

    cfg = _grocery_cfg(initial_config)
    expected = _compute_ground_truth(cfg)
    details["expected"] = expected

    try:
        num = int(answer.get("num_stores"))
        checks["num_stores_correct"] = num == expected["min_k"]
    except (TypeError, ValueError):
        checks["num_stores_correct"] = False

    given_stores = answer.get("stores_selected") or []
    if not isinstance(given_stores, list):
        given_stores = []
    checks["stores_selected_correct"] = (
        sorted(given_stores) == expected["stores_selected"]
    )

    assignment = answer.get("assignment") or []
    if not isinstance(assignment, list):
        assignment = []
    assigned_items: set[str] = set()
    assignment_valid = True
    chosen_set = set(expected["stores_selected"])
    for a in assignment:
        if not isinstance(a, dict):
            assignment_valid = False
            continue
        item = a.get("item")
        sid = a.get("store_id")
        if item in assigned_items:
            assignment_valid = False
        assigned_items.add(item)
        if sid not in chosen_set:
            assignment_valid = False
        store = cfg["stores"].get(sid)
        if store is None:
            assignment_valid = False
        else:
            inv = store.get("inventory", {})
            if item not in inv or not inv[item].get("in_stock", True):
                assignment_valid = False
    checks["assignment_covers_all_items"] = assigned_items == set(cfg["shopping_list"])
    checks["assignment_valid_stores"] = assignment_valid

    try:
        got_total = float(answer.get("total_gift_card_applied"))
        checks["gift_card_total_correct"] = (
            abs(got_total - expected["total_gift_card_applied"]) <= 0.01
        )
    except (TypeError, ValueError):
        checks["gift_card_total_correct"] = False

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks, details=details,
    )
