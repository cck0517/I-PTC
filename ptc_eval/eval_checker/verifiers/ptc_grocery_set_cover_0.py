"""Verifier for ptc_grocery_set_cover_0.

The model must find the minimum-size set of stores whose inventories
cover every item on the shopping list, breaking ties by lowest total
delivery fee.  We recompute the optimal solution from initial_config
and compare structurally.
"""

from __future__ import annotations

import itertools
from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


def _solve(config: dict) -> dict:
    """Brute-force optimal set cover from the scenario config."""
    shopping = set(config["shopping_list"])
    stores = config["stores"]

    store_items = {}
    for sid, s in stores.items():
        store_items[sid] = {
            item for item, d in s["inventory"].items() if d.get("in_stock", True)
        }

    best = None
    for r in range(1, len(stores) + 1):
        for combo in itertools.combinations(stores.keys(), r):
            covered = set()
            for sid in combo:
                covered |= store_items[sid]
            if covered >= shopping:
                cost = round(sum(stores[sid]["delivery_fee"] for sid in combo), 2)
                if best is None or r < best[0] or (r == best[0] and cost < best[1]):
                    best = (r, cost, combo)
        if best and best[0] == r:
            break

    return {"num_stores": best[0], "total_delivery_fee": best[1], "store_ids": set(best[2])}


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
    optimal = _solve(cfg)

    checks: dict[str, bool] = {}

    # Check num_stores
    checks["num_stores"] = answer.get("num_stores") == optimal["num_stores"]

    # Check total delivery fee (tolerance $0.10)
    ans_fee = answer.get("total_delivery_fee")
    if isinstance(ans_fee, (int, float)):
        checks["total_delivery_fee"] = abs(ans_fee - optimal["total_delivery_fee"]) <= 0.10
    else:
        checks["total_delivery_fee"] = False

    # Check that the chosen stores actually cover all items
    shopping = set(cfg["shopping_list"])
    ans_stores = answer.get("stores", [])
    ans_store_ids = set()
    covered_items = set()
    for s in ans_stores:
        sid = s.get("store_id", "")
        ans_store_ids.add(sid)
        for item in s.get("items", []):
            covered_items.add(item)
    checks["covers_all_items"] = covered_items >= shopping

    # Check that the store set matches an optimal solution
    checks["optimal_store_set"] = ans_store_ids == optimal["store_ids"]

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"extracted_answer": answer, "optimal": {
            "num_stores": optimal["num_stores"],
            "total_delivery_fee": optimal["total_delivery_fee"],
            "store_ids": sorted(optimal["store_ids"]),
        }},
    )
