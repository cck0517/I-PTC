"""Verifier for ptc_grocery_single_vs_split_0.

Polymorphic: handles 2-store split (PTC_queries.json easier config) AND
3-store split (main.jsonl).

Auto-detects K from the agent's answer shape (best_split has store_a/store_b
=> K=2; store_a/store_b/store_c => K=3).  Recomputes optimal at the smallest
K that yields a feasible cover (try K=1, then 2, then 3).  Allows
best_single = null when no single store carries every item.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)

TOLERANCE = 0.50


def _store_cost(store: dict, items: list[str]) -> float | None:
    """Total cost of buying `items` from one store, or None if any unavailable."""
    inv = store.get("inventory", {})
    total = 0.0
    for item in items:
        if item not in inv or not inv[item].get("in_stock", True):
            return None
        total += inv[item]["price"]
    service = round(total * store.get("service_fee_rate", 0), 2)
    return round(total + service + store.get("delivery_fee", 0), 2)


def _best_single(config: dict) -> dict | None:
    shopping = config["shopping_list"]
    best = None
    for sid, s in config["stores"].items():
        cost = _store_cost(s, shopping)
        if cost is None:
            continue
        if best is None or cost < best["grand_total"]:
            best = {"store_id": sid, "store_name": s["name"], "grand_total": cost}
    return best


def _greedy_split(stores_subset: list[tuple[str, dict]], shopping: list[str]
                  ) -> dict | None:
    """For a given combination of K stores, assign each item to whichever of
    the K offers the lowest unit price.  Returns the per-store breakdown +
    combined_total, or None if any item is uncovered or any store ends up
    with zero items (degenerate split)."""
    invs = [(sid, s, s.get("inventory", {})) for sid, s in stores_subset]
    items_per: dict[str, list[str]] = {sid: [] for sid, _ in stores_subset}
    for item in shopping:
        candidates = []
        for sid, s, inv in invs:
            if item in inv and inv[item].get("in_stock", True):
                candidates.append((inv[item]["price"], sid))
        if not candidates:
            return None
        candidates.sort()  # cheapest first
        items_per[candidates[0][1]].append(item)
    # Reject splits where some store gets zero items (degenerate)
    if any(len(v) == 0 for v in items_per.values()):
        return None
    breakdown = {}
    combined = 0.0
    for sid, s in stores_subset:
        c = _store_cost(s, items_per[sid])
        if c is None:
            return None
        item_total = sum(s["inventory"][i]["price"] for i in items_per[sid])
        item_total = round(item_total, 2)
        service = round(item_total * s.get("service_fee_rate", 0), 2)
        breakdown[sid] = {
            "store_id": sid,
            "store_name": s["name"],
            "items": items_per[sid],
            "item_total": item_total,
            "service_fee": service,
            "delivery_fee": s.get("delivery_fee", 0),
            "subtotal": c,
        }
        combined += c
    return {"breakdown": breakdown, "combined_total": round(combined, 2)}


def _best_k_split(config: dict, k: int) -> dict | None:
    shopping = config["shopping_list"]
    stores = list(config["stores"].items())
    best = None
    for combo in combinations(stores, k):
        result = _greedy_split(list(combo), shopping)
        if result is None:
            continue
        if best is None or result["combined_total"] < best["combined_total"]:
            best = result
    return best


def _solve(config: dict) -> dict:
    """Return the optimal best_single + best_split (at smallest feasible K up
    to 3) + cheaper_option."""
    bs = _best_single(config)
    best_split = None
    for k in (2, 3):
        bsp = _best_k_split(config, k)
        if bsp is not None:
            best_split = {"k": k, **bsp}
            break

    single_total = bs["grand_total"] if bs else float("inf")
    split_total = best_split["combined_total"] if best_split else float("inf")
    cheaper = "split" if split_total < single_total else "single"
    return {
        "best_single": bs,
        "best_split": best_split,
        "cheaper_option": cheaper,
    }


def _detect_k(answer_split: Any) -> int | None:
    """Return 2 or 3 based on the agent's best_split shape."""
    if not isinstance(answer_split, dict):
        return None
    has_a = "store_a" in answer_split
    has_b = "store_b" in answer_split
    has_c = "store_c" in answer_split
    if has_a and has_b and has_c:
        return 3
    if has_a and has_b:
        return 2
    return None


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

    try:
        cfg = initial_config.get("GroceryDeliveryAPIAlt", initial_config.get("GroceryDeliveryAPI", {}))
        optimal = _solve(cfg)
        opt_single = optimal["best_single"]
        opt_split = optimal["best_split"]

        checks: dict[str, bool] = {}

        # cheaper_option
        ans_cheaper = answer.get("cheaper_option") if isinstance(answer.get("cheaper_option"), str) else ""
        checks["cheaper_option"] = ans_cheaper.lower() == optimal["cheaper_option"]

        # best_single (allow null)
        ans_single = answer.get("best_single")
        if opt_single is None:
            checks["best_single"] = ans_single is None
        else:
            if isinstance(ans_single, dict):
                gt = ans_single.get("grand_total")
                checks["best_single"] = (
                    isinstance(gt, (int, float))
                    and abs(gt - opt_single["grand_total"]) <= TOLERANCE
                )
            else:
                checks["best_single"] = False

        # best_split: detect K from agent's shape, compare combined_total + coverage
        ans_split = answer.get("best_split") or {}
        if not isinstance(ans_split, dict):
            ans_split = {}
        if opt_split is None:
            # No feasible split exists at K=2 or 3
            checks["best_split_total"] = ans_split == {} or ans_split.get("combined_total") is None
            checks["split_covers_all"] = True  # vacuous
        else:
            ans_total = ans_split.get("combined_total")
            if isinstance(ans_total, (int, float)):
                checks["best_split_total"] = abs(ans_total - opt_split["combined_total"]) <= TOLERANCE
            else:
                checks["best_split_total"] = False
            # Coverage: items across whatever store_x slots the agent provided
            covered: set[str] = set()
            for slot in ("store_a", "store_b", "store_c"):
                s = ans_split.get(slot) or {}
                if isinstance(s, dict):
                    for it in s.get("items", []) or []:
                        covered.add(it)
            checks["split_covers_all"] = covered >= set(cfg["shopping_list"])

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={
                "extracted_answer": answer,
                "optimal_summary": {
                    "best_single_total": opt_single["grand_total"] if opt_single else None,
                    "best_split_k": opt_split.get("k") if opt_split else None,
                    "best_split_total": opt_split["combined_total"] if opt_split else None,
                    "cheaper_option": optimal["cheaper_option"],
                },
            },
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(passed=False, score=0.0,
                                  errors=[f"Verifier error: {e}"])
