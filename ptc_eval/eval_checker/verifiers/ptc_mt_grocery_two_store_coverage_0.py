"""Verifier for ptc_mt_grocery_two_store_coverage_0.

Q2: Best 2-store pair covering the most shopping-list items (tie-break:
    lowest total cost = item-prices + both delivery fees).
"""
from __future__ import annotations

import itertools

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(config: dict) -> dict:
    cfg = config["GroceryDeliveryAPI"]
    shopping_list = cfg["shopping_list"]
    stores = cfg["stores"]
    store_ids = sorted(stores.keys())

    def pair_score(a: str, b: str) -> tuple[int, float, list[dict], list[str]]:
        inv_a = stores[a]["inventory"]
        inv_b = stores[b]["inventory"]
        covered = []
        uncovered = []
        cost = float(stores[a]["delivery_fee"]) + float(stores[b]["delivery_fee"])
        for item in shopping_list:
            ra = inv_a.get(item)
            rb = inv_b.get(item)
            pa = float(ra["price"]) if ra and ra.get("in_stock") else None
            pb = float(rb["price"]) if rb and rb.get("in_stock") else None
            if pa is None and pb is None:
                uncovered.append(item)
                continue
            if pa is not None and (pb is None or pa <= pb):
                covered.append({"item_name": item, "store_id": a, "price": round(pa, 2)})
                cost += pa
            else:
                covered.append({"item_name": item, "store_id": b, "price": round(pb, 2)})
                cost += pb
        return len(covered), round(cost, 2), covered, uncovered

    best = None
    for a, b in itertools.combinations(store_ids, 2):
        cov_n, cost, cov, unc = pair_score(a, b)
        key = (-cov_n, cost, a, b)
        if best is None or key < best[0]:
            best = (key, (a, b, cov_n, cost, cov, unc))
    _, (a, b, cov_n, cost, cov, unc) = best
    return {
        "pair": [a, b],
        "items_covered": cov_n,
        "assignments": cov,
        "total_cost": cost,
        "items_not_covered": unc,
    }


def _approx_eq(a, b, tol=0.05) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    ans_pair = sorted(answer.get("pair") or [])
    exp_pair = sorted(expected["pair"])
    checks: dict[str, bool] = {
        "pair_correct":         ans_pair == exp_pair,
        "items_covered_correct": answer.get("items_covered") == expected["items_covered"],
        "total_cost_close":     _approx_eq(answer.get("total_cost"), expected["total_cost"], tol=0.1),
        "uncovered_set":        set(answer.get("items_not_covered") or []) == set(expected["items_not_covered"]),
    }
    # Check assignment mapping item_name -> store_id matches
    ans_assign = answer.get("assignments") or []
    ans_map = {a.get("item_name"): a.get("store_id") for a in ans_assign if isinstance(a, dict)}
    exp_map = {a["item_name"]: a["store_id"] for a in expected["assignments"]}
    checks["assignment_mapping"] = ans_map == exp_map

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
