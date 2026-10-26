"""Verifier for ptc_mt_grocery_store_comparison_0 (multi-turn).

Q1: Per-store price breakdown for the shopping list: items (name, price,
    in_stock), missing_items, delivery_fee.
Q2: Of stores with ALL items in stock, find cheapest (item_total + delivery)
    and the price delta for each other fully-stocked store.

Expected JSON keys:
    cheapest_store_id, cheapest_store_name, cheapest_total,
    comparison (array of {store_id, store_name, total, difference_from_cheapest}
                sorted by difference asc).
"""

from __future__ import annotations

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


def _store_total(store: dict, shopping_list: list[str]) -> tuple[float, bool]:
    """Return (item_total + delivery_fee, all_in_stock)."""
    inv = store.get("inventory", {})
    all_ok = True
    item_sum = 0.0
    for item in shopping_list:
        row = inv.get(item)
        if not row or not row.get("in_stock", False):
            all_ok = False
        else:
            item_sum += float(row.get("price", 0.0))
    if not all_ok:
        return 0.0, False
    return round(item_sum + float(store.get("delivery_fee", 0.0)), 2), True


def _compute_expected(config: dict) -> dict | None:
    cfg = config["GroceryDeliveryAPI"]
    shopping_list = cfg["shopping_list"]
    fully_stocked: list[dict] = []
    for sid, store in cfg["stores"].items():
        total, ok = _store_total(store, shopping_list)
        if not ok:
            continue
        fully_stocked.append({
            "store_id": sid,
            "store_name": store["name"],
            "total": total,
        })

    if not fully_stocked:
        return None

    cheapest = min(fully_stocked, key=lambda s: s["total"])
    others = [s for s in fully_stocked if s["store_id"] != cheapest["store_id"]]
    comparison = [
        {
            "store_id": s["store_id"],
            "store_name": s["store_name"],
            "total": s["total"],
            "difference_from_cheapest": round(s["total"] - cheapest["total"], 2),
        }
        for s in others
    ]
    comparison.sort(key=lambda c: c["difference_from_cheapest"])

    return {
        "cheapest_store_id": cheapest["store_id"],
        "cheapest_store_name": cheapest["store_name"],
        "cheapest_total": cheapest["total"],
        "comparison": comparison,
    }


def _approx_eq(a, b, tol=0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
    turn_answers: list[str | None] | None = None,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )

    expected = _compute_expected(initial_config)
    if expected is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["No store stocks the full shopping list"],
            details={"expected": None, "answer": answer},
        )

    checks: dict[str, bool] = {
        "cheapest_id_correct":   answer.get("cheapest_store_id")   == expected["cheapest_store_id"],
        "cheapest_name_correct": answer.get("cheapest_store_name") == expected["cheapest_store_name"],
        "cheapest_total_close":  _approx_eq(answer.get("cheapest_total"), expected["cheapest_total"]),
    }

    comp_ans = answer.get("comparison") or []
    checks["comparison_count"] = len(comp_ans) == len(expected["comparison"])

    exp_by_id = {c["store_id"]: c for c in expected["comparison"]}
    diff_ok = True
    for c in comp_ans:
        if not isinstance(c, dict):
            diff_ok = False; break
        e = exp_by_id.get(c.get("store_id"))
        if e is None:
            diff_ok = False; break
        if not _approx_eq(c.get("total"), e["total"]):
            diff_ok = False; break
        if not _approx_eq(c.get("difference_from_cheapest"), e["difference_from_cheapest"]):
            diff_ok = False; break
    checks["comparison_values_match"] = diff_ok

    # Q2 asks: sort comparison by difference ascending.
    diffs = [c.get("difference_from_cheapest", 0) for c in comp_ans]
    checks["comparison_sorted_asc"] = all(
        diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1)
    ) if len(diffs) >= 2 else True

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
