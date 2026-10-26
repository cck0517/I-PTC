"""Verifier for ptc_mt_grocery_gift_card_reopt_0 (multi-turn).

Q1: For each fully-stocked store, out_of_pocket = max(0, items + delivery -
    gift_card), sorted asc. Each row: store_id, store_name, item_total,
    delivery_fee, gift_card_balance, out_of_pocket, items[{item_name, price}].
Q2: "Just received $15 gift card for DailyGrocer (store_3)." Recompute the
    per-store out_of_pocket with the updated gift card for store_3 and
    re-rank asc.  Same JSON shape as Q1.
"""

from __future__ import annotations

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)

DAILYGROCER_ID = "store_3"
ADDED_GIFT_CARD = 15.0


def _compute_expected(config: dict) -> list[dict]:
    cfg = config["GroceryDeliveryAPI"]
    shopping_list = cfg["shopping_list"]
    gift_cards = dict(cfg.get("gift_cards") or {})
    gift_cards[DAILYGROCER_ID] = float(gift_cards.get(DAILYGROCER_ID, 0.0)) + ADDED_GIFT_CARD

    results: list[dict] = []
    for sid, store in cfg["stores"].items():
        inv = store.get("inventory", {})
        all_ok = True
        items = []
        item_total = 0.0
        for it in shopping_list:
            row = inv.get(it)
            if not row or not row.get("in_stock", False):
                all_ok = False
                break
            price = float(row.get("price", 0.0))
            item_total += price
            items.append({"item_name": it, "price": price})
        if not all_ok:
            continue
        delivery_fee = float(store.get("delivery_fee", 0.0))
        gc = float(gift_cards.get(sid, 0.0))
        out = max(0.0, item_total + delivery_fee - gc)
        results.append({
            "store_id": sid,
            "store_name": store["name"],
            "item_total": round(item_total, 2),
            "delivery_fee": round(delivery_fee, 2),
            "gift_card_balance": round(gc, 2),
            "out_of_pocket": round(out, 2),
            "items": items,
        })

    results.sort(key=lambda r: r["out_of_pocket"])
    return results


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
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )
    if isinstance(answer, dict):
        for k in ("stores", "results", "items"):
            if isinstance(answer.get(k), list):
                answer = answer[k]
                break
        else:
            answer = [answer]

    expected = _compute_expected(initial_config)

    checks: dict[str, bool] = {
        "correct_count":  len(answer) == len(expected),
    }
    ans_ids = [a.get("store_id") for a in answer if isinstance(a, dict)]
    exp_ids = [e["store_id"] for e in expected]
    checks["correct_store_set"] = set(ans_ids) == set(exp_ids)
    checks["correct_order"] = ans_ids == exp_ids

    exp_by_id = {e["store_id"]: e for e in expected}
    field_ok = True
    for a in answer:
        if not isinstance(a, dict):
            field_ok = False; break
        e = exp_by_id.get(a.get("store_id"))
        if e is None:
            continue
        if not _approx_eq(a.get("out_of_pocket"),  e["out_of_pocket"]):
            field_ok = False; break
        if not _approx_eq(a.get("item_total"),     e["item_total"]):
            field_ok = False; break
        if not _approx_eq(a.get("delivery_fee"),   e["delivery_fee"]):
            field_ok = False; break
        if not _approx_eq(a.get("gift_card_balance"), e["gift_card_balance"]):
            field_ok = False; break
    checks["fields_match"] = field_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer, "expected_count": len(expected)},
    )
