"""Verifier for ptc_mt_grocery_gift_card_0 (multi-turn).

Q1 (parity): minimize total out-of-pocket given gift card balances.
Q2: per-item alternatives across all stores + 5 rarest items.
"""
from __future__ import annotations

from typing import Any

from ._grocery_helpers import item_to_carriers, shopping_list, stores
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["GroceryDeliveryAPI"]
    shopping = shopping_list(cfg)
    carriers = item_to_carriers(cfg)

    item_options = []
    for item in shopping:
        opts = carriers.get(item, [])
        item_options.append({
            "item_name": item,
            "options": [
                {"store_id": sid, "store_name": sname, "unit_price": round(price, 2)}
                for sid, sname, price in opts
            ],
        })

    counts = [(item, len(carriers.get(item, []))) for item in set(shopping)]
    counts.sort(key=lambda r: (r[1], r[0]))
    rarest = [{"item_name": it, "store_count": n} for it, n in counts[:5]]
    return {"item_options": item_options, "rarest_items": rarest}


def _options_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if g.get("store_id") != e["store_id"]:
            return False
        if g.get("store_name") != e["store_name"]:
            return False
        gp = g.get("unit_price")
        if not isinstance(gp, (int, float)) or abs(gp - e["unit_price"]) > 0.005:
            return False
    return True


def _item_options_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if g.get("item_name") != e["item_name"]:
            return False
        if not _options_match(g.get("options"), e["options"]):
            return False
    return True


def _rarest_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if g.get("item_name") != e["item_name"] or g.get("store_count") != e["store_count"]:
            return False
    return True


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    checks = {
        "item_options_correct": _item_options_match(answer.get("item_options"),
                                                    expected["item_options"]),
        "rarest_items_correct": _rarest_match(answer.get("rarest_items"),
                                              expected["rarest_items"]),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected_summary": {
            "item_count": len(expected["item_options"]),
            "rarest_first": expected["rarest_items"][0] if expected["rarest_items"] else None,
        }, "answer": answer},
    )
