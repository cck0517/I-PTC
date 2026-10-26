"""Verifier for ptc_mt_grocery_set_cover_0 (multi-turn).

Q1 (parity): minimum-size set of stores covering the shopping list.
Q2: rarest_items, most_common_items, bottleneck_items (carried by exactly 1).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ._grocery_helpers import shopping_list, stores
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["GroceryDeliveryAPI"]
    shopping = list(shopping_list(cfg))
    sts = stores(cfg)

    # carriers[item] = list of (store_id, store_name)
    carriers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sid, s in sts.items():
        inv = s.get("inventory", {})
        for item in shopping:
            if item in inv and inv[item].get("in_stock", True):
                carriers[item].append((sid, s.get("name", sid)))

    rarest_pool = [(item, len(carriers[item])) for item in set(shopping)]
    rarest = sorted(rarest_pool, key=lambda r: (r[1], r[0]))[:5]
    most_common = sorted(rarest_pool, key=lambda r: (-r[1], r[0]))[:5]

    bottleneck = []
    for item in sorted(set(shopping)):
        cs = carriers.get(item, [])
        if len(cs) == 1:
            sid, sname = cs[0]
            bottleneck.append({"item_name": item, "store_id": sid, "store_name": sname})

    return {
        "rarest_items": [{"item_name": i, "store_count": n} for i, n in rarest],
        "most_common_items": [{"item_name": i, "store_count": n} for i, n in most_common],
        "bottleneck_items": bottleneck,
    }


def _list_match(got: Any, exp: list, keys: tuple[str, ...]) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        for k in keys:
            if g.get(k) != e[k]:
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
        "rarest_correct": _list_match(answer.get("rarest_items"),
                                      expected["rarest_items"], ("item_name", "store_count")),
        "most_common_correct": _list_match(answer.get("most_common_items"),
                                           expected["most_common_items"], ("item_name", "store_count")),
        "bottleneck_correct": _list_match(answer.get("bottleneck_items"),
                                          expected["bottleneck_items"],
                                          ("item_name", "store_id", "store_name")),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
