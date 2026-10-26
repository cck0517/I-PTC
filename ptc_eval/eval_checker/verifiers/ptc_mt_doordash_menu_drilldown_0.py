"""Verifier for ptc_mt_doordash_menu_drilldown_0.

Q2: From restaurants with estimated_delivery_time <= 30 min, list every
    popular menu item priced < $12, grouped by restaurant, sorted by rating
    desc; items within a restaurant sorted by price asc.
"""
from __future__ import annotations

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(config: dict) -> list[dict]:
    cfg = config["DoorDashAPI"]
    out = []
    for rid, rest in cfg["restaurants"].items():
        if rest.get("estimated_delivery_time", 999) > 30:
            continue
        menu = cfg["menus"].get(rid, [])
        items = [
            {"item_id": it["item_id"], "name": it["name"],
             "price": round(float(it["price"]), 2), "category": it["category"]}
            for it in menu
            if it.get("is_popular") and float(it.get("price", 0)) < 12
        ]
        if not items:
            continue
        items.sort(key=lambda x: (x["price"], x["name"]))
        out.append({
            "restaurant_id": rid, "name": rest["name"],
            "rating": rest["rating"], "items": items,
        })
    out.sort(key=lambda r: (-r["rating"], r["name"]))
    return out


def _approx_eq(a, b, tol=0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if isinstance(answer, dict) and isinstance(answer.get("restaurants"), list):
        ans = answer["restaurants"]
    elif isinstance(answer, list):
        ans = answer
    else:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected {restaurants: [...]} or a list at top level"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    ans_ids = [a.get("restaurant_id") for a in ans if isinstance(a, dict)]
    exp_ids = [e["restaurant_id"] for e in expected]
    checks: dict[str, bool] = {
        "restaurant_count": len(ans) == len(expected),
        "restaurant_set":   set(ans_ids) == set(exp_ids),
        "restaurant_order": ans_ids == exp_ids,
    }

    exp_by_rid = {e["restaurant_id"]: e for e in expected}
    items_ok = True
    for a in ans:
        if not isinstance(a, dict):
            items_ok = False; break
        e = exp_by_rid.get(a.get("restaurant_id"))
        if e is None:
            continue
        a_items = a.get("items") or []
        if len(a_items) != len(e["items"]):
            items_ok = False; break
        a_item_ids = {x.get("item_id") for x in a_items if isinstance(x, dict)}
        e_item_ids = {x["item_id"] for x in e["items"]}
        if a_item_ids != e_item_ids:
            items_ok = False; break
        e_item_by_id = {x["item_id"]: x for x in e["items"]}
        for x in a_items:
            exp_item = e_item_by_id.get(x.get("item_id"))
            if not _approx_eq(x.get("price"), exp_item["price"]):
                items_ok = False; break
    checks["items_match"] = items_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": ans},
    )
