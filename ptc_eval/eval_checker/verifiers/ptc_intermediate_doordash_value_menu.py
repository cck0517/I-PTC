"""Verifier for ptc_intermediate_doordash_value_menu_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


CUISINES = ["pizza", "sushi", "thai", "mexican"]


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "DoorDashAPIAlt", initial_config.get("DoorDashAPI", {}),
    )


def _compute_ground_truth(cfg: dict) -> dict:
    restaurants = cfg["restaurants"]
    menus = cfg["menus"]
    candidates = []
    for rid, r in restaurants.items():
        rating = float(r["rating"])
        for item in menus.get(rid, []):
            if not item.get("is_popular"):
                continue
            price = float(item["price"])
            if price >= 12.0:
                continue
            value = round(rating / price, 4)
            candidates.append({
                "item_id": item["item_id"],
                "restaurant_id": rid,
                "price": round(price, 2),
                "rating": round(rating, 2),
                "value": value,
            })
    candidates.sort(key=lambda c: (-c["value"], c["price"], c["item_id"]))
    return {
        "top_5": candidates[:5],
        "total_candidates": len(candidates),
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected"] = expected

    try:
        checks["total_candidates_correct"] = (
            int(answer.get("total_candidates")) == expected["total_candidates"]
        )
    except (TypeError, ValueError):
        checks["total_candidates_correct"] = False

    got = answer.get("top_5") or []
    if not isinstance(got, list):
        got = []
    got_ids = [str(x.get("item_id")) for x in got if isinstance(x, dict)]
    checks["top_5_order_correct"] = got_ids == [x["item_id"] for x in expected["top_5"]]

    nums_ok = bool(got)
    for g, e in zip(got, expected["top_5"]):
        if not isinstance(g, dict):
            nums_ok = False
            break
        try:
            if abs(float(g.get("value")) - e["value"]) > 1e-3:
                nums_ok = False
            if abs(float(g.get("price")) - e["price"]) > 0.01:
                nums_ok = False
            if abs(float(g.get("rating")) - e["rating"]) > 0.02:
                nums_ok = False
        except (TypeError, ValueError):
            nums_ok = False
    checks["top_5_fields_correct"] = nums_ok
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
