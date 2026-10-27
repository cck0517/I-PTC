"""Verifier for ptc_yelp_category_best_nearby_0.

Check that the model returned the best-rated business per category for all
businesses within 2 miles. The expected answer is recomputed from the
scenario:

- Category names are compared case-insensitively ("Indian" == "indian").
- When several businesses tie for a category's top rating, any of them counts.
- Every category with businesses tagged "restaurant" must be present. Bakery,
  Cafe, Dessert and Bar businesses are not tagged "restaurant", so those
  categories may be included or left out; if included they must be correct.
- No other categories are allowed.
"""

from __future__ import annotations

import copy
from typing import Any

from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)
from ptc_eval.eval_checker.multi_turn_eval.func_source_code.yelp_api import YelpAPI

_RADIUS_MILES = 2.0


def _norm(cat: Any) -> str:
    return " ".join(str(cat).lower().split())


def _expected(initial_config: dict) -> tuple[dict[str, dict], set[str]]:
    """Return ({category: {"rating": top, "winners": {business_id: name}}},
    set of required categories)."""
    api = YelpAPI()
    api._load_scenario(copy.deepcopy(initial_config["YelpAPI"]))
    groups: dict[str, list[dict]] = {}
    required: set[str] = set()
    for biz in api.businesses.values():
        dist = api.get_distance_to_business(biz["business_id"])["distance_miles"]
        if dist > _RADIUS_MILES:
            continue
        tags = [c for c in biz.get("categories", []) if _norm(c) != "restaurant"]
        if not tags:
            continue
        cat = _norm(tags[0])
        groups.setdefault(cat, []).append(biz)
        if any(_norm(c) == "restaurant" for c in biz.get("categories", [])):
            required.add(cat)
    expected = {}
    for cat, bizs in groups.items():
        top = max(b["rating"] for b in bizs)
        expected[cat] = {
            "rating": top,
            "winners": {b["business_id"]: b["name"] for b in bizs if b["rating"] == top},
        }
    return expected, required


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    try:
        answer = extract_json_from_text(final_answer)
        if answer is None:
            return VerificationResult(
                passed=False, score=0.0,
                errors=["Could not extract JSON from final answer"],
                details={"raw_answer": final_answer},
            )
        if isinstance(answer, list):
            # Try to convert list-of-dicts to dict keyed by category
            converted = {}
            for item in answer:
                cat = item.get("category") if isinstance(item, dict) else None
                if cat:
                    converted[cat] = item
            answer = converted
        if isinstance(answer, dict) and isinstance(answer.get("categories"), dict):
            answer = answer["categories"]
        answer = {_norm(k): v for k, v in answer.items() if isinstance(v, dict)}

        expected, required = _expected(initial_config)
        checks: dict[str, bool] = {}
        errors: list[str] = []

        for cat in sorted(required):
            if cat not in answer:
                checks[f"{cat}_present"] = False
                errors.append(f"Missing category '{cat}'")
        for cat, info in answer.items():
            exp = expected.get(cat)
            if exp is None:
                checks[f"{cat}_known_category"] = False
                errors.append(f"Unexpected category '{cat}'")
                continue
            bid = info.get("business_id")
            ok_biz = bid in exp["winners"]
            checks[f"{cat}_business_id"] = ok_biz
            checks[f"{cat}_rating"] = info.get("rating") == exp["rating"]
            checks[f"{cat}_name"] = ok_biz and info.get("name") == exp["winners"][bid]
            if not ok_biz:
                errors.append(f"'{cat}': {bid} is not a top-rated business "
                              f"(top {exp['rating']}: {sorted(exp['winners'])})")

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks, errors=errors,
            details={"extracted_answer": answer,
                     "required_categories": sorted(required),
                     "optional_categories": sorted(set(expected) - required)},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
