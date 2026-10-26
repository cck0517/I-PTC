"""Verifier for ptc_mt_yelp_category_pairs_0 (multi-turn).

Q1 (parity): top-5 category pairs by co-occurrence across SF food businesses.
Q2: per Q1 pair, list businesses that have BOTH categories.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

from ._yelp_helpers import SF_FOOD_CATS, businesses, has_categories
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _q1_top5_pairs(initial_config: dict) -> list[list[str]]:
    cfg = initial_config["YelpAPI"]
    counter: Counter = Counter()
    for b in businesses(cfg).values():
        if not has_categories(b, SF_FOOD_CATS):
            continue
        cats = sorted(set(b.get("categories", [])))
        for a, c in combinations(cats, 2):
            counter[(a, c)] += 1
    pairs = [
        {"pair": list(p), "co_occurrence_count": n}
        for p, n in counter.items()
    ]
    pairs.sort(key=lambda r: (-r["co_occurrence_count"], r["pair"]))
    return [p["pair"] for p in pairs[:5]]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    top5 = _q1_top5_pairs(initial_config)

    out = []
    for pair in top5:
        a, c = pair[0], pair[1]
        matches = []
        for bid, b in bizs.items():
            if not has_categories(b, SF_FOOD_CATS):
                continue
            cats = b.get("categories", [])
            if a in cats and c in cats:
                matches.append({"business_id": bid, "name": b["name"]})
        matches.sort(key=lambda r: r["name"])
        out.append({
            "pair": pair,
            "businesses": matches,
            "count": len(matches),
        })
    return {"pair_businesses": out}


def _entry_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if list(got.get("pair") or []) != exp["pair"]:
        return False
    if got.get("count") != exp["count"]:
        return False
    g_b = got.get("businesses")
    if not isinstance(g_b, list) or len(g_b) != len(exp["businesses"]):
        return False
    for g, e in zip(g_b, exp["businesses"]):
        if not isinstance(g, dict):
            return False
        if g.get("business_id") != e["business_id"] or g.get("name") != e["name"]:
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
    got = answer.get("pair_businesses")
    exp = expected["pair_businesses"]
    checks = {"count_match": isinstance(got, list) and len(got) == len(exp)}
    if not checks["count_match"]:
        got = []
    entries_ok = checks["count_match"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_match(g, e):
                entries_ok = False; break
    checks["entries_correct"] = entries_ok
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
