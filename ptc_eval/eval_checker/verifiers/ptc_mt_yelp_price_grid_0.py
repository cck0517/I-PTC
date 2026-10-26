"""Verifier for ptc_mt_yelp_price_grid_0 (multi-turn).

Q1 (parity): per-tier {business_count, total_review_count, mean_rating}.
Q2: per-tier top-3 by rating + weighted mean rating.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import businesses, round4
from .core import VerificationResult, extract_json_from_text, score_from_checks

PRICE_TIERS = ("$", "$$", "$$$", "$$$$")


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    by_tier: dict[str, list[dict]] = {t: [] for t in PRICE_TIERS}
    for bid, b in bizs.items():
        tier = b.get("price_level")
        if tier in by_tier:
            by_tier[tier].append({
                "business_id": bid,
                "name": b["name"],
                "rating": b["rating"],
                "review_count": b["review_count"],
            })

    out: dict[str, dict] = {}
    for tier in PRICE_TIERS:
        bs = by_tier[tier]
        if not bs:
            out[tier] = {"top3_by_rating": [], "weighted_mean_rating": 0.0}
            continue
        # rating desc, review_count desc, name asc
        bs_sorted = sorted(bs, key=lambda r: (-r["rating"], -r["review_count"], r["name"]))
        top3 = bs_sorted[:3]
        total_reviews = sum(b["review_count"] for b in bs)
        if total_reviews == 0:
            wmean = 0.0
        else:
            wmean = round4(sum(b["rating"] * b["review_count"] for b in bs) / total_reviews)
        out[tier] = {
            "top3_by_rating": [
                {"business_id": b["business_id"], "name": b["name"],
                 "rating": b["rating"], "review_count": b["review_count"]}
                for b in top3
            ],
            "weighted_mean_rating": wmean,
        }
    return {"by_price_level": out}


def _top3_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    keys = ("business_id", "name", "rating", "review_count")
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
    bpl_got = answer.get("by_price_level") or {}
    bpl_exp = expected["by_price_level"]

    top3_ok = True
    wmean_ok = True
    for tier in PRICE_TIERS:
        g = bpl_got.get(tier) or {}
        e = bpl_exp[tier]
        if not _top3_match(g.get("top3_by_rating"), e["top3_by_rating"]):
            top3_ok = False
        if g.get("weighted_mean_rating") != e["weighted_mean_rating"]:
            wmean_ok = False

    checks = {"top3_per_tier_correct": top3_ok, "weighted_mean_correct": wmean_ok}
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
