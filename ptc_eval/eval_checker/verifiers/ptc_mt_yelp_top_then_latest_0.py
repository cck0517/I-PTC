"""Verifier for ptc_mt_yelp_top_then_latest_0 (multi-turn).

Q1 (parity): top-3 Italian businesses by business.review_count + their latest
review.
Q2: same top-3, plus oldest review and highest-useful_count review per business.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import businesses, reviews_by_business
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _snippet(text: str | None) -> str:
    return (text or "")[:80]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    bizs = businesses(cfg)
    rbb = reviews_by_business(cfg)

    italians = [
        {"business_id": bid, "name": b["name"], "review_count": b["review_count"]}
        for bid, b in bizs.items()
        if "Italian" in b.get("categories", [])
    ]
    italians.sort(key=lambda r: (-r["review_count"], r["name"]))
    top3 = italians[:3]

    out = []
    for biz in top3:
        bid = biz["business_id"]
        revs = rbb.get(bid, [])
        if not revs:
            oldest = highest = None
        else:
            o = min(revs, key=lambda r: (r["created_at"], r["review_id"]))
            # highest useful: max useful_count, then created_at desc, then review_id asc
            h = revs[0]
            for r in revs[1:]:
                if r["useful_count"] > h["useful_count"]:
                    h = r
                elif r["useful_count"] == h["useful_count"]:
                    if r["created_at"] > h["created_at"]:
                        h = r
                    elif r["created_at"] == h["created_at"] and r["review_id"] < h["review_id"]:
                        h = r
            oldest = {
                "review_id": o["review_id"],
                "rating": o["rating"],
                "user_id": o["user_id"],
                "useful_count": o["useful_count"],
                "created_at": o["created_at"],
                "text_snippet": _snippet(o.get("text")),
            }
            highest = {
                "review_id": h["review_id"],
                "rating": h["rating"],
                "user_id": h["user_id"],
                "useful_count": h["useful_count"],
                "created_at": h["created_at"],
                "text_snippet": _snippet(h.get("text")),
            }
        out.append({
            "business_id": bid,
            "name": biz["name"],
            "review_count": biz["review_count"],
            "oldest_review": oldest,
            "highest_useful_review": highest,
        })
    return {"top3": out}


def _review_match(got: Any, exp: dict | None) -> bool:
    if exp is None:
        return got is None
    if not isinstance(got, dict):
        return False
    for k in ("review_id", "rating", "user_id", "useful_count", "created_at", "text_snippet"):
        if got.get(k) != exp[k]:
            return False
    return True


def _entry_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("business_id", "name", "review_count"):
        if got.get(k) != exp[k]:
            return False
    return _review_match(got.get("oldest_review"), exp["oldest_review"]) \
        and _review_match(got.get("highest_useful_review"), exp["highest_useful_review"])


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
    got = answer.get("top3")
    exp = expected["top3"]
    checks = {"top3_count_correct": isinstance(got, list) and len(got) == len(exp)}
    if not checks["top3_count_correct"]:
        got = []
    entries_ok = checks["top3_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_match(g, e):
                entries_ok = False; break
    checks["top3_entries_correct"] = entries_ok
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
