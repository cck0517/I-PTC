"""Verifier for ptc_mt_yelp_review_threshold_0 (multi-turn).

Q1 (parity): largest T from {5,10,15,20,25,30,40,50} where >=5 businesses qualify.
Q2: full threshold table + same search but for >=10 businesses.
"""
from __future__ import annotations

from typing import Any

from ._yelp_helpers import businesses
from .core import VerificationResult, extract_json_from_text, score_from_checks

THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["YelpAPI"]
    counts = [b["review_count"] for b in businesses(cfg).values()]

    table = []
    for t in THRESHOLDS:
        n = sum(1 for c in counts if c >= t)
        table.append({"threshold": t, "qualifying_count": n})

    t10 = -1
    n_at_t10 = 0
    for entry in table:
        if entry["qualifying_count"] >= 10 and entry["threshold"] > t10:
            t10 = entry["threshold"]
            n_at_t10 = entry["qualifying_count"]
    return {
        "threshold_table": table,
        "threshold_T_at_least_10": t10,
        "qualifying_count_at_T": n_at_t10,
    }


def _table_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not isinstance(g, dict):
            return False
        if g.get("threshold") != e["threshold"]:
            return False
        if g.get("qualifying_count") != e["qualifying_count"]:
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
        "table_correct": _table_match(answer.get("threshold_table"),
                                      expected["threshold_table"]),
        "T_at_least_10_correct":
            answer.get("threshold_T_at_least_10") == expected["threshold_T_at_least_10"],
        "count_at_T_correct":
            answer.get("qualifying_count_at_T") == expected["qualifying_count_at_T"],
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
