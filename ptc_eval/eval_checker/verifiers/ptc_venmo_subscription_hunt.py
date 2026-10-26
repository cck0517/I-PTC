"""Verifier for ptc_venmo_subscription_hunt_0.

The model returns a list of recurring-payment groups. Each group must
match the expected (recipient, amount, charge_count, total_paid,
first_date, last_date) with small float tolerances, and the list must be
in the specified sort order (total_paid desc, recipient asc).
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_AMT_TOL = 0.01


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("recipient") != exp["recipient"]:
        return False
    for k in ("first_date", "last_date"):
        if got.get(k) != exp[k]:
            return False
    if got.get("charge_count") != exp["charge_count"]:
        return False
    for k in ("amount", "total_paid"):
        a = _num(got.get(k))
        if a is None or abs(a - exp[k]) > _AMT_TOL:
            return False
    return True


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got = answer.get("subscriptions")
    exp = ground_truth.get("subscriptions", [])

    checks["has_subscriptions_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []

    checks["count_correct"] = len(got) == len(exp)

    order_ok = checks["count_correct"]
    if order_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                order_ok = False
                break
    checks["entries_and_order_correct"] = order_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
