"""Verifier for ptc_venmo_monthly_budget_breach_0.

Checks the ordered ``breaches`` list (over_by desc, friend asc, month
asc), ``worst_breach`` scalar record, and ``total_over_budget`` scalar.
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


def _entry_matches(got: Any, exp: dict | None) -> bool:
    if exp is None:
        return got is None
    if not isinstance(got, dict):
        return False
    if got.get("friend") != exp["friend"] or got.get("month") != exp["month"]:
        return False
    for k in ("sent", "over_by"):
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

    got_b = answer.get("breaches")
    exp_b = ground_truth.get("breaches", [])
    checks["has_breaches_key"] = isinstance(got_b, list)
    if not isinstance(got_b, list):
        got_b = []
    checks["breaches_count_correct"] = len(got_b) == len(exp_b)

    order_ok = checks["breaches_count_correct"]
    if order_ok:
        for g, e in zip(got_b, exp_b):
            if not _entry_matches(g, e):
                order_ok = False
                break
    checks["breaches_entries_correct"] = order_ok

    got_worst = answer.get("worst_breach")
    exp_worst = ground_truth.get("worst_breach")
    checks["worst_breach_correct"] = _entry_matches(got_worst, exp_worst)

    got_tot = _num(answer.get("total_over_budget"))
    exp_tot = ground_truth.get("total_over_budget")
    checks["total_over_budget_correct"] = (
        got_tot is not None and abs(got_tot - exp_tot) <= _AMT_TOL
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
