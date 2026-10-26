"""Verifier for ptc_venmo_beach_house_reconciliation_0.

Checks the ``per_person`` list (ordered by still_owes desc, name asc) plus
the ``total_collected`` and ``total_still_owed`` scalars.
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
    if got.get("name") != exp["name"]:
        return False
    for k in ("paid_so_far", "still_owes"):
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

    got_pp = answer.get("per_person")
    exp_pp = ground_truth.get("per_person", [])
    checks["has_per_person_key"] = isinstance(got_pp, list)
    if not isinstance(got_pp, list):
        got_pp = []
    checks["per_person_count_correct"] = len(got_pp) == len(exp_pp)

    order_ok = checks["per_person_count_correct"]
    if order_ok:
        for g, e in zip(got_pp, exp_pp):
            if not _entry_matches(g, e):
                order_ok = False
                break
    checks["per_person_entries_correct"] = order_ok

    got_tc = _num(answer.get("total_collected"))
    exp_tc = ground_truth.get("total_collected")
    checks["total_collected_correct"] = (
        got_tc is not None and abs(got_tc - exp_tc) <= _AMT_TOL
    )

    got_to = _num(answer.get("total_still_owed"))
    exp_to = ground_truth.get("total_still_owed")
    checks["total_still_owed_correct"] = (
        got_to is not None and abs(got_to - exp_to) <= _AMT_TOL
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
