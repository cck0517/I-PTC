"""Verifier for ptc_venmo_reciprocity_audit_0.

Checks:
- ``out_of_balance``: list of friends with sent_to >= 80 and received_from
  < 30% of sent_to, sorted by sent_to descending. Each entry must match
  the expected field-by-field with float tolerance.
- ``most_reciprocal_friend``: single object with the friend that maximises
  ``min(in,out)/max(in,out)``.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_AMT_TOL = 0.01
_RATIO_TOL = 0.01


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("friend") != exp["friend"]:
        return False
    for k, tol in (("sent_to", _AMT_TOL),
                   ("received_from", _AMT_TOL),
                   ("reciprocity_ratio", _RATIO_TOL)):
        a = _num(got.get(k))
        if a is None or abs(a - exp[k]) > tol:
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

    got_oob = answer.get("out_of_balance")
    exp_oob = ground_truth.get("out_of_balance", [])
    checks["has_out_of_balance_key"] = isinstance(got_oob, list)
    if not isinstance(got_oob, list):
        got_oob = []
    checks["out_of_balance_count_correct"] = len(got_oob) == len(exp_oob)

    order_ok = checks["out_of_balance_count_correct"]
    if order_ok:
        for g, e in zip(got_oob, exp_oob):
            if not _entry_matches(g, e):
                order_ok = False
                break
    checks["out_of_balance_entries_correct"] = order_ok

    got_mrf = answer.get("most_reciprocal_friend")
    exp_mrf = ground_truth.get("most_reciprocal_friend", {})
    checks["most_reciprocal_correct"] = _entry_matches(got_mrf, exp_mrf)

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
