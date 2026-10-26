"""Verifier for ptc_venmo_core_financial_circle_0.

Checks each of the four quarterly top-5 lists (strict order: total_flow
desc, |net| desc, name asc) and the ``core_friends`` intersection.
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
    if got.get("friend") != exp["friend"]:
        return False
    for k in ("total_flow", "net"):
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

    got_top5 = answer.get("top5_by_quarter")
    exp_top5 = ground_truth.get("top5_by_quarter", {})
    checks["has_top5_key"] = isinstance(got_top5, dict)
    if not isinstance(got_top5, dict):
        got_top5 = {}

    for qk in ("Q1", "Q2", "Q3", "Q4"):
        got_q = got_top5.get(qk) if isinstance(got_top5.get(qk), list) else []
        exp_q = exp_top5.get(qk, [])
        ok = len(got_q) == len(exp_q)
        if ok:
            for g, e in zip(got_q, exp_q):
                if not _entry_matches(g, e):
                    ok = False
                    break
        checks[f"{qk.lower()}_top5_correct"] = ok

    got_core = answer.get("core_friends")
    exp_core = ground_truth.get("core_friends", [])
    if not isinstance(got_core, list):
        got_core = []
    checks["core_friends_correct"] = got_core == exp_core

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
