"""Verifier for ptc_org_chart_span_deltas_0.

Checks growth_top3 (delta desc, name asc) and shrinkage_top3
(delta asc, name asc).
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k in ("manager_id", "manager_name", "count_d1", "count_d2", "delta"):
        if got.get(k) != exp[k]:
            return False
    return True


def _list_match(got: Any, exp: list) -> bool:
    if not isinstance(got, list) or len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if not _entry_matches(g, e):
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

    checks["has_growth_key"] = isinstance(answer.get("growth_top3"), list)
    checks["has_shrinkage_key"] = isinstance(answer.get("shrinkage_top3"), list)
    checks["growth_correct"] = _list_match(
        answer.get("growth_top3"), ground_truth["growth_top3"]
    )
    checks["shrinkage_correct"] = _list_match(
        answer.get("shrinkage_top3"), ground_truth["shrinkage_top3"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
