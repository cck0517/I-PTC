"""Verifier for ptc_org_chart_deepest_chain_0.

Checks the single deepest employee's id/name, chain_length, root-first
chain_by_name, and the scalar max_depth.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    exp_deep = ground_truth["deepest"]
    got_deep = answer.get("deepest")

    checks["has_deepest_key"] = isinstance(got_deep, dict)
    if not isinstance(got_deep, dict):
        got_deep = {}

    checks["employee_id_correct"] = got_deep.get("employee_id") == exp_deep["employee_id"]
    checks["employee_name_correct"] = got_deep.get("employee_name") == exp_deep["employee_name"]
    checks["chain_length_correct"] = got_deep.get("chain_length") == exp_deep["chain_length"]
    checks["chain_by_name_correct"] = got_deep.get("chain_by_name") == exp_deep["chain_by_name"]
    checks["max_depth_correct"] = answer.get("max_depth") == ground_truth["max_depth"]

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
