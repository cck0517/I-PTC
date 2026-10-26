"""Verifier for ptc_org_chart_lca_pairs_0.

Checks each of the 5 pairs (in given order) and the single tightest
pair scalar.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("pair") != exp["pair"]:
        return False
    if got.get("lca_manager_id") != exp["lca_manager_id"]:
        return False
    if got.get("lca_manager_name") != exp["lca_manager_name"]:
        return False
    if got.get("lca_depth") != exp["lca_depth"]:
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

    got = answer.get("pairs")
    exp = ground_truth["pairs"]
    checks["has_pairs_key"] = isinstance(got, list)
    if not isinstance(got, list):
        got = []
    checks["pairs_count_correct"] = len(got) == len(exp)
    entries_ok = checks["pairs_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["pairs_entries_correct"] = entries_ok

    checks["tightest_correct"] = _entry_matches(
        answer.get("tightest"), ground_truth["tightest"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
