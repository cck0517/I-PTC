"""Verifier for ptc_org_chart_span_of_control_0.

Checks the ordered top-5 managers (by direct_report_count desc, name asc)
and the 4-bucket histogram.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _entry_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    return (
        got.get("manager_id") == exp["manager_id"]
        and got.get("manager_name") == exp["manager_name"]
        and got.get("direct_report_count") == exp["direct_report_count"]
    )


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    got_top = answer.get("top5")
    exp_top = ground_truth["top5"]
    checks["has_top5_key"] = isinstance(got_top, list)
    if not isinstance(got_top, list):
        got_top = []
    checks["top5_count_correct"] = len(got_top) == len(exp_top)
    entries_ok = checks["top5_count_correct"]
    if entries_ok:
        for g, e in zip(got_top, exp_top):
            if not _entry_matches(g, e):
                entries_ok = False; break
    checks["top5_entries_correct"] = entries_ok

    got_hist = answer.get("histogram")
    exp_hist = ground_truth["histogram"]
    checks["has_histogram_key"] = isinstance(got_hist, dict)
    if not isinstance(got_hist, dict):
        got_hist = {}
    checks["histogram_correct"] = all(
        got_hist.get(k) == exp_hist[k] for k in exp_hist
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
