"""Verifier for ptc_calendar_self_conflict_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


_PAIR_FIELDS = ("title_a", "start_a", "end_a", "title_b", "start_b", "end_b")


def _pair_key(p: dict) -> tuple:
    """Order-insensitive key: a pair {A,B} matches whether the model labels A first or B first."""
    a = (p.get("title_a"), p.get("start_a"), p.get("end_a"))
    b = (p.get("title_b"), p.get("start_b"), p.get("end_b"))
    return tuple(sorted([a, b]))


def _emp_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("employee_id") != exp["employee_id"]:
        return False
    if got.get("display_name") != exp["display_name"]:
        return False
    if got.get("conflict_count") != exp["conflict_count"]:
        return False
    g_conf = got.get("conflicts")
    if not isinstance(g_conf, list) or len(g_conf) != len(exp["conflicts"]):
        return False
    g_keys = {_pair_key(c) for c in g_conf if isinstance(c, dict)}
    e_keys = {_pair_key(c) for c in exp["conflicts"]}
    return g_keys == e_keys


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    checks["total_pairs_correct"] = (
        answer.get("total_conflict_pairs") == ground_truth["total_conflict_pairs"])

    got_emps = answer.get("employees_with_conflicts")
    exp_emps = ground_truth["employees_with_conflicts"]
    checks["has_employees_key"] = isinstance(got_emps, list)
    if not isinstance(got_emps, list):
        got_emps = []
    checks["employees_length_correct"] = len(got_emps) == len(exp_emps)

    # Order-insensitive over employees: index by employee_id then check each.
    got_by_id = {e.get("employee_id"): e for e in got_emps if isinstance(e, dict)}
    entries_ok = True
    for exp_e in exp_emps:
        g = got_by_id.get(exp_e["employee_id"])
        if g is None or not _emp_matches(g, exp_e):
            entries_ok = False; break
    checks["employees_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
