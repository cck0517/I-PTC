"""Verifier for ptc_intermediate_org_shared_managers_0.

Computes per-employee manager sets from reporting_history, finds the
unordered pair with the largest intersection, and checks the agent's
answer.  Tie-break: the pair whose sorted (employee_id, employee_id)
tuple is lexicographically smallest.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


def _org_cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "OrgChartAPIAlt", initial_config.get("OrgChartAPI", {}),
    )


def _compute_ground_truth(cfg: dict) -> dict:
    employees = cfg["employees"]
    history = cfg["reporting_history"]

    mgrs_by_emp: dict[str, set[str]] = {}
    mgr_names: dict[str, str] = {}
    for row in history:
        eid = row["employee_id"]
        mid = row["manager_id"]
        mgrs_by_emp.setdefault(eid, set()).add(mid)
        mgr_names[mid] = row.get("manager_name", "")

    name_by_emp = {e["employee_id"]: e["employee_name"] for e in employees}
    all_ids = sorted(name_by_emp.keys())

    best_count = -1
    best_pair: tuple[str, str] | None = None
    best_shared: list[str] = []
    for a, b in combinations(all_ids, 2):
        shared = mgrs_by_emp.get(a, set()) & mgrs_by_emp.get(b, set())
        cnt = len(shared)
        if cnt > best_count or (
            cnt == best_count and best_pair is not None and (a, b) < best_pair
        ):
            best_count = cnt
            best_pair = (a, b)
            best_shared = sorted(shared)

    assert best_pair is not None
    a, b = best_pair
    return {
        "employee_1": {"id": a, "name": name_by_emp[a]},
        "employee_2": {"id": b, "name": name_by_emp[b]},
        "shared_managers": best_shared,
        "shared_count": best_count,
    }


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]

    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0, checks={}, details=details,
            errors=["Could not extract JSON object from final answer"],
        )

    cfg = _org_cfg(initial_config)
    expected = _compute_ground_truth(cfg)
    details["expected"] = expected

    def _id(obj: Any) -> str | None:
        if isinstance(obj, dict):
            return obj.get("id") or obj.get("employee_id")
        return None

    got_1 = _id(answer.get("employee_1"))
    got_2 = _id(answer.get("employee_2"))
    exp_1 = expected["employee_1"]["id"]
    exp_2 = expected["employee_2"]["id"]

    # Accept either order; the sorted (a, b) must match.
    checks["employee_pair_correct"] = (
        {got_1, got_2} == {exp_1, exp_2} and None not in {got_1, got_2}
    )

    try:
        cnt = int(answer.get("shared_count"))
        checks["shared_count_correct"] = cnt == expected["shared_count"]
    except (TypeError, ValueError):
        checks["shared_count_correct"] = False

    given_shared = answer.get("shared_managers") or []
    if not isinstance(given_shared, list):
        given_shared = []
    checks["shared_managers_correct"] = (
        sorted(str(x) for x in given_shared) == expected["shared_managers"]
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks, details=details,
    )
