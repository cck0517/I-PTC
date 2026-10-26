"""Verifier for ptc_mt_org_chart_deepest_chain_0.

Q1: deepest single employee on 2026-01-15 (chain to root).
Q2: top-5 deepest employees with their root-first chain_by_name.
"""
from __future__ import annotations

from datetime import date

from ._orgchart_helpers import (
    active_manager_on, chain_to_root, name_by_id,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


SNAPSHOT = date(2026, 1, 15)


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    am = active_manager_on(cfg["reporting_history"], SNAPSHOT)

    candidates = []
    for eid in am:
        ids = chain_to_root(eid, am)
        candidates.append({
            "employee_id": eid,
            "employee_name": nbi.get(eid, ""),
            "chain_length": len(ids) - 1,
            "chain_by_name": [nbi.get(i, "") for i in ids],
        })
    candidates.sort(key=lambda c: (-c["chain_length"], c["employee_name"]))
    return {"top5": candidates[:5]}


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    top5 = answer.get("top5") or []
    exp = expected["top5"]

    checks: dict[str, bool] = {
        "top5_count": len(top5) == len(exp),
    }
    ans_ids = [t.get("employee_id") for t in top5 if isinstance(t, dict)]
    exp_ids = [t["employee_id"] for t in exp]
    checks["employee_set"] = set(ans_ids) == set(exp_ids)
    checks["employee_order"] = ans_ids == exp_ids

    exp_by_id = {t["employee_id"]: t for t in exp}
    chain_ok = True
    length_ok = True
    for t in top5:
        if not isinstance(t, dict):
            chain_ok = length_ok = False; break
        e = exp_by_id.get(t.get("employee_id"))
        if e is None:
            continue
        if t.get("chain_length") != e["chain_length"]:
            length_ok = False
        if t.get("chain_by_name") != e["chain_by_name"]:
            chain_ok = False
    checks["chain_length_match"] = length_ok
    checks["chain_by_name_match"] = chain_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
