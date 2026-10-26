"""Verifier for ptc_mt_org_chart_span_deltas_0.

Q1: growth_top3 + shrinkage_top3 between 2025-07-01 and 2026-01-01.
Q2: net_change (sum of all 6 deltas) + most_volatile (largest |delta|).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import (
    active_manager_on, direct_report_counts, name_by_id,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


D1 = date(2025, 7, 1)
D2 = date(2026, 1, 1)


def _q1_top3s(cfg: dict) -> tuple[list[dict], list[dict]]:
    history = cfg["reporting_history"]
    nbi = name_by_id(cfg)
    am1 = active_manager_on(history, D1)
    am2 = active_manager_on(history, D2)
    c1 = direct_report_counts(am1)
    c2 = direct_report_counts(am2)

    deltas = []
    for mid in set(c1) & set(c2):
        if c1[mid] >= 1 and c2[mid] >= 1:
            deltas.append({
                "manager_id": mid,
                "manager_name": nbi.get(mid, ""),
                "count_d1": c1[mid],
                "count_d2": c2[mid],
                "delta": c2[mid] - c1[mid],
            })
    growth = sorted([d for d in deltas if d["delta"] > 0],
                    key=lambda d: (-d["delta"], d["manager_name"]))[:3]
    shrink = sorted([d for d in deltas if d["delta"] < 0],
                    key=lambda d: (d["delta"], d["manager_name"]))[:3]
    return growth, shrink


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    growth, shrink = _q1_top3s(cfg)
    six = list(growth) + list(shrink)
    if not six:
        return {"net_change": 0, "most_volatile": None}
    net = sum(d["delta"] for d in six)
    most_volatile = sorted(six, key=lambda d: (-abs(d["delta"]), d["manager_name"]))[0]
    return {
        "net_change": net,
        "most_volatile": {
            "manager_id": most_volatile["manager_id"],
            "manager_name": most_volatile["manager_name"],
            "count_d1": most_volatile["count_d1"],
            "count_d2": most_volatile["count_d2"],
            "delta": most_volatile["delta"],
        },
    }


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

    checks: dict[str, bool] = {
        "net_change_match": answer.get("net_change") == expected["net_change"],
    }
    mv_ans = answer.get("most_volatile")
    mv_exp = expected["most_volatile"]
    if mv_exp is None:
        checks["most_volatile_match"] = mv_ans is None
    else:
        checks["most_volatile_match"] = (
            isinstance(mv_ans, dict)
            and mv_ans.get("manager_id") == mv_exp["manager_id"]
            and mv_ans.get("delta") == mv_exp["delta"]
            and mv_ans.get("count_d1") == mv_exp["count_d1"]
            and mv_ans.get("count_d2") == mv_exp["count_d2"]
        )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
