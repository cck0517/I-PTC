"""Verifier for ptc_mt_org_chart_levels_histogram_0.

Q1: levels list with count + avg_direct_reports on 2026-01-15.
Q2: top-heavy level (max avg_direct_reports, tiebreak smaller level number)
    plus members at that level with direct_report_count.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import (
    active_manager_on, chain_lengths, direct_report_counts, name_by_id,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


SNAPSHOT = date(2026, 1, 15)


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    am = active_manager_on(cfg["reporting_history"], SNAPSHOT)
    drc = direct_report_counts(am)
    levels = chain_lengths(am)

    # Include the root(s) — anyone listed as manager_id but not in `am`.
    roots = {mid for _, (_, mid) in am.items() if mid not in am}
    for root in roots:
        levels[root] = 0

    by_level: dict[int, list[str]] = defaultdict(list)
    for eid, lvl in levels.items():
        by_level[lvl].append(eid)

    summary = []
    for lvl in sorted(by_level):
        emps = by_level[lvl]
        avg = round(sum(drc.get(e, 0) for e in emps) / len(emps), 2)
        summary.append({"level": lvl, "count": len(emps), "avg_direct_reports": avg})

    non_root = [s for s in summary if s["level"] >= 1]
    if not non_root:
        return {"top_heavy_level": None, "members": []}
    top_heavy = sorted(non_root, key=lambda x: (-x["avg_direct_reports"], x["level"]))[0]
    members = []
    for eid in by_level[top_heavy["level"]]:
        members.append({
            "employee_id": eid,
            "employee_name": nbi.get(eid, ""),
            "direct_report_count": drc.get(eid, 0),
        })
    members.sort(key=lambda m: (-m["direct_report_count"], m["employee_name"]))

    return {
        "top_heavy_level": top_heavy,
        "members": members,
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

    thl_ans = answer.get("top_heavy_level") or {}
    thl_exp = expected["top_heavy_level"]
    checks: dict[str, bool] = {
        "top_heavy_level_match": (
            isinstance(thl_ans, dict)
            and thl_ans.get("level") == thl_exp["level"]
            and thl_ans.get("count") == thl_exp["count"]
        ),
    }

    mem_ans = answer.get("members") or []
    mem_exp = expected["members"]
    checks["members_count"] = len(mem_ans) == len(mem_exp)
    ans_ids = [m.get("employee_id") for m in mem_ans if isinstance(m, dict)]
    exp_ids = [m["employee_id"] for m in mem_exp]
    checks["members_set"] = set(ans_ids) == set(exp_ids)
    checks["members_order"] = ans_ids == exp_ids

    exp_by_id = {m["employee_id"]: m for m in mem_exp}
    drc_ok = True
    for m in mem_ans:
        if not isinstance(m, dict):
            drc_ok = False; break
        e = exp_by_id.get(m.get("employee_id"))
        if e is None:
            continue
        if m.get("direct_report_count") != e["direct_report_count"]:
            drc_ok = False; break
    checks["direct_report_count_match"] = drc_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
