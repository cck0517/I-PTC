"""Verifier for ptc_mt_org_chart_reorg_churn_0.

Q1: top-5 employees by distinct manager_ids in [2025-01-01, 2025-12-31].
Q2: among those 5, the most common landing-spot manager_name(s)
    + total distinct managers across the 5 lists combined.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import name_by_id, parse_date
from .core import VerificationResult, extract_json_from_text, score_from_checks


WIN_LO = date(2025, 1, 1)
WIN_HI = date(2025, 12, 31)


def _q1_top5(cfg: dict) -> list[dict]:
    history = cfg["reporting_history"]
    nbi = name_by_id(cfg)
    per_emp: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in history:
        s = parse_date(r["start_date"])
        e = parse_date(r["end_date"])
        if s <= WIN_HI and e >= WIN_LO:
            per_emp[r["employee_id"]].add((r["manager_id"], r["manager_name"]))

    candidates = []
    for eid, mgrs in per_emp.items():
        candidates.append({
            "employee_id": eid,
            "employee_name": nbi.get(eid, ""),
            "distinct_manager_count": len(mgrs),
            "manager_names": sorted(m[1] for m in mgrs),
        })
    candidates.sort(key=lambda c: (-c["distinct_manager_count"], c["employee_name"]))
    return candidates[:5]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    top5 = _q1_top5(cfg)
    freq: dict[str, int] = defaultdict(int)
    distinct = set()
    for c in top5:
        for nm in c["manager_names"]:
            freq[nm] += 1
            distinct.add(nm)
    if not freq:
        return {"common_landing_spots": [], "total_distinct_managers": 0}
    max_f = max(freq.values())
    common = sorted(
        [{"manager_name": nm, "frequency": f} for nm, f in freq.items() if f == max_f],
        key=lambda x: x["manager_name"],
    )
    return {
        "common_landing_spots": common,
        "total_distinct_managers": len(distinct),
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

    cls_ans = answer.get("common_landing_spots") or []
    cls_exp = expected["common_landing_spots"]
    checks: dict[str, bool] = {
        "landing_count": len(cls_ans) == len(cls_exp),
        "total_distinct_managers": (
            answer.get("total_distinct_managers") == expected["total_distinct_managers"]
        ),
    }
    ans_names = [c.get("manager_name") for c in cls_ans if isinstance(c, dict)]
    exp_names = [c["manager_name"] for c in cls_exp]
    checks["landing_names_set"] = set(ans_names) == set(exp_names)
    checks["landing_names_order"] = ans_names == exp_names

    exp_by_name = {c["manager_name"]: c for c in cls_exp}
    freq_ok = True
    for c in cls_ans:
        if not isinstance(c, dict):
            freq_ok = False; break
        e = exp_by_name.get(c.get("manager_name"))
        if e is None:
            continue
        if c.get("frequency") != e["frequency"]:
            freq_ok = False; break
    checks["landing_frequency_match"] = freq_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
