"""Verifier for ptc_mt_org_chart_matrix_reports_0.

Q1: per-employee overlapping pairs of reporting rows with different managers.
Q2: top-3 manager pairs by distinct employee count across the matrix-reports list.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import (
    OPEN_END, name_by_id, parse_date,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


CAP_END = date(2026, 3, 31)


def _q1_matrix(cfg: dict) -> list[dict]:
    history = cfg["reporting_history"]
    nbi = name_by_id(cfg)
    by_emp: dict[str, list[dict]] = defaultdict(list)
    for r in history:
        by_emp[r["employee_id"]].append(r)

    out = []
    for eid, rows in by_emp.items():
        intervals = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                ri, rj = rows[i], rows[j]
                if ri["manager_id"] == rj["manager_id"]:
                    continue
                si = parse_date(ri["start_date"])
                ei = parse_date(ri["end_date"])
                sj = parse_date(rj["start_date"])
                ej = parse_date(rj["end_date"])
                ov_s = max(si, sj)
                ov_e = min(ei, ej)
                if ov_s > ov_e:
                    continue
                end_str = "" if ov_e == OPEN_END else ov_e.isoformat()
                ov_e_for_days = CAP_END if ov_e == OPEN_END else ov_e
                days = (ov_e_for_days - ov_s).days + 1
                mids = sorted([ri["manager_id"], rj["manager_id"]])
                intervals.append({
                    "start": ov_s.isoformat(),
                    "end": end_str,
                    "manager_ids": mids,
                    "days_in_overlap": days,
                })
        if intervals:
            intervals.sort(key=lambda x: (x["start"], x["end"], x["manager_ids"]))
            out.append({
                "employee_id": eid,
                "employee_name": nbi.get(eid, ""),
                "overlap_intervals": intervals,
                "total_overlap_days": sum(iv["days_in_overlap"] for iv in intervals),
            })
    out.sort(key=lambda r: (-r["total_overlap_days"], r["employee_name"]))
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    matrix = _q1_matrix(cfg)
    pair_emps: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rec in matrix:
        for iv in rec["overlap_intervals"]:
            pair = (iv["manager_ids"][0], iv["manager_ids"][1])
            pair_emps[pair].add(rec["employee_id"])

    out = []
    for pair, emps in pair_emps.items():
        out.append({
            "manager_ids": list(pair),
            "manager_names": [nbi.get(pair[0], ""), nbi.get(pair[1], "")],
            "distinct_employee_count": len(emps),
            "employee_ids": sorted(emps),
        })
    out.sort(key=lambda p: (-p["distinct_employee_count"], p["manager_ids"]))
    return {"top_manager_pairs": out[:3]}


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

    tp_ans = answer.get("top_manager_pairs") or []
    tp_exp = expected["top_manager_pairs"]
    checks: dict[str, bool] = {
        "top_pairs_count": len(tp_ans) == len(tp_exp),
    }
    ans_pairs = [tuple(p.get("manager_ids") or []) for p in tp_ans if isinstance(p, dict)]
    exp_pairs = [tuple(p["manager_ids"]) for p in tp_exp]
    checks["pair_set"] = set(ans_pairs) == set(exp_pairs)
    checks["pair_order"] = ans_pairs == exp_pairs

    exp_by_pair = {tuple(p["manager_ids"]): p for p in tp_exp}
    fields_ok = True
    for p in tp_ans:
        if not isinstance(p, dict):
            fields_ok = False; break
        e = exp_by_pair.get(tuple(p.get("manager_ids") or []))
        if e is None:
            continue
        if p.get("distinct_employee_count") != e["distinct_employee_count"]:
            fields_ok = False; break
        if set(p.get("employee_ids") or []) != set(e["employee_ids"]):
            fields_ok = False; break
    checks["pair_fields_match"] = fields_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
