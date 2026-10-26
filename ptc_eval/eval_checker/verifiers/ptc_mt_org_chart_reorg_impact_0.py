"""Verifier for ptc_mt_org_chart_reorg_impact_0.

Q1: chains differ between 2025-12-01 (BEFORE) and 2026-02-01 (AFTER).
Q2: depth_histogram + top_depth + {employee_name, old_parent, new_parent}
    triples at the most common depth.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ._orgchart_helpers import (
    active_manager_on, chain_to_root, name_by_id,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


BEFORE = date(2025, 12, 1)
AFTER = date(2026, 2, 1)


def _chain_names(eid: str, am, nbi):
    return [nbi.get(i, "") for i in chain_to_root(eid, am)]


def _q1_changes(cfg: dict) -> list[dict]:
    history = cfg["reporting_history"]
    nbi = name_by_id(cfg)
    am_b = active_manager_on(history, BEFORE)
    am_a = active_manager_on(history, AFTER)
    common = set(am_b) & set(am_a)
    out = []
    for eid in common:
        cb = _chain_names(eid, am_b, nbi)
        ca = _chain_names(eid, am_a, nbi)
        if cb == ca:
            continue
        i = 0
        while i < min(len(cb), len(ca)) and cb[i] == ca[i]:
            i += 1
        out.append({
            "employee_id": eid,
            "employee_name": nbi.get(eid, ""),
            "divergence_depth": i,
            "chain_before": cb,
            "chain_after": ca,
        })
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    changes = _q1_changes(cfg)
    if not changes:
        return {"depth_histogram": {}, "top_depth": None, "top_depth_changes": []}
    hist: dict[int, int] = defaultdict(int)
    for c in changes:
        hist[c["divergence_depth"]] += 1
    # Most common depth; tiebreak: smaller depth wins.
    top_depth = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    triples = []
    for c in changes:
        if c["divergence_depth"] != top_depth:
            continue
        cb = c["chain_before"]
        ca = c["chain_after"]
        old_p = cb[top_depth] if top_depth < len(cb) else None
        new_p = ca[top_depth] if top_depth < len(ca) else None
        triples.append({
            "employee_name": c["employee_name"],
            "old_parent": old_p,
            "new_parent": new_p,
        })
    triples.sort(key=lambda x: (
        x["old_parent"] or "", x["new_parent"] or "", x["employee_name"]
    ))
    return {
        "depth_histogram": {str(k): v for k, v in hist.items()},
        "top_depth": top_depth,
        "top_depth_changes": triples,
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

    # Histogram: agents may emit either int or str keys; normalize both.
    hist_ans = answer.get("depth_histogram") or {}
    hist_ans_norm = {str(k): v for k, v in hist_ans.items()} if isinstance(hist_ans, dict) else {}
    checks: dict[str, bool] = {
        "histogram_match": hist_ans_norm == expected["depth_histogram"],
        "top_depth_match": answer.get("top_depth") == expected["top_depth"],
    }

    tdc_ans = answer.get("top_depth_changes") or []
    tdc_exp = expected["top_depth_changes"]
    checks["tdc_count"] = len(tdc_ans) == len(tdc_exp)
    ans_keys = [(c.get("employee_name"), c.get("old_parent"), c.get("new_parent"))
                for c in tdc_ans if isinstance(c, dict)]
    exp_keys = [(c["employee_name"], c["old_parent"], c["new_parent"]) for c in tdc_exp]
    checks["tdc_set"] = set(ans_keys) == set(exp_keys)
    checks["tdc_order"] = ans_keys == exp_keys

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
