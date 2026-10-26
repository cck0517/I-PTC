"""Verifier for ptc_mt_org_chart_span_of_control_0.

Q1: top-5 managers by direct report count + bucket histogram on 2026-01-15.
Q2: stats {mean, median, max, count_ge_10} + next_highest_outside_top5.
"""
from __future__ import annotations

import statistics
from datetime import date

from ._orgchart_helpers import (
    active_manager_on, approx, direct_report_counts, name_by_id,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


SNAPSHOT = date(2026, 1, 15)


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    am = active_manager_on(cfg["reporting_history"], SNAPSHOT)
    counts = direct_report_counts(am)

    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], nbi.get(kv[0], "")),
    )
    top5_ids = {mid for mid, _ in ranked[:5]}

    values = list(counts.values())
    if not values:
        stats = {"mean": 0.0, "median": 0.0, "max": 0, "count_ge_10": 0}
    else:
        stats = {
            "mean": round(sum(values) / len(values), 2),
            "median": round(statistics.median(values), 2),
            "max": max(values),
            "count_ge_10": sum(1 for v in values if v >= 10),
        }

    next_h = None
    for mid, c in ranked:
        if mid not in top5_ids:
            next_h = {
                "manager_id": mid,
                "manager_name": nbi.get(mid, ""),
                "direct_report_count": c,
            }
            break

    return {"stats": stats, "next_highest_outside_top5": next_h}


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

    s_ans = (answer.get("stats") or {}) if isinstance(answer.get("stats"), dict) else {}
    s_exp = expected["stats"]
    checks: dict[str, bool] = {
        "stats_mean":     approx(s_ans.get("mean"),   s_exp["mean"],   tol=0.05),
        "stats_median":   approx(s_ans.get("median"), s_exp["median"], tol=0.05),
        "stats_max":      s_ans.get("max") == s_exp["max"],
        "stats_count_ge_10": s_ans.get("count_ge_10") == s_exp["count_ge_10"],
    }

    nh_ans = answer.get("next_highest_outside_top5")
    nh_exp = expected["next_highest_outside_top5"]
    if nh_exp is None:
        checks["next_highest_correct"] = nh_ans is None
    else:
        ok = (
            isinstance(nh_ans, dict)
            and nh_ans.get("manager_id") == nh_exp["manager_id"]
            and nh_ans.get("direct_report_count") == nh_exp["direct_report_count"]
        )
        checks["next_highest_correct"] = ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
