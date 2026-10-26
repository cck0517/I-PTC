"""Verifier for ptc_mt_calendar_fragmentation_0.

Q1: top-3 most-fragmented + bottom-3 least-fragmented employees.
Q2: summary stats over all 12 employees + middle3 (ranks 5-7 by score desc).
"""
from __future__ import annotations

import statistics

from ._calendar_helpers import employees_by_id, event_duration_minutes
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_employee(cfg: dict) -> list[dict]:
    emps = employees_by_id(cfg)
    counts: dict[str, dict] = {eid: {"events": 0, "minutes": 0} for eid in emps}
    for ev in cfg["events"]:
        d = counts.get(ev["employee_id"])
        if d is None:
            continue
        d["events"] += 1
        d["minutes"] += event_duration_minutes(ev)
    rows = []
    for eid, d in counts.items():
        if d["events"] == 0:
            continue
        hours = round(d["minutes"] / 60, 2)
        score = round(d["events"] / hours, 4) if hours else 0.0
        rows.append({
            "employee_id": eid,
            "display_name": emps[eid]["display_name"],
            "event_count": d["events"],
            "total_meeting_hours": hours,
            "fragmentation_score": score,
        })
    return rows


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    rows = _per_employee(cfg)
    if not rows:
        return {
            "summary": {"mean_score": 0.0, "median_score": 0.0,
                        "max_score": 0.0, "min_score": 0.0},
            "middle3": [],
        }
    scores = [r["fragmentation_score"] for r in rows]
    summary = {
        "mean_score": round(sum(scores) / len(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "max_score": round(max(scores), 4),
        "min_score": round(min(scores), 4),
    }
    ranked = sorted(rows, key=lambda r: (-r["fragmentation_score"], r["display_name"]))
    return {"summary": summary, "middle3": ranked[4:7]}


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

    s_ans = answer.get("summary") or {}
    s_exp = expected["summary"]
    def _close(a, b, tol=0.0005):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False
    checks: dict[str, bool] = {
        "summary_mean":   _close(s_ans.get("mean_score"),   s_exp["mean_score"]),
        "summary_median": _close(s_ans.get("median_score"), s_exp["median_score"]),
        "summary_max":    _close(s_ans.get("max_score"),    s_exp["max_score"]),
        "summary_min":    _close(s_ans.get("min_score"),    s_exp["min_score"]),
    }

    m_ans = answer.get("middle3") or []
    m_exp = expected["middle3"]
    checks["middle3_count"] = len(m_ans) == len(m_exp)
    ans_ids = [m.get("employee_id") for m in m_ans if isinstance(m, dict)]
    exp_ids = [m["employee_id"] for m in m_exp]
    checks["middle3_set"] = set(ans_ids) == set(exp_ids)
    checks["middle3_order"] = ans_ids == exp_ids

    exp_by_id = {m["employee_id"]: m for m in m_exp}
    score_ok = True
    for m in m_ans:
        if not isinstance(m, dict):
            score_ok = False; break
        e = exp_by_id.get(m.get("employee_id"))
        if e is None:
            continue
        if not _close(m.get("fragmentation_score"), e["fragmentation_score"]):
            score_ok = False; break
    checks["middle3_score_match"] = score_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
