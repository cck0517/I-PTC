"""Verifier for ptc_mt_calendar_weekly_recurring_0.

Q1: categories with total_events >= 8 across the team.
Q2: top-3 categories by total_events, drilldown of distinct titles per category.
"""
from __future__ import annotations

from collections import defaultdict

from ._calendar_helpers import event_duration_minutes
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _per_category(cfg: dict) -> dict[str, dict]:
    """{category: {events, employees, titles}}."""
    out: dict[str, dict] = defaultdict(lambda: {
        "events": [], "employees": set(), "titles": defaultdict(set),
    })
    for ev in cfg["events"]:
        title = ev["title"]
        cat = title.split(maxsplit=1)[0] if title else ""
        if not cat:
            continue
        d = out[cat]
        d["events"].append(ev)
        d["employees"].add(ev["employee_id"])
        d["titles"][title].add(ev["employee_id"])
    return dict(out)


def _q1_qualifying(cfg: dict) -> list[dict]:
    raw = _per_category(cfg)
    rows = []
    for cat, d in raw.items():
        n = len(d["events"])
        if n < 8:
            continue
        avg = round(sum(event_duration_minutes(e) for e in d["events"]) / n, 1)
        rows.append({
            "category": cat,
            "total_events": n,
            "distinct_employees": len(d["employees"]),
            "distinct_titles": len(d["titles"]),
            "avg_duration_minutes": avg,
            "_titles": d["titles"],
        })
    rows.sort(key=lambda r: (-r["total_events"], r["category"]))
    return rows


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    qual = _q1_qualifying(cfg)
    top3 = qual[:3]
    out = []
    for r in top3:
        titles = []
        for title, emps in r["_titles"].items():
            titles.append({"title": title, "distinct_employees": len(emps)})
        titles.sort(key=lambda t: (-t["distinct_employees"], t["title"]))
        out.append({"category": r["category"], "titles": titles})
    return {"top3_categories_drilldown": out}


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

    d_ans = answer.get("top3_categories_drilldown") or []
    d_exp = expected["top3_categories_drilldown"]
    checks: dict[str, bool] = {
        "count_match": len(d_ans) == len(d_exp),
    }
    ans_cats = [d.get("category") for d in d_ans if isinstance(d, dict)]
    exp_cats = [d["category"] for d in d_exp]
    checks["category_set"] = set(ans_cats) == set(exp_cats)
    checks["category_order"] = ans_cats == exp_cats

    exp_by_cat = {d["category"]: d for d in d_exp}
    titles_ok = True
    for d in d_ans:
        if not isinstance(d, dict):
            titles_ok = False; break
        e = exp_by_cat.get(d.get("category"))
        if e is None:
            continue
        ans_pairs = {
            (t.get("title"), t.get("distinct_employees"))
            for t in (d.get("titles") or []) if isinstance(t, dict)
        }
        exp_pairs = {(t["title"], t["distinct_employees"]) for t in e["titles"]}
        if ans_pairs != exp_pairs:
            titles_ok = False; break
    checks["titles_match"] = titles_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
