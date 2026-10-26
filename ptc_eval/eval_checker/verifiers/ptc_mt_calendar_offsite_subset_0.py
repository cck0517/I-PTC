"""Verifier for ptc_mt_calendar_offsite_subset_0.

Q1: Per business day over 3 weeks, who is FULLY FREE 9-17 local.
Q2: Best date + recommended 5-of-8 invitees (highest total level among free).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _events_by_emp(cfg: dict) -> dict[str, list[tuple[datetime, datetime]]]:
    by_id = {e["id"]: ZoneInfo(e["timezone"]) for e in cfg["employees"]}
    out: dict[str, list[tuple[datetime, datetime]]] = {eid: [] for eid in by_id}
    for ev in cfg["events"]:
        eid = ev["employee_id"]
        if eid not in by_id:
            continue
        tz = by_id[eid]
        s = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz).astimezone(ZoneInfo("UTC"))
        e = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz).astimezone(ZoneInfo("UTC"))
        out[eid].append((s, e))
    return out


def _is_free_during_day(emp: dict, day: date, events_utc: list[tuple[datetime, datetime]]) -> bool:
    tz = ZoneInfo(emp["timezone"])
    w9 = datetime(day.year, day.month, day.day, 9, 0, tzinfo=tz).astimezone(ZoneInfo("UTC"))
    w17 = datetime(day.year, day.month, day.day, 17, 0, tzinfo=tz).astimezone(ZoneInfo("UTC"))
    for s, e in events_utc:
        if s < w17 and e > w9:
            return False
    return True


def _compute_free_days(config: dict) -> list[dict]:
    cfg = config["EnterpriseCalendarAPI"]
    week_start = date.fromisoformat(cfg["metadata"]["current_week_start_date"])
    week_end = date.fromisoformat(cfg["metadata"]["current_week_end_date"])
    events = _events_by_emp(cfg)

    days = []
    d = week_start
    while d <= week_end:
        if d.weekday() < 5:  # weekday
            free_ids = []
            for emp in cfg["employees"]:
                if _is_free_during_day(emp, d, events[emp["id"]]):
                    free_ids.append(emp["id"])
            days.append({"date": d.isoformat(), "free_count": len(free_ids),
                          "free_employee_ids": free_ids})
        d += timedelta(days=1)
    days.sort(key=lambda x: (-x["free_count"], x["date"]))
    return days


def _choose_best_date(days: list[dict], emp_levels: dict[str, int]) -> dict:
    """Pick date with highest free_count; break ties by max sum of top-5 levels
    among free employees."""
    def score(d: dict) -> tuple[int, int]:
        lvls = sorted((emp_levels.get(eid, 0) for eid in d["free_employee_ids"]),
                       reverse=True)[:5]
        return d["free_count"], sum(lvls)
    best = max(days, key=score)
    lvl_pairs = sorted(
        [(eid, emp_levels.get(eid, 0)) for eid in best["free_employee_ids"]],
        key=lambda t: (-t[1], t[0]),
    )[:5]
    return {
        "chosen_date": best["date"],
        "free_count": best["free_count"],
        "recommended_ids": [p[0] for p in lvl_pairs],
        "total_level_score": sum(p[1] for p in lvl_pairs),
    }


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object"],
            details={"raw_answer": final_answer},
        )
    cfg = initial_config["EnterpriseCalendarAPI"]
    emp_levels = {e["id"]: e.get("level", 0) for e in cfg["employees"]}
    days = _compute_free_days(initial_config)
    expected = _choose_best_date(days, emp_levels)

    rec_ans = answer.get("recommended_invitees") or []
    rec_ids_ans = [r.get("employee_id") for r in rec_ans if isinstance(r, dict)]

    checks = {
        "chosen_date_correct": answer.get("chosen_date") == expected["chosen_date"],
        "free_count_correct":  answer.get("free_count") == expected["free_count"],
        "invitees_count":      len(rec_ans) == 5,
        "invitees_set":        set(rec_ids_ans) == set(expected["recommended_ids"]),
        "total_level_correct": answer.get("total_level_score") == expected["total_level_score"],
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer,
                 "all_days_top5": days[:5]},
    )
