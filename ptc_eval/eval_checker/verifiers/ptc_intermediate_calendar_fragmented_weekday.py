"""Verifier for ptc_intermediate_calendar_fragmented_weekday_0."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .core import VerificationResult, extract_json_from_text, score_from_checks


WEEK_START = date(2026, 5, 4)
NUM_WEEKS = 4
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "EnterpriseCalendarAPIAlt",
        initial_config.get("EnterpriseCalendarAPI", {}),
    )


def _free_blocks_for_day(emp, events, day) -> int:
    tz = ZoneInfo(emp["timezone"])
    day_start = datetime(day.year, day.month, day.day, 9, tzinfo=tz)
    day_end = datetime(day.year, day.month, day.day, 17, tzinfo=tz)
    intervals = []
    for ev in events:
        if ev["employee_id"] != emp["id"]:
            continue
        es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
        ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
        s = max(es, day_start)
        e = min(ee, day_end)
        if s < e:
            intervals.append((s, e))
    intervals.sort()
    # Merge
    merged: list[tuple] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    # Count free blocks
    blocks = 0
    cursor = day_start
    for s, e in merged:
        if cursor < s:
            blocks += 1
        cursor = max(cursor, e)
    if cursor < day_end:
        blocks += 1
    return blocks


def _compute_ground_truth(cfg: dict) -> dict:
    employees = cfg["employees"]
    events = cfg["events"]
    emp = employees[0]  # single-user task
    best_wd = -1
    best_avg = -1.0
    best_per_week: list[int] = []
    for wd in range(5):
        per_week: list[int] = []
        for w in range(NUM_WEEKS):
            d = WEEK_START + timedelta(days=7 * w + wd)
            per_week.append(_free_blocks_for_day(emp, events, d))
        avg = sum(per_week) / NUM_WEEKS
        if avg > best_avg or (avg == best_avg and wd < best_wd):
            best_wd = wd
            best_avg = avg
            best_per_week = per_week
    return {
        "weekday": WEEKDAY_NAMES[best_wd],
        "avg_free_blocks": round(best_avg, 2),
        "per_week_blocks": best_per_week,
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected"] = expected
    checks["weekday_correct"] = str(answer.get("weekday")) == expected["weekday"]
    try:
        checks["avg_correct"] = abs(float(answer.get("avg_free_blocks")) - expected["avg_free_blocks"]) < 0.05
    except (TypeError, ValueError):
        checks["avg_correct"] = False
    got = answer.get("per_week_blocks") or []
    if not isinstance(got, list):
        got = []
    try:
        checks["per_week_correct"] = [int(x) for x in got] == expected["per_week_blocks"]
    except (TypeError, ValueError):
        checks["per_week_correct"] = False
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
