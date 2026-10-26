"""Verifier for ptc_intermediate_calendar_cross_tz_0.

Finds the earliest 1-hour UTC slot on the hour grid over the next 10
business days starting 2026-04-20 where all 15 teammates are free in
their local 9:00-17:00 working hours and have no conflicting events.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)


START_DATE = date(2026, 4, 20)
NUM_BUSINESS_DAYS = 10


def _cal_cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "EnterpriseCalendarAPIAlt",
        initial_config.get("EnterpriseCalendarAPI", {}),
    )


def _business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _has_conflict(emp: dict, events: list[dict],
                  local_start: datetime, local_end: datetime) -> bool:
    tz = ZoneInfo(emp["timezone"])
    for ev in events:
        if ev["employee_id"] != emp["id"]:
            continue
        es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
        ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
        if local_start < ee and local_end > es:
            return True
    return False


def _compute_ground_truth(cfg: dict) -> dict:
    employees = cfg["employees"]
    events = cfg["events"]
    for day in _business_days(START_DATE, NUM_BUSINESS_DAYS):
        for utc_h in range(0, 24):
            utc_start = datetime(day.year, day.month, day.day, utc_h,
                                 tzinfo=ZoneInfo("UTC"))
            utc_end = utc_start + timedelta(hours=1)
            ok = True
            for emp in employees:
                tz = ZoneInfo(emp["timezone"])
                ls = utc_start.astimezone(tz)
                le = utc_end.astimezone(tz)
                if ls.date() != le.date() or ls.hour < 9 or le.hour > 17:
                    ok = False
                    break
                if _has_conflict(emp, events, ls.replace(tzinfo=tz),
                                 le.replace(tzinfo=tz)):
                    ok = False
                    break
            if ok:
                per_attendee = []
                for emp in employees:
                    tz = ZoneInfo(emp["timezone"])
                    ls = utc_start.astimezone(tz)
                    le = utc_end.astimezone(tz)
                    per_attendee.append({
                        "name": emp["display_name"],
                        "timezone": emp["timezone"],
                        "start_local": ls.strftime("%Y-%m-%dT%H:%M:%S"),
                        "end_local": le.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                return {
                    "start_utc": utc_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_utc": utc_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "per_attendee_local": per_attendee,
                }
    return {}


def _normalize_utc(s: str) -> str:
    return (s or "").replace("Z", "").strip()


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]

    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0, checks={}, details=details,
            errors=["Could not extract JSON object from final answer"],
        )

    cfg = _cal_cfg(initial_config)
    expected = _compute_ground_truth(cfg)
    details["expected"] = expected

    if not expected:
        return VerificationResult(
            passed=False, score=0.0, checks={}, details=details,
            errors=["No valid slot exists in initial_config; check task setup."],
        )

    checks["start_utc_correct"] = (
        _normalize_utc(answer.get("start_utc", ""))
        == _normalize_utc(expected["start_utc"])
    )
    checks["end_utc_correct"] = (
        _normalize_utc(answer.get("end_utc", ""))
        == _normalize_utc(expected["end_utc"])
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks, details=details,
    )
