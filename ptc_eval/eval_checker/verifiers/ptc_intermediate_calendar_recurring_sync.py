"""Verifier for ptc_intermediate_calendar_recurring_sync_0."""

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


def _busy(emp, events, day, hour) -> bool:
    tz = ZoneInfo(emp["timezone"])
    ls = datetime(day.year, day.month, day.day, hour, tzinfo=tz)
    le = ls + timedelta(hours=1)
    for ev in events:
        if ev["employee_id"] != emp["id"]:
            continue
        es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
        ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
        if ls < ee and le > es:
            return True
    return False


def _compute_ground_truth(cfg: dict) -> dict:
    employees = cfg["employees"]
    events = cfg["events"]
    for wd in range(5):
        for hour in range(9, 17):
            ok = True
            occs = []
            for w in range(NUM_WEEKS):
                d = WEEK_START + timedelta(days=7 * w + wd)
                occs.append(d.isoformat())
                for emp in employees:
                    if _busy(emp, events, d, hour):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return {
                    "weekday": WEEKDAY_NAMES[wd],
                    "hour": hour,
                    "occurrences": occs,
                }
    return {}


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
    if not expected:
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["No valid recurring slot in initial_config"])
    checks["weekday_correct"] = str(answer.get("weekday")) == expected["weekday"]
    try:
        checks["hour_correct"] = int(answer.get("hour")) == expected["hour"]
    except (TypeError, ValueError):
        checks["hour_correct"] = False
    got = answer.get("occurrences") or []
    if not isinstance(got, list):
        got = []
    checks["occurrences_correct"] = [str(x) for x in got] == expected["occurrences"]
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
